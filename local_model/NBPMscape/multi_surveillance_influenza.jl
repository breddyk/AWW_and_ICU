# ============================================================================
# multi_surveillance.jl
# ============================================================================

using Pkg
const _PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(_PROJECT_DIR)

using NBPMscape
using DataFrames
using Statistics
using Distributions
using CSV
using Distributed
using Dates
using Printf

let n = haskey(ENV, "SLURM_CPUS_PER_TASK") ?
    min(parse(Int, ENV["SLURM_CPUS_PER_TASK"]) - 1, 24) : 24
addprocs(n)
end

@everywhere using Pkg
@everywhere Pkg.activate($_PROJECT_DIR)
@everywhere using NBPMscape
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Distributions
@everywhere using CSV

@everywhere const CONFIG_REL_PATH = "config/outbreak_params_influenza_like.yaml"
@everywhere const CONFIG_ABS_PATH = joinpath(pkgdir(NBPMscape), CONFIG_REL_PATH)
@everywhere const CONFIG_DATA     = NBPMscape.load_config(CONFIG_ABS_PATH)

@everywhere const INFECTIVITY_FOR_R0_INFLUENZA = Dict(
    1.5 => 9.861059848839812,
    2.0 => 13.509938212723691,
    2.5 => 18.574936904423197,
    3.0 => 22.985815931675493,
)

@everywhere function apply_yaml_scalars(P::NamedTuple, config::Dict)
    haskey(config, "parameters") || return P
    params = config["parameters"]
    P_dict = Dict{Symbol, Any}(pairs(P))
    name_map = Dict(
        "rho_hosp"         => :ρ_hosp,
        "rho_asymptomatic" => :ρ_asymptomatic,
        "mu"               => :μ,
        "omega"            => :ω,
    )
    for (key, value) in params
        sym = get(name_map, key, Symbol(key))
        if haskey(P_dict, sym)
            if sym === :dowcont
                value = tuple(value...)
            end
            P_dict[sym] = value
        end
    end
    return NamedTuple(P_dict)
end

@everywhere const P_FROM_CONFIG = let
    p = apply_yaml_scalars(NBPMscape.P, CONFIG_DATA)
    cfg = get(CONFIG_DATA, "parameters", Dict())
    default_adult = p.ed_ari_destinations_adult.proportion_of_attendances
    default_child = p.ed_ari_destinations_child.proportion_of_attendances
    ed_adult = DataFrame(
        destination               = [:discharged, :short_stay, :longer_stay],
        proportion_of_attendances = [
            get(cfg, "ed_ari_destinations_adult_p_discharged",  default_adult[1]),
            get(cfg, "ed_ari_destinations_adult_p_short_stay",  default_adult[2]),
            get(cfg, "ed_ari_destinations_adult_p_longer_stay", default_adult[3]),
        ],
    )
    ed_child = DataFrame(
        destination               = [:discharged, :short_stay, :longer_stay],
        proportion_of_attendances = [
            get(cfg, "ed_ari_destinations_child_p_discharged",  default_child[1]),
            get(cfg, "ed_ari_destinations_child_p_short_stay",  default_child[2]),
            get(cfg, "ed_ari_destinations_child_p_longer_stay", default_child[3]),
        ],
    )
    merge(p, (ed_ari_destinations_adult = ed_adult,
              ed_ari_destinations_child = ed_child))
end

@everywhere const HARISS_SITES_FROM_CONFIG = let
    rel = get(get(CONFIG_DATA, "parameters", Dict()),
              "hariss_nhs_trust_sampling_sites_file", nothing)
    if rel === nothing
        NBPMscape.P.hariss_nhs_trust_sampling_sites
    else
        full = joinpath(pkgdir(NBPMscape), rel)
        isfile(full) ? CSV.read(full, DataFrame) : NBPMscape.P.hariss_nhs_trust_sampling_sites
    end
end

@everywhere function sample_poisson_direct(lambda::Float64)
    return Float64(rand(Poisson(lambda)))
end

@everywhere function sample_daily_imports_poisson(
    country_data::DataFrame,
    sample_id::Int,
    import_column::Symbol,
)
    n_times = nrow(country_data)
    daily_imports = zeros(Int, n_times)
    total_imports = 0
    for (idx, row) in enumerate(eachrow(country_data))
        has_mean = !ismissing(row[import_column]) &&
                   isfinite(row[import_column])   &&
                   !isnan(row[import_column])      &&
                   row[import_column] > 0
        if has_mean
            sampled = round(Int, sample_poisson_direct(row[import_column]))
            daily_imports[idx] = sampled
            total_imports += sampled
        end
    end
    return daily_imports, total_imports
end

# ============================================================================
# Helper: call secondary_care_td and return first finite SC_TD
# Defined at top level with @everywhere so workers have it in their world age
# at startup — avoids Julia 1.12 world age errors when called from pmap.
# ============================================================================

@everywhere function _run_hariss(
    base_params, hariss_df, bg_cache,
    n_samples, phl_dow, turnaround_time,
    country_name, sample_id, max_observation_time,
)
    try
        result = redirect_stdout(devnull) do
            NBPMscape.secondary_care_td(;
                p                                = base_params,
                sims                             = [hariss_df],
                pathogen_type                    = P_FROM_CONFIG.pathogen_type,
                initial_dow                      = P_FROM_CONFIG.initial_dow,
                hariss_courier_to_analysis       = P_FROM_CONFIG.hariss_courier_to_analysis,
                hariss_turnaround_time           = [turnaround_time, turnaround_time + 1e-6],
                n_hosp_samples_per_week          = n_samples,
                sample_allocation                = P_FROM_CONFIG.sample_allocation,
                sample_proportion_adult          = P_FROM_CONFIG.sample_proportion_adult,
                hariss_nhs_trust_sampling_sites  = HARISS_SITES_FROM_CONFIG,
                weight_samples_by                = P_FROM_CONFIG.weight_samples_by,
                phl_collection_dow               = Vector{Int64}(phl_dow),
                phl_collection_time              = Float64(P_FROM_CONFIG.phl_collection_time),
                hosp_to_phl_cutoff_time_relative = P_FROM_CONFIG.hosp_to_phl_cutoff_time_relative,
                swab_time_mode                   = P_FROM_CONFIG.swab_time_mode,
                swab_proportion_at_48h           = P_FROM_CONFIG.swab_proportion_at_48h,
                proportion_hosp_swabbed          = P_FROM_CONFIG.proportion_hosp_swabbed,
                only_sample_before_death         = P_FROM_CONFIG.hariss_only_sample_before_death,
                ed_discharge_limit               = Float64(P_FROM_CONFIG.tdischarge_ed_upper_limit),
                hosp_short_stay_limit            = Float64(P_FROM_CONFIG.tdischarge_hosp_short_stay_upper_limit),
                hosp_ari_admissions              = Int(P_FROM_CONFIG.hosp_ari_admissions),
                hosp_ari_admissions_adult_p      = Float64(P_FROM_CONFIG.hosp_ari_admissions_adult_p),
                hosp_ari_admissions_child_p      = Float64(P_FROM_CONFIG.hosp_ari_admissions_child_p),
                ed_ari_destinations_adult        = P_FROM_CONFIG.ed_ari_destinations_adult,
                ed_ari_destinations_child        = P_FROM_CONFIG.ed_ari_destinations_child,
                precomputed_ari_bg               = bg_cache,
            )
        end
        if nrow(result) > 0 && :SC_TD in propertynames(result)
            finite_tds = filter(x -> !ismissing(x) && isfinite(x), result.SC_TD)
            if !isempty(finite_tds)
                td_min = minimum(finite_tds)
                return td_min <= max_observation_time ? td_min : Inf
            end
        end
    catch err
        @warn "HARISS call failed" country=country_name sample=sample_id err=err
    end
    return Inf
end

# ============================================================================
# Core per-sample simulation
# ============================================================================

@everywhere function simulate_multitype_sample(
    country_data::DataFrame,
    country_name::String,
    sample_id::Int,
    R0::Float64,
    mean_generation_time::Float64,
    icu_sampling_proportion::Float64,
    p_det::Float64,
    false_positive_rate::Float64,
    max_observation_time::Float64,
    hariss_bg_cache,
    hariss_bg_cache_enhanced;
    mean_infectious_period::Float64          = 0.99,
    turnaround_time::Float64                 = 3.0,
    n_hosp_samples_per_week::Int             = Int(P_FROM_CONFIG.n_hosp_samples_per_week),
    n_hosp_samples_per_week_enhanced::Int    = 1200,
    phl_collection_dow_baseline::Vector{Int} = [2, 5],
    phl_collection_dow_enhanced::Vector{Int} = [2, 3, 4, 5, 6],
    max_cases::Int                           = 10000,
)
    infectious_period = mean_infectious_period
    latent_period     = mean_generation_time - 0.5 * infectious_period
    latent_period < 0 && error("latent_period < 0 for gen_time=$mean_generation_time")

    fixed_shape = 1000.0

    base_params = merge(P_FROM_CONFIG, (
        infectivity      = INFECTIVITY_FOR_R0_INFLUENZA[R0],
        latent_scale     = latent_period / fixed_shape,
        infectious_scale = infectious_period / fixed_shape,
        infectious_shape = fixed_shape,
        latent_shape     = fixed_shape,
        importrate       = 0.0,
        turnaroundtime   = turnaround_time,
    ))
    icu_params = merge(base_params, (p_sampled_icu = icu_sampling_proportion,))
    infectious_params = merge(base_params, (
        latent_scale     = 1e-6,
        infectious_scale = (infectious_period / 2.0) / fixed_shape,
        latent_shape     = fixed_shape,
    ))

    latent_import_counts,     total_latent     = sample_daily_imports_poisson(country_data, sample_id, :daily_latent_imports)
    infectious_import_counts, total_infectious = sample_daily_imports_poisson(country_data, sample_id, :daily_infectious_imports)
    detectable_import_counts, total_detectable = sample_daily_imports_poisson(country_data, sample_id, :daily_detectable_imports)

    # ── Lean accumulator vectors ──────────────────────────────────────────────
    local_tinf          = Float64[]
    icu_ticu            = Float64[]
    icu_trecovered      = Float64[]
    hariss_pid          = String[]
    hariss_tinf         = Float64[]
    hariss_tgp          = Float64[]
    hariss_ted          = Float64[]
    hariss_thospital    = Float64[]
    hariss_ticu         = Float64[]
    hariss_trecovered   = Float64[]
    hariss_tdeceased    = Float64[]
    hariss_severity     = Symbol[]
    hariss_iscommuter   = Bool[]
    hariss_homeregion   = String[]
    hariss_simid        = String[]
    hariss_tstepdown    = Float64[]
    hariss_tdischarge   = Float64[]
    hariss_fatal        = Bool[]
    hariss_infectee_age = Int8[]
    hariss_importedinf  = Bool[]

    # ── AWW state ─────────────────────────────────────────────────────────────
    aww_ctt1_detected         = false
    first_aww_ctt1_time       = Inf
    aww_ctt2_detected         = false
    first_aww_ctt2_time       = Inf
    prev_aww_positive         = false
    aww_escalation_triggered  = false
    first_aww_escalation_day  = Inf
    first_aww_escalation_time = Inf

    # ── Main per-day loop ─────────────────────────────────────────────────────
    for (idx, row) in enumerate(eachrow(country_data))
        time = row.time
        time >= max_observation_time && break

        daily_detectable_count = detectable_import_counts[idx]
        p_true        = daily_detectable_count > 0 ?
                            1.0 - (1.0 - p_det)^daily_detectable_count : 0.0
        true_positive = rand() < p_true

        if !aww_ctt1_detected && true_positive
            aww_ctt1_detected   = true
            first_aww_ctt1_time = time + turnaround_time
        end

        reported_positive = true_positive || (!true_positive && rand() < false_positive_rate)

        if reported_positive
            if prev_aww_positive && !aww_ctt2_detected
                aww_ctt2_detected   = true
                first_aww_ctt2_time = time + turnaround_time
            end
            prev_aww_positive = true
        else
            prev_aww_positive = false
        end

        if !aww_escalation_triggered && reported_positive
            aww_escalation_triggered  = true
            first_aww_escalation_day  = time
            first_aww_escalation_time = time + turnaround_time
        end

        for (import_counts, params, t0_offset) in (
                (latent_import_counts,     base_params,       -latent_period / 2.0),
                (infectious_import_counts, infectious_params,  0.0),
            )
            n = import_counts[idx]
            n == 0 && continue
            t0 = Float64(time) + t0_offset
            for _ in 1:n
                results = NBPMscape.simtree(params,
                    initialtime    = t0,
                    maxtime        = max_observation_time,
                    maxgenerations = 100,
                    initialcontact = :G,
                    max_cases      = max_cases,
                )
                G = results.G
                nrow(G) == 0 && continue
                for gr in eachrow(G)
                    gr.generation == 0 && continue
                    push!(local_tinf, gr.tinf)
                    if isfinite(gr.ticu) && isfinite(gr.trecovered)
                        push!(icu_ticu,       gr.ticu)
                        push!(icu_trecovered, gr.trecovered)
                    end
                    if isfinite(gr.ted) || isfinite(gr.thospital)
                        push!(hariss_pid,          gr.pid)
                        push!(hariss_tinf,         gr.tinf)
                        push!(hariss_tgp,          gr.tgp)
                        push!(hariss_ted,          gr.ted)
                        push!(hariss_thospital,    gr.thospital)
                        push!(hariss_ticu,         gr.ticu)
                        push!(hariss_trecovered,   gr.trecovered)
                        push!(hariss_tdeceased,    gr.tdeceased)
                        push!(hariss_severity,     gr.severity)
                        push!(hariss_iscommuter,   gr.iscommuter)
                        push!(hariss_homeregion,   gr.homeregion)
                        push!(hariss_simid,        gr.simid)
                        push!(hariss_tstepdown,    gr.tstepdown)
                        push!(hariss_tdischarge,   gr.tdischarge)
                        push!(hariss_fatal,        gr.fatal)
                        push!(hariss_infectee_age, gr.infectee_age)
                        push!(hariss_importedinf,  gr.importedinfection)
                    end
                end
            end
        end
    end

    # ── Baseline ICU ──────────────────────────────────────────────────────────
    icu_detected   = false
    first_icu_time = Inf
    if !isempty(icu_ticu)
        icu_fo      = (G = DataFrame(ticu = icu_ticu, trecovered = icu_trecovered),)
        icu_sampled = NBPMscape.sampleforest(icu_fo, icu_params)
        finite_reports = filter(isfinite, icu_sampled.treport)
        if !isempty(finite_reports)
            icu_detected   = true
            first_icu_time = minimum(finite_reports)
        end
    end

    # ── Build hariss_df once — reused for baseline and triggered ─────────────
    hariss_df = nothing
    if !isempty(hariss_pid)
        hariss_df = DataFrame(
            pid               = hariss_pid,
            tinf              = hariss_tinf,
            tgp               = hariss_tgp,
            ted               = hariss_ted,
            thospital         = hariss_thospital,
            ticu              = hariss_ticu,
            tstepdown         = hariss_tstepdown,
            tdischarge        = hariss_tdischarge,
            trecovered        = hariss_trecovered,
            tdeceased         = hariss_tdeceased,
            severity          = hariss_severity,
            fatal             = hariss_fatal,
            iscommuter        = hariss_iscommuter,
            homeregion        = hariss_homeregion,
            simid             = hariss_simid,
            infectee_age      = hariss_infectee_age,
            importedinfection = hariss_importedinf,
        )
    end

    # ── Baseline HARISS ───────────────────────────────────────────────────────
    hariss_detected   = false
    first_hariss_time = Inf
    if hariss_df !== nothing && hariss_bg_cache !== nothing
        first_hariss_time = _run_hariss(base_params, hariss_df, hariss_bg_cache,
                                        n_hosp_samples_per_week,
                                        phl_collection_dow_baseline,
                                        turnaround_time,
                                        country_name, sample_id, max_observation_time)
        hariss_detected = isfinite(first_hariss_time)
    end

    # ── AWW-triggered enhanced HARISS ─────────────────────────────────────────
    hariss_triggered_detected   = false
    first_hariss_triggered_time = Inf
    aww_escalation_lag          = NaN

    if aww_escalation_triggered && hariss_df !== nothing && hariss_bg_cache_enhanced !== nothing
        td_enhanced = _run_hariss(base_params, hariss_df, hariss_bg_cache_enhanced,
                                  n_hosp_samples_per_week_enhanced,
                                  phl_collection_dow_enhanced,
                                  turnaround_time,
                                  country_name, sample_id, max_observation_time)
        if isfinite(td_enhanced) && td_enhanced > first_aww_escalation_time
            hariss_triggered_detected   = true
            first_hariss_triggered_time = td_enhanced
            aww_escalation_lag          = first_hariss_triggered_time - first_aww_escalation_time
        end
    end

    # ── Local case counts at detection ───────────────────────────────────────
    icu_local_cases      = (icu_detected      && isfinite(first_icu_time))      ? Float64(count(<=(first_icu_time),      local_tinf)) : NaN
    hariss_local_cases   = (hariss_detected   && isfinite(first_hariss_time))   ? Float64(count(<=(first_hariss_time),   local_tinf)) : NaN
    aww_ctt1_local_cases = (aww_ctt1_detected && isfinite(first_aww_ctt1_time)) ? Float64(count(<=(first_aww_ctt1_time), local_tinf)) : NaN
    aww_ctt2_local_cases = (aww_ctt2_detected && isfinite(first_aww_ctt2_time)) ? Float64(count(<=(first_aww_ctt2_time), local_tinf)) : NaN
    aww_esc_local_cases  = (aww_escalation_triggered && isfinite(first_aww_escalation_time)) ?
        Float64(count(<=(first_aww_escalation_time), local_tinf)) : NaN
    hariss_trig_local    = (hariss_triggered_detected && isfinite(first_hariss_triggered_time)) ?
        Float64(count(<=(first_hariss_triggered_time), local_tinf)) : NaN

    # ── Earliest-detection summaries ─────────────────────────────────────────
    t_icu      = icu_detected             && isfinite(first_icu_time)             ? Float64(first_icu_time)             : Inf
    t_har      = hariss_detected          && isfinite(first_hariss_time)          ? Float64(first_hariss_time)          : Inf
    t_ctt1     = aww_ctt1_detected        && isfinite(first_aww_ctt1_time)        ? Float64(first_aww_ctt1_time)        : Inf
    t_ctt2     = aww_ctt2_detected        && isfinite(first_aww_ctt2_time)        ? Float64(first_aww_ctt2_time)        : Inf
    t_esc      = aww_escalation_triggered && isfinite(first_aww_escalation_time)  ? Float64(first_aww_escalation_time)  : Inf
    t_har_trig = hariss_triggered_detected && isfinite(first_hariss_triggered_time) ?
        Float64(first_hariss_triggered_time) : Inf

    function _earliest_type(t_aww, t_icu, t_har)
        t_min = min(t_aww, t_icu, t_har)
        t_min == Inf                                        ? "" :
        t_min == t_aww && t_aww <= t_icu && t_aww <= t_har ? "AWW" :
        t_min == t_icu && t_icu <= t_har                    ? "ICU" : "HARISS"
    end

    function _earliest_type_triggered(t_aww_esc, t_icu, t_har_trig)
        t_min = min(t_aww_esc, t_icu, t_har_trig)
        t_min == Inf       ? "" :
        t_min == t_aww_esc ? "AWW_escalation" :
        t_min == t_icu     ? "ICU" : "HARISS_enhanced"
    end

    t_min_ctt1      = min(t_ctt1, t_icu, t_har)
    t_min_ctt2      = min(t_ctt2, t_icu, t_har)
    t_min_triggered = aww_escalation_triggered ? min(t_esc, t_har_trig, t_icu) : Inf

    return (
        sample_id                            = sample_id,
        country                              = country_name,
        R0                                   = R0,
        gen_time                             = mean_generation_time,
        p_det                                = p_det,
        false_positive_rate                  = false_positive_rate,
        icu_detection_time                   = first_icu_time,
        icu_local_cases                      = icu_local_cases,
        hariss_detection_time                = first_hariss_time,
        hariss_local_cases                   = hariss_local_cases,
        aww_ctt1_detection_time              = first_aww_ctt1_time,
        aww_ctt1_local_cases                 = aww_ctt1_local_cases,
        aww_ctt2_detection_time              = first_aww_ctt2_time,
        aww_ctt2_local_cases                 = aww_ctt2_local_cases,
        aww_escalation_detection_time        = first_aww_escalation_time,
        aww_escalation_local_cases           = aww_esc_local_cases,
        hariss_triggered_detection_time      = first_hariss_triggered_time,
        hariss_triggered_local_cases         = hariss_trig_local,
        aww_to_hariss_escalation_lag         = aww_escalation_lag,
        earliest_detection_time_ctt1         = t_min_ctt1 == Inf ? NaN : t_min_ctt1,
        earliest_surveillance_type_ctt1      = _earliest_type(t_ctt1, t_icu, t_har),
        earliest_detection_time_ctt2         = t_min_ctt2 == Inf ? NaN : t_min_ctt2,
        earliest_surveillance_type_ctt2      = _earliest_type(t_ctt2, t_icu, t_har),
        earliest_detection_time_triggered    = t_min_triggered == Inf ? NaN : t_min_triggered,
        earliest_surveillance_type_triggered = _earliest_type_triggered(t_esc, t_icu, t_har_trig),
        total_latent                         = Float64(total_latent),
        total_infectious                     = Float64(total_infectious),
        total_detectable                     = Float64(total_detectable),
    )
end

# ============================================================================
# Driver
# ============================================================================

function run_multitype_comparison(;
    csv_path::String,
    output_path::String,
    num_samples::Int,
    turnaround_time::Float64,
    max_detection_time_threshold::Float64,
    extra_time::Float64,
    icu_sampling_proportion::Float64,
    n_hosp_samples_per_week::Int,
    n_hosp_samples_per_week_enhanced::Int,
    phl_collection_dow_baseline::Vector{Int},
    phl_collection_dow_enhanced::Vector{Int},
    R0::Float64,
    gen_time::Float64,
    base_pdet::Float64,
    sampling_fraction::Float64,
    country::String,
    false_positive_rate::Float64,
    max_cases::Int = 10000,
)
    p_det = base_pdet * sampling_fraction
    println("="^80)
    println("MULTITYPE COMPARISON (single scenario)")
    println("="^80)
    println("Country: $country | R0=$R0 | gen_time=$gen_time")
    println("AWW: base_pdet=$base_pdet × sampling_fraction=$sampling_fraction => p_det=$p_det")
    println("AWW false positive rate: $false_positive_rate")
    println("  CTT=1: first true positive (FPR=0)")
    println("  CTT=2: two consecutive reported positives (FPR=$false_positive_rate)")
    println("  Escalation: first reported positive (FPR=$false_positive_rate)")
    println("ICU sampling proportion: $(icu_sampling_proportion*100)%")
    println("HARISS baseline: $n_hosp_samples_per_week samples/week, PHL DOW: $phl_collection_dow_baseline")
    println("HARISS enhanced: $n_hosp_samples_per_week_enhanced samples/week, PHL DOW: $phl_collection_dow_enhanced")
    println("Max cases per tree: $max_cases")
    println("Samples: $num_samples | Workers: $(nworkers())")
    println("="^80)

    merged_data = CSV.read(csv_path, DataFrame)
    _country_eq(r, c) = string(r.outbreak_country) == string(c)

    param_row = filter(
        r -> Float64(r.R0) == R0 &&
             Float64(r.generation_time) == gen_time &&
             _country_eq(r, country),
        merged_data,
    )
    nrow(param_row) == 0 && error("No rows in CSV for country=$country R0=$R0 gen_time=$gen_time")

    mdts = unique(collect(skipmissing(param_row.mean_detection_time)))
    isempty(mdts) && error("mean_detection_time missing for all matching rows")
    length(mdts) != 1 &&
        error("Expected single mean_detection_time; got $(length(mdts)) distinct values")
    mean_det_time = Float64(first(mdts))
    (isnan(mean_det_time) || mean_det_time > max_detection_time_threshold) &&
        error("Invalid mean_detection_time=$mean_det_time")

    max_obs_time = mean_det_time + extra_time
    country_data = filter(
        r -> Float64(r.R0) == R0 &&
             Float64(r.generation_time) == gen_time &&
             _country_eq(r, country),
        merged_data,
    )
    sort!(country_data, :time)
    country_trimmed = filter(r -> r.time <= max_obs_time, country_data)
    nrow(country_trimmed) == 0 && error("No country data within observation window")

    out_dir = dirname(output_path)
    !isempty(out_dir) && !isdir(out_dir) && mkpath(out_dir)

    println("\nPre-building HARISS ARI background caches (baseline + enhanced, shared patient pool)...")
    flush(stdout)

    shared_kwargs = (
        sample_allocation               = P_FROM_CONFIG.sample_allocation,
        sample_proportion_adult         = P_FROM_CONFIG.sample_proportion_adult,
        hariss_nhs_trust_sampling_sites = HARISS_SITES_FROM_CONFIG,
        weight_samples_by               = P_FROM_CONFIG.weight_samples_by,
        swab_time_mode                  = P_FROM_CONFIG.swab_time_mode,
        swab_proportion_at_48h          = P_FROM_CONFIG.swab_proportion_at_48h,
        proportion_hosp_swabbed         = P_FROM_CONFIG.proportion_hosp_swabbed,
        ed_discharge_limit              = Float64(P_FROM_CONFIG.tdischarge_ed_upper_limit),
        hosp_short_stay_limit           = Float64(P_FROM_CONFIG.tdischarge_hosp_short_stay_upper_limit),
        hosp_ari_admissions             = Int(P_FROM_CONFIG.hosp_ari_admissions),
        hosp_ari_admissions_adult_p     = Float64(P_FROM_CONFIG.hosp_ari_admissions_adult_p),
        hosp_ari_admissions_child_p     = Float64(P_FROM_CONFIG.hosp_ari_admissions_child_p),
        ed_ari_destinations_adult       = P_FROM_CONFIG.ed_ari_destinations_adult,
        ed_ari_destinations_child       = P_FROM_CONFIG.ed_ari_destinations_child,
    )

    t_bg1 = @elapsed hariss_bg_cache = NBPMscape.build_ari_background(;
        max_observation_time    = Float64(max_obs_time),
        n_hosp_samples_per_week = n_hosp_samples_per_week,
        shared_kwargs...
    )

    t_bg2 = @elapsed hariss_bg_cache_enhanced = NBPMscape.build_ari_background(;
        max_observation_time    = Float64(max_obs_time),
        n_hosp_samples_per_week = n_hosp_samples_per_week_enhanced,
        prebuilt_bg             = hariss_bg_cache,
        shared_kwargs...
    )

    @printf("  baseline done in %.1fs, enhanced done in %.1fs\n", t_bg1, t_bg2)
    println("  Shared background patient pool: $(nrow(hariss_bg_cache.bg_df)) background swabs")
    println("  Baseline phl_sample_targets: $(sum(hariss_bg_cache.phl_sample_targets.sample_target_per_week)) samples/week total")
    println("  Enhanced phl_sample_targets: $(sum(hariss_bg_cache_enhanced.phl_sample_targets.sample_target_per_week)) samples/week total")
    println("max_observation_time = $(round(max_obs_time, digits=1)) days")
    println("\nDispatching $num_samples samples to $(nworkers()) workers...")
    flush(stdout)

    t_start = time()
    pool    = CachingPool(workers())
    all_rows = pmap(pool, 1:num_samples) do sample_id
        try
            res = simulate_multitype_sample(
                country_trimmed,
                country,
                sample_id,
                R0,
                gen_time,
                icu_sampling_proportion,
                p_det,
                false_positive_rate,
                max_obs_time,
                hariss_bg_cache,
                hariss_bg_cache_enhanced;
                turnaround_time                  = turnaround_time,
                n_hosp_samples_per_week          = n_hosp_samples_per_week,
                n_hosp_samples_per_week_enhanced = n_hosp_samples_per_week_enhanced,
                phl_collection_dow_baseline      = phl_collection_dow_baseline,
                phl_collection_dow_enhanced      = phl_collection_dow_enhanced,
                max_cases                        = max_cases,
            )
            icu_str  = isfinite(res.icu_detection_time) ?
                string(round(res.icu_detection_time,  digits=1)) * "d" : "—"
            har_str  = isfinite(res.hariss_detection_time) ?
                string(round(res.hariss_detection_time, digits=1)) * "d" : "—"
            esc_str  = isfinite(res.aww_escalation_detection_time) ?
                string(round(res.aww_escalation_detection_time, digits=1)) * "d" : "—"
            trig_str = isfinite(res.hariss_triggered_detection_time) ?
                string(round(res.hariss_triggered_detection_time, digits=1)) * "d" : "—"
            lag_str  = !isnan(res.aww_to_hariss_escalation_lag) ?
                string(round(res.aww_to_hariss_escalation_lag, digits=1)) * "d" : "—"
            println("  [$sample_id/$num_samples] ICU=$icu_str HARISS=$har_str " *
                    "AWW_esc=$esc_str HARISS_trig=$trig_str lag=$lag_str")
            flush(stdout)
            return res
        catch e
            @warn "Sample $sample_id failed" err=e
            return nothing
        end
    end
    clear!(pool)

    elapsed = round(time() - t_start, digits=1)
    valid   = filter(!isnothing, all_rows)
    println("\n$(length(valid))/$num_samples samples completed in $(elapsed)s")

    df = DataFrame(valid)
    CSV.write(output_path, df)
    println("Wrote $(nrow(df)) rows to $output_path")
    return df
end

# ============================================================================
# Scenario parameters
# ============================================================================
const SCENARIO_R0                = 1.5
const SCENARIO_GEN_TIME          = 3.5
const SCENARIO_BASE_PDET         = 0.16
const SCENARIO_SAMPLING_FRACTION = 0.01 # Change to 5pct for comparison
const SCENARIO_COUNTRY           = "Switzerland"
const AWW_FALSE_POSITIVE_RATE    = 0.04

const N_HOSP_SAMPLES_BASELINE    = Int(P_FROM_CONFIG.n_hosp_samples_per_week)  # 600 from YAML
const N_HOSP_SAMPLES_ENHANCED    = 1200
const PHL_COLLECTION_BASELINE    = [2, 5]
const PHL_COLLECTION_ENHANCED    = [2, 3, 4, 5, 6]

input_csv_path  = "global_model/pgfgleam/all_results/global/daily_imports_influenza.csv"
output_csv_path = "global_model/pgfgleam/all_results/local/multi_influenza_1pct.csv"

run_multitype_comparison(;
    csv_path                         = input_csv_path,
    output_path                      = output_csv_path,
    num_samples                      = 250,
    turnaround_time                  = 3.0,
    max_detection_time_threshold     = 100.0,
    extra_time                       = 35.0,
    icu_sampling_proportion          = 0.20,
    n_hosp_samples_per_week          = N_HOSP_SAMPLES_BASELINE,
    n_hosp_samples_per_week_enhanced = N_HOSP_SAMPLES_ENHANCED,
    phl_collection_dow_baseline      = PHL_COLLECTION_BASELINE,
    phl_collection_dow_enhanced      = PHL_COLLECTION_ENHANCED,
    R0                               = SCENARIO_R0,
    gen_time                         = SCENARIO_GEN_TIME,
    base_pdet                        = SCENARIO_BASE_PDET,
    sampling_fraction                = SCENARIO_SAMPLING_FRACTION,
    country                          = SCENARIO_COUNTRY,
    false_positive_rate              = AWW_FALSE_POSITIVE_RATE,
    max_cases                        = 100000,
)

println("\n✓ multi_surveillance_influenza complete!")
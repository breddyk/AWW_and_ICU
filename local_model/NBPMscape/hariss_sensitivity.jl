# ============================================================================
# hariss_sensitivity.jl
# ============================================================================
#
# HARISS sensitivity analysis for SI figures, built on multi_surveillance.jl
# (same infectivity calibration, outbreak growth, ICU sampling, AWW CTT=1/CTT=2
# logic and HARISS call). For a few fixed origin countries it varies:
#
#   (1) Healthcare pathway rates
#         original  - rates in the YAML config (main analysis)
#         corrected - upstream emvolz/NBPMscape@34031ab: competing routes to
#                     ED/hospital (direct vs via GP) halved so the mean
#                     onset-to-admission matches the literature value
#   (2) Share of ED ARI attendances sequenced = n_hosp_samples_per_week / B,
#       where B (`hosp_ari_admissions`) is the weekly ED ARI background
#   (3) Swab ratio admitted : discharged from ED (admitted fixed at 90%),
#       applied identically to pathogen X cases and to the background
#
# Per sample, imports are drawn once and the outbreak is grown once per rate
# set (same random seed for both); every HARISS scenario is evaluated on that
# same outbreak. AWW depends only on imports, so it is computed once.
#
# Swab proportions are applied here: the background is built with everyone
# swabbed and thinned by ED destination per sample, pathogen X ED/hospital
# cases are thinned by whether they were admitted, and secondary_care_td is
# called with proportion_hosp_swabbed = 1.0.
#
# Output: one row per (country, sample, rate_set, B, swab ratio).
# Plot with local_model/scripts/plot_hariss_sensitivity.py.
#
# Settings can be overridden with environment variables, e.g.
#   N_SAMPLES=4 COUNTRIES="Switzerland" julia --project=local_model local_model/NBPMscape/hariss_sensitivity.jl
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
using Random

let n = haskey(ENV, "N_WORKERS")           ? parse(Int, ENV["N_WORKERS"]) :
        haskey(ENV, "SLURM_CPUS_PER_TASK") ? parse(Int, ENV["SLURM_CPUS_PER_TASK"]) - 1 : 24
addprocs(clamp(n, 1, 24))
end

@everywhere using Pkg
@everywhere Pkg.activate($_PROJECT_DIR)
@everywhere using NBPMscape
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Distributions
@everywhere using CSV
@everywhere using Random

@everywhere const CONFIG_REL_PATH = "config/outbreak_params_covid19_like.yaml"
@everywhere const CONFIG_ABS_PATH = joinpath(pkgdir(NBPMscape), CONFIG_REL_PATH)
@everywhere const CONFIG_DATA     = NBPMscape.load_config(CONFIG_ABS_PATH)

@everywhere const INFECTIVITY_FOR_R0_COVID = Dict(
    1.5 => 5.740442959665717,
    2.0 => 7.909172103056187,
    2.5 => 9.763825466244938,
    3.0 => 12.314589631224113,
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

# ── Healthcare pathway rate sets ─────────────────────────────────────────────
# Corrected COVID-like rates from upstream emvolz/NBPMscape@34031ab
# (outbreak_params_covid19_like_20261008.yaml).
@everywhere const RATE_SETS = (
    original  = P_FROM_CONFIG,
    corrected = merge(P_FROM_CONFIG, (
        ed_direct_rate         = 0.5 * (1/5),   # 1/10
        gp_before_hosp_rate    = 0.5 * (1/3),   # 1/6
        hosp_admit_direct_rate = 0.5 * (1/4),   # 1/8
    )),
)

# ============================================================================
# SETTINGS
# ============================================================================

const N_SAMPLES = parse(Int, get(ENV, "N_SAMPLES", "100"))
const COUNTRIES = String.(split(get(ENV, "COUNTRIES", "Switzerland;Spain;Germany;United States"), ";"))

const SCENARIO_R0                = 2.0
const SCENARIO_GEN_TIME          = 4.0
const SCENARIO_BASE_PDET         = 0.16
const SCENARIO_SAMPLING_FRACTION = 0.01
const AWW_FALSE_POSITIVE_RATE    = 0.04
const ICU_SAMPLING_PROPORTION    = 0.20
const TURNAROUND_TIME            = 3.0
const EXTRA_TIME                 = 60.0
const MAX_DETECTION_THRESHOLD    = 100.0
const MAX_CASES                  = parse(Int, get(ENV, "MAX_CASES", "10000"))

const N_HOSP_SAMPLES = Int(P_FROM_CONFIG.n_hosp_samples_per_week)
const MAIN_B         = Int(P_FROM_CONFIG.hosp_ari_admissions)

# Grid axis 1: share of ED ARI attendances sequenced (main analysis first).
# Each share sets B = N_HOSP_SAMPLES / share.
const SEQ_SHARES = [N_HOSP_SAMPLES / MAIN_B, 0.03, 0.02, 0.015, 0.01, 0.005]
const B_VALUES   = [MAIN_B; [round(Int, N_HOSP_SAMPLES / s) for s in SEQ_SHARES[2:end]]]

# Grid axis 2: swab ratio admitted : discharged, admitted fixed at 90%.
const SWAB_SCENARIOS = [
    (name = "ratio_1",   discharged = 0.90, admitted = 0.90),   # main analysis
    (name = "ratio_2",   discharged = 0.45, admitted = 0.90),
    (name = "ratio_4.5", discharged = 0.20, admitted = 0.90),
    (name = "ratio_18",  discharged = 0.05, admitted = 0.90),
]
const MAIN_SWAB = "ratio_1"

input_csv_path  = get(ENV, "INPUT_CSV", "/home/users/breddyk/AWW_and_ICU/global_model/pgfgleam/all_results/global/daily_imports_sensitivity.csv")
output_csv_path = get(ENV, "OUTPUT_CSV", "/home/users/breddyk/AWW_and_ICU/global_model/pgfgleam/all_results/local/hariss_sensitivity_covid19.csv")

# ============================================================================
# HELPERS
# ============================================================================

@everywhere function sample_daily_imports_poisson(country_data::DataFrame, import_column::Symbol)
    daily_imports = zeros(Int, nrow(country_data))
    for (idx, row) in enumerate(eachrow(country_data))
        v = row[import_column]
        if !ismissing(v) && isfinite(v) && v > 0
            daily_imports[idx] = rand(Poisson(v))
        end
    end
    return daily_imports
end

# Keep each background swab with the probability for its ED destination.
@everywhere function thin_background(bg, swab)
    keep_p = ifelse.(bg.bg_df.ED_destination .== :discharged, swab.discharged, swab.admitted)
    bg_df  = bg.bg_df[rand(nrow(bg.bg_df)) .< keep_p, :]
    bg_by_phl = Dict{String, DataFrame}()
    for g in groupby(bg_df, :public_health_laboratory)
        bg_by_phl[String(first(g.public_health_laboratory))] = DataFrame(g)
    end
    return merge(bg, (bg_df = bg_df, bg_by_phl = bg_by_phl))
end

# Keep each pathogen X ED/hospital case with the probability for its route.
@everywhere function thin_cases(hariss_df::DataFrame, swab)
    keep_p = ifelse.(isfinite.(hariss_df.thospital), swab.admitted, swab.discharged)
    return hariss_df[rand(nrow(hariss_df)) .< keep_p, :]
end

@everywhere function _run_hariss(base_params, hariss_df, bg_cache, B, turnaround_time,
                                 country_name, sample_id, max_observation_time)
    nrow(hariss_df) == 0 && return Inf
    try
        result = redirect_stdout(devnull) do
            NBPMscape.secondary_care_td(;
                p                                = base_params,
                sims                             = [hariss_df],
                pathogen_type                    = P_FROM_CONFIG.pathogen_type,
                initial_dow                      = P_FROM_CONFIG.initial_dow,
                hariss_courier_to_analysis       = P_FROM_CONFIG.hariss_courier_to_analysis,
                hariss_turnaround_time           = [turnaround_time, turnaround_time + 1e-6],
                n_hosp_samples_per_week          = Int(P_FROM_CONFIG.n_hosp_samples_per_week),
                sample_allocation                = P_FROM_CONFIG.sample_allocation,
                sample_proportion_adult          = P_FROM_CONFIG.sample_proportion_adult,
                hariss_nhs_trust_sampling_sites  = HARISS_SITES_FROM_CONFIG,
                weight_samples_by                = P_FROM_CONFIG.weight_samples_by,
                phl_collection_dow               = Vector{Int64}(P_FROM_CONFIG.phl_collection_dow),
                phl_collection_time              = Float64(P_FROM_CONFIG.phl_collection_time),
                hosp_to_phl_cutoff_time_relative = P_FROM_CONFIG.hosp_to_phl_cutoff_time_relative,
                swab_time_mode                   = P_FROM_CONFIG.swab_time_mode,
                swab_proportion_at_48h           = P_FROM_CONFIG.swab_proportion_at_48h,
                proportion_hosp_swabbed          = 1.0,   # already thinned
                only_sample_before_death         = P_FROM_CONFIG.hariss_only_sample_before_death,
                ed_discharge_limit               = Float64(P_FROM_CONFIG.tdischarge_ed_upper_limit),
                hosp_short_stay_limit            = Float64(P_FROM_CONFIG.tdischarge_hosp_short_stay_upper_limit),
                hosp_ari_admissions              = B,
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

# Grow the local outbreak from the sampled imports (as in multi_surveillance.jl).
@everywhere function grow_outbreak(country_data, latent_counts, infectious_counts,
                                   base_params, infectious_params, latent_period,
                                   max_observation_time, max_cases)
    local_tinf     = Float64[]
    icu_ticu       = Float64[]
    icu_trecovered = Float64[]
    hosp_rows      = DataFrame[]
    for (idx, row) in enumerate(eachrow(country_data))
        time = row.time
        time >= max_observation_time && break
        for (import_counts, params, t0_offset) in (
                (latent_counts,     base_params,       -latent_period / 2.0),
                (infectious_counts, infectious_params,  0.0),
            )
            n = import_counts[idx]
            n == 0 && continue
            t0 = Float64(time) + t0_offset
            for _ in 1:n
                G = NBPMscape.simtree(params,
                    initialtime    = t0,
                    maxtime        = max_observation_time,
                    maxgenerations = 100,
                    initialcontact = :G,
                    max_cases      = max_cases,
                ).G
                nrow(G) == 0 && continue
                G = G[G.generation .!= 0, :]
                append!(local_tinf, G.tinf)
                icu = isfinite.(G.ticu) .& isfinite.(G.trecovered)
                append!(icu_ticu,       G.ticu[icu])
                append!(icu_trecovered, G.trecovered[icu])
                hosp = isfinite.(G.ted) .| isfinite.(G.thospital)
                any(hosp) && push!(hosp_rows, G[hosp, [:pid, :tinf, :tgp, :ted, :thospital, :ticu,
                    :tstepdown, :tdischarge, :trecovered, :tdeceased, :severity, :fatal,
                    :iscommuter, :homeregion, :simid, :infectee_age, :importedinfection]])
            end
        end
    end
    hariss_df = isempty(hosp_rows) ? DataFrame() : reduce(vcat, hosp_rows)
    return (local_tinf = local_tinf, icu_ticu = icu_ticu,
            icu_trecovered = icu_trecovered, hariss_df = hariss_df)
end

@everywhere cases_by(t, tinf) = isfinite(t) ? Float64(count(<=(t), tinf)) : NaN

# ============================================================================
# PER-SAMPLE SIMULATION
# ============================================================================

@everywhere function simulate_sensitivity_sample(
    country_data::DataFrame, country_name::String, sample_id::Int,
    R0::Float64, mean_generation_time::Float64, icu_sampling_proportion::Float64,
    p_det::Float64, false_positive_rate::Float64, max_observation_time::Float64,
    bg_caches::Dict, b_values, swab_scenarios, main_b::Int, main_swab::String;
    mean_infectious_period::Float64 = 8.0/3.0,
    turnaround_time::Float64        = 3.0,
    max_cases::Int                  = 10000,
)
    seed = hash((country_name, sample_id))
    Random.seed!(seed)

    infectious_period = mean_infectious_period
    latent_period     = mean_generation_time - 0.5 * infectious_period
    latent_period < 0 && error("latent_period < 0 for gen_time=$mean_generation_time")
    fixed_shape = 1000.0

    latent_counts     = sample_daily_imports_poisson(country_data, :daily_latent_imports)
    infectious_counts = sample_daily_imports_poisson(country_data, :daily_infectious_imports)
    detectable_counts = sample_daily_imports_poisson(country_data, :daily_detectable_imports)

    # ── AWW (same logic as multi_surveillance.jl) ────────────────────────────
    first_aww_ctt1_time = Inf
    first_aww_ctt2_time = Inf
    prev_aww_positive   = false
    for (idx, row) in enumerate(eachrow(country_data))
        time = row.time
        time >= max_observation_time && break
        k = detectable_counts[idx]
        p_true        = k > 0 ? 1.0 - (1.0 - p_det)^k : 0.0
        true_positive = rand() < p_true
        if isinf(first_aww_ctt1_time) && true_positive
            first_aww_ctt1_time = time + turnaround_time
        end
        reported_positive = true_positive || rand() < false_positive_rate
        if reported_positive
            if prev_aww_positive && isinf(first_aww_ctt2_time)
                first_aww_ctt2_time = time + turnaround_time
            end
            prev_aww_positive = true
        else
            prev_aww_positive = false
        end
    end

    rows = NamedTuple[]
    for rate_set in (:original, :corrected)
        p = RATE_SETS[rate_set]
        base_params = merge(p, (
            infectivity      = INFECTIVITY_FOR_R0_COVID[R0],
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

        # Same random stream for both rate sets (common random numbers)
        Random.seed!(hash((seed, :tree)))
        ob = grow_outbreak(country_data, latent_counts, infectious_counts,
                           base_params, infectious_params, latent_period,
                           max_observation_time, max_cases)

        first_icu_time = Inf
        if !isempty(ob.icu_ticu)
            icu_sampled = NBPMscape.sampleforest(
                (G = DataFrame(ticu = ob.icu_ticu, trecovered = ob.icu_trecovered),), icu_params)
            finite_reports = filter(isfinite, icu_sampled.treport)
            !isempty(finite_reports) && (first_icu_time = minimum(finite_reports))
        end

        for B in b_values
            rate_set === :original && B != main_b && continue
            scens = rate_set === :corrected ? swab_scenarios :
                    filter(s -> s.name == main_swab, swab_scenarios)
            for swab in scens
                t_har = Inf
                if nrow(ob.hariss_df) > 0
                    t_har = _run_hariss(base_params, thin_cases(ob.hariss_df, swab),
                                        thin_background(bg_caches[B], swab), B,
                                        turnaround_time, country_name, sample_id,
                                        max_observation_time)
                end
                push!(rows, (
                    country            = country_name,
                    R0                 = R0,
                    gen_time           = mean_generation_time,
                    sample_id          = sample_id,
                    rate_set           = String(rate_set),
                    ed_ari_background  = B,
                    seq_share          = P_FROM_CONFIG.n_hosp_samples_per_week / B,
                    swab_scenario      = swab.name,
                    swab_ratio         = swab.admitted / swab.discharged,
                    swab_discharged    = swab.discharged,
                    swab_admitted      = swab.admitted,
                    is_main_analysis   = rate_set === :original && B == main_b && swab.name == main_swab,
                    max_obs_time       = max_observation_time,
                    hariss_time        = t_har,
                    hariss_local_cases = cases_by(t_har, ob.local_tinf),
                    icu_time           = first_icu_time,
                    icu_local_cases    = cases_by(first_icu_time, ob.local_tinf),
                    aww_ctt1_time      = first_aww_ctt1_time,
                    aww_ctt2_time      = first_aww_ctt2_time,
                    n_local_infections = length(ob.local_tinf),
                    n_hariss_cases     = nrow(ob.hariss_df),
                ))
            end
        end
    end
    return rows
end

# ============================================================================
# DRIVER
# ============================================================================

function run_sensitivity()
    p_det = SCENARIO_BASE_PDET * SCENARIO_SAMPLING_FRACTION
    println("="^80)
    println("HARISS SENSITIVITY (R0=$SCENARIO_R0, gen_time=$SCENARIO_GEN_TIME)")
    println("Countries: ", join(COUNTRIES, ", "), " | Samples: $N_SAMPLES | Workers: $(nworkers())")
    println("AWW p_det=$p_det, FPR=$AWW_FALSE_POSITIVE_RATE | ICU sampling $(ICU_SAMPLING_PROPORTION*100)% | max_cases=$MAX_CASES")
    println("HARISS samples/week: $N_HOSP_SAMPLES | main-analysis ED ARI background: $MAIN_B")
    println("ED ARI background grid: ", join(["$b ($(round(100*N_HOSP_SAMPLES/b, digits=2))%)" for b in B_VALUES], ", "))
    println("Swab ratios: ", join([s.name for s in SWAB_SCENARIOS], ", "))
    for rs in (:original, :corrected)
        r = RATE_SETS[rs]
        @printf("  %-9s rates: ed_direct=%.4f gp_before_hosp=%.4f hosp_admit_direct=%.4f hosp_admit_from_gp=%.4f\n",
                rs, r.ed_direct_rate, r.gp_before_hosp_rate, r.hosp_admit_direct_rate, r.hosp_admit_from_gp_rate)
    end
    println("Output: $output_csv_path")
    println("="^80)

    merged_data = CSV.read(input_csv_path, DataFrame)
    merged_data = filter(r -> Float64(r.R0) == SCENARIO_R0 &&
                              Float64(r.generation_time) == SCENARIO_GEN_TIME, merged_data)
    out_dir = dirname(output_csv_path)
    !isempty(out_dir) && !isdir(out_dir) && mkpath(out_dir)
    isfile(output_csv_path) && rm(output_csv_path)

    for country in COUNTRIES
        country_data = sort(filter(r -> string(r.outbreak_country) == country, merged_data), :time)
        if nrow(country_data) == 0
            @warn "No rows in CSV for this country - skipping" country
            continue
        end
        mdts = unique(collect(skipmissing(country_data.mean_detection_time)))
        mean_det_time = Float64(first(mdts))
        if isnan(mean_det_time) || mean_det_time > MAX_DETECTION_THRESHOLD
            @warn "Invalid mean_detection_time - skipping" country mean_det_time
            continue
        end
        max_obs_time = mean_det_time + EXTRA_TIME
        country_trimmed = filter(r -> r.time <= max_obs_time, country_data)

        # One background per B, built with everyone swabbed; thinned per sample.
        t_bg = @elapsed bg_caches = Dict(B => NBPMscape.build_ari_background(;
                max_observation_time            = Float64(max_obs_time),
                n_hosp_samples_per_week         = N_HOSP_SAMPLES,
                sample_allocation               = P_FROM_CONFIG.sample_allocation,
                sample_proportion_adult         = P_FROM_CONFIG.sample_proportion_adult,
                hariss_nhs_trust_sampling_sites = HARISS_SITES_FROM_CONFIG,
                weight_samples_by               = P_FROM_CONFIG.weight_samples_by,
                swab_time_mode                  = P_FROM_CONFIG.swab_time_mode,
                swab_proportion_at_48h          = P_FROM_CONFIG.swab_proportion_at_48h,
                proportion_hosp_swabbed         = 1.0,
                ed_discharge_limit              = Float64(P_FROM_CONFIG.tdischarge_ed_upper_limit),
                hosp_short_stay_limit           = Float64(P_FROM_CONFIG.tdischarge_hosp_short_stay_upper_limit),
                hosp_ari_admissions             = B,
                hosp_ari_admissions_adult_p     = Float64(P_FROM_CONFIG.hosp_ari_admissions_adult_p),
                hosp_ari_admissions_child_p     = Float64(P_FROM_CONFIG.hosp_ari_admissions_child_p),
                ed_ari_destinations_adult       = P_FROM_CONFIG.ed_ari_destinations_adult,
                ed_ari_destinations_child       = P_FROM_CONFIG.ed_ari_destinations_child,
            ) for B in B_VALUES)
        println("\n$(now())  $country  window=$(round(max_obs_time, digits=1)) days, backgrounds built in $(round(t_bg, digits=1))s")
        flush(stdout)

        t_start = time()
        pool = CachingPool(workers())
        results = pmap(pool, 1:N_SAMPLES) do sample_id
            try
                rows = simulate_sensitivity_sample(
                    country_trimmed, country, sample_id,
                    SCENARIO_R0, SCENARIO_GEN_TIME, ICU_SAMPLING_PROPORTION,
                    p_det, AWW_FALSE_POSITIVE_RATE, max_obs_time,
                    bg_caches, B_VALUES, SWAB_SCENARIOS, MAIN_B, MAIN_SWAB;
                    turnaround_time = TURNAROUND_TIME,
                    max_cases       = MAX_CASES,
                )
                println("  [$sample_id/$N_SAMPLES] done")
                flush(stdout)
                return rows
            catch e
                @warn "Sample $sample_id failed" country err=e
                return NamedTuple[]
            end
        end
        clear!(pool)

        all_rows = reduce(vcat, results)
        if isempty(all_rows)
            @warn "No results for country" country
            continue
        end
        df = DataFrame(all_rows)
        CSV.write(output_csv_path, df; append = isfile(output_csv_path))
        println("  $country: $(nrow(df)) rows in $(round(time() - t_start, digits=1))s")
        for g in groupby(df, [:rate_set, :ed_ari_background, :swab_scenario])
            det = filter(isfinite, g.hariss_time)
            med = isempty(det) ? NaN : round(median(det), digits = 1)
            @printf("   %-9s B=%6d %-10s HARISS detected %3.0f%%  median %s d\n",
                    first(g.rate_set), first(g.ed_ari_background), first(g.swab_scenario),
                    100 * length(det) / nrow(g), med)
        end
        flush(stdout)
    end
    println("\nDone. Results: $output_csv_path")
end

run_sensitivity()

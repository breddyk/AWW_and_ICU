# ============================================================================
# multitype_comparison.jl
#
# Single-scenario driver: ICU + one AWW arm + HARISS.
#
# AWW: effective p_det = base_pdet × sampling_fraction (true detection from
#      CSV-seeded detectable imports only). False positives (FPR) apply only
#      to the AWW *test outcome* layer and do not seed the epidemic.
#
# AWW "detection" time: first day index t2 such that days t1 and t2 are both
# reported positive with t2 = t1 + 1 (consecutive calendar rows in the
# country time series). Recorded time is t2 + turnaround (aligned with other
# channels). After a positive with no positive next day, the streak resets.
#
# Per sample we also store min(ICU time, HARISS time, AWW two-hit time) and
# which channel achieved that minimum (ties broken AWW < ICU < HARISS).
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

# Increase for production runs (e.g. addprocs(180))
addprocs(1)

@everywhere using Pkg
@everywhere Pkg.activate($_PROJECT_DIR)
@everywhere using NBPMscape
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Distributions
@everywhere using CSV

@everywhere const CONFIG_REL_PATH = "config/outbreak_params_covid19_like.yaml"
@everywhere const CONFIG_ABS_PATH = joinpath(pkgdir(NBPMscape), CONFIG_REL_PATH)
@everywhere const CONFIG_DATA = NBPMscape.load_config(CONFIG_ABS_PATH)

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
            get(cfg, "ed_ari_destinations_adult_p_discharged",   default_adult[1]),
            get(cfg, "ed_ari_destinations_adult_p_short_stay",   default_adult[2]),
            get(cfg, "ed_ari_destinations_adult_p_longer_stay",  default_adult[3]),
        ],
    )
    ed_child = DataFrame(
        destination               = [:discharged, :short_stay, :longer_stay],
        proportion_of_attendances = [
            get(cfg, "ed_ari_destinations_child_p_discharged",   default_child[1]),
            get(cfg, "ed_ari_destinations_child_p_short_stay",   default_child[2]),
            get(cfg, "ed_ari_destinations_child_p_longer_stay",  default_child[3]),
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
    import_column::Symbol
)
    n_times = nrow(country_data)
    daily_imports = zeros(Int, n_times)
    total_imports = 0
    for (idx, row) in enumerate(eachrow(country_data))
        has_mean = !ismissing(row[import_column]) &&
                   isfinite(row[import_column]) &&
                   !isnan(row[import_column]) &&
                   row[import_column] > 0
        if has_mean
            sampled = round(Int, sample_poisson_direct(row[import_column]))
            daily_imports[idx] = sampled
            total_imports += sampled
        else
            daily_imports[idx] = 0
        end
    end
    return daily_imports, total_imports
end

# One Monte Carlo sample. AWW uses p_det on true imports only; false_positive_rate
# is Bernoulli on days with no true positive draw. Two consecutive reported
# positives set AWW detection on the second day.
# (Triple-quoted docstrings cannot be attached to @everywhere functions.)
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
    hariss_bg_cache;
    mean_infectious_period = 8/3,
    turnaround_time::Float64 = 3.0,
    n_hosp_samples_per_week::Int = Int(P_FROM_CONFIG.n_hosp_samples_per_week),
)
    infectious_period = mean_infectious_period
    latent_period = mean_generation_time - (0.5 * infectious_period)
    latent_period < 0 && error("Invalid parameters: latent_period < 0 for gen_time=$mean_generation_time")

    fixed_shape = 1000.0
    latent_scale = latent_period / fixed_shape
    infectious_scale = infectious_period / fixed_shape
    baseline_R0 = 2.03
    infectivity_scaling = (R0 / baseline_R0) * P_FROM_CONFIG.infectivity

    base_params = merge(P_FROM_CONFIG, (
        infectivity = infectivity_scaling,
        latent_scale = latent_scale,
        infectious_scale = infectious_scale,
        infectious_shape = fixed_shape,
        latent_shape = fixed_shape,
        importrate = 0.0,
        turnaroundtime = turnaround_time,
    ))
    icu_params = merge(base_params, (psampled = icu_sampling_proportion,))
    infectious_params = merge(base_params, (
        latent_scale = 1e-6,
        infectious_scale = (infectious_period / 2.0) / fixed_shape,
        latent_shape = fixed_shape,
    ))

    latent_import_counts, total_latent = sample_daily_imports_poisson(
        country_data, sample_id, :daily_latent_imports)
    infectious_import_counts, total_infectious = sample_daily_imports_poisson(
        country_data, sample_id, :daily_infectious_imports)
    detectable_import_counts, total_detectable = sample_daily_imports_poisson(
        country_data, sample_id, :daily_detectable_imports)

    sample_infections = DataFrame()
    icu_detected = false
    first_icu_time = Inf
    hariss_detected = false
    first_hariss_time = Inf

    # hariss_bg_cache is built after the loop, only when local cases exist,
    # so we never pay for it on samples with no local spread.

    aww_detected = false
    first_aww_time = Inf
    prev_aww_positive = false

    for (idx, row) in enumerate(eachrow(country_data))
        time = row.time
        time >= max_observation_time && break

        # AWW: always evaluate — cheap Bernoulli draws, independent of the tree.
        daily_detectable_count = detectable_import_counts[idx]
        p_true = daily_detectable_count > 0 ? 1.0 - (1.0 - p_det)^daily_detectable_count : 0.0
        true_positive = rand() < p_true
        reported_positive = true_positive || (!true_positive && rand() < false_positive_rate)

        if reported_positive
            if prev_aww_positive && !aww_detected
                aww_detected = true
                first_aww_time = time + turnaround_time
            end
            prev_aww_positive = true
        else
            prev_aww_positive = false
        end

        # Tree building and ICU: stop once ICU has detected.
        # This caps sample_infections growth — critical for high R0 where each
        # simtree call produces a large subtree and sampleforest cost scales with
        # the total number of accumulated infections.
        if !icu_detected
            daily_latent_count = latent_import_counts[idx]
            daily_infectious_count = infectious_import_counts[idx]

            for j in 1:daily_latent_count
                results = NBPMscape.simtree(base_params,
                    initialtime = Float64(time) - latent_period / 2.0,
                    maxtime = max_observation_time,
                    maxgenerations = 100,
                    initialcontact = :G,
                )
                if nrow(results.G) > 0
                    results.G[1, :generation] = 0
                    results.G.import_event .= Int(time)
                    results.G.import_time .= Float64(time)
                    if isempty(sample_infections)
                        sample_infections = results.G
                    else
                        append!(sample_infections, results.G)
                    end
                end
            end
            for j in 1:daily_infectious_count
                results = NBPMscape.simtree(infectious_params,
                    initialtime = Float64(time),
                    maxtime = max_observation_time,
                    maxgenerations = 100,
                    initialcontact = :G,
                )
                if nrow(results.G) > 0
                    results.G[1, :generation] = 0
                    results.G.import_event .= Int(time)
                    results.G.import_time .= Float64(time)
                    if isempty(sample_infections)
                        sample_infections = results.G
                    else
                        append!(sample_infections, results.G)
                    end
                end
            end

            if !isempty(sample_infections)
                icu_sampled = NBPMscape.sampleforest((G = sample_infections,), icu_params)
                if !isempty(icu_sampled.treport)
                    icu_detected = true
                    first_icu_time = minimum(icu_sampled.treport)
                end
            end
        end

        # Exit once both ICU and AWW have fired. AWW is checked above regardless,
        # so running the loop past ICU detection while waiting for AWW is cheap.
        icu_detected && aww_detected && break
    end

    # HARISS: only meaningful when local UK cases exist. hariss_bg_cache is
    # pre-built once on the main process and passed in — no per-sample cost.
    n_local = isempty(sample_infections) ? 0 : sum(sample_infections.generation .> 0)

    if n_local > 0 && hariss_bg_cache !== nothing
        try
            sims_for_hariss = sample_infections
            if !(:simid in propertynames(sims_for_hariss))
                sims_for_hariss = copy(sample_infections)
                sims_for_hariss.simid .= "sim-$(sample_id)"
            end
            hariss_result = redirect_stdout(devnull) do
                NBPMscape.secondary_care_td(;
                    p = base_params,
                    sims = [sims_for_hariss],
                    pathogen_type                    = P_FROM_CONFIG.pathogen_type,
                    initial_dow                      = P_FROM_CONFIG.initial_dow,
                    hariss_courier_to_analysis       = P_FROM_CONFIG.hariss_courier_to_analysis,
                    hariss_turnaround_time           = [turnaround_time, turnaround_time + 1e-6],
                    n_hosp_samples_per_week          = n_hosp_samples_per_week,
                    sample_allocation                = P_FROM_CONFIG.sample_allocation,
                    sample_proportion_adult          = P_FROM_CONFIG.sample_proportion_adult,
                    hariss_nhs_trust_sampling_sites  = HARISS_SITES_FROM_CONFIG,
                    weight_samples_by                = P_FROM_CONFIG.weight_samples_by,
                    phl_collection_dow               = Vector{Int64}(P_FROM_CONFIG.phl_collection_dow),
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
                    precomputed_ari_bg               = hariss_bg_cache,
                )
            end
            if nrow(hariss_result) > 0 && :SC_TD in propertynames(hariss_result)
                sc_td_finite = filter(x -> !ismissing(x) && isfinite(x), hariss_result.SC_TD)
                if !isempty(sc_td_finite)
                    td_min = minimum(sc_td_finite)
                    if td_min <= max_observation_time
                        hariss_detected = true
                        first_hariss_time = td_min
                    end
                end
            end
        catch hariss_err
            @warn "HARISS sampling failed" country = country_name sample = sample_id err = hariss_err
        end
    end

    icu_local_cases = NaN
    hariss_local_cases = NaN
    aww_local_cases = NaN
    if !isempty(sample_infections)
        local_mask = sample_infections.generation .> 0
        if icu_detected && isfinite(first_icu_time)
            icu_local_cases = Float64(sum(local_mask .& (sample_infections.tinf .<= first_icu_time)))
        end
        if hariss_detected && isfinite(first_hariss_time)
            hariss_local_cases = Float64(sum(local_mask .& (sample_infections.tinf .<= first_hariss_time)))
        end
        if aww_detected && isfinite(first_aww_time)
            aww_local_cases = Float64(sum(local_mask .& (sample_infections.tinf .<= first_aww_time)))
        end
    end

    t_icu = icu_detected && isfinite(first_icu_time) ? Float64(first_icu_time) : Inf
    t_har = hariss_detected && isfinite(first_hariss_time) ? Float64(first_hariss_time) : Inf
    t_aww = aww_detected && isfinite(first_aww_time) ? Float64(first_aww_time) : Inf
    t_min = min(t_icu, t_har, t_aww)
    earliest_type = if t_min == Inf
        ""
    elseif t_min == t_aww && t_aww <= t_icu && t_aww <= t_har
        "AWW"
    elseif t_min == t_icu && t_icu <= t_har
        "ICU"
    else
        "HARISS"
    end

    return (
        sample_id = sample_id,
        country = country_name,
        R0 = R0,
        gen_time = mean_generation_time,
        p_det = p_det,
        false_positive_rate = false_positive_rate,
        icu_detection_time = first_icu_time,
        icu_local_cases = icu_local_cases,
        hariss_detection_time = first_hariss_time,
        hariss_local_cases = hariss_local_cases,
        aww_detection_time = first_aww_time,
        aww_local_cases = aww_local_cases,
        earliest_detection_time = t_min == Inf ? NaN : t_min,
        earliest_surveillance_type = earliest_type,
        total_latent = Float64(total_latent),
        total_infectious = Float64(total_infectious),
        total_detectable = Float64(total_detectable),
    )
end

function run_multitype_comparison(;
    csv_path::String,
    output_path::String,
    num_samples::Int,
    turnaround_time::Float64,
    max_detection_time_threshold::Float64,
    extra_time::Float64,
    icu_sampling_proportion::Float64,
    n_hosp_samples_per_week::Int,
    R0::Float64,
    gen_time::Float64,
    base_pdet::Float64,
    sampling_fraction::Float64,
    country::String,
    false_positive_rate::Float64,
)
    p_det = base_pdet * sampling_fraction
    println("="^80)
    println("MULTITYPE COMPARISON (single scenario)")
    println("="^80)
    println("Country: $country | R0=$R0 | gen_time=$gen_time")
    println("AWW: base_pdet=$base_pdet × sampling_fraction=$sampling_fraction => p_det=$p_det")
    println("AWW false positive rate (test layer only): $false_positive_rate")
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
    isempty(mdts) && error("mean_detection_time is missing for all matching CSV rows")
    length(mdts) != 1 &&
        error("Expected a single mean_detection_time for this scenario; got $(length(mdts)) distinct values")
    mean_det_time = Float64(first(mdts))
    (isnan(mean_det_time) || mean_det_time > max_detection_time_threshold) &&
        error("Invalid mean_detection_time=$mean_det_time for scenario")

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

    println("\nPre-building HARISS ARI background (shared across all samples)...")
    flush(stdout)
    t_bg = @elapsed hariss_bg_cache = NBPMscape.build_ari_background(;
        max_observation_time            = Float64(max_obs_time),
        n_hosp_samples_per_week         = n_hosp_samples_per_week,
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
    @printf("  done in %.1fs\n", t_bg)
    println("max_observation_time = $(round(max_obs_time, digits=1)) days")
    println("\nStarting pmap over $num_samples samples on $(nworkers()) workers...")
    flush(stdout)

    pool       = CachingPool(workers())
    batch_size = max(1, min(50, num_samples ÷ nworkers()))
    all_rows   = Vector{Any}()
    sizehint!(all_rows, num_samples)
    t_start = time()

    for batch_start in 1:batch_size:num_samples
        batch_end = min(batch_start + batch_size - 1, num_samples)
        batch_ids = collect(batch_start:batch_end)

        batch_rows = pmap(pool, batch_ids) do sample_id
            simulate_multitype_sample(
                country_trimmed,
                country,
                sample_id,
                R0,
                gen_time,
                icu_sampling_proportion,
                p_det,
                false_positive_rate,
                max_obs_time,
                hariss_bg_cache;
                turnaround_time         = turnaround_time,
                n_hosp_samples_per_week = n_hosp_samples_per_week,
            )
        end
        append!(all_rows, batch_rows)

        elapsed = time() - t_start
        done    = length(all_rows)
        rate    = done / elapsed
        eta     = (num_samples - done) / rate
        @printf("  [%4d/%4d] %.1fs elapsed | %.2f samples/s | ETA %.1fs\n",
                done, num_samples, elapsed, rate, eta)
        flush(stdout)
    end
    clear!(pool)

    df = DataFrame(all_rows)
    CSV.write(output_path, df)
    total = round(time() - t_start, digits=1)
    println("\nWrote $(nrow(df)) rows to $output_path  (total: $(total)s)")
    return df
end


const SCENARIO_R0 = 2.0
const SCENARIO_GEN_TIME = 4.0
const SCENARIO_BASE_PDET = 0.16
const SCENARIO_SAMPLING_FRACTION = 0.5
const SCENARIO_COUNTRY = "Switzerland"
const AWW_FALSE_POSITIVE_RATE = 0.04

project_root    = normpath(joinpath(@__DIR__, "..", ".."))
input_csv_path  = joinpath(project_root, "global_model/pgfgleam/all_results/global/daily_imports_sensitivity.csv")
output_csv_path = joinpath(project_root, "global_model/pgfgleam/all_results/local/multi_surveillance_result.csv")

run_multitype_comparison(;
    csv_path = input_csv_path,
    output_path = output_csv_path,
    num_samples = 1000,
    turnaround_time = 3.0,
    max_detection_time_threshold = 200.0,
    extra_time = 50.0,
    icu_sampling_proportion = 0.10,
    n_hosp_samples_per_week = Int(NBPMscape.P.n_hosp_samples_per_week),
    R0 = SCENARIO_R0,
    gen_time = SCENARIO_GEN_TIME,
    base_pdet = SCENARIO_BASE_PDET,
    sampling_fraction = SCENARIO_SAMPLING_FRACTION,
    country = SCENARIO_COUNTRY,
    false_positive_rate = AWW_FALSE_POSITIVE_RATE,
)

println("\n✓ multitype_comparison complete!")

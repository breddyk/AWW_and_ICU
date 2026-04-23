# ============================================================================
# full_ICU_AWW_HARISS_update.jl
#
# Three-channel surveillance driver: ICU sampling + Airport Wastewater (AWW)
# + HARISS (Hospital Admission Respiratory Infection Surveillance Sampling).
#
# Same epidemiological model as `full_ICU_AWW_HARISS.jl`, but parallelises
# across Monte Carlo *samples* (not only countries) via `simulate_single_sample`
# + `pmap` in `run_simulations_from_merged_csv` — see the PARALLELISATION NOTE
# below `simulate_single_sample`.
#
# Mirrors `full_ICU_WW.jl` but adds HARISS as a third detection channel via
# `NBPMscape.secondary_care_td`. Note: the `WW_*` column prefix is retained
# for compatibility with the existing AWW analysis pipeline -- semantically
# `WW_*` here denotes the airport wastewater channel (AWW).
#
# Parameter wiring
# ----------------
# All HARISS-related parameters (sample allocation, PHL collection days,
# turnaround time, weekly sample budget, ARI background, etc.) are loaded
# from the real YAML config under `local_model/config/` rather than being
# hardcoded against the package's defaults in `NBPMscape.P`. The active
# config is `config/outbreak_params_covid19_like.yaml`; switch to one of
# the `config/HARISS/covid19_like_params_HARISS_*.yaml` scenarios by
# changing `CONFIG_REL_PATH` below.
#
# NHS Trust / geography data (HARISS)
# -----------------------------------
# The HARISS hospital network is `data/hariss_nhs_trust_sampling_sites.csv`
# (YAML key `hariss_nhs_trust_sampling_sites_file`). NBPMscape supplies
# supporting lookup tables (`AE_12M`, catchment populations, ITL2→trust
# admission probabilities) from `data/nhs_trust_data/*.csv`. After replacing
# the upstream NHS source files, regenerate the derived CSVs with:
#     python3 scripts/build_hariss_nhs_trust_lookups.py
# ============================================================================

# Activate the NBPMscape project environment on the MAIN process BEFORE
# loading NBPMscape (it's a local, unregistered package, so it won't be
# findable otherwise). Workers are activated below after addprocs.
using Pkg
const _PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(_PROJECT_DIR)
# If this is the first run in a fresh checkout (or deps changed), uncomment:
# Pkg.instantiate()

using NBPMscape
# using Plots
using DataFrames
using Statistics
using Distributions
using CSV
# using StatsPlots
# using LaTeXStrings
# using KernelDensity
using Distributed
using Dates

addprocs(180)

@everywhere using Pkg
@everywhere Pkg.activate($_PROJECT_DIR)

# Load packages on all workers
@everywhere using NBPMscape
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Distributions
@everywhere using CSV

# default(fontfamily="Times New Roman")

# ============================================================================
# REAL CONFIG LOADING (must run on every worker so HARISS uses the same
# parameters everywhere `simulate_country_detection` is invoked).
# ============================================================================

# Path to the YAML config -- relative to the NBPMscape package root so it
# resolves the same way on every worker regardless of CWD.
@everywhere const CONFIG_REL_PATH = "config/outbreak_params_covid19_like.yaml"

@everywhere const CONFIG_ABS_PATH = joinpath(pkgdir(NBPMscape), CONFIG_REL_PATH)

@everywhere const CONFIG_DATA = NBPMscape.load_config(CONFIG_ABS_PATH)

# Map a YAML config Dict onto a parameter NamedTuple by overwriting only the
# scalar/vector keys that already exist in `P`. Mirrors the conversions done
# in `NBPMscape.update_configurable_parameters` (rho_hosp -> ρ_hosp etc.) but
# avoids the requirement of supplying `default_configurable_params` and skips
# the dataframe-loading step (which expects ICU site files we do not have).
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
            # `dowcont` is stored as a Tuple inside P but YAML loads it as Vector
            if sym === :dowcont
                value = tuple(value...)
            end
            P_dict[sym] = value
        end
    end
    return NamedTuple(P_dict)
end

# Parameters with config-file overrides applied.
@everywhere const P_FROM_CONFIG = let
    p = apply_yaml_scalars(NBPMscape.P, CONFIG_DATA)
    # Rebuild the ED ARI destination DataFrames from the YAML config scalars.
    # These scalars (`ed_ari_destinations_adult_p_discharged` etc.) exist in the
    # YAML but NOT in NBPMscape.P, so apply_yaml_scalars skips them. Read them
    # directly from the config dict, falling back to the existing DataFrame
    # values in P.
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

# Load the HARISS NHS Trust sampling sites file pointed to by the YAML. If the
# file is absent, fall back to `NBPMscape.P.hariss_nhs_trust_sampling_sites`
# (the package-bundled default). Resolved relative to the package root so the path
# works from any CWD on every worker.
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

if myid() == 1
    println("Loaded HARISS config from: ", CONFIG_ABS_PATH)
    println("HARISS sites loaded: ", nrow(HARISS_SITES_FROM_CONFIG), " rows")
    if nrow(HARISS_SITES_FROM_CONFIG) > 0 &&
       startswith(string(first(HARISS_SITES_FROM_CONFIG[!, 1])), "DM")
        println("⚠️  HARISS sites file appears to still contain DUMMY trust codes.")
        println("    Replace data/hariss_nhs_trust_sampling_sites.csv with the real list",
                " for accurate geographic matching.")
    end
end

# ============================================================================
# SAMPLING FUNCTIONS (must be @everywhere for workers)
# ============================================================================

@everywhere function sample_poisson_direct(lambda::Float64)
    """
    Sample directly from Poisson(lambda) for daily imports
    """
    return Float64(rand(Poisson(lambda)))
end

@everywhere function sample_daily_imports_poisson(
    country_data::DataFrame,
    sample_id::Int,
    import_column::Symbol
)
    """
    Sample directly from Poisson distribution using specified import column
    
    Args:
        country_data: DataFrame with import data
        sample_id: Sample identifier
        import_column: Either :daily_latent_imports, :daily_infectious_imports, or :daily_detectable_imports
    
    Returns: Vector of daily import counts (integers) and total imports for this sample
    """
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

# ============================================================================
# COMBINED ICU + WW DETECTION WITH SPLIT IMPORT TYPES
# ============================================================================
#
# PARALLELISATION NOTE
# --------------------
# Each Monte-Carlo sample is independent, so we expose the per-sample work via
# `simulate_single_sample` and let the top-level `pmap` in
# `run_simulations_from_merged_csv` dispatch (combination × sample) tuples.
# This means big-population countries (Brazil/USA/...) no longer block a whole
# worker serially on 100 samples -- their samples are spread across all
# workers, dramatically improving tail latency. `simulate_country_detection`
# is kept as a thin serial wrapper for back-compat / direct calls.

@everywhere function simulate_single_sample(
    country_data::DataFrame,
    country_name::String,
    sample_id::Int,
    R0::Float64,
    mean_generation_time::Float64,
    icu_sampling_proportion::Float64,
    airport_detection_probs::Vector{Float64},
    max_observation_time::Float64,
    hariss_bg_cache;
    mean_infectious_period = 8/3,
    turnaround_time::Float64 = 3.0,
    n_hosp_samples_per_week::Int = Int(P_FROM_CONFIG.n_hosp_samples_per_week)
)
    """
    Run ONE Monte-Carlo sample of the ICU + AWW + HARISS simulation.

    Returns a NamedTuple of per-sample outcomes (Inf / NaN encode "not detected").
    Aggregate N sample results into the format expected by the rest of the
    pipeline with `aggregate_sample_results`.
    """

    # --- FIXED INFECTIOUS PERIOD ---
    infectious_period = mean_infectious_period
    latent_period = mean_generation_time - (0.5 * infectious_period)

    if latent_period < 0
        error("Invalid parameters: latent_period < 0 for gen_time=$mean_generation_time")
    end

    # --- DETERMINISTIC PERIODS ---
    fixed_shape = 1000.0
    latent_scale = latent_period / fixed_shape
    infectious_scale = infectious_period / fixed_shape

    # --- SCALE INFECTIVITY ---
    baseline_R0 = 2.03
    infectivity_scaling = (R0 / baseline_R0) * P_FROM_CONFIG.infectivity

    base_params = merge(P_FROM_CONFIG, (
        infectivity = infectivity_scaling,
        latent_scale = latent_scale,
        infectious_scale = infectious_scale,
        infectious_shape = fixed_shape,
        latent_shape = fixed_shape,
        importrate = 0.0,
        turnaroundtime = turnaround_time
    ))

    icu_params = merge(base_params, (psampled = icu_sampling_proportion,))

    # Hoisted out of the per-day loop: same for every day of this sample.
    infectious_params = merge(base_params, (
        latent_scale = 1e-6,
        infectious_scale = (infectious_period / 2.0) / fixed_shape,
        latent_shape = fixed_shape
    ))

    # Sample the three Poisson import streams
    latent_import_counts, total_latent = sample_daily_imports_poisson(
        country_data, sample_id, :daily_latent_imports)
    infectious_import_counts, total_infectious = sample_daily_imports_poisson(
        country_data, sample_id, :daily_infectious_imports)
    detectable_import_counts, total_detectable = sample_daily_imports_poisson(
        country_data, sample_id, :daily_detectable_imports)

    # Per-sample state
    sample_infections = DataFrame()
    icu_detected = false
    first_icu_time = Inf
    hariss_detected = false
    first_hariss_time = Inf
    # HARISS is now a single end-of-sample check (see comment below the
    # per-day loop), so no weekly-throttle bookkeeping is needed.

    # hariss_bg_cache is built after the per-day loop, only when local cases
    # exist, so we never pay for it on samples with no local spread.

    airport_detected = Dict{Float64, Bool}()
    first_airport_time = Dict{Float64, Float64}()
    for p_det in airport_detection_probs
        airport_detected[p_det] = false
        first_airport_time[p_det] = Inf
    end

    for (idx, row) in enumerate(eachrow(country_data))
        time = row.time
        if time >= max_observation_time
            break
        end

        # AIRPORT DETECTION: detectable imports (I + P)
        daily_detectable_count = detectable_import_counts[idx]
        if daily_detectable_count > 0
            for p_det in airport_detection_probs
                if !airport_detected[p_det]
                    prob_detection_today = 1.0 - (1.0 - p_det)^daily_detectable_count
                    if rand() < prob_detection_today
                        airport_detected[p_det] = true
                        first_airport_time[p_det] = time + turnaround_time
                    end
                end
            end
        end

        # LOCAL TRANSMISSION + ICU: skip once ICU has detected.
        # Capping tree growth here is critical for high R0 — without this
        # guard, sample_infections grows exponentially and sampleforest
        # becomes O(n²) overall.
        if !icu_detected
            daily_latent_count = latent_import_counts[idx]
            daily_infectious_count = infectious_import_counts[idx]

            for j in 1:daily_latent_count
                results = NBPMscape.simtree(base_params,
                    initialtime = Float64(time) - latent_period / 2.0,
                    maxtime = max_observation_time,
                    maxgenerations = 100,
                    initialcontact = :G
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
                    initialcontact = :G
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

            # ICU detection check (per-day)
            if !isempty(sample_infections)
                icu_sampled = NBPMscape.sampleforest(
                    (G = sample_infections,), icu_params
                )
                if !isempty(icu_sampled.treport)
                    icu_detected = true
                    first_icu_time = minimum(icu_sampled.treport)
                end
            end
        end

        # HARISS is NOT checked inside the per-day loop -- see the single
        # terminal call after this loop. HARISS detection cannot influence
        # transmission, and running it weekly means re-processing the whole
        # trajectory each call (the dominant cost per sample). A single
        # end-of-sample call gives min(treport), which is mathematically
        # identical to the first weekly check that would have fired.

        # EARLY STOPPING — ICU + every AWW arm. HARISS is resolved after the
        # loop, so it no longer gates early exit here. The per-day loop is
        # cheap (Poisson import draws + bounded simtree calls), so running
        # until ICU and AWW have both fired is dominated by the simtree
        # cost we were already paying.
        all_airport_detected = all(airport_detected[p] for p in airport_detection_probs)
        if icu_detected && all_airport_detected
            break
        end
    end

    # ── Single end-of-sample HARISS check ──
    # HARISS requires local (generation > 0) UK cases; skip when there are
    # none (legitimate Inf). hariss_bg_cache is passed in pre-built.
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
            @warn "HARISS sampling failed" country=country_name sample=sample_id err=hariss_err
        end
    end

    # Compute per-sample local-cases-at-detection counts
    icu_local_cases = NaN
    hariss_local_cases = NaN
    airport_local_cases = Dict{Float64, Float64}()
    for p_det in airport_detection_probs
        airport_local_cases[p_det] = NaN
    end

    if !isempty(sample_infections)
        local_mask = sample_infections.generation .> 0
        if icu_detected && isfinite(first_icu_time)
            icu_local_cases = Float64(sum(local_mask .& (sample_infections.tinf .<= first_icu_time)))
        end
        if hariss_detected && isfinite(first_hariss_time)
            hariss_local_cases = Float64(sum(local_mask .& (sample_infections.tinf .<= first_hariss_time)))
        end
        for p_det in airport_detection_probs
            if airport_detected[p_det] && isfinite(first_airport_time[p_det])
                airport_local_cases[p_det] = Float64(sum(local_mask .& (sample_infections.tinf .<= first_airport_time[p_det])))
            end
        end
    end

    return (
        sample_id = sample_id,
        icu_detected = icu_detected,
        icu_detection_time = first_icu_time,
        icu_local_cases = icu_local_cases,
        hariss_detected = hariss_detected,
        hariss_detection_time = first_hariss_time,
        hariss_local_cases = hariss_local_cases,
        airport_detected = airport_detected,
        airport_detection_times = first_airport_time,
        airport_local_cases = airport_local_cases,
        total_latent = Float64(total_latent),
        total_infectious = Float64(total_infectious),
        total_detectable = Float64(total_detectable),
    )
end

@everywhere function aggregate_sample_results(
    sample_results::AbstractVector,
    airport_detection_probs::Vector{Float64}
)
    """
    Combine a vector of `simulate_single_sample` outputs into the same
    NamedTuple that `simulate_country_detection` historically returned.
    """
    icu_detection_times = Float64[]
    icu_local_cases_at_detection = Float64[]
    hariss_detection_times = Float64[]
    hariss_local_cases_at_detection = Float64[]

    airport_results = Dict{Float64, Dict{Symbol, Any}}()
    for p_det in airport_detection_probs
        airport_results[p_det] = Dict(
            :detection_times => Float64[],
            :local_cases_at_detection => Float64[],
            :total_detections => 0
        )
    end

    latent_imports = Float64[]
    infectious_imports = Float64[]
    detectable_imports = Float64[]

    for r in sample_results
        push!(latent_imports, r.total_latent)
        push!(infectious_imports, r.total_infectious)
        push!(detectable_imports, r.total_detectable)

        if r.icu_detected && isfinite(r.icu_detection_time) && !isnan(r.icu_local_cases)
            push!(icu_detection_times, r.icu_detection_time)
            push!(icu_local_cases_at_detection, r.icu_local_cases)
        end
        if r.hariss_detected && isfinite(r.hariss_detection_time) && !isnan(r.hariss_local_cases)
            push!(hariss_detection_times, r.hariss_detection_time)
            push!(hariss_local_cases_at_detection, r.hariss_local_cases)
        end
        for p_det in airport_detection_probs
            if r.airport_detected[p_det] && isfinite(r.airport_detection_times[p_det])
                airport_results[p_det][:total_detections] += 1
                push!(airport_results[p_det][:detection_times], r.airport_detection_times[p_det])
                if !isnan(r.airport_local_cases[p_det])
                    push!(airport_results[p_det][:local_cases_at_detection], r.airport_local_cases[p_det])
                end
            end
        end
    end

    mean_or_nan(v) = isempty(v) ? NaN : mean(v)

    return (
        icu_detection_times = icu_detection_times,
        icu_local_cases_at_detection = icu_local_cases_at_detection,
        hariss_detection_times = hariss_detection_times,
        hariss_local_cases_at_detection = hariss_local_cases_at_detection,
        airport_results = airport_results,
        mean_latent_imports_per_sample = mean_or_nan(latent_imports),
        mean_infectious_imports_per_sample = mean_or_nan(infectious_imports),
        mean_detectable_imports_per_sample = mean_or_nan(detectable_imports)
    )
end

# Back-compat wrapper: serial loop over samples. Prefer calling
# `simulate_single_sample` directly (e.g. from pmap) for parallelism.
@everywhere function simulate_country_detection(
    country_data::DataFrame,
    country_name::String,
    num_samples::Int,
    R0::Float64,
    mean_generation_time::Float64,
    icu_sampling_proportion::Float64,
    airport_detection_probs::Vector{Float64},
    max_observation_time::Float64;
    mean_infectious_period = 8/3,
    turnaround_time::Float64 = 3.0,
    n_hosp_samples_per_week::Int = Int(P_FROM_CONFIG.n_hosp_samples_per_week),
    verbose::Bool = true
)
    if verbose
        println("  Processing $country_name (R0=$R0, gen_time=$mean_generation_time)")
        println("    Max observation time: $max_observation_time days")
        println("    ICU sampling: $(icu_sampling_proportion*100)%")
        println("    AWW detection probs: $(airport_detection_probs)")
        println("    HARISS samples/week: $(n_hosp_samples_per_week)")
    end

    hariss_bg_cache = NBPMscape.build_ari_background(;
        max_observation_time             = Float64(max_observation_time),
        initial_dow                      = P_FROM_CONFIG.initial_dow,
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
        ed_discharge_limit               = Float64(P_FROM_CONFIG.tdischarge_ed_upper_limit),
        hosp_short_stay_limit            = Float64(P_FROM_CONFIG.tdischarge_hosp_short_stay_upper_limit),
        hosp_ari_admissions              = Int(P_FROM_CONFIG.hosp_ari_admissions),
        hosp_ari_admissions_adult_p      = Float64(P_FROM_CONFIG.hosp_ari_admissions_adult_p),
        hosp_ari_admissions_child_p      = Float64(P_FROM_CONFIG.hosp_ari_admissions_child_p),
        ed_ari_destinations_adult        = P_FROM_CONFIG.ed_ari_destinations_adult,
        ed_ari_destinations_child        = P_FROM_CONFIG.ed_ari_destinations_child,
    )

    sample_results = Vector{Any}(undef, num_samples)
    for sample in 1:num_samples
        if verbose && (sample % 10 == 0 || sample == 1)
            println("    Sample $sample/$num_samples")
            flush(stdout)
        end
        sample_results[sample] = simulate_single_sample(
            country_data, country_name, sample,
            R0, mean_generation_time, icu_sampling_proportion,
            airport_detection_probs, max_observation_time,
            hariss_bg_cache;
            mean_infectious_period = mean_infectious_period,
            turnaround_time = turnaround_time,
            n_hosp_samples_per_week = n_hosp_samples_per_week
        )
    end

    agg = aggregate_sample_results(sample_results, airport_detection_probs)

    if verbose
        for p_det in airport_detection_probs
            airport_pct = round(100*agg.airport_results[p_det][:total_detections]/num_samples, digits=1)
            println("    p_det=$p_det: AWW=$airport_pct% detected")
        end
        println("    ICU=$(round(100*length(agg.icu_detection_times)/num_samples, digits=1))% detected")
        println("    HARISS=$(round(100*length(agg.hariss_detection_times)/num_samples, digits=1))% detected")
        println("    Mean latent imports/sample: $(round(agg.mean_latent_imports_per_sample, digits=2))")
        println("    Mean infectious imports/sample: $(round(agg.mean_infectious_imports_per_sample, digits=2))")
        println("    Mean detectable imports/sample: $(round(agg.mean_detectable_imports_per_sample, digits=2))")
    end

    return agg
end

# ============================================================================
# SAFE CSV WRITE FUNCTION
# ============================================================================

function safe_csv_write(output_path::String, df::DataFrame)
    """
    Safely write DataFrame to CSV with proper vector handling and error recovery
    """
    # Convert vector columns to strings
    df_to_save = copy(df)
    vector_cols = [:ICU_detection_times, :ICU_local_cases_samples,
                   :HARISS_detection_times, :HARISS_local_cases_samples,
                   :WW_008_10pct_detection_times, :WW_008_10pct_local_cases_samples,
                   :WW_008_25pct_detection_times, :WW_008_25pct_local_cases_samples,
                   :WW_008_50pct_detection_times, :WW_008_50pct_local_cases_samples,
                   :WW_008_100pct_detection_times, :WW_008_100pct_local_cases_samples,
                   :WW_016_10pct_detection_times, :WW_016_10pct_local_cases_samples,
                   :WW_016_25pct_detection_times, :WW_016_25pct_local_cases_samples,
                   :WW_016_50pct_detection_times, :WW_016_50pct_local_cases_samples,
                   :WW_016_100pct_detection_times, :WW_016_100pct_local_cases_samples]
    
    for col in vector_cols
        if col in names(df_to_save)
            df_to_save[!, col] = string.(df_to_save[!, col])
        end
    end
    
    # Try writing to temporary file first
    temp_path = output_path * ".tmp"
    
    try
        # Ensure directory exists
        output_dir = dirname(output_path)
        if !isdir(output_dir) && output_dir != ""
            mkpath(output_dir)
            println("Created output directory: $output_dir")
        end
        
        # Write to temp file
        CSV.write(temp_path, df_to_save)
        
        # Move temp to final location (atomic operation)
        mv(temp_path, output_path; force=true)
        
        return true
    catch e
        println("❌ ERROR writing CSV: $e")
        
        # Try backup location
        backup_path = output_path * ".backup_" * string(Dates.now())
        try
            CSV.write(backup_path, df_to_save)
            println("✓ Saved to backup: $backup_path")
            return false
        catch backup_error
            println("❌ Backup also failed: $backup_error")
            
            # Clean up temp file if it exists
            if isfile(temp_path)
                rm(temp_path; force=true)
            end
            
            rethrow(e)
        end
    end
end

# ============================================================================
# MAIN SIMULATION FROM MERGED CSV (BATCHED WITH SAVE EVERY 125)
# ============================================================================

function run_simulations_from_merged_csv(
    csv_path::String;
    num_samples::Int = 100,
    turnaround_time::Float64 = 3.0,
    max_detection_time_threshold::Float64 = 120.0,
    extra_time::Float64 = 20.0,
    icu_sampling_proportion::Float64 = 0.10,
    n_hosp_samples_per_week::Int = Int(P_FROM_CONFIG.n_hosp_samples_per_week),
    output_path::String = "results_aww_icu_hariss.csv",
    batch_size::Int = 125
)
    """
    Run AWW + ICU + HARISS simulations.

    Airport (AWW) detection model:
    - base_pdet ∈ {0.08, 0.16} (per-flight detection probability)
    - sampling_fraction ∈ {10%, 25%, 50%, 100%} (proportion of flights tested)
    - effective p_det = base_pdet × sampling_fraction

    This gives 8 AWW configurations.

    HARISS detection:
    - All HARISS parameters (sample allocation, PHL collection days, hospital
      ARI background, sampling sites, etc.) are loaded from the YAML config
      pointed at by `CONFIG_REL_PATH` at the top of this file.
    - n_hosp_samples_per_week defaults to the value in that config.

    Features:
    - Early stopping: Stops each sample when all three channels have detected
    - Batched processing: Processes batch_size combinations at a time
    - Saves after each batch completes
    - Crash recovery: Can resume from existing output file
    """

    println("="^80)
    println("AWW + ICU + HARISS SIMULATIONS (8 AWW TYPES, BATCHED, SAVE EVERY $batch_size)")
    println("="^80)
    println("Input CSV: $csv_path")
    println("Max detection time threshold: $max_detection_time_threshold days")
    println("Extra simulation time: $extra_time days")
    println("Samples per combination: $num_samples")
    println("ICU sampling proportion: $(icu_sampling_proportion*100)%")
    println("HARISS samples per week: $n_hosp_samples_per_week")
    println("HARISS config: $CONFIG_REL_PATH")
    println("HARISS sample allocation: $(P_FROM_CONFIG.sample_allocation)")
    println("HARISS PHL collection DOW: $(P_FROM_CONFIG.phl_collection_dow)")
    println("HARISS hosp ARI background/wk: $(P_FROM_CONFIG.hosp_ari_admissions)")
    println("AWW base p_det values: 8%, 16%")
    println("AWW sampling fractions: 10%, 25%, 50%, 100%")
    println("Workers: $(nworkers())")
    println("Batch size: $batch_size combinations")
    println("="^80)
    
    # Ensure output directory exists
    output_dir = dirname(output_path)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    # Check write permissions
    println("\nChecking output path permissions...")
    println("Output path: $output_path")
    println("Directory exists: $(isdir(output_dir))")
    if output_dir != ""
        println("Directory writable: $(iswritable(output_dir))")
    end
    
    # Load merged data
    println("\nLoading merged data...")
    merged_data = CSV.read(csv_path, DataFrame)
    println("Loaded $(nrow(merged_data)) rows")
    
    # Get unique parameter combinations
    param_combinations = unique(merged_data[:, [:R0, :generation_time, :outbreak_country, :mean_detection_time]])
    println("\nFound $(nrow(param_combinations)) unique parameter combinations")
    
    # Filter out combinations where mean_detection_time is missing or > threshold
    valid_combinations = filter(row -> 
        !ismissing(row.mean_detection_time) && 
        !isnan(row.mean_detection_time) && 
        row.mean_detection_time <= max_detection_time_threshold,
        param_combinations
    )

    # # Restrict to a specific whitelist of countries
    # allowed_countries = Set(["Paraguay", "Ghana", "Switzerland"])
    # valid_combinations = filter(row ->
    #     row.outbreak_country ∈ allowed_countries,
    #     valid_combinations
    # )

    # # Remove R0 = 3.0, gen time = 4.0
    # valid_combinations = filter(row -> 
    #     !(Float64(row.R0) == 3.0) && !(Float64(row.generation_time) == 4.0),
    #     valid_combinations
    # )

    # Filter to R0 = 2.0, gen time = 6.0
    valid_combinations = filter(row -> 
        (Float64(row.R0) == 2.0) && (Float64(row.generation_time) == 6.0),
        valid_combinations
    )

    println("After filtering: $(nrow(valid_combinations)) valid combinations")
    println("(Excluded $(nrow(param_combinations) - nrow(valid_combinations)) with mean_detection_time > $max_detection_time_threshold or missing)")
    
    # SKIP PROBLEMATIC COMBINATIONS
    problematic_combinations = [
        ("Uzbekistan", 2.5, 4.0),
        ("Solomon Islands", 2.5, 4.0),
        ("Sint Maarten", 2.5, 4.0),
        ("Saint Maarten", 2.5, 4.0),
        ("Uzbekistan", 3.0, 4.0),
        ("Solomon Islands", 3.0, 4.0),
        ("Sint Maarten", 3.0, 4.0),
        ("Saint Maarten", 3.0, 4.0)
    ]

    problematic_countries = Set(["North Korea", "Uzbekistan"])

    valid_combinations = filter(row -> 
        (row.outbreak_country, Float64(row.R0), Float64(row.generation_time)) ∉ problematic_combinations,
        valid_combinations
    )

    valid_combinations = filter(row ->
        (row.outbreak_country) ∉ problematic_countries,
        valid_combinations
    )

    println("Remaining combinations to process: $(nrow(valid_combinations))")

    if isfile(output_path)
        println("\n⚠️  Found existing output file: $output_path")
        println("Loading existing results to resume...")
        
        try
            existing_results = CSV.read(output_path, DataFrame)
            
            # Parse the string representations back to vectors
            vector_cols = [:ICU_detection_times, :ICU_local_cases_samples,
                        :HARISS_detection_times, :HARISS_local_cases_samples,
                        :WW_008_10pct_detection_times, :WW_008_10pct_local_cases_samples,
                        :WW_008_25pct_detection_times, :WW_008_25pct_local_cases_samples,
                        :WW_008_50pct_detection_times, :WW_008_50pct_local_cases_samples,
                        :WW_008_100pct_detection_times, :WW_008_100pct_local_cases_samples,
                        :WW_016_10pct_detection_times, :WW_016_10pct_local_cases_samples,
                        :WW_016_25pct_detection_times, :WW_016_25pct_local_cases_samples,
                        :WW_016_50pct_detection_times, :WW_016_50pct_local_cases_samples,
                        :WW_016_100pct_detection_times, :WW_016_100pct_local_cases_samples]
            
            # Create a NEW DataFrame with correct types
            results_df = DataFrame(
                country = String[],
                R0 = Float64[],
                gen_time = Float64[],
                mean_detection_time_from_csv = Float64[],
                max_observation_time = Float64[],
                ICU_mean_detection_time = Float64[],
                ICU_detection_times = Vector{Float64}[],
                ICU_mean_local_cases = Float64[],
                ICU_local_cases_samples = Vector{Float64}[],
                HARISS_mean_detection_time = Float64[],
                HARISS_detection_times = Vector{Float64}[],
                HARISS_mean_local_cases = Float64[],
                HARISS_local_cases_samples = Vector{Float64}[],
                HARISS_detections = Int[],
                # Base p_det = 0.08
                WW_008_10pct_mean_detection_time = Float64[],
                WW_008_10pct_detection_times = Vector{Float64}[],
                WW_008_10pct_mean_local_cases = Float64[],
                WW_008_10pct_local_cases_samples = Vector{Float64}[],
                WW_008_10pct_detections = Int[],
                WW_008_25pct_mean_detection_time = Float64[],
                WW_008_25pct_detection_times = Vector{Float64}[],
                WW_008_25pct_mean_local_cases = Float64[],
                WW_008_25pct_local_cases_samples = Vector{Float64}[],
                WW_008_25pct_detections = Int[],
                WW_008_50pct_mean_detection_time = Float64[],
                WW_008_50pct_detection_times = Vector{Float64}[],
                WW_008_50pct_mean_local_cases = Float64[],
                WW_008_50pct_local_cases_samples = Vector{Float64}[],
                WW_008_50pct_detections = Int[],
                WW_008_100pct_mean_detection_time = Float64[],
                WW_008_100pct_detection_times = Vector{Float64}[],
                WW_008_100pct_mean_local_cases = Float64[],
                WW_008_100pct_local_cases_samples = Vector{Float64}[],
                WW_008_100pct_detections = Int[],
                # Base p_det = 0.16
                WW_016_10pct_mean_detection_time = Float64[],
                WW_016_10pct_detection_times = Vector{Float64}[],
                WW_016_10pct_mean_local_cases = Float64[],
                WW_016_10pct_local_cases_samples = Vector{Float64}[],
                WW_016_10pct_detections = Int[],
                WW_016_25pct_mean_detection_time = Float64[],
                WW_016_25pct_detection_times = Vector{Float64}[],
                WW_016_25pct_mean_local_cases = Float64[],
                WW_016_25pct_local_cases_samples = Vector{Float64}[],
                WW_016_25pct_detections = Int[],
                WW_016_50pct_mean_detection_time = Float64[],
                WW_016_50pct_detection_times = Vector{Float64}[],
                WW_016_50pct_mean_local_cases = Float64[],
                WW_016_50pct_local_cases_samples = Vector{Float64}[],
                WW_016_50pct_detections = Int[],
                WW_016_100pct_mean_detection_time = Float64[],
                WW_016_100pct_detection_times = Vector{Float64}[],
                WW_016_100pct_mean_local_cases = Float64[],
                WW_016_100pct_local_cases_samples = Vector{Float64}[],
                WW_016_100pct_detections = Int[],
                mean_latent_imports_per_sample = Float64[],
                mean_infectious_imports_per_sample = Float64[],
                mean_detectable_imports_per_sample = Float64[]
            )
            
            # Copy data row by row with proper type conversion
            for row in eachrow(existing_results)
                push!(results_df, (
                    country = row.country,
                    R0 = row.R0,
                    gen_time = row.gen_time,
                    mean_detection_time_from_csv = row.mean_detection_time_from_csv,
                    max_observation_time = row.max_observation_time,
                    ICU_mean_detection_time = row.ICU_mean_detection_time,
                    ICU_detection_times = eval(Meta.parse(row.ICU_detection_times)),
                    ICU_mean_local_cases = row.ICU_mean_local_cases,
                    ICU_local_cases_samples = eval(Meta.parse(row.ICU_local_cases_samples)),
                    HARISS_mean_detection_time = row.HARISS_mean_detection_time,
                    HARISS_detection_times = eval(Meta.parse(row.HARISS_detection_times)),
                    HARISS_mean_local_cases = row.HARISS_mean_local_cases,
                    HARISS_local_cases_samples = eval(Meta.parse(row.HARISS_local_cases_samples)),
                    HARISS_detections = row.HARISS_detections,
                    # Base p_det = 0.08
                    WW_008_10pct_mean_detection_time = row.WW_008_10pct_mean_detection_time,
                    WW_008_10pct_detection_times = eval(Meta.parse(row.WW_008_10pct_detection_times)),
                    WW_008_10pct_mean_local_cases = row.WW_008_10pct_mean_local_cases,
                    WW_008_10pct_local_cases_samples = eval(Meta.parse(row.WW_008_10pct_local_cases_samples)),
                    WW_008_10pct_detections = row.WW_008_10pct_detections,
                    WW_008_25pct_mean_detection_time = row.WW_008_25pct_mean_detection_time,
                    WW_008_25pct_detection_times = eval(Meta.parse(row.WW_008_25pct_detection_times)),
                    WW_008_25pct_mean_local_cases = row.WW_008_25pct_mean_local_cases,
                    WW_008_25pct_local_cases_samples = eval(Meta.parse(row.WW_008_25pct_local_cases_samples)),
                    WW_008_25pct_detections = row.WW_008_25pct_detections,
                    WW_008_50pct_mean_detection_time = row.WW_008_50pct_mean_detection_time,
                    WW_008_50pct_detection_times = eval(Meta.parse(row.WW_008_50pct_detection_times)),
                    WW_008_50pct_mean_local_cases = row.WW_008_50pct_mean_local_cases,
                    WW_008_50pct_local_cases_samples = eval(Meta.parse(row.WW_008_50pct_local_cases_samples)),
                    WW_008_50pct_detections = row.WW_008_50pct_detections,
                    WW_008_100pct_mean_detection_time = row.WW_008_100pct_mean_detection_time,
                    WW_008_100pct_detection_times = eval(Meta.parse(row.WW_008_100pct_detection_times)),
                    WW_008_100pct_mean_local_cases = row.WW_008_100pct_mean_local_cases,
                    WW_008_100pct_local_cases_samples = eval(Meta.parse(row.WW_008_100pct_local_cases_samples)),
                    WW_008_100pct_detections = row.WW_008_100pct_detections,
                    # Base p_det = 0.16
                    WW_016_10pct_mean_detection_time = row.WW_016_10pct_mean_detection_time,
                    WW_016_10pct_detection_times = eval(Meta.parse(row.WW_016_10pct_detection_times)),
                    WW_016_10pct_mean_local_cases = row.WW_016_10pct_mean_local_cases,
                    WW_016_10pct_local_cases_samples = eval(Meta.parse(row.WW_016_10pct_local_cases_samples)),
                    WW_016_10pct_detections = row.WW_016_10pct_detections,
                    WW_016_25pct_mean_detection_time = row.WW_016_25pct_mean_detection_time,
                    WW_016_25pct_detection_times = eval(Meta.parse(row.WW_016_25pct_detection_times)),
                    WW_016_25pct_mean_local_cases = row.WW_016_25pct_mean_local_cases,
                    WW_016_25pct_local_cases_samples = eval(Meta.parse(row.WW_016_25pct_local_cases_samples)),
                    WW_016_25pct_detections = row.WW_016_25pct_detections,
                    WW_016_50pct_mean_detection_time = row.WW_016_50pct_mean_detection_time,
                    WW_016_50pct_detection_times = eval(Meta.parse(row.WW_016_50pct_detection_times)),
                    WW_016_50pct_mean_local_cases = row.WW_016_50pct_mean_local_cases,
                    WW_016_50pct_local_cases_samples = eval(Meta.parse(row.WW_016_50pct_local_cases_samples)),
                    WW_016_50pct_detections = row.WW_016_50pct_detections,
                    WW_016_100pct_mean_detection_time = row.WW_016_100pct_mean_detection_time,
                    WW_016_100pct_detection_times = eval(Meta.parse(row.WW_016_100pct_detection_times)),
                    WW_016_100pct_mean_local_cases = row.WW_016_100pct_mean_local_cases,
                    WW_016_100pct_local_cases_samples = eval(Meta.parse(row.WW_016_100pct_local_cases_samples)),
                    WW_016_100pct_detections = row.WW_016_100pct_detections,
                    mean_latent_imports_per_sample = row.mean_latent_imports_per_sample,
                    mean_infectious_imports_per_sample = row.mean_infectious_imports_per_sample,
                    mean_detectable_imports_per_sample = row.mean_detectable_imports_per_sample
                ))
            end
            
            println("Found $(nrow(results_df)) existing results")
            
            # Filter out already completed combinations
            completed_keys = Set(zip(results_df.country, results_df.R0, results_df.gen_time))
            valid_combinations = filter(row -> 
                (row.outbreak_country, Float64(row.R0), Float64(row.generation_time)) ∉ completed_keys,
                valid_combinations
            )
            
        catch e
            println("ERROR loading existing file: $e")
            println("Starting fresh...")
            # Initialize empty results DataFrame if loading failed
            results_df = DataFrame(
                country = String[],
                R0 = Float64[],
                gen_time = Float64[],
                mean_detection_time_from_csv = Float64[],
                max_observation_time = Float64[],
                ICU_mean_detection_time = Float64[],
                ICU_detection_times = Vector{Float64}[],
                ICU_mean_local_cases = Float64[],
                ICU_local_cases_samples = Vector{Float64}[],
                HARISS_mean_detection_time = Float64[],
                HARISS_detection_times = Vector{Float64}[],
                HARISS_mean_local_cases = Float64[],
                HARISS_local_cases_samples = Vector{Float64}[],
                HARISS_detections = Int[],
                # Base p_det = 0.08
                WW_008_10pct_mean_detection_time = Float64[],
                WW_008_10pct_detection_times = Vector{Float64}[],
                WW_008_10pct_mean_local_cases = Float64[],
                WW_008_10pct_local_cases_samples = Vector{Float64}[],
                WW_008_10pct_detections = Int[],
                WW_008_25pct_mean_detection_time = Float64[],
                WW_008_25pct_detection_times = Vector{Float64}[],
                WW_008_25pct_mean_local_cases = Float64[],
                WW_008_25pct_local_cases_samples = Vector{Float64}[],
                WW_008_25pct_detections = Int[],
                WW_008_50pct_mean_detection_time = Float64[],
                WW_008_50pct_detection_times = Vector{Float64}[],
                WW_008_50pct_mean_local_cases = Float64[],
                WW_008_50pct_local_cases_samples = Vector{Float64}[],
                WW_008_50pct_detections = Int[],
                WW_008_100pct_mean_detection_time = Float64[],
                WW_008_100pct_detection_times = Vector{Float64}[],
                WW_008_100pct_mean_local_cases = Float64[],
                WW_008_100pct_local_cases_samples = Vector{Float64}[],
                WW_008_100pct_detections = Int[],
                # Base p_det = 0.16
                WW_016_10pct_mean_detection_time = Float64[],
                WW_016_10pct_detection_times = Vector{Float64}[],
                WW_016_10pct_mean_local_cases = Float64[],
                WW_016_10pct_local_cases_samples = Vector{Float64}[],
                WW_016_10pct_detections = Int[],
                WW_016_25pct_mean_detection_time = Float64[],
                WW_016_25pct_detection_times = Vector{Float64}[],
                WW_016_25pct_mean_local_cases = Float64[],
                WW_016_25pct_local_cases_samples = Vector{Float64}[],
                WW_016_25pct_detections = Int[],
                WW_016_50pct_mean_detection_time = Float64[],
                WW_016_50pct_detection_times = Vector{Float64}[],
                WW_016_50pct_mean_local_cases = Float64[],
                WW_016_50pct_local_cases_samples = Vector{Float64}[],
                WW_016_50pct_detections = Int[],
                WW_016_100pct_mean_detection_time = Float64[],
                WW_016_100pct_detection_times = Vector{Float64}[],
                WW_016_100pct_mean_local_cases = Float64[],
                WW_016_100pct_local_cases_samples = Vector{Float64}[],
                WW_016_100pct_detections = Int[],
                mean_latent_imports_per_sample = Float64[],
                mean_infectious_imports_per_sample = Float64[],
                mean_detectable_imports_per_sample = Float64[]
            )
        end
    else
        # Initialize empty results DataFrame
        results_df = DataFrame(
            country = String[],
            R0 = Float64[],
            gen_time = Float64[],
            mean_detection_time_from_csv = Float64[],
            max_observation_time = Float64[],
            ICU_mean_detection_time = Float64[],
            ICU_detection_times = Vector{Float64}[],
            ICU_mean_local_cases = Float64[],
            ICU_local_cases_samples = Vector{Float64}[],
            HARISS_mean_detection_time = Float64[],
            HARISS_detection_times = Vector{Float64}[],
            HARISS_mean_local_cases = Float64[],
            HARISS_local_cases_samples = Vector{Float64}[],
            HARISS_detections = Int[],
            # Base p_det = 0.08
            WW_008_10pct_mean_detection_time = Float64[],
            WW_008_10pct_detection_times = Vector{Float64}[],
            WW_008_10pct_mean_local_cases = Float64[],
            WW_008_10pct_local_cases_samples = Vector{Float64}[],
            WW_008_10pct_detections = Int[],
            WW_008_25pct_mean_detection_time = Float64[],
            WW_008_25pct_detection_times = Vector{Float64}[],
            WW_008_25pct_mean_local_cases = Float64[],
            WW_008_25pct_local_cases_samples = Vector{Float64}[],
            WW_008_25pct_detections = Int[],
            WW_008_50pct_mean_detection_time = Float64[],
            WW_008_50pct_detection_times = Vector{Float64}[],
            WW_008_50pct_mean_local_cases = Float64[],
            WW_008_50pct_local_cases_samples = Vector{Float64}[],
            WW_008_50pct_detections = Int[],
            WW_008_100pct_mean_detection_time = Float64[],
            WW_008_100pct_detection_times = Vector{Float64}[],
            WW_008_100pct_mean_local_cases = Float64[],
            WW_008_100pct_local_cases_samples = Vector{Float64}[],
            WW_008_100pct_detections = Int[],
            # Base p_det = 0.16
            WW_016_10pct_mean_detection_time = Float64[],
            WW_016_10pct_detection_times = Vector{Float64}[],
            WW_016_10pct_mean_local_cases = Float64[],
            WW_016_10pct_local_cases_samples = Vector{Float64}[],
            WW_016_10pct_detections = Int[],
            WW_016_25pct_mean_detection_time = Float64[],
            WW_016_25pct_detection_times = Vector{Float64}[],
            WW_016_25pct_mean_local_cases = Float64[],
            WW_016_25pct_local_cases_samples = Vector{Float64}[],
            WW_016_25pct_detections = Int[],
            WW_016_50pct_mean_detection_time = Float64[],
            WW_016_50pct_detection_times = Vector{Float64}[],
            WW_016_50pct_mean_local_cases = Float64[],
            WW_016_50pct_local_cases_samples = Vector{Float64}[],
            WW_016_50pct_detections = Int[],
            WW_016_100pct_mean_detection_time = Float64[],
            WW_016_100pct_detection_times = Vector{Float64}[],
            WW_016_100pct_mean_local_cases = Float64[],
            WW_016_100pct_local_cases_samples = Vector{Float64}[],
            WW_016_100pct_detections = Int[],
            mean_latent_imports_per_sample = Float64[],
            mean_infectious_imports_per_sample = Float64[],
            mean_detectable_imports_per_sample = Float64[]
        )
    end
    
    total_combinations = nrow(valid_combinations)
    
    if total_combinations == 0
        println("\n✓ All combinations already processed!")
        return results_df
    end
    
    # Calculate number of batches
    num_batches = ceil(Int, total_combinations / batch_size)
    println("\nProcessing in $num_batches batches of $batch_size combinations each")
    println("="^80)
    flush(stdout)
    
    # Airport detection model:
    # - base_pdet ∈ {0.08, 0.16} (per-flight detection probability)
    # - sampling_fraction ∈ {10%, 25%, 50%, 100%} (proportion of flights tested)
    # - effective p_det = base_pdet × sampling_fraction
    base_pdets = [0.08, 0.16]
    sampling_fractions = [0.10, 0.25, 0.50, 1.0]  # 10%, 25%, 50%, 100%
    
    # Generate all combinations
    p_dets = Float64[]
    config_labels = String[]
    for base_pdet in base_pdets
        for sampling_frac in sampling_fractions
            push!(p_dets, base_pdet * sampling_frac)
            # Format: base_pdet as percentage (e.g., 0.08 → "008", 0.16 → "016")
            base_pct = Int(round(base_pdet * 100))  # 0.08 → 4, 0.16 → 16
            base_str = lpad(string(base_pct), 3, '0')  # 4 → "008", 16 → "016"
            sampling_pct = Int(round(sampling_frac * 100))  # 0.10 → 10, 0.25 → 25
            pct_str = string(sampling_pct) * "pct"
            push!(config_labels, "$(base_str)_$(pct_str)")
        end
    end
    
    println("\nAirport detection parameters:")
    println("  Base p_det values: $base_pdets")
    println("  Sampling fractions: $(sampling_fractions .* 100)%")
    println("  Total WW configurations: $(length(p_dets))")
    for (i, (label, pdet)) in enumerate(zip(config_labels, p_dets))
        println("    $label: p_det = $pdet")
    end
    println()
    
    # Process in batches
    for batch_num in 1:num_batches
        batch_start = (batch_num - 1) * batch_size + 1
        batch_end = min(batch_num * batch_size, total_combinations)
        batch_combinations = valid_combinations[batch_start:batch_end, :]
        
        println("\n" * "="^80)
        println("BATCH $batch_num/$num_batches: Processing combinations $batch_start to $batch_end")
        println("="^80)
        flush(stdout)
        
        # --------------------------------------------------------------
        # Phase 1 (main process): validate each combination, pre-filter the
        # country data once, and build a per-combo spec. Doing the filter
        # here (instead of inside every pmap task) avoids re-scanning the
        # 900k-row merged table from every worker, every sample.
        # --------------------------------------------------------------
        # Phase 1a: data filtering (sequential, fast — avoids re-scanning
        # the merged table from workers).
        combo_specs_no_cache = NamedTuple[]
        for (i, param_row) in enumerate(eachrow(batch_combinations))
            global_idx = batch_start + i - 1
            R0 = Float64(param_row.R0)
            gen_time = Float64(param_row.generation_time)
            country = param_row.outbreak_country
            mean_det_time = param_row.mean_detection_time
            max_obs_time = mean_det_time + extra_time

            country_data = filter(row ->
                row.R0 == R0 &&
                row.generation_time == gen_time &&
                row.outbreak_country == country,
                merged_data
            )

            if nrow(country_data) == 0
                println("[$global_idx/$total_combinations] WARNING: No data for $country (R0=$R0, gen_time=$gen_time), skipping...")
                continue
            end

            sort!(country_data, :time)
            country_trimmed = filter(row -> row.time <= max_obs_time, country_data)

            if nrow(country_trimmed) == 0
                println("[$global_idx/$total_combinations] WARNING: No data within observation window for $country, skipping...")
                continue
            end

            push!(combo_specs_no_cache, (
                global_idx    = global_idx,
                country       = country,
                R0            = R0,
                gen_time      = gen_time,
                mean_det_time = mean_det_time,
                max_obs_time  = max_obs_time,
                country_data  = country_trimmed,
            ))
        end

        # Phase 1b: build one hariss_bg_cache per combination in parallel.
        # build_ari_background is independent of importation data — only
        # max_obs_time varies per combo; all other params are NHS config
        # constants already resident on every worker. With 180 workers all
        # 125 builds run simultaneously (~1 round × 7-10s) rather than
        # sequentially on the main process (125 × 7-10s ≈ 950s).
        println("  Pre-building $(length(combo_specs_no_cache)) HARISS background caches in parallel...")
        flush(stdout)
        hariss_caches = pmap(combo_specs_no_cache) do spec
            NBPMscape.build_ari_background(;
                max_observation_time             = Float64(spec.max_obs_time),
                initial_dow                      = P_FROM_CONFIG.initial_dow,
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
                ed_discharge_limit               = Float64(P_FROM_CONFIG.tdischarge_ed_upper_limit),
                hosp_short_stay_limit            = Float64(P_FROM_CONFIG.tdischarge_hosp_short_stay_upper_limit),
                hosp_ari_admissions              = Int(P_FROM_CONFIG.hosp_ari_admissions),
                hosp_ari_admissions_adult_p      = Float64(P_FROM_CONFIG.hosp_ari_admissions_adult_p),
                hosp_ari_admissions_child_p      = Float64(P_FROM_CONFIG.hosp_ari_admissions_child_p),
                ed_ari_destinations_adult        = P_FROM_CONFIG.ed_ari_destinations_adult,
                ed_ari_destinations_child        = P_FROM_CONFIG.ed_ari_destinations_child,
            )
        end

        combo_specs = [(; spec..., hariss_bg_cache = cache)
                       for (spec, cache) in zip(combo_specs_no_cache, hariss_caches)]

        if isempty(combo_specs)
            println("No valid combinations in this batch, skipping.")
            continue
        end

        # --------------------------------------------------------------
        # Phase 2: flatten (combination x sample) into a single pmap task
        # list. Each task runs ONE Monte-Carlo sample, so big-population
        # countries no longer block a worker serially through 100 samples.
        # --------------------------------------------------------------
        sample_tasks = Tuple{Int, Int}[]  # (combo_idx_in_specs, sample_id)
        for (ci, _) in enumerate(combo_specs)
            for s in 1:num_samples
                push!(sample_tasks, (ci, s))
            end
        end

        println("Dispatching $(length(sample_tasks)) sample tasks ",
                "($(length(combo_specs)) combinations × $num_samples samples) ",
                "to $(nworkers()) workers...")
        flush(stdout)

        # CRITICAL: use a CachingPool so the closure (which captures the
        # ~6 MB `combo_specs` bundle of per-country DataFrames) is shipped
        # to each worker ONCE rather than being re-serialised on every
        # one of the thousands of sample tasks. Without the caching pool,
        # the driver process becomes a serialisation bottleneck and
        # workers sit idle waiting for tasks to arrive.
        # `batch_size` also groups small tasks per dispatch to amortise
        # the remaining per-task overhead.
        pool = CachingPool(workers())
        sample_outputs = try
            pmap(pool, sample_tasks; batch_size = 4) do task
                ci, sample_id = task
                spec = combo_specs[ci]
                try
                    res = simulate_single_sample(
                        spec.country_data,
                        spec.country,
                        sample_id,
                        spec.R0,
                        spec.gen_time,
                        icu_sampling_proportion,
                        p_dets,
                        spec.max_obs_time,
                        spec.hariss_bg_cache;
                        turnaround_time = turnaround_time,
                        n_hosp_samples_per_week = n_hosp_samples_per_week
                    )
                    icu_str = if res.icu_detected && isfinite(res.icu_detection_time)
                        string(round(Float64(res.icu_detection_time), digits=2)) * "d"
                    else
                        "not detected"
                    end
                    har_str = if res.hariss_detected && isfinite(res.hariss_detection_time)
                        string(round(Float64(res.hariss_detection_time), digits=2)) * "d"
                    else
                        "not detected"
                    end
                    # Runs on workers — lines may interleave; includes batch for context.
                    println("[$(spec.global_idx)/$total_combinations] [batch $batch_num/$num_batches] ",
                            "Simulation $sample_id/$num_samples complete: ",
                            "country=$(spec.country), R0=$(spec.R0), gen_time=$(spec.gen_time), ",
                            "ICU detection time=$icu_str, HARISS detection time=$har_str")
                    flush(stdout)
                    return (ci = ci, ok = true, result = res, err = nothing)
                catch e
                    println("[$(spec.global_idx)/$total_combinations] [batch $batch_num/$num_batches] ",
                            "Simulation $sample_id/$num_samples FAILED: country=$(spec.country), ",
                            "R0=$(spec.R0), gen_time=$(spec.gen_time), error=$e")
                    flush(stdout)
                    return (ci = ci, ok = false, result = nothing, err = e)
                end
            end
        finally
            clear!(pool)
        end

        # --------------------------------------------------------------
        # Phase 3 (main process): group sample outputs by combination and
        # aggregate into the same per-combo NamedTuple the old pmap used
        # to return, so the downstream CSV schema is unchanged.
        # --------------------------------------------------------------
        per_combo_samples = [Any[] for _ in 1:length(combo_specs)]
        per_combo_errors  = [Any[] for _ in 1:length(combo_specs)]
        for out in sample_outputs
            if out.ok
                push!(per_combo_samples[out.ci], out.result)
            else
                push!(per_combo_errors[out.ci], out.err)
            end
        end

        batch_results = Vector{Any}(undef, length(combo_specs))
        for (ci, spec) in enumerate(combo_specs)
            samples = per_combo_samples[ci]
            errs = per_combo_errors[ci]

            if !isempty(errs)
                # Surface the first error per combo but keep going with any
                # successful samples (same permissive behaviour as before).
                println("[$(spec.global_idx)/$total_combinations] WARN $(spec.country): $(length(errs))/$num_samples sample(s) failed -- first error: $(errs[1])")
            end

            if isempty(samples)
                println("[$(spec.global_idx)/$total_combinations] ERROR $(spec.country): all samples failed, skipping row")
                batch_results[ci] = nothing
                continue
            end

            agg = aggregate_sample_results(samples, p_dets)

            icu_mean_time = isempty(agg.icu_detection_times) ? NaN : mean(agg.icu_detection_times)
            icu_mean_cases = isempty(agg.icu_local_cases_at_detection) ? NaN : mean(agg.icu_local_cases_at_detection)
            hariss_mean_time = isempty(agg.hariss_detection_times) ? NaN : mean(agg.hariss_detection_times)
            hariss_mean_cases = isempty(agg.hariss_local_cases_at_detection) ? NaN : mean(agg.hariss_local_cases_at_detection)
            hariss_num_detections = length(agg.hariss_detection_times)

            ww_results_by_config = Dict{String, NamedTuple}()
            for (label, p_det) in zip(config_labels, p_dets)
                ww_mean_time = isempty(agg.airport_results[p_det][:detection_times]) ? NaN :
                               mean(agg.airport_results[p_det][:detection_times])
                ww_mean_cases = isempty(agg.airport_results[p_det][:local_cases_at_detection]) ? NaN :
                                mean(agg.airport_results[p_det][:local_cases_at_detection])
                ww_results_by_config[label] = (
                    detection_times = agg.airport_results[p_det][:detection_times],
                    mean_detection_time = ww_mean_time,
                    local_cases_samples = agg.airport_results[p_det][:local_cases_at_detection],
                    mean_local_cases = ww_mean_cases,
                    num_detections = length(agg.airport_results[p_det][:detection_times])
                )
            end

            println("[$(spec.global_idx)/$total_combinations] ✓ $(spec.country): ICU=$(round(icu_mean_time, digits=2))d HARISS=$(round(hariss_mean_time, digits=2))d")

            batch_results[ci] = (
                country = spec.country,
                R0 = spec.R0,
                gen_time = spec.gen_time,
                mean_detection_time_from_csv = spec.mean_det_time,
                max_observation_time = spec.max_obs_time,
                ICU_mean_detection_time = icu_mean_time,
                ICU_detection_times = agg.icu_detection_times,
                ICU_mean_local_cases = icu_mean_cases,
                ICU_local_cases_samples = agg.icu_local_cases_at_detection,
                HARISS_mean_detection_time = hariss_mean_time,
                HARISS_detection_times = agg.hariss_detection_times,
                HARISS_mean_local_cases = hariss_mean_cases,
                HARISS_local_cases_samples = agg.hariss_local_cases_at_detection,
                HARISS_detections = hariss_num_detections,
                # Base p_det = 0.08, 10% sampling
                WW_008_10pct_mean_detection_time = ww_results_by_config["008_10pct"].mean_detection_time,
                WW_008_10pct_detection_times = ww_results_by_config["008_10pct"].detection_times,
                WW_008_10pct_mean_local_cases = ww_results_by_config["008_10pct"].mean_local_cases,
                WW_008_10pct_local_cases_samples = ww_results_by_config["008_10pct"].local_cases_samples,
                WW_008_10pct_detections = ww_results_by_config["008_10pct"].num_detections,
                # Base p_det = 0.08, 25% sampling
                WW_008_25pct_mean_detection_time = ww_results_by_config["008_25pct"].mean_detection_time,
                WW_008_25pct_detection_times = ww_results_by_config["008_25pct"].detection_times,
                WW_008_25pct_mean_local_cases = ww_results_by_config["008_25pct"].mean_local_cases,
                WW_008_25pct_local_cases_samples = ww_results_by_config["008_25pct"].local_cases_samples,
                WW_008_25pct_detections = ww_results_by_config["008_25pct"].num_detections,
                # Base p_det = 0.08, 50% sampling
                WW_008_50pct_mean_detection_time = ww_results_by_config["008_50pct"].mean_detection_time,
                WW_008_50pct_detection_times = ww_results_by_config["008_50pct"].detection_times,
                WW_008_50pct_mean_local_cases = ww_results_by_config["008_50pct"].mean_local_cases,
                WW_008_50pct_local_cases_samples = ww_results_by_config["008_50pct"].local_cases_samples,
                WW_008_50pct_detections = ww_results_by_config["008_50pct"].num_detections,
                # Base p_det = 0.08, 100% sampling
                WW_008_100pct_mean_detection_time = ww_results_by_config["008_100pct"].mean_detection_time,
                WW_008_100pct_detection_times = ww_results_by_config["008_100pct"].detection_times,
                WW_008_100pct_mean_local_cases = ww_results_by_config["008_100pct"].mean_local_cases,
                WW_008_100pct_local_cases_samples = ww_results_by_config["008_100pct"].local_cases_samples,
                WW_008_100pct_detections = ww_results_by_config["008_100pct"].num_detections,
                # Base p_det = 0.16, 10% sampling
                WW_016_10pct_mean_detection_time = ww_results_by_config["016_10pct"].mean_detection_time,
                WW_016_10pct_detection_times = ww_results_by_config["016_10pct"].detection_times,
                WW_016_10pct_mean_local_cases = ww_results_by_config["016_10pct"].mean_local_cases,
                WW_016_10pct_local_cases_samples = ww_results_by_config["016_10pct"].local_cases_samples,
                WW_016_10pct_detections = ww_results_by_config["016_10pct"].num_detections,
                # Base p_det = 0.16, 25% sampling
                WW_016_25pct_mean_detection_time = ww_results_by_config["016_25pct"].mean_detection_time,
                WW_016_25pct_detection_times = ww_results_by_config["016_25pct"].detection_times,
                WW_016_25pct_mean_local_cases = ww_results_by_config["016_25pct"].mean_local_cases,
                WW_016_25pct_local_cases_samples = ww_results_by_config["016_25pct"].local_cases_samples,
                WW_016_25pct_detections = ww_results_by_config["016_25pct"].num_detections,
                # Base p_det = 0.16, 50% sampling
                WW_016_50pct_mean_detection_time = ww_results_by_config["016_50pct"].mean_detection_time,
                WW_016_50pct_detection_times = ww_results_by_config["016_50pct"].detection_times,
                WW_016_50pct_mean_local_cases = ww_results_by_config["016_50pct"].mean_local_cases,
                WW_016_50pct_local_cases_samples = ww_results_by_config["016_50pct"].local_cases_samples,
                WW_016_50pct_detections = ww_results_by_config["016_50pct"].num_detections,
                # Base p_det = 0.16, 100% sampling
                WW_016_100pct_mean_detection_time = ww_results_by_config["016_100pct"].mean_detection_time,
                WW_016_100pct_detection_times = ww_results_by_config["016_100pct"].detection_times,
                WW_016_100pct_mean_local_cases = ww_results_by_config["016_100pct"].mean_local_cases,
                WW_016_100pct_local_cases_samples = ww_results_by_config["016_100pct"].local_cases_samples,
                WW_016_100pct_detections = ww_results_by_config["016_100pct"].num_detections,
                mean_latent_imports_per_sample = agg.mean_latent_imports_per_sample,
                mean_infectious_imports_per_sample = agg.mean_infectious_imports_per_sample,
                mean_detectable_imports_per_sample = agg.mean_detectable_imports_per_sample
            )
        end

        # Filter out nothing results and append to existing results
        valid_results = filter(x -> !isnothing(x), batch_results)
        
        if !isempty(valid_results)
            new_results_df = DataFrame(valid_results)
            append!(results_df, new_results_df)
        end
        
        # SAVE AFTER EACH BATCH using safe write function
        save_success = safe_csv_write(output_path, results_df)
        
        if save_success
            println("\n💾 BATCH $batch_num/$num_batches COMPLETE - Saved progress")
            println("   Total rows in CSV: $(nrow(results_df))")
            println("   Combinations remaining: $(total_combinations - batch_end)")
        else
            println("\n⚠️  BATCH $batch_num/$num_batches COMPLETE - Saved to backup location")
            println("   Total rows processed: $(nrow(results_df))")
            println("   Combinations remaining: $(total_combinations - batch_end)")
        end
        flush(stdout)
    end
    
    println("\n" * "="^80)
    println("ALL SIMULATIONS COMPLETE")
    println("Results saved to: $output_path")
    println("Total rows: $(nrow(results_df))")
    println("="^80)
    
    return results_df
end

# ============================================================================
# RUN THE SIMULATIONS
# ============================================================================

# Resolve paths relative to the repository root so the script is portable
project_root = normpath(joinpath(@__DIR__, "..", ".."))
input_csv_path = "/Users/reddy/AWW_and_ICU/global_model/pgfgleam/all_results/global/daily_imports_sensitivity.csv"
output_csv_path = "/Users/reddy/AWW_and_ICU/global_model/pgfgleam/all_results/local/full_ICU_AWW_HARISS_result.csv"

results = run_simulations_from_merged_csv(
    input_csv_path;
    num_samples = 100,
    turnaround_time = 3.0,
    max_detection_time_threshold = 200.0,
    extra_time = 35.0,
    icu_sampling_proportion = 0.10,
    n_hosp_samples_per_week = Int(P_FROM_CONFIG.n_hosp_samples_per_week),
    output_path = output_csv_path,
    batch_size = 125
)

println("\n✓ ICU + AWW + HARISS simulations complete!")
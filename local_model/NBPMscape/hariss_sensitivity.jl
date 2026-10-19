# ============================================================================
# HARISS SENSITIVITY ANALYSIS
# ============================================================================
#
# Tests how HARISS (and ICU) detection time responds to the parameter updates
# in upstream emvolz/NBPMscape@34031ab, for a small set of fixed outbreak
# origins, instead of re-running the full all-country analysis.
#
# Parameters varied:
#   (1) Healthcare pathway rates
#         :original  - rates in config/outbreak_params_<disease>.yaml (main analysis)
#         :corrected - halved competing-route rates from *_20261008.yaml, so the
#                      mean onset-to-admission matches the literature value
#   (2) Share of ED ARI attendances sequenced (= weekly samples / ED ARI
#       background B, set via `hosp_ari_admissions`)
#   (3) Ratio of swab proportions, admitted : discharged from ED, applied
#       identically to pathogen X cases and to the background
#
# Design: per (country, sample) the imports are drawn once and the outbreak
# tree is grown once per rate set. Every HARISS scenario is then evaluated on
# that same tree, so differences between scenarios reflect the HARISS
# settings rather than tree-to-tree noise. AWW only depends on the imports,
# so it is computed once per sample.
#
# Swab proportions are applied here rather than inside NBPMscape: the
# background is built with everyone swabbed and then thinned by ED
# destination, and pathogen X hospital cases are thinned by whether they
# were admitted; secondary_care_td is then called with
# proportion_hosp_swabbed = 1.0. With a uniform proportion this matches the
# package behaviour.
#
# Output: one row per (country, sample, rate_set, B, swab ratio).
# Plot with scripts/plot_hariss_sensitivity.py.
#
# Usage (defaults shown; N_WORKERS defaults to CPU threads - 1, max 24):
#   DISEASE=covid N_SAMPLES=100 R0=2.0 GEN_TIME=4.0 MAX_CASES=1000 \
#   COUNTRIES="Spain;Germany;United States;Japan" N_WORKERS=7 \
#   julia --project=local_model local_model/NBPMscape/hariss_sensitivity.jl
#
# Quick timing test before a full run:
#   N_SAMPLES=4 COUNTRIES=Spain N_WORKERS=4 julia --project=local_model local_model/NBPMscape/hariss_sensitivity.jl
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
using Random
using Dates

# Workers: N_WORKERS if set, else the SLURM allocation minus one (main
# process), else this machine's CPU threads minus one. Capped at 24.
let n = haskey(ENV, "N_WORKERS")           ? parse(Int, ENV["N_WORKERS"]) :
        haskey(ENV, "SLURM_CPUS_PER_TASK") ? parse(Int, ENV["SLURM_CPUS_PER_TASK"]) - 1 :
                                             Sys.CPU_THREADS - 1
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

# ============================================================================
# SETTINGS
# ============================================================================

const DISEASE   = lowercase(get(ENV, "DISEASE", "covid"))       # "covid" or "influenza"
const N_SAMPLES = parse(Int, get(ENV, "N_SAMPLES", "100"))
# The influenza import CSV only contains R0 = 1.5, generation time = 3.5
const R0_VALUE  = parse(Float64, get(ENV, "R0", DISEASE == "influenza" ? "1.5" : "2.0"))
const GEN_TIME  = parse(Float64, get(ENV, "GEN_TIME", DISEASE == "influenza" ? "3.5" : "4.0"))  # 4.0 = main analysis
const COUNTRIES = String.(split(get(ENV, "COUNTRIES", "Spain;Germany;United States;Japan"), ";"))

const EXTRA_TIME = 50.0   # as in the main analysis call
@everywhere const TURNAROUND_TIME         = 3.0
@everywhere const ICU_SAMPLING_PROPORTION = 0.10
# Per-tree cap in simtree (stops a tree once it exceeds this many infections).
# Must be well above the local case counts at which ICU/HARISS detect, or
# detection is truncated.
@everywhere const MAX_CASES = $(parse(Int, get(ENV, "MAX_CASES", "1000")))

# Grid axis 1: share of ED ARI attendances sequenced. With a uniform swab
# proportion, a case's chance of being sequenced is ~ n_hosp_samples_per_week / B
# (the swab proportion cancels), so the background B enters only through this
# share. Each share sets B = 300 / share:
#   4.93% (main analysis, B = 6088), 3%, 2%, 1.5%, 1%, 0.5%.
# The updated estimates (summer 15856 -> 1.9%, winter 27894 -> 1.1%) fall inside
# the range and are marked on the plots rather than used as grid points.
const B_VALUES = [6088, 10000, 15000, 20000, 30000, 60000]

# Grid axis 2: ratio of swab proportions, admitted : discharged from ED.
# Admitted fixed at 90%; ratio 1 = same proportion for all (main analysis),
# ratio 18 = ~5% of discharged swabbed (upstream suggestion).
const SWAB_SCENARIOS = [
    (name = "ratio_1",   discharged = 0.90, admitted = 0.90),
    (name = "ratio_2",   discharged = 0.45, admitted = 0.90),
    (name = "ratio_4.5", discharged = 0.20, admitted = 0.90),
    (name = "ratio_18",  discharged = 0.05, admitted = 0.90),
]

# Full grid on corrected rates, plus the main-analysis configuration on the
# original rates; comparing the two at (6088, ratio_1) isolates the effect of
# the pathway-rate fix.
const MAIN_B    = 6088
const MAIN_SWAB = "ratio_1"

# AWW: effective p_det = base_pdet x sampling_fraction. Only the weakest
# (8% x 10%) and strongest (16% x 100%) configurations are recorded, which
# bound the AWW lead time across the main analysis.
@everywhere const AWW_PDET_LOW  = 0.08 * 0.10
@everywhere const AWW_PDET_HIGH = 0.16 * 1.0
@everywhere const AWW_PDETS     = (AWW_PDET_LOW, AWW_PDET_HIGH)

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT_CSV = joinpath(PROJECT_ROOT, "global_model", "pgfgleam", "all_results", "global",
    DISEASE == "influenza" ? "daily_imports_influenza.csv" : "daily_imports_sensitivity.csv")
const OUTPUT_CSV = joinpath(PROJECT_ROOT, "global_model", "pgfgleam", "all_results", "local",
    "hariss_sensitivity_$(DISEASE)_R0_$(R0_VALUE)_GT_$(GEN_TIME).csv")

DISEASE in ("covid", "influenza") || error("DISEASE must be \"covid\" or \"influenza\", got \"$DISEASE\"")

@everywhere const DISEASE_W = $DISEASE
@everywhere const ORIGINAL_CONFIG  = DISEASE_W == "influenza" ?
    "config/outbreak_params_influenza_like.yaml" : "config/outbreak_params_covid19_like.yaml"
@everywhere const CORRECTED_CONFIG = DISEASE_W == "influenza" ?
    "config/outbreak_params_influenza_like_20261008.yaml" : "config/outbreak_params_covid19_like_20261008.yaml"

# Calibrated infectivity for each R0 (from inftoR.jl, as in the main drivers)
@everywhere const INFECTIVITY_FOR_R0 = DISEASE_W == "influenza" ?
    Dict(1.5 => 10.813340, 2.0 => 14.417787, 2.5 => 18.022233, 3.0 => 21.626680) :
    Dict(1.5 => 0.679860,  2.0 => 0.906480,  2.5 => 1.133100,  3.0 => 1.359720)
# 8/3 days, Verity et al (2020). Influenza 0.99, Cori et al (2012)
@everywhere const MEAN_INFECTIOUS_PERIOD = DISEASE_W == "influenza" ? 0.99 : 8/3

# Only these keys differ between the two rate sets
@everywhere const RATE_KEYS = (:gp_only_rate, :ed_direct_rate, :gp_before_hosp_rate,
                               :ed_from_gp_rate, :hosp_admit_direct_rate, :hosp_admit_from_gp_rate)

# ============================================================================
# CONFIG LOADING (same mapping as full_ICU_AWW_HARISS_update.jl)
# ============================================================================

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

@everywhere function load_params(rel_path::String)
    config = NBPMscape.load_config(joinpath(pkgdir(NBPMscape), rel_path))
    p = apply_yaml_scalars(NBPMscape.P, config)
    cfg = get(config, "parameters", Dict())
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
    sites_rel = get(cfg, "hariss_nhs_trust_sampling_sites_file", nothing)
    sites = if sites_rel === nothing
        NBPMscape.P.hariss_nhs_trust_sampling_sites
    else
        full = joinpath(pkgdir(NBPMscape), sites_rel)
        isfile(full) ? CSV.read(full, DataFrame) : NBPMscape.P.hariss_nhs_trust_sampling_sites
    end
    return merge(p, (ed_ari_destinations_adult = ed_adult,
                     ed_ari_destinations_child = ed_child,
                     hariss_sites              = sites))
end

# Everything except the healthcare rates comes from the original config so
# the only differences between scenarios are the ones being tested.
@everywhere const P_ORIGINAL  = load_params(ORIGINAL_CONFIG)
@everywhere const P_CORRECTED = merge(P_ORIGINAL,
    NamedTuple{RATE_KEYS}(getfield.(Ref(load_params(CORRECTED_CONFIG)), RATE_KEYS)))
@everywhere const RATE_SETS = (original = P_ORIGINAL, corrected = P_CORRECTED)

# ============================================================================
# PER-SAMPLE SIMULATION
# ============================================================================

@everywhere function sample_imports(country_data::DataFrame, col::Symbol)
    counts = zeros(Int, nrow(country_data))
    for (i, v) in enumerate(country_data[!, col])
        if !ismissing(v) && isfinite(v) && v > 0
            counts[i] = rand(Poisson(v))
        end
    end
    return counts
end

# Grow the local outbreak from the sampled imports. Returns infection times,
# ICU (ticu, trecovered) and the hospital/ED cases HARISS can sample.
@everywhere function grow_outbreak(country_data, latent_counts, infectious_counts,
                                   base_params, infectious_params, latent_period,
                                   max_obs_time)
    local_tinf = Float64[]
    icu_ticu = Float64[]
    icu_trecovered = Float64[]
    hosp_rows = DataFrame[]
    for (idx, row) in enumerate(eachrow(country_data))
        time = row.time
        time >= max_obs_time && break
        for (counts, params, t0_offset) in (
                (latent_counts,     base_params,       -latent_period / 2.0),
                (infectious_counts, infectious_params,  0.0),
            )
            n = counts[idx]
            n == 0 && continue
            t0 = Float64(time) + t0_offset
            for _ in 1:n
                G = NBPMscape.simtree(params,
                    initialtime    = t0,
                    maxtime        = max_obs_time,
                    maxgenerations = 100,
                    initialcontact = :G,
                    max_cases      = MAX_CASES,
                ).G
                nrow(G) == 0 && continue
                G = G[G.generation .!= 0, :]
                append!(local_tinf, G.tinf)
                icu = isfinite.(G.ticu)
                append!(icu_ticu, G.ticu[icu])
                append!(icu_trecovered, G.trecovered[icu])
                hosp = isfinite.(G.ted) .| isfinite.(G.thospital)
                any(hosp) && push!(hosp_rows, G[hosp, [:pid, :tinf, :tgp, :ted, :thospital, :ticu,
                    :tstepdown, :tdischarge, :trecovered, :tdeceased, :severity, :fatal,
                    :iscommuter, :homeregion, :simid, :infectee_age, :importedinfection]])
            end
        end
    end
    hosp_df = isempty(hosp_rows) ? DataFrame() : reduce(vcat, hosp_rows)
    return (local_tinf = local_tinf, icu_ticu = icu_ticu,
            icu_trecovered = icu_trecovered, hosp_df = hosp_df)
end

@everywhere function background_for(p, B::Int, max_obs_time::Float64)
    NBPMscape.build_ari_background(;
        max_observation_time            = max_obs_time,
        n_hosp_samples_per_week         = Int(p.n_hosp_samples_per_week),
        sample_allocation               = p.sample_allocation,
        sample_proportion_adult         = p.sample_proportion_adult,
        hariss_nhs_trust_sampling_sites = p.hariss_sites,
        weight_samples_by               = p.weight_samples_by,
        swab_time_mode                  = p.swab_time_mode,
        swab_proportion_at_48h          = p.swab_proportion_at_48h,
        proportion_hosp_swabbed         = 1.0,   # thinned per scenario below
        ed_discharge_limit              = Float64(p.tdischarge_ed_upper_limit),
        hosp_short_stay_limit           = Float64(p.tdischarge_hosp_short_stay_upper_limit),
        hosp_ari_admissions             = B,
        hosp_ari_admissions_adult_p     = Float64(p.hosp_ari_admissions_adult_p),
        hosp_ari_admissions_child_p     = Float64(p.hosp_ari_admissions_child_p),
        ed_ari_destinations_adult       = p.ed_ari_destinations_adult,
        ed_ari_destinations_child       = p.ed_ari_destinations_child,
    )
end

# Keep each background swab with the probability for its ED destination
@everywhere function thin_background(bg, swab)
    keep_p = ifelse.(bg.bg_df.ED_destination .== :discharged, swab.discharged, swab.admitted)
    bg_df = bg.bg_df[rand(nrow(bg.bg_df)) .< keep_p, :]
    return merge(bg, (bg_df = bg_df, bg_by_phl = NBPMscape._group_df_by_phl(bg_df)))
end

# Keep each pathogen X ED/hospital case with the probability for its route
@everywhere function thin_cases(hosp_df::DataFrame, swab)
    nrow(hosp_df) == 0 && return hosp_df
    keep_p =ifelse.(isfinite.(hosp_df.thospital), swab.admitted, swab.discharged)
    return hosp_df[rand(nrow(hosp_df)) .< keep_p, :]
end

@everywhere function hariss_time(p, base_params, hosp_df::DataFrame, bg, B::Int, max_obs_time)
    nrow(hosp_df) == 0 && return Inf
    try
        res = redirect_stdout(devnull) do
            NBPMscape.secondary_care_td(;
                p                                = base_params,
                sims                             = [hosp_df],
                pathogen_type                    = p.pathogen_type,
                initial_dow                      = p.initial_dow,
                hariss_courier_to_analysis       = p.hariss_courier_to_analysis,
                hariss_turnaround_time           = [TURNAROUND_TIME, TURNAROUND_TIME + 1e-6],
                n_hosp_samples_per_week          = Int(p.n_hosp_samples_per_week),
                sample_allocation                = p.sample_allocation,
                sample_proportion_adult          = p.sample_proportion_adult,
                hariss_nhs_trust_sampling_sites  = p.hariss_sites,
                weight_samples_by                = p.weight_samples_by,
                phl_collection_dow               = Vector{Int64}(p.phl_collection_dow),
                phl_collection_time              = Float64(p.phl_collection_time),
                hosp_to_phl_cutoff_time_relative = p.hosp_to_phl_cutoff_time_relative,
                swab_time_mode                   = p.swab_time_mode,
                swab_proportion_at_48h           = p.swab_proportion_at_48h,
                proportion_hosp_swabbed          = 1.0,   # already thinned
                only_sample_before_death         = p.hariss_only_sample_before_death,
                ed_discharge_limit               = Float64(p.tdischarge_ed_upper_limit),
                hosp_short_stay_limit            = Float64(p.tdischarge_hosp_short_stay_upper_limit),
                hosp_ari_admissions              = B,
                hosp_ari_admissions_adult_p      = Float64(p.hosp_ari_admissions_adult_p),
                hosp_ari_admissions_child_p      = Float64(p.hosp_ari_admissions_child_p),
                ed_ari_destinations_adult        = p.ed_ari_destinations_adult,
                ed_ari_destinations_child        = p.ed_ari_destinations_child,
                precomputed_ari_bg               = bg,
            )
        end
        if nrow(res) > 0 && :SC_TD in propertynames(res)
            td = filter(x -> !ismissing(x) && isfinite(x), res.SC_TD)
            !isempty(td) && minimum(td) <= max_obs_time && return minimum(td)
        end
    catch err
        @warn "HARISS sampling failed" err
    end
    return Inf
end

@everywhere cases_by(t, tinf) = isfinite(t) ? Float64(count(<=(t), tinf)) : NaN

# All rows for one (country, sample)
@everywhere function run_sample(country::String, country_data::DataFrame, sample_id::Int,
                                max_obs_time::Float64, R0::Float64, gen_time::Float64,
                                b_values, swab_scenarios, main_b::Int, main_swab::String)
    seed = hash((country, sample_id))
    Random.seed!(seed)

    infectious_period = MEAN_INFECTIOUS_PERIOD
    latent_period = gen_time - 0.5 * infectious_period
    latent_period < 0 && error("latent_period < 0 for gen_time=$gen_time")
    fixed_shape = 1000.0

    latent_counts     = sample_imports(country_data, :daily_latent_imports)
    infectious_counts = sample_imports(country_data, :daily_infectious_imports)
    detectable_counts = sample_imports(country_data, :daily_detectable_imports)

    # AWW (independent of healthcare rates and HARISS settings)
    aww = Dict(p => Inf for p in AWW_PDETS)
    for (idx, row) in enumerate(eachrow(country_data))
        row.time >= max_obs_time && break
        k = detectable_counts[idx]
        k == 0 && continue
        for p in AWW_PDETS
            if isinf(aww[p]) && rand() < 1.0 - (1.0 - p)^k
                aww[p] = row.time + TURNAROUND_TIME
            end
        end
    end

    rows = NamedTuple[]
    for rate_set in (:original, :corrected)
        p = RATE_SETS[rate_set]
        base_params = merge(p, (
            infectivity      = INFECTIVITY_FOR_R0[R0],
            latent_scale     = latent_period / fixed_shape,
            infectious_scale = infectious_period / fixed_shape,
            infectious_shape = fixed_shape,
            latent_shape     = fixed_shape,
            importrate       = 0.0,
            turnaroundtime   = TURNAROUND_TIME,
        ))
        infectious_params = merge(base_params, (
            latent_scale     = 1e-6,
            infectious_scale = (infectious_period / 2.0) / fixed_shape,
            latent_shape     = fixed_shape,
        ))

        # Same random stream for both rate sets (common random numbers)
        Random.seed!(hash((seed, :tree)))
        ob = grow_outbreak(country_data, latent_counts, infectious_counts,
                           base_params, infectious_params, latent_period, max_obs_time)

        icu_t = Inf
        if !isempty(ob.icu_ticu)
            icu_s = NBPMscape.sampleforest(
                (G = DataFrame(ticu = ob.icu_ticu, trecovered = ob.icu_trecovered),),
                merge(base_params, (p_sampled_icu = ICU_SAMPLING_PROPORTION,)))
            !isempty(icu_s.treport) && (icu_t = minimum(icu_s.treport))
        end

        for B in b_values
            scen = rate_set === :corrected ? swab_scenarios :
                   filter(s -> s.name == main_swab, swab_scenarios)
            (rate_set === :original && B != main_b) && continue
            bg_full = background_for(p, B, max_obs_time)
            for swab in scen
                h_t = hariss_time(p, base_params, thin_cases(ob.hosp_df, swab),
                                  thin_background(bg_full, swab), B, max_obs_time)
                push!(rows, (
                    country            = country,
                    R0                 = R0,
                    generation_time    = gen_time,
                    sample_id          = sample_id,
                    rate_set           = String(rate_set),
                    ed_ari_background  = B,
                    seq_share          = p.n_hosp_samples_per_week / B,
                    swab_scenario      = swab.name,
                    swab_ratio         = swab.admitted / swab.discharged,
                    swab_discharged    = swab.discharged,
                    swab_admitted      = swab.admitted,
                    is_main_analysis   = rate_set === :original && B == main_b && swab.name == main_swab,
                    max_obs_time       = max_obs_time,
                    hariss_time        = h_t,
                    hariss_local_cases = cases_by(h_t, ob.local_tinf),
                    icu_time           = icu_t,
                    icu_local_cases    = cases_by(icu_t, ob.local_tinf),
                    aww_time_low       = aww[AWW_PDET_LOW],    # 8% x 10% sampling
                    aww_time_high      = aww[AWW_PDET_HIGH],   # 16% x 100% sampling
                    n_local_infections = length(ob.local_tinf),
                    n_hosp_cases       = nrow(ob.hosp_df),
                ))
            end
        end
    end
    return rows
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^80)
    println("HARISS SENSITIVITY ($DISEASE, R0=$R0_VALUE, gen_time=$GEN_TIME)")
    println("Countries: ", join(COUNTRIES, ", "))
    println("Samples per country: $N_SAMPLES   Workers: $(nworkers())")
    println("ED ARI background values: $B_VALUES (share sequenced: ",
            join(["$(round(100 * P_ORIGINAL.n_hosp_samples_per_week / b, digits = 2))%" for b in B_VALUES], ", "), ")")
    println("Swab scenarios: ", join([s.name for s in SWAB_SCENARIOS], ", "))
    for rs in (:original, :corrected)
        println("  $rs rates: ", join(["$k=$(round(getfield(RATE_SETS[rs], k), digits=4))" for k in RATE_KEYS], ", "))
    end
    println("Output: $OUTPUT_CSV")
    println("="^80)

    data = CSV.read(INPUT_CSV, DataFrame)
    data = data[(data.R0 .== R0_VALUE) .& (data.generation_time .== GEN_TIME), :]
    mkpath(dirname(OUTPUT_CSV))
    isfile(OUTPUT_CSV) && rm(OUTPUT_CSV)

    pool = CachingPool(workers())
    for country in COUNTRIES
        cdata = sort(data[data.outbreak_country .== country, :], :time)
        if nrow(cdata) == 0
            @warn "No rows for country at this R0/generation time - skipping" country
            continue
        end
        mdt = first(cdata.mean_detection_time)
        mdt > 64 && @warn "mean_detection_time > 64 (excluded from main analysis)" country mdt
        max_obs_time = mdt + EXTRA_TIME
        println("\n$(now())  $country  (mean_detection_time=$(round(mdt, digits=1)), window=$(round(max_obs_time, digits=1)) days)")
        flush(stdout)

        results = pmap(pool, 1:N_SAMPLES) do s
            run_sample(country, cdata, s,Float64(max_obs_time), R0_VALUE, GEN_TIME,
                       B_VALUES, SWAB_SCENARIOS, MAIN_B, MAIN_SWAB)
        end
        df = DataFrame(reduce(vcat, results))
        CSV.write(OUTPUT_CSV, df; append = isfile(OUTPUT_CSV))

        for g in groupby(df, [:rate_set, :ed_ari_background, :swab_scenario])
            det = filter(isfinite, g.hariss_time)
            med = isempty(det) ? NaN : round(median(det), digits = 1)
            println("   $(rpad(first(g.rate_set), 9)) B=$(lpad(first(g.ed_ari_background), 5)) ",
                    "$(rpad(first(g.swab_scenario), 15)) HARISS detected $(round(100*length(det)/nrow(g), digits=0))%  median $(med) d")
        end
        flush(stdout)
    end
    println("\nDone. Results: $OUTPUT_CSV")
end

main()

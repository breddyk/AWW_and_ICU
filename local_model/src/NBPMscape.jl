module NBPMscape

using JumpProcesses
using DifferentialEquations
using Random
using Distributions
using DataFrames
import UUIDs
import StatsBase
using Interpolations
import SpecialFunctions as SF
using Plots
using LinearAlgebra
using Optim   # required by misc_functions.jl

using RData
using CSV
using YAML    # required by config.jl

# TODO
using Revise
using Debugger

# ── Commuting / population data ──────────────────────────────────────────────
const COMMUTERPROBPATH   = joinpath(@__DIR__, "..", "data", "commuting_ITL2_prob_list.rds")
const COMMUTEPROB        = load(COMMUTERPROBPATH)
const COMMUTERINPROBPATH = joinpath(@__DIR__, "..", "data", "commuting_ITL2_inprob_list.rds")
const COMMUTEINPROB      = load(COMMUTERINPROBPATH)
const COMMUTERMPATH      = joinpath(@__DIR__, "..", "data", "commuting_ITL2_list.rds")
const COMMUTERM          = load(COMMUTERMPATH)
const REGKEYPATH         = joinpath(@__DIR__, "..", "data", "ITL2_key2.rds")
const REGKEY             = load(REGKEYPATH)
const CAAPATH            = joinpath(@__DIR__, "..", "data", "CAA_pax_2024_ITL2.rds")
const CAAIMPORTS         = load(CAAPATH)
itl2size = load(joinpath(@__DIR__, "..", "data", "itl2_population2022.rds"))
const ITL2SIZE           = filter(r -> r.ITL225CD in REGKEY.code, itl2size)

# ── HARISS NHS Trust sampling network ────────────────────────────────────────
const HARISS_TRUST_PATH  = joinpath(@__DIR__, "..", "data", "hariss_nhs_trust_sampling_sites.csv")
const HARISS_NHS_TRUST_SITES = CSV.read(HARISS_TRUST_PATH, DataFrame)

# ── Exports ──────────────────────────────────────────────────────────────────
export simtree, simforest, sampleforest, simgendist, Infection, infectivitytoR
export transmissionrate, sampdegree, REGKEY, COMMUTEPROB  # TODO
export secondary_care_td, icu_td, gp_td, courier_collection_times, sample_hosp_cases_n
export load_config, validate_config, update_configurable_parameters

# ── Source files ─────────────────────────────────────────────────────────────
include("misc_functions.jl")          # gamma_params_from_mode_cdf, allocate_with_rounding,
                                      # NHS_TRUST_CATCHMENT_POP_ADULT_CHILD, AE_12M,
                                      # ITL2_TO_NHS_TRUST_PROB_ADULT, ITL2_TO_NHS_TRUST_PROB_CHILD
include("core.jl")
include("hosp_sampling_functions.jl")
include("sampling_infections.jl")
include("config.jl")

# Default parameters ──────────────────────────────────────────────────────────
# NOTE: infectivity is set so that baseline_R0 ≈ 2.03 for SARS-CoV-2 parameters.
# For other pathogens (influenza) override via merge(P, (...)) in the calling script.
P = (
    # Contact rates
    fcont  = 1.0     # household contacts
    , gcont  = 0.50  # work/school contacts
    , oocont = 0.50  # casual/other contacts

    , dowcont = (0.1043502, 0.1402675, 0.1735913, 0.1437642, 0.1596205, 0.1445298, 0.1338766)
    # Infectivity (SARS-CoV-2 defaults)
    , infectivity       = 1.25
    , infectivity_shape = 2.2 * 0.75
    , infectivity_scale = 2.5 * 0.75

    # Latent / infectious periods (SARS-CoV-2 defaults)
    , latent_shape     = 3.26    # Zhao 2021
    , latent_scale     = 0.979
    , infectious_shape = 8.16    # Verity 2020, mean ~24 d
    , infectious_scale = 3.03

    # Transmission reduction in hospital
    , ρ = 0.250

    # Network dynamics
    , frate = 0.0
    , grate = 1/30.0

    # Contact degree distributions
    , fnegbinomr  = 4.45
    , fnegbinomp  = 0.77
    , gnegbinomr  = 1.44
    , gnegbinomp  = 0.1366
    , oorateshape  = 1.42
    , ooratescale  = 6.27746
    , oorateshape1 = 2.42
    , ooratescale1 = 6.27746

    # Sequencing delay (days, uniform)
    , lagseqdblb  = 3
    , lagsseqdbub = 7

    # Severity (SARS-CoV-2 defaults)
    , propmild    = 0.60
    , propsevere  = 0.05

    # Care pathway rates
    , gprate       = 1/3
    , edrate       = 1/5    # rate to ED for moderate_ED cases
    , ped          = 0.74   # fraction of moderate cases → ED (rest → GP)
    , hospadmitrate = 1/4   # Docherty 2020
    , icurate      = 1/2.5  # Knock 2021

    # ICU sampling
    , psampled      = 0.05   # proportion sampled from ICU
    , turnaroundtime = 3     # days

    # Importation (legacy, not used in main pipeline)
    , importrate   = 0.5
    , nimports     = 1000
    , import_t_df  = 2.48
    , import_t_s   = 8.75

    # Molecular evolution
    , μ = 0.001
    , ω = 0.5

    # Commuting
    , commuterate = 2.0

    # ── HARISS / secondary-care sampling parameters ───────────────────────
    , pathogen_type                = "virus"
    , initial_dow                  = 1          # Sunday
    , n_hosp_samples_per_week      = 300
    , turnaroundtime_hariss        = [3, 3]      # fixed 3 day turnaround time just to be identical to other methods
    , hariss_courier_to_analysis   = 1.0
    , phl_collection_dow           = [2, 5]     # Monday and Thursday
    , phl_collection_time          = 0.5        # midday
    , hosp_to_phl_cutoff_time_relative = 1.0
    , swab_time_mode               = 0.25
    , swab_proportion_at_48h       = 0.9
    , proportion_hosp_swabbed      = 0.9
    , sample_allocation            = "equal"
    , sample_proportion_adult      = "free"
    , weight_samples_by            = "ae_mean"
    , hosp_ari_admissions          = 6088
    , hosp_ari_admissions_adult_p  = 0.52
    , hosp_ari_admissions_child_p  = 0.48
    , hariss_only_sample_before_death = true
    , tdischarge_ed_upper_limit              = 0.5
    , tdischarge_hosp_short_stay_upper_limit = 1.0
    , hariss_nhs_trust_sampling_sites = HARISS_NHS_TRUST_SITES
    , ed_ari_destinations_adult = DataFrame(
        destination             = [:discharged, :short_stay, :longer_stay],
        proportion_of_attendances = [0.628, 0.030, 0.342]
      )
    , ed_ari_destinations_child = DataFrame(
        destination             = [:discharged, :short_stay, :longer_stay],
        proportion_of_attendances = [0.861, 0.014, 0.125]
      )

    # ── ICU / upstream sampling parameters ───────────────────────────────
    , turnaroundtime_icu            = [3, 3]
    , p_sampled_icu                 = 0.05
    , icu_sample_type               = "number"
    , icu_site_stage                = "current"
    , sample_icu_cases_version      = "number"
    , n_icu_samples_per_week        = 300
    , icu_ari_admissions            = 1440
    , icu_ari_admissions_adult_p    = 0.76
    , icu_ari_admissions_child_p    = 0.24
    , icu_only_sample_before_death  = true
    , icu_swab_lag_max              = 1.0
    , icu_nhs_trust_sampling_sites  = HARISS_NHS_TRUST_SITES   # use same dummy file

    # ── Metagenomic test sensitivity ──────────────────────────────────────
    , sensitivity_mg_virus    = 0.89
    , sensitivity_mg_bacteria = 0.97
    , sensitivity_mg_fungi    = 0.89
)

end

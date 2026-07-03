# NBPMscape.jl
module NBPMscape

using JumpProcesses
using DifferentialEquations
using Random
using Distributions
using DataFrames
import UUIDs
import StatsBase
using Interpolations
using NamedArrays
import SpecialFunctions as SF
# using Plots
using LinearAlgebra
using Optim   # required by misc_functions.jl

using RData
using CSV
using YAML    # required by config.jl

# TODO
# using Revise
# using Debugger

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

# ── Contact Distributions & Matrices (POLYMOD) ───────────────────────────────
const CONTACT_DIST_PATH = joinpath(@__DIR__, "..", "data", "contacts", "contact_setting_age_group_distributions.rds")
const _RAW_CONTACT_DIST = load(CONTACT_DIST_PATH)

const CONTACT_DISTRIBUTIONS = filter(row -> lowercase(String(row.age_group)) != "all", _RAW_CONTACT_DIST)

const HOME_MATRIX_PATH = joinpath(@__DIR__, "..", "data", "contacts", "polymod_contact_matrix_home.rds")
const CONTACT_MATRIX_HOME = load(HOME_MATRIX_PATH)

const WORK_MATRIX_PATH = joinpath(@__DIR__, "..", "data", "contacts", "polymod_contact_matrix_school_work.rds")
const CONTACT_MATRIX_SCHOOL_WORK = load(WORK_MATRIX_PATH)

const OTHER_MATRIX_PATH = joinpath(@__DIR__, "..", "data", "contacts", "polymod_contact_matrix_other.rds")
const CONTACT_MATRIX_OTHER = load(OTHER_MATRIX_PATH)

# ── HARISS NHS Trust sampling network ────────────────────────────────────────
const HARISS_TRUST_PATH  = joinpath(@__DIR__, "..", "data", "hariss_nhs_trust_sampling_sites.csv")
const HARISS_NHS_TRUST_SITES = CSV.read(HARISS_TRUST_PATH, DataFrame)

# ── Traveler Age Data (Real weights) ─────────────────────────────────────────
const TRAVELLER_AGE_PATH = joinpath(@__DIR__, "..", "data", "international_travel", "travelpac_2019_age_single_year_weights.rds")
const INT_TRAVELLERS_AGE_SINGLE_YR = load(TRAVELLER_AGE_PATH)

# ── Exports ──────────────────────────────────────────────────────────────────
export simtree, simforest, sampleforest, simgendist, Infection, infectivitytoR
export transmissionrate, sampdegree, REGKEY, COMMUTEPROB  # TODO
export secondary_care_td, icu_td, gp_td, courier_collection_times, sample_hosp_cases_n, build_ari_background
export load_config, validate_config, update_configurable_parameters

# ── Source files ─────────────────────────────────────────────────────────────
include("misc_functions.jl")          # gamma_params_from_mode_cdf, allocate_with_rounding,
                                      # NHS_TRUST_CATCHMENT_POP_ADULT_CHILD, AE_12M,
                                      # ITL2_TO_NHS_TRUST_PROB_ADULT, ITL2_TO_NHS_TRUST_PROB_CHILD
include("core.jl")
include("hosp_sampling_functions.jl")
include("sampling_infections.jl")
include("config.jl")

global P = initialize_parameters()


end
#=
Helper functions and module-level data for HARISS hospital sampling.

Functions:
- gamma_params_from_mode_cdf      Fits Gamma distribution params from mode + CDF constraint
- allocate_with_rounding          Integer allocation across groups with rounding correction

Module-level data constants (loaded from real NHS data under data/nhs_trust_data/):
- NHS_TRUST_CATCHMENT_POP_ADULT_CHILD   Per-Trust catchment share + adult/child split
                                          (2022 Trust Catchment Populations Worksheet,
                                          year 2020, AdmissionType = "Emergency")
- AE_12M                                12-month A&E Type-1 attendance per Trust
                                          (NHS England monthly SitReps,
                                          April 2024 – March 2025)
- ITL2_TO_NHS_TRUST_PROB_ADULT          ITL2 region → NHS Trust admission prob (adults)
- ITL2_TO_NHS_TRUST_PROB_CHILD          ITL2 region → NHS Trust admission prob (children)

The ITL2 → Trust probability tables are built by mapping each ITL2 region
to its NHS England Region (7 macro-regions, via `trust_to_nhs_region.csv`
and an internal ITL2→NHS-Region lookup) and then distributing admission
probability across trusts within that NHS Region proportional to each
trust's adult/child catchment size. Welsh ITL2 regions (TLL3/TLL4/TLL5),
which do not correspond to any English NHS Region, fall back to a uniform
admission probability across all English trusts.

To refresh these lookups after upstream NHS files change:
    python3 scripts/build_hariss_nhs_trust_lookups.py
=#

using Optim

"""
    gamma_params_from_mode_cdf(; mode_val, cdf_at_2, lower_shape=1.0+1e-6, upper_shape=10.0)

Fit shape and scale parameters of a Gamma distribution such that:
  - the mode equals `mode_val`
  - CDF at 2 days equals `cdf_at_2`

Uses Brent optimisation over the shape parameter interval [lower_shape, upper_shape].
Returns a `Gamma(shape, scale)` distribution object.
"""
function gamma_params_from_mode_cdf(; mode_val::Real, cdf_at_2::Real,
                                     lower_shape::Real = 1.0 + 1e-6,
                                     upper_shape::Real = 10.0)
    # mode of Gamma(α,θ) = (α-1)*θ  ⟹  θ = mode/(α-1)
    # Minimise |CDF(2; α, θ) - cdf_at_2|
    objective(log_α) = begin
        α = exp(log_α)
        if α <= 1.0
            return Inf
        end
        θ = mode_val / (α - 1.0)
        if θ <= 0.0
            return Inf
        end
        (cdf(Gamma(α, θ), 2.0) - cdf_at_2)^2
    end
    res = Optim.optimize(objective, log(lower_shape), log(upper_shape), Brent())
    α = exp(Optim.minimizer(res))
    θ = mode_val / (α - 1.0)
    return Gamma(α, θ)
end


"""
    allocate_with_rounding(; total, weights)

Allocate integer `total` across groups according to `weights` using largest-remainder
rounding so that the allocated integers sum exactly to `total`.

Returns a `Vector{Int}`.
"""
function allocate_with_rounding(; total::Int, weights::AbstractVector)
    n = length(weights)
    wsum = sum(weights)
    if wsum == 0.0
        return fill(0, n)
    end
    ideal = Float64.(weights) ./ wsum .* total
    floored = floor.(Int, ideal)
    remainders = ideal .- floored
    deficit = total - sum(floored)
    # Give the extra 1s to the groups with the largest remainders
    order = sortperm(remainders, rev=true)
    for i in 1:deficit
        floored[order[i]] += 1
    end
    return floored
end


# ============================================================================
# Real module-level constants, loaded from data/nhs_trust_data/.
#
# These three DataFrames replace the DM1–DM15 synthetic stand-ins that the
# package shipped with originally. They are keyed on real NHS Trust codes
# (matching what is in data/hariss_nhs_trust_sampling_sites.csv) so every
# innerjoin in sample_hosp_cases_n / build_ari_background succeeds against
# the HARISS sampling network as written in the config.
# ============================================================================

const _NHS_TRUST_DATA_DIR = joinpath(@__DIR__, "..", "data", "nhs_trust_data")

# --- NHS_TRUST_CATCHMENT_POP_ADULT_CHILD ---
# Per-Trust catchment size and adult/child split. Emergency-admission
# catchment for 2020 from the 2022 Trust Catchment Populations Worksheet.
# Age band 15-19 is split 1/5 child (age 15) / 4/5 adult (ages 16-19).
const NHS_TRUST_CATCHMENT_POP_ADULT_CHILD = let
    path = joinpath(_NHS_TRUST_DATA_DIR, "nhs_trust_catchment_adult_child.csv")
    df = CSV.read(path, DataFrame; stringtype = String)
    # Keep the column names the downstream code expects:
    # TrustCode, catchment_prop_of_total_sum, prop_adult, prop_child
    df
end

# --- AE_12M ---
# 12-month NHS England Type-1 A&E attendances per Trust (April 2024 –
# March 2025). Trusts without a major ED (e.g. RBV / The Christie) are
# retained with zero monthly counts and mean_12m_prop = 0 so HARISS
# innerjoins do not silently drop them from the sampling network.
const AE_12M = let
    path = joinpath(_NHS_TRUST_DATA_DIR, "ae_12m_attendances.csv")
    CSV.read(path, DataFrame; stringtype = String)
end

# --- Trust → NHS Region lookup (auxiliary, for ITL2 → Trust build below) ---
const _TRUST_TO_NHS_REGION = let
    path = joinpath(_NHS_TRUST_DATA_DIR, "trust_to_nhs_region.csv")
    df = CSV.read(path, DataFrame; stringtype = String)
    Dict(row.NHS_Trust_code => row.nhs_region for row in eachrow(df))
end

# ITL2 region → NHS England Region classification. Covers every code in
# REGKEY (39 codes, 36 English + 3 Welsh). Welsh codes are tagged "WALES"
# so the probability-table builder falls back to a uniform prior across
# all English trusts (there is no NHS England Region for Wales).
const _ITL2_TO_NHS_REGION = Dict{String, String}(
    "TLC3" => "NORTH EAST AND YORKSHIRE",   # Tees Valley
    "TLC4" => "NORTH EAST AND YORKSHIRE",   # Northumberland, Durham and Tyne
    "TLD1" => "NORTH WEST",                  # Cumbria
    "TLD3" => "NORTH WEST",                  # Greater Manchester
    "TLD4" => "NORTH WEST",                  # Lancashire
    "TLD6" => "NORTH WEST",                  # Cheshire
    "TLD7" => "NORTH WEST",                  # Merseyside
    "TLE1" => "NORTH EAST AND YORKSHIRE",   # East Yorkshire and N Lincs
    "TLE2" => "NORTH EAST AND YORKSHIRE",   # North Yorkshire
    "TLE3" => "NORTH EAST AND YORKSHIRE",   # South Yorkshire
    "TLE4" => "NORTH EAST AND YORKSHIRE",   # West Yorkshire
    "TLF1" => "MIDLANDS",                    # Derbyshire and Nottinghamshire
    "TLF2" => "MIDLANDS",                    # Leicestershire, Rutland, Northants
    "TLF3" => "MIDLANDS",                    # Lincolnshire
    "TLG1" => "MIDLANDS",                    # Herefordshire, Worcs, Warwickshire
    "TLG2" => "MIDLANDS",                    # Shropshire and Staffordshire
    "TLG3" => "MIDLANDS",                    # West Midlands
    "TLH2" => "EAST OF ENGLAND",            # Bedfordshire and Hertfordshire
    "TLH3" => "EAST OF ENGLAND",            # Essex
    "TLH4" => "EAST OF ENGLAND",            # Cambridgeshire and Peterborough
    "TLH5" => "EAST OF ENGLAND",            # Norfolk
    "TLH6" => "EAST OF ENGLAND",            # Suffolk
    "TLI3" => "LONDON",                      # Inner London - West
    "TLI4" => "LONDON",                      # Inner London - East
    "TLI5" => "LONDON",                      # Outer London - East and NE
    "TLI6" => "LONDON",                      # Outer London - South
    "TLI7" => "LONDON",                      # Outer London - West and NW
    "TLJ1" => "SOUTH EAST",                  # Berks, Bucks, Oxfordshire
    "TLJ2" => "SOUTH EAST",                  # Surrey, E & W Sussex
    "TLJ3" => "SOUTH EAST",                  # Hampshire and IoW
    "TLJ4" => "SOUTH EAST",                  # Kent
    "TLK3" => "SOUTH WEST",                  # Cornwall
    "TLK4" => "SOUTH WEST",                  # Devon
    "TLK5" => "SOUTH WEST",                  # West of England
    "TLK6" => "SOUTH WEST",                  # N Somerset, Somerset, Dorset
    "TLK7" => "SOUTH WEST",                  # Gloucestershire and Wiltshire
    "TLL3" => "WALES",                       # North Wales
    "TLL4" => "WALES",                       # Mid and South West Wales
    "TLL5" => "WALES",                       # South East Wales
)

# Build the ITL2 → NHS Trust admission probability table. For each ITL2
# region column:
#   - If the region maps to an English NHS Region, the column weights are
#     `trust_weight[T] / sum_{T' in same NHS Region}(trust_weight[T'])`
#     for trusts physically in that NHS Region, and 0 elsewhere.
#   - If the region is Welsh (or unknown), the column is uniform across
#     every trust (1/N).
# `trust_weight` is adult or child catchment respectively.
function _build_itl2_to_trust_prob(trust_codes::Vector{String},
                                    trust_weight::Vector{Float64})
    reg_codes = REGKEY.code
    n_trusts  = length(trust_codes)
    # Pre-compute per-NHS-region weight sums so the inner loop is O(1) per column.
    nhs_regions  = [get(_TRUST_TO_NHS_REGION, tc, "") for tc in trust_codes]
    unique_regs  = unique(nhs_regions)
    region_sum   = Dict(r => 0.0 for r in unique_regs)
    for (r, w) in zip(nhs_regions, trust_weight)
        region_sum[r] += w
    end

    df = DataFrame(NHS_Trust_code = trust_codes)
    for r in reg_codes
        nhs = get(_ITL2_TO_NHS_REGION, r, "")
        col = Vector{Float64}(undef, n_trusts)
        if nhs == "WALES" || nhs == "" || !haskey(region_sum, nhs) || region_sum[nhs] <= 0.0
            fill!(col, 1.0 / n_trusts)
        else
            total = region_sum[nhs]
            @inbounds for i in 1:n_trusts
                col[i] = nhs_regions[i] == nhs ? trust_weight[i] / total : 0.0
            end
        end
        df[!, Symbol(r)] = col
    end
    return df
end

const ITL2_TO_NHS_TRUST_PROB_ADULT = _build_itl2_to_trust_prob(
    String.(NHS_TRUST_CATCHMENT_POP_ADULT_CHILD.TrustCode),
    Float64.(NHS_TRUST_CATCHMENT_POP_ADULT_CHILD.catchment_prop_of_total_sum .*
             NHS_TRUST_CATCHMENT_POP_ADULT_CHILD.prop_adult),
)
const ITL2_TO_NHS_TRUST_PROB_CHILD = _build_itl2_to_trust_prob(
    String.(NHS_TRUST_CATCHMENT_POP_ADULT_CHILD.TrustCode),
    Float64.(NHS_TRUST_CATCHMENT_POP_ADULT_CHILD.catchment_prop_of_total_sum .*
             NHS_TRUST_CATCHMENT_POP_ADULT_CHILD.prop_child),
)

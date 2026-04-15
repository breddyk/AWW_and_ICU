#=
Helper functions and module-level data for HARISS hospital sampling.

Functions:
- gamma_params_from_mode_cdf      Fits Gamma distribution params from mode + CDF constraint
- allocate_with_rounding          Integer allocation across groups with rounding correction

Module-level data constants (synthetic / placeholder):
- NHS_TRUST_CATCHMENT_POP_ADULT_CHILD   Catchment population proportions per NHS Trust
- AE_12M                                12-month A&E attendance data per NHS Trust
- ITL2_TO_NHS_TRUST_PROB_ADULT          ITL2 region → NHS Trust admission prob (adults)
- ITL2_TO_NHS_TRUST_PROB_CHILD          ITL2 region → NHS Trust admission prob (children)

NOTE: The NHS Trust data here is SYNTHETIC, built from the dummy trust codes (DM1–DM15)
in hariss_nhs_trust_sampling_sites.csv. Each trust is given equal weight. Replace with
real UK NHS Trust data for production use.
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
# Synthetic module-level constants (built from dummy NHS Trust codes DM1–DM15)
# Replace with real data for production analysis.
# ============================================================================

const _HARISS_TRUST_CODES = ["DM$i" for i in 1:15]
const _N_TRUSTS = length(_HARISS_TRUST_CODES)

# --- NHS_TRUST_CATCHMENT_POP_ADULT_CHILD ---
# Equal catchment populations across all dummy trusts
const NHS_TRUST_CATCHMENT_POP_ADULT_CHILD = let
    prop = 1.0 / _N_TRUSTS
    DataFrame(
        TrustCode            = _HARISS_TRUST_CODES,
        catchment_prop_of_total_sum = fill(prop, _N_TRUSTS),
        prop_child           = fill(0.20, _N_TRUSTS),   # 20% children
        prop_adult           = fill(0.80, _N_TRUSTS),   # 80% adults
    )
end

# --- AE_12M ---
# Equal A&E attendance proportions across all dummy trusts
const AE_12M = let
    prop = 1.0 / _N_TRUSTS
    df = DataFrame(NHS_Trust_code = _HARISS_TRUST_CODES)
    # Monthly columns (12 months of equal attendances)
    month_labels = ["2024_4_function","2024_5_function","2024_6_function",
                    "2024_7_function","2024_8_function","2024_9_function",
                    "2024_10_function","2024_11_function","2024_12_function",
                    "2025_1_function","2025_2_function","2025_3_function"]
    for m in month_labels
        df[!, m] = fill(1000, _N_TRUSTS)
    end
    df[!, :mean_12m]      = fill(1000.0, _N_TRUSTS)
    df[!, :mean_12m_prop] = fill(prop, _N_TRUSTS)
    df
end

# --- ITL2 region list (from REGKEY) ---
# Build a uniform prior: any ITL2 region maps with equal probability to each dummy trust.
# This is used in sample_hosp_cases_n to assign simulated cases to NHS Trusts.
# The actual assignment is done inside sample_hosp_cases_n using wsample.
function _build_itl2_to_trust_prob(trust_codes::Vector{String})
    # Load REGKEY to get ITL2 codes
    reg_codes = REGKEY.code
    n_regions  = length(reg_codes)
    n_trusts   = length(trust_codes)
    # Uniform probability: each region sends cases equally to all trusts
    prob = fill(1.0 / n_trusts, n_trusts)
    df = DataFrame(NHS_Trust_code = trust_codes)
    for r in reg_codes
        df[!, Symbol(r)] = copy(prob)
    end
    df
end

const ITL2_TO_NHS_TRUST_PROB_ADULT = _build_itl2_to_trust_prob(_HARISS_TRUST_CODES)
const ITL2_TO_NHS_TRUST_PROB_CHILD = _build_itl2_to_trust_prob(_HARISS_TRUST_CODES)

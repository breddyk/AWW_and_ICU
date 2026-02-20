#!/usr/bin/env julia

# Load NBPMscape ONLY on main process (avoid Plots conflicts on workers)
using Pkg
Pkg.activate("/home/users/breddyk/AWW_and_ICU/NBPMscape")
using NBPMscape
using Printf
using Distributed

println("="^80)
println("CALIBRATING INFECTIVITY VALUES FOR TARGET R0 VALUES")
println("="^80)
println()

# Target R0 values for our simulations
target_R0_values = [1.5, 2.0, 2.5, 3.0]

# Default NBPMscape infectivity
default_infectivity = NBPMscape.P.infectivity
println("Default NBPMscape.P.infectivity = $default_infectivity")
println()

# Calculate R0 for default infectivity (single-threaded, uses internal parallelism)
println("Calculating R0 for default infectivity...")
flush(stdout)
default_R0 = infectivitytoR(default_infectivity; nsims=5000)
@printf("Default infectivity %.3f → R0 = %.3f\n", default_infectivity, default_R0)
println()

# Calculate infectivity for each target R0 using proportional scaling
println("PROPORTIONAL SCALING ESTIMATES:")
println("-"^80)
@printf("%-10s %-20s %-20s\n", "Target R0", "Estimated infectivity", "Formula")
println("-"^80)

calibration_dict = Dict{Float64, Float64}()

for R0 in target_R0_values
    # Proportional scaling: infectivity = (R0 / default_R0) * default_infectivity
    estimated_infectivity = (R0 / default_R0) * default_infectivity
    calibration_dict[R0] = estimated_infectivity
    
    @printf("%-10.1f %-20.4f (%.1f/%.2f) × %.3f\n", 
            R0, estimated_infectivity, R0, default_R0, default_infectivity)
end
println()

# Verify estimates by simulating R0 for each estimated infectivity
println("VERIFICATION (simulating R0 for estimated infectivity values):")
println("-"^80)
@printf("%-10s %-20s %-20s %-15s\n", "Target R0", "Infectivity", "Actual R0", "Error (%)")
println("-"^80)

verified_dict = Dict{Float64, NamedTuple}()

for R0 in target_R0_values
    infectivity = calibration_dict[R0]
    
    println("Simulating for R0=$R0, infectivity=$infectivity...")
    flush(stdout)
    
    actual_R0 = infectivitytoR(infectivity; nsims=50000)
    error_pct = abs(actual_R0 - R0) / R0 * 100
    
    verified_dict[R0] = (infectivity = infectivity, actual_R0 = actual_R0, error = error_pct)
    
    @printf("%-10.1f %-20.4f %-20.3f %-15.2f%%\n", 
            R0, infectivity, actual_R0, error_pct)
end
println()

# Summary recommendation
println("="^80)
println("RECOMMENDATION FOR YOUR SIMULATION CODE:")
println("="^80)
println()
println("Replace this code:")
println("```julia")
println("baseline_R0 = 2.0")
println("infectivity_scaling = (R0 / baseline_R0) * NBPMscape.P.infectivity")
println("```")
println()
println("With this lookup table:")
println("```julia")
println("# Calibrated infectivity values (verified via NBPMscape.infectivitytoR)")
println("# Simulated with nsims=5000")
println("infectivity_for_R0 = Dict(")
for R0 in sort(target_R0_values)
    inf_val = verified_dict[R0].infectivity
    actual = verified_dict[R0].actual_R0
    @printf("    %.1f => %.4f,  # Actual R0 ≈ %.3f\n", R0, inf_val, actual)
end
println(")")
println()
println("infectivity_scaling = infectivity_for_R0[R0]")
println("```")
println()

# Check if proportional scaling is a good approximation
max_error = maximum([verified_dict[R0].error for R0 in target_R0_values])
println("Maximum error in proportional scaling: $(@sprintf("%.2f%%", max_error))")

if max_error < 5.0
    println("✓ Proportional scaling is a good approximation (error < 5%)")
    println("  You can safely use: infectivity_scaling = (R0 / $(@sprintf("%.2f", default_R0))) × $default_infectivity")
else
    println("⚠ Proportional scaling has significant error (> 5%)")
    println("  Recommend using the lookup table above for accuracy")
end
println()
println("="^80)
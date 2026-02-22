using NBPMscape
using Plots
using DataFrames
using Statistics
using Distributions
using CSV
using StatsPlots
using LaTeXStrings
using KernelDensity
using Distributed

addprocs(45)

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

# Load packages on all workers
@everywhere using NBPMscape
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Distributions
@everywhere using CSV

default(fontfamily="Times New Roman")

# ============================================================================
# SAMPLING FUNCTIONS (must be @everywhere for workers)
# ============================================================================

@everywhere function sample_truncated_gamma(shape::Float64, rate::Float64, lower_bound::Float64)
    """
    Sample from Gamma(shape, rate) with C >= lower_bound
    Uses inverse CDF method - always succeeds in one attempt
    """
    scale = 1.0 / rate
    gamma_dist = Gamma(shape, scale)
    
    # Get CDF value at lower bound
    p_lower = cdf(gamma_dist, lower_bound)
    
    # If lower_bound is beyond the distribution (p_lower ≈ 1), return lower_bound
    if p_lower >= 0.9999
        return lower_bound
    end
    
    # Sample uniformly from [p_lower, 1]
    u = p_lower + rand() * (1.0 - p_lower)
    
    # Invert the CDF to get the sample
    return quantile(gamma_dist, u)
end

@everywhere function sample_poisson_direct(lambda::Float64)
    """
    Sample directly from Poisson(lambda) for daily imports
    Safe version that handles edge cases
    """
    # Handle invalid lambda values
    if !isfinite(lambda) || lambda <= 0
        return 0.0
    end
    
    # Cap lambda at a reasonable maximum to prevent numerical issues
    # Poisson sampler can struggle with very large lambda values
    lambda_safe = min(lambda, 1000.0)
    
    return Float64(rand(Poisson(lambda_safe)))
end

@everywhere function sample_daily_imports_from_cumulative(
    country_data::DataFrame,
    sample_id::Int
)
    """
    Sample daily imports using appropriate method:
    - If Gamma params available: Sample from cumulative Gamma with truncation, then difference
    - If only mean available: Sample directly from Poisson (already daily)
    
    Returns: Vector of daily import counts (integers)
    """
    n_times = nrow(country_data)
    daily_imports = zeros(Int, n_times)
    
    # Check if we're using Gamma for this trajectory
    has_any_gamma = any(row -> !ismissing(row.gamma_cumulative_shape) && 
                               !isnan(row.gamma_cumulative_shape) &&
                               !ismissing(row.daily_mean_imports) &&
                               row.daily_mean_imports > 1.0, 
                       eachrow(country_data))
    
    if has_any_gamma
        # Use Gamma sampling with cumulative truncation
        cumulative_samples = zeros(Float64, n_times)
        
        for (idx, row) in enumerate(eachrow(country_data))
            has_gamma = !ismissing(row.gamma_cumulative_shape) && 
                        !ismissing(row.gamma_cumulative_rate) &&
                        !isnan(row.gamma_cumulative_shape) && 
                        !isnan(row.gamma_cumulative_rate)
            
            has_mean = !ismissing(row.daily_mean_imports) && 
                       !isnan(row.daily_mean_imports) && 
                       !isinf(row.daily_mean_imports) &&
                       row.daily_mean_imports > 0
            
            # Get lower bound from previous time point
            lower_bound = (idx > 1) ? cumulative_samples[idx-1] : 0.0
            
            if has_mean && row.daily_mean_imports > 1.0 && has_gamma
                # Always sample from truncated Gamma distribution (cumulative)
                cumulative_samples[idx] = sample_truncated_gamma(
                    row.gamma_cumulative_shape,
                    row.gamma_cumulative_rate,
                    lower_bound
                )
            elseif has_mean
                # Fallback: assume cumulative grows by Poisson increment
                cumulative_samples[idx] = lower_bound + sample_poisson_direct(row.daily_mean_imports)
            else
                cumulative_samples[idx] = lower_bound
            end
        end
        
        # Calculate daily imports as differences
        for idx in 1:n_times
            if idx == 1
                daily_imports[idx] = round(Int, cumulative_samples[idx])
            else
                daily_imports[idx] = round(Int, cumulative_samples[idx] - cumulative_samples[idx-1])
            end
        end
        
    else
        # Pure Poisson sampling (direct daily, no cumulative structure needed)
        for (idx, row) in enumerate(eachrow(country_data))
            has_mean = !ismissing(row.daily_mean_imports) && 
                       !isnan(row.daily_mean_imports) && 
                       !isinf(row.daily_mean_imports) &&
                       row.daily_mean_imports > 0
            
            if has_mean
                daily_imports[idx] = round(Int, sample_poisson_direct(row.daily_mean_imports))
            else
                daily_imports[idx] = 0
            end
        end
    end
    
    return daily_imports
end

# ============================================================================
# OPTIMIZED SIMULATION WITH EARLY STOPPING (must be @everywhere for workers)
# ============================================================================

@everywhere function calculate_detections_early_stopping(
    country_data::DataFrame,
    country_name::String,
    num_samples::Int,
    R0::Float64,
    mean_generation_time::Float64,
    icu_sampling_proportion::Float64,
    airport_detection_prob::Float64;
    mean_infectious_period = 8/3,
    turnaround_time::Float64 = 3.0,
    max_observation_time = 100.0
)
    """
    Optimized version: for each sample, stop as soon as BOTH detections occur
    No need to simulate beyond when we have both airport AND ICU detection times
    """
    
    println("  Processing $country_name (R0=$R0, gen_time=$mean_generation_time)")
    println("    Optimization: Early stopping when both detections occur")
    
    # --- FIXED INFECTIOUS PERIOD (biological property) ---
    infectious_period = mean_infectious_period  # ~2.67 days - CONSTANT
    
    # --- CALCULATE LATENT PERIOD FROM GENERATION TIME ---
    # For SEIR: Gen_time = Latent + 0.5 * Infectious
    latent_period = mean_generation_time - (0.5 * infectious_period)
    
    if latent_period < 0
        error("Invalid parameters: latent_period < 0 for gen_time=$mean_generation_time")
    end
    
    println("    Latent period: $(round(latent_period, digits=2))d")
    println("    Infectious period: $(round(infectious_period, digits=2))d (fixed)")
    
    # --- DETERMINISTIC PERIODS (high shape = low variance) ---
    fixed_shape = 1000.0  # Very high shape → essentially deterministic
    
    latent_scale = latent_period / fixed_shape
    infectious_scale = infectious_period / fixed_shape
    
    # --- SCALE INFECTIVITY TO MATCH TARGET R0 ---
    # Proportional scaling validated via inftoR.jl
    # NBPMscape.P.infectivity = 1.25 corresponds to R0 ≈ 2.03
    baseline_R0 = 2.03
    infectivity_scaling = (R0 / baseline_R0) * NBPMscape.P.infectivity
    
    base_params = merge(NBPMscape.P, (
        infectivity = infectivity_scaling,
        latent_scale = latent_scale,
        infectious_scale = infectious_scale,
        infectious_shape = fixed_shape,
        latent_shape = fixed_shape,
        importrate = 0.0,
        turnaroundtime = turnaround_time
    ))
    
    icu_params = merge(base_params, (psampled = icu_sampling_proportion,))
    
    # Storage for results
    icu_detection_times = Float64[]
    icu_local_cases_at_detection = Float64[]
    airport_detection_times = Float64[]
    airport_local_cases_at_detection = Float64[]
    total_airport_detections = 0
    
    for sample in 1:num_samples
        if sample % 10 == 0 || sample == 1
            println("    Sample $sample/$num_samples")
            flush(stdout)
        end
        
        # Sample daily imports for this trajectory
        daily_import_counts = sample_daily_imports_from_cumulative(country_data, sample)
        
        # Track state for this sample
        sample_infections = DataFrame()
        airport_detected = false
        icu_detected = false
        first_airport_detection_time = Inf
        first_icu_detection_time = Inf
        
        # Process imports chronologically - STOP EARLY when both detections occur
        for (idx, row) in enumerate(eachrow(country_data))
            time = row.time
            
            # EARLY STOPPING: if both detections have occurred, we're done with this sample!
            if airport_detected && icu_detected
                break
            end
            
            if time >= max_observation_time
                break
            end
            
            daily_import_count = daily_import_counts[idx]
            
            # Check airport detection (fast check, no simulation needed)
            if !airport_detected && daily_import_count > 0
                prob_detection_today = 1.0 - (1.0 - airport_detection_prob)^daily_import_count
                if rand() < prob_detection_today
                    airport_detected = true
                    first_airport_detection_time = time + turnaround_time
                    total_airport_detections += 1
                end
            end
            
            # Simulate transmission trees for each import
            for j in 1:daily_import_count
                results = NBPMscape.simtree(base_params,
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
            
            # Check ICU detection after processing this time point
            if !icu_detected && !isempty(sample_infections)
                icu_sampled = NBPMscape.sampleforest(
                    (G = sample_infections,), 
                    icu_params
                )
                
                if !isempty(icu_sampled.treport)
                    icu_detected = true
                    first_icu_detection_time = minimum(icu_sampled.treport)
                end
            end
        end
        
        # Record results for this sample
        if airport_detected && isfinite(first_airport_detection_time)
            push!(airport_detection_times, first_airport_detection_time)
            
            if !isempty(sample_infections)
                airport_local_cases = sum((sample_infections.generation .> 0) .& 
                                         (sample_infections.tinf .<= first_airport_detection_time))
                push!(airport_local_cases_at_detection, airport_local_cases)
            end
        end
        
        if icu_detected && isfinite(first_icu_detection_time)
            push!(icu_detection_times, first_icu_detection_time)
            
            if !isempty(sample_infections)
                icu_local_cases = sum((sample_infections.generation .> 0) .& 
                                     (sample_infections.tinf .<= first_icu_detection_time))
                push!(icu_local_cases_at_detection, icu_local_cases)
            end
        end
    end
    
    println("    Results: Airport=$(round(100*total_airport_detections/num_samples, digits=1))% detected, ICU=$(round(100*length(icu_detection_times)/num_samples, digits=1))% detected")
    
    return (
        icu_mean_detection = isempty(icu_detection_times) ? NaN : mean(icu_detection_times),
        icu_detection_times = icu_detection_times,
        icu_mean_local_cases = isempty(icu_local_cases_at_detection) ? NaN : mean(icu_local_cases_at_detection),
        icu_detection_rate = length(icu_detection_times) / num_samples,
        airport_mean_detection = isempty(airport_detection_times) ? NaN : mean(airport_detection_times),
        airport_detection_times = airport_detection_times,
        airport_mean_local_cases = isempty(airport_local_cases_at_detection) ? NaN : mean(airport_local_cases_at_detection),
        airport_detection_rate = total_airport_detections / num_samples
    )
end

# ============================================================================
# MAIN PARAMETER SWEEP
# ============================================================================

function run_parameter_sweep_optimized(
    csv_path::String,
    countries::Vector{String};
    num_samples::Int = 100,
    turnaround_time::Float64 = 3.0,
    time_buffer::Float64 = 20.0,
    output_path::String = "results_optimized.csv",
    R0_values = [1.5, 2.0, 2.5, 3.0],
    generation_times = [4.0, 6.0, 8.0, 10.0],
    airport_probs = [0.04, 0.08, 0.16],
    icu_sampling_proportion::Float64 = 0.10
)
    """
    Optimized parameter sweep that stops each sample early once both detections occur
    """
    
    println("="^80)
    println("OPTIMIZED PARAMETER SWEEP (Early Stopping)")
    println("="^80)
    println("Countries: $(join(countries, ", "))")
    println("R0 values: $R0_values")
    println("Generation times: $generation_times")
    println("Airport detection probs: $airport_probs")
    println("ICU sampling: $(icu_sampling_proportion*100)% (FIXED)")
    println("Samples per combination: $num_samples")
    println("Workers: $(nworkers())")
    println("Optimization: Stops each sample when BOTH detections occur")
    println("="^80)
    
    # Load data
    println("\nLoading data from: $csv_path")
    all_data = CSV.read(csv_path, DataFrame)
    println("Loaded $(nrow(all_data)) total rows")
    
    # Start fresh
    println("🆕 Starting with fresh results DataFrame")
    results_df = DataFrame(
        country = String[],
        R0 = Float64[],
        generation_time = Float64[],
        airport_detection_prob = Float64[],
        icu_sampling_proportion = Float64[],
        airport_mean_detection_time = Float64[],
        airport_std_detection_time = Float64[],
        airport_mean_local_cases = Float64[],
        airport_detection_rate = Float64[],
        icu_mean_detection_time = Float64[],
        icu_std_detection_time = Float64[],
        icu_mean_local_cases = Float64[],
        icu_detection_rate = Float64[]
    )
    
    total_combinations = length(R0_values) * length(generation_times) * length(airport_probs) * length(countries)
    current_combination = 0
    
    for R0 in R0_values
        for gen_time in generation_times
            for p_det in airport_probs
                
                println("\n" * "="^80)
                println("R0=$R0, generation_time=$gen_time, airport_p_det=$p_det")
                println("="^80)
                
                # Filter data
                daily_imports_data = filter(row -> 
                    row.R0 == R0 && 
                    row.generation_time == gen_time && 
                    row.p_det == p_det &&
                    row.outbreak_country in countries,
                    all_data
                )
                
                if nrow(daily_imports_data) == 0
                    println("WARNING: No data for this combination")
                    continue
                end
                
                println("Found $(nrow(daily_imports_data)) rows for this combination")
                
                # Trim data
                trimmed_data = DataFrame()
                for country in unique(daily_imports_data.outbreak_country)
                    country_data = filter(row -> row.outbreak_country == country, daily_imports_data)
                    detection_time = first(country_data).mean_detection_time
                    cutoff_time = detection_time + time_buffer
                    country_trimmed = filter(row -> row.time <= cutoff_time, country_data)
                    
                    if isempty(trimmed_data)
                        trimmed_data = country_trimmed
                    else
                        append!(trimmed_data, country_trimmed)
                    end
                end
                
                # Process each country
                for country in countries
                    current_combination += 1
                    
                    println("\n" * "-"^60)
                    println("COMBINATION $current_combination/$total_combinations")
                    println("COUNTRY: $country")
                    println("-"^60)
                    
                    country_data = filter(row -> row.outbreak_country == country, trimmed_data)
                    
                    if nrow(country_data) == 0
                        println("  No data for $country, skipping...")
                        continue
                    end
                    
                    sort!(country_data, :time)
                    
                    # Run optimized simulation with early stopping
                    result = calculate_detections_early_stopping(
                        country_data,
                        country,
                        num_samples,
                        R0,
                        gen_time,
                        icu_sampling_proportion,
                        p_det;
                        turnaround_time = turnaround_time
                    )
                    
                    # Add result
                    push!(results_df, (
                        country,
                        R0,
                        gen_time,
                        p_det,
                        icu_sampling_proportion,
                        result.airport_mean_detection,
                        isempty(result.airport_detection_times) ? NaN : std(result.airport_detection_times),
                        result.airport_mean_local_cases,
                        result.airport_detection_rate,
                        result.icu_mean_detection,
                        isempty(result.icu_detection_times) ? NaN : std(result.icu_detection_times),
                        result.icu_mean_local_cases,
                        result.icu_detection_rate
                    ))
                    
                    # Save after each country
                    CSV.write(output_path, results_df)
                    
                    println("  ✓ Completed $country")
                    println("    Airport: $(round(result.airport_mean_detection, digits=2)) days ($(round(result.airport_detection_rate*100, digits=1))% detection)")
                    println("    ICU: $(round(result.icu_mean_detection, digits=2)) days ($(round(result.icu_detection_rate*100, digits=1))% detection)")
                    println("  💾 Saved to: $output_path ($(nrow(results_df)) rows)")
                end
            end
        end
    end
    
    println("\n" * "="^80)
    println("PARAMETER SWEEP COMPLETE")
    println("Results saved to: $output_path")
    println("Total rows: $(nrow(results_df))")
    println("="^80)
    
    return results_df
end

# ============================================================================
# RUN THE SWEEP
# ============================================================================

results = run_parameter_sweep_optimized(
    "global_model/pgfgleam_code/all_results/global/daily_imports_specific_sensitivity_gamma.csv",
    ["Paraguay", "South Korea", "Switzerland"];
    num_samples = 100,
    turnaround_time = 3.0,
    time_buffer = 60.0,
    output_path = "global_model/pgfgleam_code/all_results/local/specific_results_with_gamma.csv",
    R0_values = [1.5, 2.0, 2.5, 3.0],
    generation_times = [4.0, 6.0, 8.0, 10.0],
    airport_probs = [0.04, 0.08, 0.16],
    icu_sampling_proportion = 0.10 
)

println("\n✓ Optimized parameter sweep complete!")
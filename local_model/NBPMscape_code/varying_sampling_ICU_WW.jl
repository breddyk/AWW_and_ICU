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

@everywhere function sample_poisson_direct(lambda::Float64)
    """
    Sample directly from Poisson(lambda) for daily imports
    Safe version with bounds checking
    """
    if !isfinite(lambda) || lambda <= 0
        return 0.0
    end
    lambda_safe = min(lambda, 1000.0)
    return Float64(rand(Poisson(lambda_safe)))
end

@everywhere function sample_importation_count(row)
    """
    Sample importation count based on mean_imports value
    Handle missing values properly
    """
    if ismissing(row.mean_imports) || !isfinite(row.mean_imports) || row.mean_imports <= 0
        return 0
    end
    return round(Int, sample_poisson_direct(row.mean_imports))
end

function simulate_infection_trees_once(
    country_data::DataFrame,
    country_name::String,
    num_samples::Int,
    R0::Float64,
    mean_generation_time::Float64;
    mean_infectious_period = 8/3,
    turnaround_time = 3.0,
    max_observation_time = 100.0
)
    """
    Generate infection trees ONCE for a country at given R0/gen_time.
    These will be reused for all detection probability combinations.
    """
    println("  Generating infection trees for $country_name (R0=$R0, gen_time=$mean_generation_time)")
    
    if nrow(country_data) == 0
        println("    Warning: No valid time points for $country_name")
        return nothing
    end
    
    # --- FIXED INFECTIOUS PERIOD (biological property) ---
    infectious_period = mean_infectious_period  # ~2.67 days - CONSTANT
    
    # --- CALCULATE LATENT PERIOD FROM GENERATION TIME ---
    latent_period = mean_generation_time - (0.5 * infectious_period)
    
    if latent_period < 0
        error("Invalid parameters: latent_period < 0 for gen_time=$mean_generation_time")
    end
    
    println("    Latent: $(round(latent_period, digits=2))d, Infectious: $(round(infectious_period, digits=2))d")
    
    # --- DETERMINISTIC PERIODS (high shape = low variance) ---
    fixed_shape = 1000.0
    latent_scale = latent_period / fixed_shape
    infectious_scale = infectious_period / fixed_shape
    
    # --- SCALE INFECTIVITY TO MATCH TARGET R0 ---
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
    
    # Storage
    all_sample_infections = Vector{DataFrame}()
    all_daily_imports = Vector{Vector{Tuple{Float64, Int}}}()
    
    for sample in 1:num_samples
        if sample % 20 == 0
            println("    Sample $sample/$num_samples")
            flush(stdout)
        end
        
        sample_infections = DataFrame()
        daily_imports_this_sample = Vector{Tuple{Float64, Int}}()
        
        # Process imports chronologically
        for (idx, row) in enumerate(eachrow(country_data))
            time = row.time
            if time >= max_observation_time
                break
            end
            
            # Sample imports for this specific day
            daily_import_count = sample_importation_count(row)
            push!(daily_imports_this_sample, (time, daily_import_count))
            
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
        end
        
        push!(all_sample_infections, sample_infections)
        push!(all_daily_imports, daily_imports_this_sample)
    end
    
    non_empty = sum(!isempty(df) for df in all_sample_infections)
    println("    Generated $(length(all_sample_infections)) samples ($non_empty non-empty)")
    
    return (
        infections = all_sample_infections,
        daily_imports = all_daily_imports,
        base_params = base_params
    )
end

@everywhere function calculate_detections_from_trees(
    tree_data,
    icu_sampling_proportion::Float64,
    airport_detection_prob::Float64;
    turnaround_time::Float64 = 3.0
)
    """
    Calculate detection times from pre-generated infection trees.
    This is MUCH faster than regenerating trees!
    """
    
    if isnothing(tree_data)
        return (
            icu_mean_detection = NaN,
            icu_detection_times = Float64[],
            icu_mean_local_cases = NaN,
            icu_local_cases_samples = Float64[],
            icu_detection_rate = 0.0,
            airport_mean_detection = NaN,
            airport_detection_times = Float64[],
            airport_mean_local_cases = NaN,
            airport_local_cases_samples = Float64[],
            airport_detection_rate = 0.0
        )
    end
    
    infections = tree_data.infections
    daily_imports = tree_data.daily_imports
    base_params = tree_data.base_params
    
    # Update sampling parameter
    icu_params = merge(base_params, (psampled = icu_sampling_proportion,))
    
    # Storage for results
    icu_detection_times = Float64[]
    icu_local_cases_at_detection = Float64[]
    airport_detection_times = Float64[]
    airport_local_cases_at_detection = Float64[]
    total_airport_detections = 0
    
    num_samples = length(infections)
    
    for sample in 1:num_samples
        sample_infections = infections[sample]
        sample_daily_imports = daily_imports[sample]
        
        # Airport detection: test each day independently
        airport_detected_this_sample = false
        first_airport_detection_time = Inf
        
        for (time, daily_import_count) in sample_daily_imports
            if !airport_detected_this_sample && daily_import_count > 0
                prob_detection_today = 1.0 - (1.0 - airport_detection_prob)^daily_import_count
                
                if rand() < prob_detection_today
                    airport_detected_this_sample = true
                    first_airport_detection_time = time + turnaround_time
                    total_airport_detections += 1
                    break
                end
            end
        end
        
        # Process detections only if we have infections
        if !isempty(sample_infections)
            # ICU detection
            icu_sampled = NBPMscape.sampleforest(
                (G = sample_infections,), 
                icu_params
            )
            
            if !isempty(icu_sampled.treport)
                icu_first_detection = minimum(icu_sampled.treport)
                push!(icu_detection_times, icu_first_detection)
                
                icu_local_cases = sum((sample_infections.generation .> 0) .& 
                                     (sample_infections.tinf .<= icu_first_detection))
                push!(icu_local_cases_at_detection, icu_local_cases)
            end
            
            # Airport local cases
            if airport_detected_this_sample && isfinite(first_airport_detection_time)
                push!(airport_detection_times, first_airport_detection_time)
                
                airport_local_cases = sum((sample_infections.generation .> 0) .& 
                                         (sample_infections.tinf .<= first_airport_detection_time))
                push!(airport_local_cases_at_detection, airport_local_cases)
            end
        end
    end
    
    return (
        icu_mean_detection = isempty(icu_detection_times) ? NaN : mean(icu_detection_times),
        icu_detection_times = icu_detection_times,
        icu_mean_local_cases = isempty(icu_local_cases_at_detection) ? NaN : mean(icu_local_cases_at_detection),
        icu_local_cases_samples = icu_local_cases_at_detection,
        icu_detection_rate = length(icu_detection_times) / num_samples,
        airport_mean_detection = isempty(airport_detection_times) ? NaN : mean(airport_detection_times),
        airport_detection_times = airport_detection_times,
        airport_mean_local_cases = isempty(airport_local_cases_at_detection) ? NaN : mean(airport_local_cases_at_detection),
        airport_local_cases_samples = airport_local_cases_at_detection,
        airport_detection_rate = total_airport_detections / num_samples
    )
end

function run_sensitivity_analysis(
    csv_path::String,
    countries::Vector{String};
    num_samples::Int = 100,
    turnaround_time::Float64 = 3.0,
    time_buffer::Float64 = 20.0,
    output_path::String = "sensitivity_analysis_results.csv",
    R0_values = [1.5, 2.0, 2.5, 3.0],
    generation_times = [4.0, 6.0, 8.0, 10.0],
    airport_probs = 0.01:0.01:1.0,
    icu_props = 0.01:0.01:1.0
)
    """
    Run sensitivity analysis over R0, gen_time, airport detection probabilities and ICU sampling proportions.
    EFFICIENT: Generate infection trees once per R0×gen_time×country, then reuse for all parameter combinations!
    """
    
    println("="^80)
    println("SENSITIVITY ANALYSIS WITH R0/GEN_TIME SWEEP")
    println("="^80)
    println("Countries: $(join(countries, ", "))")
    println("R0 values: $R0_values")
    println("Generation times: $generation_times")
    println("Samples per combination: $num_samples")
    println("Airport detection probs: $(length(airport_probs)) values from $(minimum(airport_probs)) to $(maximum(airport_probs))")
    println("ICU sampling props: $(length(icu_props)) values from $(minimum(icu_props)) to $(maximum(icu_props))")
    println("Total combinations per R0×gen_time×country: $(length(airport_probs) * length(icu_props))")
    println("Output: $output_path")
    println("="^80)
    
    # Load and prepare data
    println("\nLoading data...")
    daily_imports_data = CSV.read(csv_path, DataFrame)
    
    if "outbreak_country" in names(daily_imports_data)
        rename!(daily_imports_data, :outbreak_country => :country)
    end
    
    # Filter for specified countries
    daily_imports_data = filter(row -> row.country in countries, daily_imports_data)
    
    # Initialize results dataframe
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
    
    # Process each R0×gen_time combination
    for R0 in R0_values
        for gen_time in generation_times
            println("\n" * "="^80)
            println("R0=$R0, GENERATION_TIME=$gen_time")
            println("="^80)
            
            # Process each country
            for country in countries
                println("\n" * "-"^60)
                println("COUNTRY: $country")
                println("-"^60)
                
                # Get data for this country
                country_data = filter(row -> row.country == country, daily_imports_data)
                
                if nrow(country_data) == 0
                    println("  No data for $country, skipping...")
                    continue
                end
                
                # Trim data based on detection time
                detection_time = first(country_data).mean_detection_time
                cutoff_time = detection_time + time_buffer
                country_trimmed = filter(row -> row.time <= cutoff_time, country_data)
                sort!(country_trimmed, :time)
                
                # STEP 1: Generate infection trees ONCE (this is the slow part)
                println("  STEP 1: Generating infection trees (this takes time)...")
                tree_data = simulate_infection_trees_once(
                    country_trimmed,
                    country,
                    num_samples,
                    R0,
                    gen_time;
                    turnaround_time = turnaround_time
                )
                
                if isnothing(tree_data)
                    println("  Skipping $country due to no valid data")
                    continue
                end
                
                # STEP 2: Rapidly evaluate all parameter combinations (this is fast!)
                println("  STEP 2: Evaluating all parameter combinations in parallel...")
                
                # Create all combinations to evaluate
                param_combinations = [(airport_prob, icu_prop) 
                                     for airport_prob in airport_probs 
                                     for icu_prop in icu_props]
                
                total_combos = length(param_combinations)
                println("    Total combinations: $total_combos")
                println("    Using $(nworkers()) workers")
                flush(stdout)
                
                # Evaluate in parallel using pmap
                results_list = pmap(param_combinations) do (airport_prob, icu_prop)
                    calculate_detections_from_trees(
                        tree_data,
                        icu_prop,
                        airport_prob;
                        turnaround_time = turnaround_time
                    )
                end
                
                # Store all results
                for (idx, (airport_prob, icu_prop)) in enumerate(param_combinations)
                    result = results_list[idx]
                    
                    push!(results_df, (
                        country,
                        R0,
                        gen_time,
                        airport_prob,
                        icu_prop,
                        result.airport_mean_detection,
                        isempty(result.airport_detection_times) ? NaN : std(result.airport_detection_times),
                        result.airport_mean_local_cases,
                        result.airport_detection_rate,
                        result.icu_mean_detection,
                        isempty(result.icu_detection_times) ? NaN : std(result.icu_detection_times),
                        result.icu_mean_local_cases,
                        result.icu_detection_rate
                    ))
                    
                    if idx % 1000 == 0
                        println("    Stored $idx/$total_combos results")
                        flush(stdout)
                    end
                end
                
                println("  ✓ Completed all $total_combos combinations for $country")
                
                # Save after each country
                CSV.write(output_path, results_df)
                println("  💾 Saved progress ($(nrow(results_df)) rows)")
            end
        end
    end
    
    println("\n" * "="^80)
    println("ANALYSIS COMPLETE")
    println("Results saved to: $output_path")
    println("Total rows: $(nrow(results_df))")
    println("="^80)
    
    return results_df
end

# Run the sensitivity analysis
println("Starting sensitivity analysis...")

results = run_sensitivity_analysis(
    "/Users/reddy/AWW_and_ICU/pgfgleam/daily_imports_specific_sensitivity_gamma.csv",
    ["Congo", "South Korea", "Switzerland"];
    num_samples = 100,
    turnaround_time = 3.0,
    time_buffer = 45.0,
    output_path = "/Users/reddy/AWW_and_ICU/pgfgleam/pgfgleam/datasets/varying_sampling_params.csv",
    R0_values = [2.0],
    generation_times = [4.0],
    airport_probs = 0.01:0.01:1.0,
    icu_props = 0.01:0.01:1.0
)

println("\n✓ Analysis complete!")
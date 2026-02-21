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
        import_column: Either :daily_transmission_capable_imports or :daily_detectable_imports
    
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
# TIME SERIES TRACKING SIMULATION
# ============================================================================

@everywhere function simulate_country_timeseries(
    country_data::DataFrame,
    country_name::String,
    num_samples::Int,
    R0::Float64,
    mean_generation_time::Float64,
    icu_sampling_proportion::Float64,
    airport_detection_prob::Float64,
    max_observation_time::Float64;
    mean_infectious_period = 8/3,
    turnaround_time::Float64 = 3.0,
    verbose::Bool = true
)
    """
    Simulate and track time series for each sample
    
    Returns:
        Vector of sample results, each containing:
        - times: time points
        - detectable_imports: daily detectable imports over time
        - transmission_imports: daily transmission-capable imports over time
        - local_cases: cumulative local cases over time
        - arrival_time: first import time
        - aww_detection_time: airport detection time (or NaN)
        - icu_detection_time: ICU detection time (or NaN)
    """
    
    if verbose
        println("  Processing $country_name (R0=$R0, gen_time=$mean_generation_time)")
        println("    Max observation time: $max_observation_time days")
        println("    ICU sampling: $(icu_sampling_proportion*100)%")
        println("    AWW detection prob: $(airport_detection_prob)")
    end
    
    # --- FIXED INFECTIOUS PERIOD ---
    infectious_period = mean_infectious_period
    latent_period = mean_generation_time - (0.5 * infectious_period)
    
    if latent_period < 0
        error("Invalid parameters: latent_period < 0 for gen_time=$mean_generation_time")
    end
    
    if verbose
        println("    Latent period: $(round(latent_period, digits=2))d")
        println("    Infectious period: $(round(infectious_period, digits=2))d (fixed)")
    end
    
    # --- DETERMINISTIC PERIODS ---
    fixed_shape = 1000.0
    latent_scale = latent_period / fixed_shape
    infectious_scale = infectious_period / fixed_shape
    
    # --- SCALE INFECTIVITY ---
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
    
    # Storage for all sample results
    all_sample_results = []
    
    times = country_data.time
    
    for sample in 1:num_samples
        if verbose && (sample % 10 == 0 || sample == 1)
            println("    Sample $sample/$num_samples")
            flush(stdout)
        end
        
        # Sample import time series
        transmission_import_counts, _ = sample_daily_imports_poisson(
            country_data, sample, :daily_transmission_capable_imports
        )
        
        detectable_import_counts, _ = sample_daily_imports_poisson(
            country_data, sample, :daily_detectable_imports
        )
        
        # Initialize time series
        local_cases_timeseries = zeros(Int, length(times))
        
        # Track detection events
        arrival_time = NaN
        aww_detection_time = NaN
        icu_detection_time = NaN
        
        aww_detected = false
        icu_detected = false
        
        # Track state for this sample
        sample_infections = DataFrame()
        
        # Process imports chronologically
        for (idx, row) in enumerate(eachrow(country_data))
            time = row.time
            if time >= max_observation_time
                break
            end
            
            # Record first arrival
            if isnan(arrival_time)
                if transmission_import_counts[idx] > 0 || detectable_import_counts[idx] > 0
                    arrival_time = time
                end
            end
            
            # ========================================
            # AIRPORT DETECTION: Uses detectable imports (I + P)
            # ========================================
            daily_detectable_count = detectable_import_counts[idx]
            
            if daily_detectable_count > 0 && !aww_detected
                prob_detection_today = 1.0 - (1.0 - airport_detection_prob)^daily_detectable_count
                
                if rand() < prob_detection_today
                    aww_detected = true
                    aww_detection_time = time + turnaround_time
                end
            end
            
            # ========================================
            # LOCAL TRANSMISSION: Seeds from transmission-capable imports (L + I)
            # ========================================
            daily_transmission_count = transmission_import_counts[idx]
            
            for j in 1:daily_transmission_count
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
            
            # Update local cases time series
            if !isempty(sample_infections)
                # Count local cases (generation > 0) that have been infected by this time
                local_cases_timeseries[idx] = sum((sample_infections.generation .> 0) .& 
                                                   (sample_infections.tinf .<= time))
            end
            
            # Check ICU detection after processing this day's imports
            if !icu_detected && !isempty(sample_infections)
                icu_sampled = NBPMscape.sampleforest(
                    (G = sample_infections,), 
                    icu_params
                )
                
                if !isempty(icu_sampled.treport)
                    icu_detected = true
                    icu_detection_time = minimum(icu_sampled.treport)
                end
            end
            
            # EARLY STOPPING
            if icu_detected && aww_detected
                if verbose && sample <= 3
                    println("    Early stop at day $time: All detections occurred")
                end
                break
            end
        end
        
        # Store this sample's results
        push!(all_sample_results, (
            sample_id = sample,
            times = times,
            detectable_imports = detectable_import_counts,
            transmission_imports = transmission_import_counts,
            local_cases = local_cases_timeseries,
            arrival_time = arrival_time,
            aww_detection_time = aww_detection_time,
            icu_detection_time = icu_detection_time
        ))
    end
    
    if verbose
        n_aww = sum([!isnan(r.aww_detection_time) for r in all_sample_results])
        n_icu = sum([!isnan(r.icu_detection_time) for r in all_sample_results])
        println("    AWW detected: $(n_aww)/$(num_samples) ($(round(100*n_aww/num_samples, digits=1))%)")
        println("    ICU detected: $(n_icu)/$(num_samples) ($(round(100*n_icu/num_samples, digits=1))%)")
    end
    
    return all_sample_results
end

# ============================================================================
# MAIN SIMULATION WITH TIME SERIES OUTPUT
# ============================================================================

function run_timeseries_simulations(
    csv_path::String,
    selected_countries::Vector{String},
    R0_val::Float64,
    gen_time_val::Float64;
    num_samples::Int = 100,
    turnaround_time::Float64 = 3.0,
    extra_time::Float64 = 45.0,
    icu_sampling_proportion::Float64 = 0.10,
    airport_detection_prob::Float64 = 0.08,  # 16% base × 50% sampling
    output_dir::String = "AWW_and_ICU/pgfgleam/datasets/timeseries"
)
    """
    Run time series simulations for specified countries and parameters
    
    Args:
        csv_path: Path to input CSV with import data
        selected_countries: Vector of country names
        R0_val: R0 value
        gen_time_val: Generation time value
        airport_detection_prob: 0.08 for 16% base × 50% sampling
    """
    
    println("="^80)
    println("TIME SERIES SIMULATIONS")
    println("="^80)
    println("R0: $R0_val")
    println("Generation time: $gen_time_val days")
    println("Countries: $selected_countries")
    println("Samples per country: $num_samples")
    println("AWW detection prob: $(airport_detection_prob) (16% base × 50% sampling)")
    println("ICU sampling: $(icu_sampling_proportion*100)%")
    println("Workers: $(nworkers())")
    println("="^80)
    
    # Create output directory
    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    # Load merged data
    println("\nLoading data...")
    merged_data = CSV.read(csv_path, DataFrame)
    println("Loaded $(nrow(merged_data)) rows")
    
    # Process each country
    for country in selected_countries
        println("\n" * "="^80)
        println("Processing: $country")
        println("="^80)
        
        # Filter data for this country and parameters
        country_data = filter(row -> 
            row.R0 == R0_val && 
            row.generation_time == gen_time_val && 
            row.outbreak_country == country,
            merged_data
        )
        
        if nrow(country_data) == 0
            println("WARNING: No data for $country (R0=$R0_val, gen_time=$gen_time_val)")
            continue
        end
        
        sort!(country_data, :time)
        
        # Get mean detection time for max observation window
        mean_det_time = country_data[1, :mean_detection_time]
        if ismissing(mean_det_time) || isnan(mean_det_time)
            println("WARNING: Invalid mean_detection_time for $country")
            continue
        end
        
        max_obs_time = mean_det_time + extra_time
        
        # Filter to max observation time
        country_trimmed = filter(row -> row.time <= max_obs_time, country_data)
        
        if nrow(country_trimmed) == 0
            println("WARNING: No data within observation window for $country")
            continue
        end
        
        # Run simulation
        try
            sample_results = simulate_country_timeseries(
                country_trimmed,
                country,
                num_samples,
                R0_val,
                gen_time_val,
                icu_sampling_proportion,
                airport_detection_prob,
                max_obs_time;
                turnaround_time = turnaround_time,
                verbose = true
            )
            
            # Save results for this country
            output_file = joinpath(output_dir, 
                "timeseries_$(country)_R0_$(R0_val)_gentime_$(gen_time_val).csv")
            
            # Convert to DataFrame format for saving
            println("\nSaving results...")
            
            results_df = DataFrame(
                sample_id = Int[],
                times = String[],  # Will store as string representation of array
                detectable_imports = String[],
                transmission_imports = String[],
                local_cases = String[],
                arrival_time = Float64[],
                aww_detection_time = Float64[],
                icu_detection_time = Float64[]
            )
            
            for result in sample_results
                push!(results_df, (
                    sample_id = result.sample_id,
                    times = string(result.times),
                    detectable_imports = string(result.detectable_imports),
                    transmission_imports = string(result.transmission_imports),
                    local_cases = string(result.local_cases),
                    arrival_time = result.arrival_time,
                    aww_detection_time = result.aww_detection_time,
                    icu_detection_time = result.icu_detection_time
                ))
            end
            
            CSV.write(output_file, results_df)
            println("✓ Saved: $output_file")
            println("  $(nrow(results_df)) samples saved")
            
        catch e
            println("ERROR processing $country: $e")
            continue
        end
    end
    
    println("\n" * "="^80)
    println("ALL TIME SERIES SIMULATIONS COMPLETE")
    println("Results saved to: $output_dir")
    println("="^80)
end

# ============================================================================
# RUN THE TIME SERIES SIMULATIONS
# ============================================================================

# User-specified parameters
selected_countries = ["Ghana", "Nigeria", "Kenya"]  # Specify countries
R0_val = 2.0  # Specify R0
gen_time_val = 4.0  # Specify generation time

run_timeseries_simulations(
    "AWW_and_ICU/pgfgleam/daily_imports_sensitivity_complete.csv",
    selected_countries,
    R0_val,
    gen_time_val;
    num_samples = 100,
    turnaround_time = 3.0,
    extra_time = 45.0,
    icu_sampling_proportion = 0.10,
    airport_detection_prob = 0.08,  # 16% base × 50% sampling = 8%
    output_dir = "AWW_and_ICU/pgfgleam/datasets/timeseries"
)

println("\n✓ Time series simulations complete!")
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
        import_column: Either :daily_detectable_imports_top3 or :daily_detectable_imports_all
    
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
# AWW DETECTION COMPARISON: ALL vs TOP-3
# ============================================================================

@everywhere function simulate_country_aww_comparison(
    country_data::DataFrame,
    country_name::String,
    num_samples::Int,
    R0::Float64,
    mean_generation_time::Float64,
    airport_detection_probs::Vector{Float64},
    max_observation_time::Float64;
    mean_infectious_period = 8/3,
    turnaround_time::Float64 = 3.0,
    verbose::Bool = true
)
    """
    Simulate AWW detection comparing ALL countries vs TOP-3 countries
    
    Compares:
    - AWW with all countries (using daily_detectable_imports_all)
    - AWW with top-3 countries (using daily_detectable_imports_top3)
    
    For each scenario, tests 4 sampling fractions with base p_det = 0.16
    """
    
    if verbose
        println("  Processing $country_name (R0=$R0, gen_time=$mean_generation_time)")
        println("    Max observation time: $max_observation_time days")
        println("    AWW detection probs: $(airport_detection_probs)")
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
    
    # Storage for results - separate for ALL vs TOP3
    aww_all_results = Dict{Float64, Dict{Symbol, Any}}()
    aww_top3_results = Dict{Float64, Dict{Symbol, Any}}()
    
    for p_det in airport_detection_probs
        aww_all_results[p_det] = Dict(
            :detection_times => Float64[],
            :local_cases_at_detection => Float64[],
            :total_detections => 0
        )
        aww_top3_results[p_det] = Dict(
            :detection_times => Float64[],
            :local_cases_at_detection => Float64[],
            :total_detections => 0
        )
    end
    
    all_imports_per_sample = Float64[]
    top3_imports_per_sample = Float64[]
    
    for sample in 1:num_samples
        if verbose && (sample % 10 == 0 || sample == 1)
            println("    Sample $sample/$num_samples")
            flush(stdout)
        end
        
        # ========================================
        # Sample TWO different import scenarios
        # ========================================
        
        # 1. ALL countries - can trigger airport detection
        all_import_counts, total_all = sample_daily_imports_poisson(
            country_data, sample, :daily_detectable_imports_all
        )
        push!(all_imports_per_sample, total_all)
        
        # 2. TOP-3 countries - can trigger airport detection
        top3_import_counts, total_top3 = sample_daily_imports_poisson(
            country_data, sample, :daily_detectable_imports_top3
        )
        push!(top3_imports_per_sample, total_top3)
        
        # Track state for this sample (use ALL imports for local transmission)
        sample_infections = DataFrame()
        
        # Track AWW detection for each p_det (ALL countries)
        aww_all_detected = Dict{Float64, Bool}()
        first_aww_all_detection_time = Dict{Float64, Float64}()
        for p_det in airport_detection_probs
            aww_all_detected[p_det] = false
            first_aww_all_detection_time[p_det] = Inf
        end
        
        # Track AWW detection for each p_det (TOP-3 countries)
        aww_top3_detected = Dict{Float64, Bool}()
        first_aww_top3_detection_time = Dict{Float64, Float64}()
        for p_det in airport_detection_probs
            aww_top3_detected[p_det] = false
            first_aww_top3_detection_time[p_det] = Inf
        end
        
        # Process imports chronologically
        for (idx, row) in enumerate(eachrow(country_data))
            time = row.time
            if time >= max_observation_time
                break
            end
            
            # ========================================
            # AIRPORT DETECTION - ALL COUNTRIES
            # ========================================
            daily_all_count = all_import_counts[idx]
            
            if daily_all_count > 0
                for p_det in airport_detection_probs
                    if !aww_all_detected[p_det]
                        prob_detection_today = 1.0 - (1.0 - p_det)^daily_all_count
                        
                        if rand() < prob_detection_today
                            aww_all_detected[p_det] = true
                            first_aww_all_detection_time[p_det] = time + turnaround_time
                            aww_all_results[p_det][:total_detections] += 1
                        end
                    end
                end
            end
            
            # ========================================
            # AIRPORT DETECTION - TOP-3 COUNTRIES
            # ========================================
            daily_top3_count = top3_import_counts[idx]
            
            if daily_top3_count > 0
                for p_det in airport_detection_probs
                    if !aww_top3_detected[p_det]
                        prob_detection_today = 1.0 - (1.0 - p_det)^daily_top3_count
                        
                        if rand() < prob_detection_today
                            aww_top3_detected[p_det] = true
                            first_aww_top3_detection_time[p_det] = time + turnaround_time
                            aww_top3_results[p_det][:total_detections] += 1
                        end
                    end
                end
            end
            
            # ========================================
            # LOCAL TRANSMISSION: Seeds from ALL imports
            # (using ALL imports to represent actual disease spread)
            # ========================================
            daily_transmission_count = all_import_counts[idx]
            
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
            
            # EARLY STOPPING
            all_aww_all_detected = all(aww_all_detected[p] for p in airport_detection_probs)
            all_aww_top3_detected = all(aww_top3_detected[p] for p in airport_detection_probs)
            if all_aww_all_detected && all_aww_top3_detected
                if verbose && sample <= 3
                    println("    Early stop at day $time: All detections occurred")
                end
                break
            end
        end
        
        # Record detections
        if !isempty(sample_infections)
            # AWW detection - ALL countries
            for p_det in airport_detection_probs
                if aww_all_detected[p_det] && isfinite(first_aww_all_detection_time[p_det])
                    push!(aww_all_results[p_det][:detection_times], first_aww_all_detection_time[p_det])
                    
                    aww_all_local_cases = sum((sample_infections.generation .> 0) .& 
                                             (sample_infections.tinf .<= first_aww_all_detection_time[p_det]))
                    push!(aww_all_results[p_det][:local_cases_at_detection], aww_all_local_cases)
                end
            end
            
            # AWW detection - TOP-3 countries
            for p_det in airport_detection_probs
                if aww_top3_detected[p_det] && isfinite(first_aww_top3_detection_time[p_det])
                    push!(aww_top3_results[p_det][:detection_times], first_aww_top3_detection_time[p_det])
                    
                    aww_top3_local_cases = sum((sample_infections.generation .> 0) .& 
                                              (sample_infections.tinf .<= first_aww_top3_detection_time[p_det]))
                    push!(aww_top3_results[p_det][:local_cases_at_detection], aww_top3_local_cases)
                end
            end
        end
    end
    
    if verbose
        for p_det in airport_detection_probs
            all_pct = round(100*aww_all_results[p_det][:total_detections]/num_samples, digits=1)
            top3_pct = round(100*aww_top3_results[p_det][:total_detections]/num_samples, digits=1)
            println("    p_det=$p_det: AWW-ALL=$all_pct%, AWW-TOP3=$top3_pct%")
        end
        println("    Mean ALL imports/sample: $(round(mean(all_imports_per_sample), digits=2))")
        println("    Mean TOP-3 imports/sample: $(round(mean(top3_imports_per_sample), digits=2))")
    end
    
    return (
        aww_all_results = aww_all_results,
        aww_top3_results = aww_top3_results,
        mean_all_imports_per_sample = mean(all_imports_per_sample),
        mean_top3_imports_per_sample = mean(top3_imports_per_sample)
    )
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
    vector_cols = [:AWW_ALL_016_10pct_detection_times, :AWW_ALL_016_10pct_local_cases_samples,
                   :AWW_ALL_016_25pct_detection_times, :AWW_ALL_016_25pct_local_cases_samples,
                   :AWW_ALL_016_50pct_detection_times, :AWW_ALL_016_50pct_local_cases_samples,
                   :AWW_ALL_016_100pct_detection_times, :AWW_ALL_016_100pct_local_cases_samples,
                   :AWW_TOP3_016_10pct_detection_times, :AWW_TOP3_016_10pct_local_cases_samples,
                   :AWW_TOP3_016_25pct_detection_times, :AWW_TOP3_016_25pct_local_cases_samples,
                   :AWW_TOP3_016_50pct_detection_times, :AWW_TOP3_016_50pct_local_cases_samples,
                   :AWW_TOP3_016_100pct_detection_times, :AWW_TOP3_016_100pct_local_cases_samples]
    
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
        backup_path = output_path * ".backup"
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
# MAIN SIMULATION FROM T3 CSV (BATCHED WITH SAVE)
# ============================================================================

function run_aww_t3_comparison(
    csv_path::String;
    num_samples::Int = 100,
    turnaround_time::Float64 = 3.0,
    max_detection_time_threshold::Float64 = 120.0,
    extra_time::Float64 = 20.0,
    base_pdet::Float64 = 0.16,
    sampling_fractions::Vector{Float64} = [0.10, 0.25, 0.50, 1.0],
    output_path::String = "results_aww_t3_comparison.csv",
    batch_size::Int = 125
)
    """
    Compare AWW detection between ALL countries vs TOP-3 countries
    
    Airport detection model:
    - base_pdet = 0.16 (per-flight detection probability)
    - sampling_fractions = [10%, 25%, 50%, 100%] (proportion of flights tested)
    - effective p_det = base_pdet × sampling_fraction
    
    This gives 8 AWW configurations:
    - 4 for ALL countries: 0.016, 0.04, 0.08, 0.16
    - 4 for TOP-3 countries: 0.016, 0.04, 0.08, 0.16
    
    Features:
    - Early stopping: Stops each sample when all detections occur
    - Batched processing: Processes batch_size combinations at a time
    - Saves after each batch completes
    - Crash recovery: Can resume from existing output file
    """
    
    println("="^80)
    println("AWW ALL vs TOP-3 COMPARISON (BATCHED, SAVE EVERY $batch_size)")
    println("="^80)
    println("Input CSV: $csv_path")
    println("Max detection time threshold: $max_detection_time_threshold days")
    println("Extra simulation time: $extra_time days")
    println("Samples per combination: $num_samples")
    println("Base p_det: $base_pdet")
    println("Sampling fractions: $(sampling_fractions .* 100)%")
    println("Workers: $(nworkers())")
    println("Batch size: $batch_size combinations")
    println("="^80)
    
    # Generate effective detection probabilities
    p_dets = base_pdet .* sampling_fractions
    config_labels = String[]
    for sampling_frac in sampling_fractions
        base_pct = Int(round(base_pdet * 100))  # 0.16 → 16
        base_str = lpad(string(base_pct), 3, '0')  # 16 → "016"
        sampling_pct = Int(round(sampling_frac * 100))  # 0.10 → 10, 0.25 → 25
        pct_str = string(sampling_pct) * "pct"
        push!(config_labels, "$(base_str)_$(pct_str)")
    end
    
    println("\nAirport detection parameters:")
    println("  Base p_det: $base_pdet")
    println("  Sampling fractions: $(sampling_fractions .* 100)%")
    println("  Effective p_det values: $p_dets")
    for (i, (label, pdet)) in enumerate(zip(config_labels, p_dets))
        println("    $label: p_det = $pdet")
    end
    println()
    
    # Ensure output directory exists
    output_dir = dirname(output_path)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    # Load T3 data
    println("\nLoading T3 comparison data...")
    t3_data = CSV.read(csv_path, DataFrame)
    println("Loaded $(nrow(t3_data)) rows")
    
    # Get unique parameter combinations
    param_combinations = unique(t3_data[:, [:R0, :generation_time, :outbreak_country, :mean_detection_time_all]])
    println("\nFound $(nrow(param_combinations)) unique parameter combinations")
    
    # Filter out combinations where mean_detection_time_all is missing or > threshold
    valid_combinations = filter(row -> 
        !ismissing(row.mean_detection_time_all) && 
        !isnan(row.mean_detection_time_all) && 
        row.mean_detection_time_all <= max_detection_time_threshold,
        param_combinations
    )
    
    println("After filtering: $(nrow(valid_combinations)) valid combinations")
    
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
            
            # Create results DataFrame structure
            results_df = DataFrame(
                country = String[],
                R0 = Float64[],
                gen_time = Float64[],
                mean_detection_time_all = Float64[],
                mean_detection_time_top3 = Float64[],
                max_observation_time = Float64[],
                # ALL countries - base 0.16
                AWW_ALL_016_10pct_mean_detection_time = Float64[],
                AWW_ALL_016_10pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_10pct_mean_local_cases = Float64[],
                AWW_ALL_016_10pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_10pct_detections = Int[],
                AWW_ALL_016_25pct_mean_detection_time = Float64[],
                AWW_ALL_016_25pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_25pct_mean_local_cases = Float64[],
                AWW_ALL_016_25pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_25pct_detections = Int[],
                AWW_ALL_016_50pct_mean_detection_time = Float64[],
                AWW_ALL_016_50pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_50pct_mean_local_cases = Float64[],
                AWW_ALL_016_50pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_50pct_detections = Int[],
                AWW_ALL_016_100pct_mean_detection_time = Float64[],
                AWW_ALL_016_100pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_100pct_mean_local_cases = Float64[],
                AWW_ALL_016_100pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_100pct_detections = Int[],
                # TOP-3 countries - base 0.16
                AWW_TOP3_016_10pct_mean_detection_time = Float64[],
                AWW_TOP3_016_10pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_10pct_mean_local_cases = Float64[],
                AWW_TOP3_016_10pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_10pct_detections = Int[],
                AWW_TOP3_016_25pct_mean_detection_time = Float64[],
                AWW_TOP3_016_25pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_25pct_mean_local_cases = Float64[],
                AWW_TOP3_016_25pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_25pct_detections = Int[],
                AWW_TOP3_016_50pct_mean_detection_time = Float64[],
                AWW_TOP3_016_50pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_50pct_mean_local_cases = Float64[],
                AWW_TOP3_016_50pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_50pct_detections = Int[],
                AWW_TOP3_016_100pct_mean_detection_time = Float64[],
                AWW_TOP3_016_100pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_100pct_mean_local_cases = Float64[],
                AWW_TOP3_016_100pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_100pct_detections = Int[],
                mean_all_imports_per_sample = Float64[],
                mean_top3_imports_per_sample = Float64[]
            )
            
            # Copy data row by row with proper type conversion
            for row in eachrow(existing_results)
                push!(results_df, (
                    country = row.country,
                    R0 = row.R0,
                    gen_time = row.gen_time,
                    mean_detection_time_all = row.mean_detection_time_all,
                    mean_detection_time_top3 = row.mean_detection_time_top3,
                    max_observation_time = row.max_observation_time,
                    # ALL - 0.16 base
                    AWW_ALL_016_10pct_mean_detection_time = row.AWW_ALL_016_10pct_mean_detection_time,
                    AWW_ALL_016_10pct_detection_times = eval(Meta.parse(row.AWW_ALL_016_10pct_detection_times)),
                    AWW_ALL_016_10pct_mean_local_cases = row.AWW_ALL_016_10pct_mean_local_cases,
                    AWW_ALL_016_10pct_local_cases_samples = eval(Meta.parse(row.AWW_ALL_016_10pct_local_cases_samples)),
                    AWW_ALL_016_10pct_detections = row.AWW_ALL_016_10pct_detections,
                    AWW_ALL_016_25pct_mean_detection_time = row.AWW_ALL_016_25pct_mean_detection_time,
                    AWW_ALL_016_25pct_detection_times = eval(Meta.parse(row.AWW_ALL_016_25pct_detection_times)),
                    AWW_ALL_016_25pct_mean_local_cases = row.AWW_ALL_016_25pct_mean_local_cases,
                    AWW_ALL_016_25pct_local_cases_samples = eval(Meta.parse(row.AWW_ALL_016_25pct_local_cases_samples)),
                    AWW_ALL_016_25pct_detections = row.AWW_ALL_016_25pct_detections,
                    AWW_ALL_016_50pct_mean_detection_time = row.AWW_ALL_016_50pct_mean_detection_time,
                    AWW_ALL_016_50pct_detection_times = eval(Meta.parse(row.AWW_ALL_016_50pct_detection_times)),
                    AWW_ALL_016_50pct_mean_local_cases = row.AWW_ALL_016_50pct_mean_local_cases,
                    AWW_ALL_016_50pct_local_cases_samples = eval(Meta.parse(row.AWW_ALL_016_50pct_local_cases_samples)),
                    AWW_ALL_016_50pct_detections = row.AWW_ALL_016_50pct_detections,
                    AWW_ALL_016_100pct_mean_detection_time = row.AWW_ALL_016_100pct_mean_detection_time,
                    AWW_ALL_016_100pct_detection_times = eval(Meta.parse(row.AWW_ALL_016_100pct_detection_times)),
                    AWW_ALL_016_100pct_mean_local_cases = row.AWW_ALL_016_100pct_mean_local_cases,
                    AWW_ALL_016_100pct_local_cases_samples = eval(Meta.parse(row.AWW_ALL_016_100pct_local_cases_samples)),
                    AWW_ALL_016_100pct_detections = row.AWW_ALL_016_100pct_detections,
                    # TOP3 - 0.16 base
                    AWW_TOP3_016_10pct_mean_detection_time = row.AWW_TOP3_016_10pct_mean_detection_time,
                    AWW_TOP3_016_10pct_detection_times = eval(Meta.parse(row.AWW_TOP3_016_10pct_detection_times)),
                    AWW_TOP3_016_10pct_mean_local_cases = row.AWW_TOP3_016_10pct_mean_local_cases,
                    AWW_TOP3_016_10pct_local_cases_samples = eval(Meta.parse(row.AWW_TOP3_016_10pct_local_cases_samples)),
                    AWW_TOP3_016_10pct_detections = row.AWW_TOP3_016_10pct_detections,
                    AWW_TOP3_016_25pct_mean_detection_time = row.AWW_TOP3_016_25pct_mean_detection_time,
                    AWW_TOP3_016_25pct_detection_times = eval(Meta.parse(row.AWW_TOP3_016_25pct_detection_times)),
                    AWW_TOP3_016_25pct_mean_local_cases = row.AWW_TOP3_016_25pct_mean_local_cases,
                    AWW_TOP3_016_25pct_local_cases_samples = eval(Meta.parse(row.AWW_TOP3_016_25pct_local_cases_samples)),
                    AWW_TOP3_016_25pct_detections = row.AWW_TOP3_016_25pct_detections,
                    AWW_TOP3_016_50pct_mean_detection_time = row.AWW_TOP3_016_50pct_mean_detection_time,
                    AWW_TOP3_016_50pct_detection_times = eval(Meta.parse(row.AWW_TOP3_016_50pct_detection_times)),
                    AWW_TOP3_016_50pct_mean_local_cases = row.AWW_TOP3_016_50pct_mean_local_cases,
                    AWW_TOP3_016_50pct_local_cases_samples = eval(Meta.parse(row.AWW_TOP3_016_50pct_local_cases_samples)),
                    AWW_TOP3_016_50pct_detections = row.AWW_TOP3_016_50pct_detections,
                    AWW_TOP3_016_100pct_mean_detection_time = row.AWW_TOP3_016_100pct_mean_detection_time,
                    AWW_TOP3_016_100pct_detection_times = eval(Meta.parse(row.AWW_TOP3_016_100pct_detection_times)),
                    AWW_TOP3_016_100pct_mean_local_cases = row.AWW_TOP3_016_100pct_mean_local_cases,
                    AWW_TOP3_016_100pct_local_cases_samples = eval(Meta.parse(row.AWW_TOP3_016_100pct_local_cases_samples)),
                    AWW_TOP3_016_100pct_detections = row.AWW_TOP3_016_100pct_detections,
                    mean_all_imports_per_sample = row.mean_all_imports_per_sample,
                    mean_top3_imports_per_sample = row.mean_top3_imports_per_sample
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
            # Initialize empty results DataFrame
            results_df = DataFrame(
                country = String[],
                R0 = Float64[],
                gen_time = Float64[],
                mean_detection_time_all = Float64[],
                mean_detection_time_top3 = Float64[],
                max_observation_time = Float64[],
                # ALL countries - base 0.16
                AWW_ALL_016_10pct_mean_detection_time = Float64[],
                AWW_ALL_016_10pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_10pct_mean_local_cases = Float64[],
                AWW_ALL_016_10pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_10pct_detections = Int[],
                AWW_ALL_016_25pct_mean_detection_time = Float64[],
                AWW_ALL_016_25pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_25pct_mean_local_cases = Float64[],
                AWW_ALL_016_25pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_25pct_detections = Int[],
                AWW_ALL_016_50pct_mean_detection_time = Float64[],
                AWW_ALL_016_50pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_50pct_mean_local_cases = Float64[],
                AWW_ALL_016_50pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_50pct_detections = Int[],
                AWW_ALL_016_100pct_mean_detection_time = Float64[],
                AWW_ALL_016_100pct_detection_times = Vector{Float64}[],
                AWW_ALL_016_100pct_mean_local_cases = Float64[],
                AWW_ALL_016_100pct_local_cases_samples = Vector{Float64}[],
                AWW_ALL_016_100pct_detections = Int[],
                # TOP-3 countries - base 0.16
                AWW_TOP3_016_10pct_mean_detection_time = Float64[],
                AWW_TOP3_016_10pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_10pct_mean_local_cases = Float64[],
                AWW_TOP3_016_10pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_10pct_detections = Int[],
                AWW_TOP3_016_25pct_mean_detection_time = Float64[],
                AWW_TOP3_016_25pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_25pct_mean_local_cases = Float64[],
                AWW_TOP3_016_25pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_25pct_detections = Int[],
                AWW_TOP3_016_50pct_mean_detection_time = Float64[],
                AWW_TOP3_016_50pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_50pct_mean_local_cases = Float64[],
                AWW_TOP3_016_50pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_50pct_detections = Int[],
                AWW_TOP3_016_100pct_mean_detection_time = Float64[],
                AWW_TOP3_016_100pct_detection_times = Vector{Float64}[],
                AWW_TOP3_016_100pct_mean_local_cases = Float64[],
                AWW_TOP3_016_100pct_local_cases_samples = Vector{Float64}[],
                AWW_TOP3_016_100pct_detections = Int[],
                mean_all_imports_per_sample = Float64[],
                mean_top3_imports_per_sample = Float64[]
            )
        end
    else
        # Initialize empty results DataFrame
        results_df = DataFrame(
            country = String[],
            R0 = Float64[],
            gen_time = Float64[],
            mean_detection_time_all = Float64[],
            mean_detection_time_top3 = Float64[],
            max_observation_time = Float64[],
            # ALL countries - base 0.16
            AWW_ALL_016_10pct_mean_detection_time = Float64[],
            AWW_ALL_016_10pct_detection_times = Vector{Float64}[],
            AWW_ALL_016_10pct_mean_local_cases = Float64[],
            AWW_ALL_016_10pct_local_cases_samples = Vector{Float64}[],
            AWW_ALL_016_10pct_detections = Int[],
            AWW_ALL_016_25pct_mean_detection_time = Float64[],
            AWW_ALL_016_25pct_detection_times = Vector{Float64}[],
            AWW_ALL_016_25pct_mean_local_cases = Float64[],
            AWW_ALL_016_25pct_local_cases_samples = Vector{Float64}[],
            AWW_ALL_016_25pct_detections = Int[],
            AWW_ALL_016_50pct_mean_detection_time = Float64[],
            AWW_ALL_016_50pct_detection_times = Vector{Float64}[],
            AWW_ALL_016_50pct_mean_local_cases = Float64[],
            AWW_ALL_016_50pct_local_cases_samples = Vector{Float64}[],
            AWW_ALL_016_50pct_detections = Int[],
            AWW_ALL_016_100pct_mean_detection_time = Float64[],
            AWW_ALL_016_100pct_detection_times = Vector{Float64}[],
            AWW_ALL_016_100pct_mean_local_cases = Float64[],
            AWW_ALL_016_100pct_local_cases_samples = Vector{Float64}[],
            AWW_ALL_016_100pct_detections = Int[],
            # TOP-3 countries - base 0.16
            AWW_TOP3_016_10pct_mean_detection_time = Float64[],
            AWW_TOP3_016_10pct_detection_times = Vector{Float64}[],
            AWW_TOP3_016_10pct_mean_local_cases = Float64[],
            AWW_TOP3_016_10pct_local_cases_samples = Vector{Float64}[],
            AWW_TOP3_016_10pct_detections = Int[],
            AWW_TOP3_016_25pct_mean_detection_time = Float64[],
            AWW_TOP3_016_25pct_detection_times = Vector{Float64}[],
            AWW_TOP3_016_25pct_mean_local_cases = Float64[],
            AWW_TOP3_016_25pct_local_cases_samples = Vector{Float64}[],
            AWW_TOP3_016_25pct_detections = Int[],
            AWW_TOP3_016_50pct_mean_detection_time = Float64[],
            AWW_TOP3_016_50pct_detection_times = Vector{Float64}[],
            AWW_TOP3_016_50pct_mean_local_cases = Float64[],
            AWW_TOP3_016_50pct_local_cases_samples = Vector{Float64}[],
            AWW_TOP3_016_50pct_detections = Int[],
            AWW_TOP3_016_100pct_mean_detection_time = Float64[],
            AWW_TOP3_016_100pct_detection_times = Vector{Float64}[],
            AWW_TOP3_016_100pct_mean_local_cases = Float64[],
            AWW_TOP3_016_100pct_local_cases_samples = Vector{Float64}[],
            AWW_TOP3_016_100pct_detections = Int[],
            mean_all_imports_per_sample = Float64[],
            mean_top3_imports_per_sample = Float64[]
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
    
    # Process in batches
    for batch_num in 1:num_batches
        batch_start = (batch_num - 1) * batch_size + 1
        batch_end = min(batch_num * batch_size, total_combinations)
        batch_combinations = valid_combinations[batch_start:batch_end, :]
        
        println("\n" * "="^80)
        println("BATCH $batch_num/$num_batches: Processing combinations $batch_start to $batch_end")
        println("="^80)
        flush(stdout)
        
        # Create indexed combinations for this batch
        indexed_combinations = [(i, row) for (i, row) in enumerate(eachrow(batch_combinations))]
        
        # Process batch in parallel
        batch_results = pmap(indexed_combinations) do idx_and_row
            idx, param_row = idx_and_row
            global_idx = batch_start + idx - 1
            
            R0 = Float64(param_row.R0)
            gen_time = Float64(param_row.generation_time)
            country = param_row.outbreak_country
            mean_det_time_all = param_row.mean_detection_time_all
            
            # Calculate max observation time for this simulation
            max_obs_time = mean_det_time_all + extra_time
            
            println("[$global_idx/$total_combinations] Processing: $country | R0=$R0 | gen_time=$gen_time")
            
            # Get data for this combination
            country_data = filter(row -> 
                row.R0 == R0 && 
                row.generation_time == gen_time && 
                row.outbreak_country == country,
                t3_data
            )
            
            if nrow(country_data) == 0
                println("[$global_idx/$total_combinations] WARNING: No data for $country (R0=$R0, gen_time=$gen_time), skipping...")
                return nothing
            end
            
            sort!(country_data, :time)
            
            # Filter to max observation time
            country_trimmed = filter(row -> row.time <= max_obs_time, country_data)
            
            if nrow(country_trimmed) == 0
                println("[$global_idx/$total_combinations] WARNING: No data within observation window for $country, skipping...")
                return nothing
            end
            
            # Get mean detection times from CSV
            mean_det_time_top3 = first(country_trimmed.mean_detection_time_top3)
            
            try
                result = simulate_country_aww_comparison(
                    country_trimmed,
                    country,
                    num_samples,
                    R0,
                    gen_time,
                    p_dets,  # Pass vector of all effective p_det values
                    max_obs_time;
                    turnaround_time = turnaround_time,
                    verbose = false
                )
                
                # Process results for each configuration
                aww_results_by_config = Dict{String, NamedTuple}()
                
                for (i, (label, p_det)) in enumerate(zip(config_labels, p_dets))
                    # ALL countries
                    aww_all_mean_time = isempty(result.aww_all_results[p_det][:detection_times]) ? NaN : 
                                       mean(result.aww_all_results[p_det][:detection_times])
                    aww_all_mean_cases = isempty(result.aww_all_results[p_det][:local_cases_at_detection]) ? NaN : 
                                        mean(result.aww_all_results[p_det][:local_cases_at_detection])
                    
                    # TOP-3 countries
                    aww_top3_mean_time = isempty(result.aww_top3_results[p_det][:detection_times]) ? NaN : 
                                        mean(result.aww_top3_results[p_det][:detection_times])
                    aww_top3_mean_cases = isempty(result.aww_top3_results[p_det][:local_cases_at_detection]) ? NaN : 
                                         mean(result.aww_top3_results[p_det][:local_cases_at_detection])
                    
                    aww_results_by_config[label] = (
                        all_detection_times = result.aww_all_results[p_det][:detection_times],
                        all_mean_detection_time = aww_all_mean_time,
                        all_local_cases_samples = result.aww_all_results[p_det][:local_cases_at_detection],
                        all_mean_local_cases = aww_all_mean_cases,
                        all_num_detections = length(result.aww_all_results[p_det][:detection_times]),
                        top3_detection_times = result.aww_top3_results[p_det][:detection_times],
                        top3_mean_detection_time = aww_top3_mean_time,
                        top3_local_cases_samples = result.aww_top3_results[p_det][:local_cases_at_detection],
                        top3_mean_local_cases = aww_top3_mean_cases,
                        top3_num_detections = length(result.aww_top3_results[p_det][:detection_times])
                    )
                end
                
                println("[$global_idx/$total_combinations] ✓ $country: AWW-ALL-0.016=$(round(aww_results_by_config["016_10pct"].all_mean_detection_time, digits=2))d, AWW-TOP3-0.016=$(round(aww_results_by_config["016_10pct"].top3_mean_detection_time, digits=2))d")
                
                return (
                    country = country,
                    R0 = R0,
                    gen_time = gen_time,
                    mean_detection_time_all = mean_det_time_all,
                    mean_detection_time_top3 = mean_det_time_top3,
                    max_observation_time = max_obs_time,
                    # ALL - 10% sampling
                    AWW_ALL_016_10pct_mean_detection_time = aww_results_by_config["016_10pct"].all_mean_detection_time,
                    AWW_ALL_016_10pct_detection_times = aww_results_by_config["016_10pct"].all_detection_times,
                    AWW_ALL_016_10pct_mean_local_cases = aww_results_by_config["016_10pct"].all_mean_local_cases,
                    AWW_ALL_016_10pct_local_cases_samples = aww_results_by_config["016_10pct"].all_local_cases_samples,
                    AWW_ALL_016_10pct_detections = aww_results_by_config["016_10pct"].all_num_detections,
                    # ALL - 25% sampling
                    AWW_ALL_016_25pct_mean_detection_time = aww_results_by_config["016_25pct"].all_mean_detection_time,
                    AWW_ALL_016_25pct_detection_times = aww_results_by_config["016_25pct"].all_detection_times,
                    AWW_ALL_016_25pct_mean_local_cases = aww_results_by_config["016_25pct"].all_mean_local_cases,
                    AWW_ALL_016_25pct_local_cases_samples = aww_results_by_config["016_25pct"].all_local_cases_samples,
                    AWW_ALL_016_25pct_detections = aww_results_by_config["016_25pct"].all_num_detections,
                    # ALL - 50% sampling
                    AWW_ALL_016_50pct_mean_detection_time = aww_results_by_config["016_50pct"].all_mean_detection_time,
                    AWW_ALL_016_50pct_detection_times = aww_results_by_config["016_50pct"].all_detection_times,
                    AWW_ALL_016_50pct_mean_local_cases = aww_results_by_config["016_50pct"].all_mean_local_cases,
                    AWW_ALL_016_50pct_local_cases_samples = aww_results_by_config["016_50pct"].all_local_cases_samples,
                    AWW_ALL_016_50pct_detections = aww_results_by_config["016_50pct"].all_num_detections,
                    # ALL - 100% sampling
                    AWW_ALL_016_100pct_mean_detection_time = aww_results_by_config["016_100pct"].all_mean_detection_time,
                    AWW_ALL_016_100pct_detection_times = aww_results_by_config["016_100pct"].all_detection_times,
                    AWW_ALL_016_100pct_mean_local_cases = aww_results_by_config["016_100pct"].all_mean_local_cases,
                    AWW_ALL_016_100pct_local_cases_samples = aww_results_by_config["016_100pct"].all_local_cases_samples,
                    AWW_ALL_016_100pct_detections = aww_results_by_config["016_100pct"].all_num_detections,
                    # TOP3 - 10% sampling
                    AWW_TOP3_016_10pct_mean_detection_time = aww_results_by_config["016_10pct"].top3_mean_detection_time,
                    AWW_TOP3_016_10pct_detection_times = aww_results_by_config["016_10pct"].top3_detection_times,
                    AWW_TOP3_016_10pct_mean_local_cases = aww_results_by_config["016_10pct"].top3_mean_local_cases,
                    AWW_TOP3_016_10pct_local_cases_samples = aww_results_by_config["016_10pct"].top3_local_cases_samples,
                    AWW_TOP3_016_10pct_detections = aww_results_by_config["016_10pct"].top3_num_detections,
                    # TOP3 - 25% sampling
                    AWW_TOP3_016_25pct_mean_detection_time = aww_results_by_config["016_25pct"].top3_mean_detection_time,
                    AWW_TOP3_016_25pct_detection_times = aww_results_by_config["016_25pct"].top3_detection_times,
                    AWW_TOP3_016_25pct_mean_local_cases = aww_results_by_config["016_25pct"].top3_mean_local_cases,
                    AWW_TOP3_016_25pct_local_cases_samples = aww_results_by_config["016_25pct"].top3_local_cases_samples,
                    AWW_TOP3_016_25pct_detections = aww_results_by_config["016_25pct"].top3_num_detections,
                    # TOP3 - 50% sampling
                    AWW_TOP3_016_50pct_mean_detection_time = aww_results_by_config["016_50pct"].top3_mean_detection_time,
                    AWW_TOP3_016_50pct_detection_times = aww_results_by_config["016_50pct"].top3_detection_times,
                    AWW_TOP3_016_50pct_mean_local_cases = aww_results_by_config["016_50pct"].top3_mean_local_cases,
                    AWW_TOP3_016_50pct_local_cases_samples = aww_results_by_config["016_50pct"].top3_local_cases_samples,
                    AWW_TOP3_016_50pct_detections = aww_results_by_config["016_50pct"].top3_num_detections,
                    # TOP3 - 100% sampling
                    AWW_TOP3_016_100pct_mean_detection_time = aww_results_by_config["016_100pct"].top3_mean_detection_time,
                    AWW_TOP3_016_100pct_detection_times = aww_results_by_config["016_100pct"].top3_detection_times,
                    AWW_TOP3_016_100pct_mean_local_cases = aww_results_by_config["016_100pct"].top3_mean_local_cases,
                    AWW_TOP3_016_100pct_local_cases_samples = aww_results_by_config["016_100pct"].top3_local_cases_samples,
                    AWW_TOP3_016_100pct_detections = aww_results_by_config["016_100pct"].top3_num_detections,
                    mean_all_imports_per_sample = result.mean_all_imports_per_sample,
                    mean_top3_imports_per_sample = result.mean_top3_imports_per_sample
                )
            catch e
                println("[$global_idx/$total_combinations] ERROR processing $country: $e")
                return nothing
            end
        end
        
        # Filter out nothing results and append to existing results
        valid_results = filter(x -> !isnothing(x), batch_results)
        
        if !isempty(valid_results)
            new_results_df = DataFrame(valid_results)
            append!(results_df, new_results_df)
        end
        
        # SAVE AFTER EACH BATCH
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

results = run_aww_t3_comparison(
    "/Users/reddy/AWW_and_ICU/pgfgleam/daily_imports_sensitivity_T3.csv";
    num_samples = 100,
    turnaround_time = 3.0,
    max_detection_time_threshold = 200.0,
    extra_time = 45.0,
    base_pdet = 0.16,
    sampling_fractions = [0.10, 0.25, 0.50, 1.0],
    output_path = "/Users/reddy/AWW_and_ICU/pgfgleam/datasets/aww_t3_comparison_results.csv",
    batch_size = 200
)

println("\n✓ AWW ALL vs TOP-3 comparison complete!")
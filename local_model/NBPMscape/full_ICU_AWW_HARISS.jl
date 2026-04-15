# ============================================================================
# full_ICU_AWW_HARISS.jl
#
# Three-channel surveillance driver: ICU sampling + Airport Wastewater (AWW)
# + HARISS (Hospital Admission Respiratory Infection Surveillance Sampling).
#
# Mirrors `full_ICU_WW.jl` but adds HARISS as a third detection channel via
# `NBPMscape.secondary_care_td`. Note: the `WW_*` column prefix is retained
# for compatibility with the existing AWW analysis pipeline -- semantically
# `WW_*` here denotes the airport wastewater channel (AWW).
# ============================================================================

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
        import_column: Either :daily_latent_imports, :daily_infectious_imports, or :daily_detectable_imports
    
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
# COMBINED ICU + WW DETECTION WITH SPLIT IMPORT TYPES
# ============================================================================

@everywhere function simulate_country_detection(
    country_data::DataFrame,
    country_name::String,
    num_samples::Int,
    R0::Float64,
    mean_generation_time::Float64,
    icu_sampling_proportion::Float64,
    airport_detection_probs::Vector{Float64},
    max_observation_time::Float64;
    mean_infectious_period = 8/3,
    turnaround_time::Float64 = 3.0,
    n_hosp_samples_per_week::Int = NBPMscape.P.n_hosp_samples_per_week,
    verbose::Bool = true
)
    """
    Simulate ICU + AWW + HARISS detection with CORRECT import types

    KEY:
    - Seeding local transmission uses:
      * daily_latent_imports (L) - starts from latent period (normal seeding)
      * daily_infectious_imports (I) - starts immediately infectious (no latent period)
    - Airport (AWW) detection uses daily_detectable_imports (I + P)
    - HARISS detection samples hospital admissions from the locally simulated
      outbreak via NBPMscape.secondary_care_td.
    """

    if verbose
        println("  Processing $country_name (R0=$R0, gen_time=$mean_generation_time)")
        println("    Max observation time: $max_observation_time days")
        println("    ICU sampling: $(icu_sampling_proportion*100)%")
        println("    AWW detection probs: $(airport_detection_probs)")
        println("    HARISS samples/week: $(n_hosp_samples_per_week)")
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
    
    # Storage for results
    icu_detection_times = Float64[]
    icu_local_cases_at_detection = Float64[]

    hariss_detection_times = Float64[]
    hariss_local_cases_at_detection = Float64[]

    airport_results = Dict{Float64, Dict{Symbol, Any}}()
    for p_det in airport_detection_probs
        airport_results[p_det] = Dict(
            :detection_times => Float64[],
            :local_cases_at_detection => Float64[],
            :total_detections => 0
        )
    end
    
    latent_imports_per_sample = Float64[]
    infectious_imports_per_sample = Float64[]
    detectable_imports_per_sample = Float64[]
    
    for sample in 1:num_samples
        if verbose && (sample % 10 == 0 || sample == 1)
            println("    Sample $sample/$num_samples")
            flush(stdout)
        end
        
        # ========================================
        # KEY CHANGE: Sample THREE different import types
        # ========================================
        
        # 1. Latent imports (L) - seeds local transmission, starts from latent period
        latent_import_counts, total_latent = sample_daily_imports_poisson(
            country_data, sample, :daily_latent_imports
        )
        
        # 2. Infectious imports (I) - seeds local transmission, starts immediately infectious
        infectious_import_counts, total_infectious = sample_daily_imports_poisson(
            country_data, sample, :daily_infectious_imports
        )
        
        push!(latent_imports_per_sample, total_latent)
        push!(infectious_imports_per_sample, total_infectious)
        
        # 3. Detectable imports (I + P) - can trigger airport detection
        detectable_import_counts, total_detectable = sample_daily_imports_poisson(
            country_data, sample, :daily_detectable_imports
        )
        push!(detectable_imports_per_sample, total_detectable)
        
        # Track state for this sample
        sample_infections = DataFrame()
        icu_detected_this_sample = false
        first_icu_detection_time = Inf
        hariss_detected_this_sample = false
        first_hariss_detection_time = Inf
        
        # Track WW detection for each p_det
        airport_detected = Dict{Float64, Bool}()
        first_airport_detection_time = Dict{Float64, Float64}()
        for p_det in airport_detection_probs
            airport_detected[p_det] = false
            first_airport_detection_time[p_det] = Inf
        end
        
        # Process imports chronologically
        for (idx, row) in enumerate(eachrow(country_data))
            time = row.time
            if time >= max_observation_time
                break
            end
            
            # ========================================
            # AIRPORT DETECTION: Uses detectable imports (I + P)
            # ========================================
            daily_detectable_count = detectable_import_counts[idx]
            
            if daily_detectable_count > 0
                for p_det in airport_detection_probs
                    if !airport_detected[p_det]
                        prob_detection_today = 1.0 - (1.0 - p_det)^daily_detectable_count
                        
                        if rand() < prob_detection_today
                            airport_detected[p_det] = true
                            first_airport_detection_time[p_det] = time + turnaround_time
                            airport_results[p_det][:total_detections] += 1
                        end
                    end
                end
            end
            
            # ========================================
            # LOCAL TRANSMISSION: Seeds from latent (L) and infectious (I) imports
            # ========================================
            daily_latent_count = latent_import_counts[idx]
            daily_infectious_count = infectious_import_counts[idx]
            
            # Process latent imports (L) - normal seeding (starts from midpoint of latent period)
            for j in 1:daily_latent_count
                results = NBPMscape.simtree(base_params,
                    initialtime = Float64(time) - latent_period / 2.0,
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
            
            # Process infectious imports (I) - start immediately infectious, and midway through (no latent period)
            # Use modified parameters with essentially zero latent period
            # Set latent_scale to very small value so laglatent ≈ 0, making tinfectious ≈ tinf
            infectious_params = merge(base_params, (
                latent_scale = 1e-6,
                infectious_scale = (infectious_period / 2.0) / fixed_shape,  # Half infectious period remaining
                latent_shape = fixed_shape
            ))
            
            for j in 1:daily_infectious_count
                # Start at arrival time, not before
                results = NBPMscape.simtree(infectious_params,
                    initialtime = Float64(time),
                    maxtime = max_observation_time,
                    maxgenerations = 100,
                    initialcontact = :G
                )
                
                if nrow(results.G) > 0
                    results.G[1, :generation] = 0
                    results.G.import_event .= Int(time)
                    results.G.import_time .= Float64(time)
                    # With very small latent_scale, tinfectious ≈ tinf (immediately infectious)
                    # and tfin ≈ tinf + lagrecovery (no latent period in total duration)
                    
                    if isempty(sample_infections)
                        sample_infections = results.G
                    else
                        append!(sample_infections, results.G)
                    end
                end
            end
            
            # Check ICU detection after processing this day's imports
            if !icu_detected_this_sample && !isempty(sample_infections)
                icu_sampled = NBPMscape.sampleforest(
                    (G = sample_infections,),
                    icu_params
                )

                if !isempty(icu_sampled.treport)
                    icu_detected_this_sample = true
                    first_icu_detection_time = minimum(icu_sampled.treport)
                end
            end

            # Check HARISS (secondary care) detection after processing this day's imports
            if !hariss_detected_this_sample && !isempty(sample_infections)
                try
                    sims_for_hariss = sample_infections
                    if !(:simid in propertynames(sims_for_hariss))
                        sims_for_hariss = copy(sample_infections)
                        sims_for_hariss.simid .= 1
                    end

                    hariss_result = NBPMscape.secondary_care_td(;
                        p = base_params,
                        sims = [(G = sims_for_hariss,)],
                        n_hosp_samples_per_week = n_hosp_samples_per_week,
                    )

                    if nrow(hariss_result) > 0 && :SC_TD in propertynames(hariss_result)
                        sc_td_finite = filter(x -> !ismissing(x) && isfinite(x), hariss_result.SC_TD)
                        if !isempty(sc_td_finite)
                            hariss_detected_this_sample = true
                            first_hariss_detection_time = minimum(sc_td_finite)
                        end
                    end
                catch hariss_err
                    if verbose && sample <= 3
                        println("    HARISS sampling skipped (day $time): $hariss_err")
                    end
                end
            end

            # EARLY STOPPING
            all_airport_detected = all(airport_detected[p] for p in airport_detection_probs)
            if icu_detected_this_sample && all_airport_detected && hariss_detected_this_sample
                if verbose && sample <= 3
                    println("    Early stop at day $time: All detections occurred")
                end
                break
            end
        end
        
        # Record detections
        if !isempty(sample_infections)
            # ICU detection
            if icu_detected_this_sample && isfinite(first_icu_detection_time)
                push!(icu_detection_times, first_icu_detection_time)

                icu_local_cases = sum((sample_infections.generation .> 0) .&
                                     (sample_infections.tinf .<= first_icu_detection_time))
                push!(icu_local_cases_at_detection, icu_local_cases)
            end

            # HARISS detection
            if hariss_detected_this_sample && isfinite(first_hariss_detection_time)
                push!(hariss_detection_times, first_hariss_detection_time)

                hariss_local_cases = sum((sample_infections.generation .> 0) .&
                                         (sample_infections.tinf .<= first_hariss_detection_time))
                push!(hariss_local_cases_at_detection, hariss_local_cases)
            end

            # Airport/WW detection for each p_det
            for p_det in airport_detection_probs
                if airport_detected[p_det] && isfinite(first_airport_detection_time[p_det])
                    push!(airport_results[p_det][:detection_times], first_airport_detection_time[p_det])
                    
                    airport_local_cases = sum((sample_infections.generation .> 0) .& 
                                             (sample_infections.tinf .<= first_airport_detection_time[p_det]))
                    push!(airport_results[p_det][:local_cases_at_detection], airport_local_cases)
                end
            end
        end
    end
    
    if verbose
        for p_det in airport_detection_probs
            airport_pct = round(100*airport_results[p_det][:total_detections]/num_samples, digits=1)
            println("    p_det=$p_det: AWW=$airport_pct% detected")
        end
        icu_pct = round(100*length(icu_detection_times)/num_samples, digits=1)
        println("    ICU=$icu_pct% detected")
        hariss_pct = round(100*length(hariss_detection_times)/num_samples, digits=1)
        println("    HARISS=$hariss_pct% detected")
        println("    Mean latent imports/sample: $(round(mean(latent_imports_per_sample), digits=2))")
        println("    Mean infectious imports/sample: $(round(mean(infectious_imports_per_sample), digits=2))")
        println("    Mean detectable imports/sample: $(round(mean(detectable_imports_per_sample), digits=2))")
    end

    return (
        icu_detection_times = icu_detection_times,
        icu_local_cases_at_detection = icu_local_cases_at_detection,
        hariss_detection_times = hariss_detection_times,
        hariss_local_cases_at_detection = hariss_local_cases_at_detection,
        airport_results = airport_results,
        mean_latent_imports_per_sample = mean(latent_imports_per_sample),
        mean_infectious_imports_per_sample = mean(infectious_imports_per_sample),
        mean_detectable_imports_per_sample = mean(detectable_imports_per_sample)
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
    vector_cols = [:ICU_detection_times, :ICU_local_cases_samples,
                   :HARISS_detection_times, :HARISS_local_cases_samples,
                   :WW_008_10pct_detection_times, :WW_008_10pct_local_cases_samples,
                   :WW_008_25pct_detection_times, :WW_008_25pct_local_cases_samples,
                   :WW_008_50pct_detection_times, :WW_008_50pct_local_cases_samples,
                   :WW_008_100pct_detection_times, :WW_008_100pct_local_cases_samples,
                   :WW_016_10pct_detection_times, :WW_016_10pct_local_cases_samples,
                   :WW_016_25pct_detection_times, :WW_016_25pct_local_cases_samples,
                   :WW_016_50pct_detection_times, :WW_016_50pct_local_cases_samples,
                   :WW_016_100pct_detection_times, :WW_016_100pct_local_cases_samples]
    
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
        backup_path = output_path * ".backup_" * string(Dates.now())
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
# MAIN SIMULATION FROM MERGED CSV (BATCHED WITH SAVE EVERY 125)
# ============================================================================

function run_simulations_from_merged_csv(
    csv_path::String;
    num_samples::Int = 100,
    turnaround_time::Float64 = 3.0,
    max_detection_time_threshold::Float64 = 120.0,
    extra_time::Float64 = 20.0,
    icu_sampling_proportion::Float64 = 0.10,
    n_hosp_samples_per_week::Int = NBPMscape.P.n_hosp_samples_per_week,
    output_path::String = "results_aww_icu_hariss.csv",
    batch_size::Int = 125
)
    """
    Run AWW + ICU + HARISS simulations.

    Airport (AWW) detection model:
    - base_pdet ∈ {0.08, 0.16} (per-flight detection probability)
    - sampling_fraction ∈ {10%, 25%, 50%, 100%} (proportion of flights tested)
    - effective p_det = base_pdet × sampling_fraction

    This gives 8 AWW configurations.

    HARISS detection:
    - Single configuration using n_hosp_samples_per_week (default from
      NBPMscape.P), all other HARISS parameters from NBPMscape.P.

    Features:
    - Early stopping: Stops each sample when all three channels have detected
    - Batched processing: Processes batch_size combinations at a time
    - Saves after each batch completes
    - Crash recovery: Can resume from existing output file
    """

    println("="^80)
    println("AWW + ICU + HARISS SIMULATIONS (8 AWW TYPES, BATCHED, SAVE EVERY $batch_size)")
    println("="^80)
    println("Input CSV: $csv_path")
    println("Max detection time threshold: $max_detection_time_threshold days")
    println("Extra simulation time: $extra_time days")
    println("Samples per combination: $num_samples")
    println("ICU sampling proportion: $(icu_sampling_proportion*100)%")
    println("HARISS samples per week: $n_hosp_samples_per_week")
    println("AWW base p_det values: 8%, 16%")
    println("AWW sampling fractions: 10%, 25%, 50%, 100%")
    println("Workers: $(nworkers())")
    println("Batch size: $batch_size combinations")
    println("="^80)
    
    # Ensure output directory exists
    output_dir = dirname(output_path)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    # Check write permissions
    println("\nChecking output path permissions...")
    println("Output path: $output_path")
    println("Directory exists: $(isdir(output_dir))")
    if output_dir != ""
        println("Directory writable: $(iswritable(output_dir))")
    end
    
    # Load merged data
    println("\nLoading merged data...")
    merged_data = CSV.read(csv_path, DataFrame)
    println("Loaded $(nrow(merged_data)) rows")
    
    # Get unique parameter combinations
    param_combinations = unique(merged_data[:, [:R0, :generation_time, :outbreak_country, :mean_detection_time]])
    println("\nFound $(nrow(param_combinations)) unique parameter combinations")
    
    # Filter out combinations where mean_detection_time is missing or > threshold
    valid_combinations = filter(row -> 
        !ismissing(row.mean_detection_time) && 
        !isnan(row.mean_detection_time) && 
        row.mean_detection_time <= max_detection_time_threshold,
        param_combinations
    )

    # # Filter out R0 = 3, gen_time = 4.0
    # valid_combinations = filter(row -> 
    #     !(Float64(row.R0) == 3.0 && Float64(row.generation_time) == 4.0),
    #     valid_combinations
    # )

    println("After filtering: $(nrow(valid_combinations)) valid combinations")
    println("(Excluded $(nrow(param_combinations) - nrow(valid_combinations)) with mean_detection_time > $max_detection_time_threshold or missing)")
    
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
            
            # Parse the string representations back to vectors
            vector_cols = [:ICU_detection_times, :ICU_local_cases_samples,
                        :HARISS_detection_times, :HARISS_local_cases_samples,
                        :WW_008_10pct_detection_times, :WW_008_10pct_local_cases_samples,
                        :WW_008_25pct_detection_times, :WW_008_25pct_local_cases_samples,
                        :WW_008_50pct_detection_times, :WW_008_50pct_local_cases_samples,
                        :WW_008_100pct_detection_times, :WW_008_100pct_local_cases_samples,
                        :WW_016_10pct_detection_times, :WW_016_10pct_local_cases_samples,
                        :WW_016_25pct_detection_times, :WW_016_25pct_local_cases_samples,
                        :WW_016_50pct_detection_times, :WW_016_50pct_local_cases_samples,
                        :WW_016_100pct_detection_times, :WW_016_100pct_local_cases_samples]
            
            # Create a NEW DataFrame with correct types
            results_df = DataFrame(
                country = String[],
                R0 = Float64[],
                gen_time = Float64[],
                mean_detection_time_from_csv = Float64[],
                max_observation_time = Float64[],
                ICU_mean_detection_time = Float64[],
                ICU_detection_times = Vector{Float64}[],
                ICU_mean_local_cases = Float64[],
                ICU_local_cases_samples = Vector{Float64}[],
                HARISS_mean_detection_time = Float64[],
                HARISS_detection_times = Vector{Float64}[],
                HARISS_mean_local_cases = Float64[],
                HARISS_local_cases_samples = Vector{Float64}[],
                HARISS_detections = Int[],
                # Base p_det = 0.08
                WW_008_10pct_mean_detection_time = Float64[],
                WW_008_10pct_detection_times = Vector{Float64}[],
                WW_008_10pct_mean_local_cases = Float64[],
                WW_008_10pct_local_cases_samples = Vector{Float64}[],
                WW_008_10pct_detections = Int[],
                WW_008_25pct_mean_detection_time = Float64[],
                WW_008_25pct_detection_times = Vector{Float64}[],
                WW_008_25pct_mean_local_cases = Float64[],
                WW_008_25pct_local_cases_samples = Vector{Float64}[],
                WW_008_25pct_detections = Int[],
                WW_008_50pct_mean_detection_time = Float64[],
                WW_008_50pct_detection_times = Vector{Float64}[],
                WW_008_50pct_mean_local_cases = Float64[],
                WW_008_50pct_local_cases_samples = Vector{Float64}[],
                WW_008_50pct_detections = Int[],
                WW_008_100pct_mean_detection_time = Float64[],
                WW_008_100pct_detection_times = Vector{Float64}[],
                WW_008_100pct_mean_local_cases = Float64[],
                WW_008_100pct_local_cases_samples = Vector{Float64}[],
                WW_008_100pct_detections = Int[],
                # Base p_det = 0.16
                WW_016_10pct_mean_detection_time = Float64[],
                WW_016_10pct_detection_times = Vector{Float64}[],
                WW_016_10pct_mean_local_cases = Float64[],
                WW_016_10pct_local_cases_samples = Vector{Float64}[],
                WW_016_10pct_detections = Int[],
                WW_016_25pct_mean_detection_time = Float64[],
                WW_016_25pct_detection_times = Vector{Float64}[],
                WW_016_25pct_mean_local_cases = Float64[],
                WW_016_25pct_local_cases_samples = Vector{Float64}[],
                WW_016_25pct_detections = Int[],
                WW_016_50pct_mean_detection_time = Float64[],
                WW_016_50pct_detection_times = Vector{Float64}[],
                WW_016_50pct_mean_local_cases = Float64[],
                WW_016_50pct_local_cases_samples = Vector{Float64}[],
                WW_016_50pct_detections = Int[],
                WW_016_100pct_mean_detection_time = Float64[],
                WW_016_100pct_detection_times = Vector{Float64}[],
                WW_016_100pct_mean_local_cases = Float64[],
                WW_016_100pct_local_cases_samples = Vector{Float64}[],
                WW_016_100pct_detections = Int[],
                mean_latent_imports_per_sample = Float64[],
                mean_infectious_imports_per_sample = Float64[],
                mean_detectable_imports_per_sample = Float64[]
            )
            
            # Copy data row by row with proper type conversion
            for row in eachrow(existing_results)
                push!(results_df, (
                    country = row.country,
                    R0 = row.R0,
                    gen_time = row.gen_time,
                    mean_detection_time_from_csv = row.mean_detection_time_from_csv,
                    max_observation_time = row.max_observation_time,
                    ICU_mean_detection_time = row.ICU_mean_detection_time,
                    ICU_detection_times = eval(Meta.parse(row.ICU_detection_times)),
                    ICU_mean_local_cases = row.ICU_mean_local_cases,
                    ICU_local_cases_samples = eval(Meta.parse(row.ICU_local_cases_samples)),
                    HARISS_mean_detection_time = row.HARISS_mean_detection_time,
                    HARISS_detection_times = eval(Meta.parse(row.HARISS_detection_times)),
                    HARISS_mean_local_cases = row.HARISS_mean_local_cases,
                    HARISS_local_cases_samples = eval(Meta.parse(row.HARISS_local_cases_samples)),
                    HARISS_detections = row.HARISS_detections,
                    # Base p_det = 0.08
                    WW_008_10pct_mean_detection_time = row.WW_008_10pct_mean_detection_time,
                    WW_008_10pct_detection_times = eval(Meta.parse(row.WW_008_10pct_detection_times)),
                    WW_008_10pct_mean_local_cases = row.WW_008_10pct_mean_local_cases,
                    WW_008_10pct_local_cases_samples = eval(Meta.parse(row.WW_008_10pct_local_cases_samples)),
                    WW_008_10pct_detections = row.WW_008_10pct_detections,
                    WW_008_25pct_mean_detection_time = row.WW_008_25pct_mean_detection_time,
                    WW_008_25pct_detection_times = eval(Meta.parse(row.WW_008_25pct_detection_times)),
                    WW_008_25pct_mean_local_cases = row.WW_008_25pct_mean_local_cases,
                    WW_008_25pct_local_cases_samples = eval(Meta.parse(row.WW_008_25pct_local_cases_samples)),
                    WW_008_25pct_detections = row.WW_008_25pct_detections,
                    WW_008_50pct_mean_detection_time = row.WW_008_50pct_mean_detection_time,
                    WW_008_50pct_detection_times = eval(Meta.parse(row.WW_008_50pct_detection_times)),
                    WW_008_50pct_mean_local_cases = row.WW_008_50pct_mean_local_cases,
                    WW_008_50pct_local_cases_samples = eval(Meta.parse(row.WW_008_50pct_local_cases_samples)),
                    WW_008_50pct_detections = row.WW_008_50pct_detections,
                    WW_008_100pct_mean_detection_time = row.WW_008_100pct_mean_detection_time,
                    WW_008_100pct_detection_times = eval(Meta.parse(row.WW_008_100pct_detection_times)),
                    WW_008_100pct_mean_local_cases = row.WW_008_100pct_mean_local_cases,
                    WW_008_100pct_local_cases_samples = eval(Meta.parse(row.WW_008_100pct_local_cases_samples)),
                    WW_008_100pct_detections = row.WW_008_100pct_detections,
                    # Base p_det = 0.16
                    WW_016_10pct_mean_detection_time = row.WW_016_10pct_mean_detection_time,
                    WW_016_10pct_detection_times = eval(Meta.parse(row.WW_016_10pct_detection_times)),
                    WW_016_10pct_mean_local_cases = row.WW_016_10pct_mean_local_cases,
                    WW_016_10pct_local_cases_samples = eval(Meta.parse(row.WW_016_10pct_local_cases_samples)),
                    WW_016_10pct_detections = row.WW_016_10pct_detections,
                    WW_016_25pct_mean_detection_time = row.WW_016_25pct_mean_detection_time,
                    WW_016_25pct_detection_times = eval(Meta.parse(row.WW_016_25pct_detection_times)),
                    WW_016_25pct_mean_local_cases = row.WW_016_25pct_mean_local_cases,
                    WW_016_25pct_local_cases_samples = eval(Meta.parse(row.WW_016_25pct_local_cases_samples)),
                    WW_016_25pct_detections = row.WW_016_25pct_detections,
                    WW_016_50pct_mean_detection_time = row.WW_016_50pct_mean_detection_time,
                    WW_016_50pct_detection_times = eval(Meta.parse(row.WW_016_50pct_detection_times)),
                    WW_016_50pct_mean_local_cases = row.WW_016_50pct_mean_local_cases,
                    WW_016_50pct_local_cases_samples = eval(Meta.parse(row.WW_016_50pct_local_cases_samples)),
                    WW_016_50pct_detections = row.WW_016_50pct_detections,
                    WW_016_100pct_mean_detection_time = row.WW_016_100pct_mean_detection_time,
                    WW_016_100pct_detection_times = eval(Meta.parse(row.WW_016_100pct_detection_times)),
                    WW_016_100pct_mean_local_cases = row.WW_016_100pct_mean_local_cases,
                    WW_016_100pct_local_cases_samples = eval(Meta.parse(row.WW_016_100pct_local_cases_samples)),
                    WW_016_100pct_detections = row.WW_016_100pct_detections,
                    mean_latent_imports_per_sample = row.mean_latent_imports_per_sample,
                    mean_infectious_imports_per_sample = row.mean_infectious_imports_per_sample,
                    mean_detectable_imports_per_sample = row.mean_detectable_imports_per_sample
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
            # Initialize empty results DataFrame if loading failed
            results_df = DataFrame(
                country = String[],
                R0 = Float64[],
                gen_time = Float64[],
                mean_detection_time_from_csv = Float64[],
                max_observation_time = Float64[],
                ICU_mean_detection_time = Float64[],
                ICU_detection_times = Vector{Float64}[],
                ICU_mean_local_cases = Float64[],
                ICU_local_cases_samples = Vector{Float64}[],
                HARISS_mean_detection_time = Float64[],
                HARISS_detection_times = Vector{Float64}[],
                HARISS_mean_local_cases = Float64[],
                HARISS_local_cases_samples = Vector{Float64}[],
                HARISS_detections = Int[],
                # Base p_det = 0.08
                WW_008_10pct_mean_detection_time = Float64[],
                WW_008_10pct_detection_times = Vector{Float64}[],
                WW_008_10pct_mean_local_cases = Float64[],
                WW_008_10pct_local_cases_samples = Vector{Float64}[],
                WW_008_10pct_detections = Int[],
                WW_008_25pct_mean_detection_time = Float64[],
                WW_008_25pct_detection_times = Vector{Float64}[],
                WW_008_25pct_mean_local_cases = Float64[],
                WW_008_25pct_local_cases_samples = Vector{Float64}[],
                WW_008_25pct_detections = Int[],
                WW_008_50pct_mean_detection_time = Float64[],
                WW_008_50pct_detection_times = Vector{Float64}[],
                WW_008_50pct_mean_local_cases = Float64[],
                WW_008_50pct_local_cases_samples = Vector{Float64}[],
                WW_008_50pct_detections = Int[],
                WW_008_100pct_mean_detection_time = Float64[],
                WW_008_100pct_detection_times = Vector{Float64}[],
                WW_008_100pct_mean_local_cases = Float64[],
                WW_008_100pct_local_cases_samples = Vector{Float64}[],
                WW_008_100pct_detections = Int[],
                # Base p_det = 0.16
                WW_016_10pct_mean_detection_time = Float64[],
                WW_016_10pct_detection_times = Vector{Float64}[],
                WW_016_10pct_mean_local_cases = Float64[],
                WW_016_10pct_local_cases_samples = Vector{Float64}[],
                WW_016_10pct_detections = Int[],
                WW_016_25pct_mean_detection_time = Float64[],
                WW_016_25pct_detection_times = Vector{Float64}[],
                WW_016_25pct_mean_local_cases = Float64[],
                WW_016_25pct_local_cases_samples = Vector{Float64}[],
                WW_016_25pct_detections = Int[],
                WW_016_50pct_mean_detection_time = Float64[],
                WW_016_50pct_detection_times = Vector{Float64}[],
                WW_016_50pct_mean_local_cases = Float64[],
                WW_016_50pct_local_cases_samples = Vector{Float64}[],
                WW_016_50pct_detections = Int[],
                WW_016_100pct_mean_detection_time = Float64[],
                WW_016_100pct_detection_times = Vector{Float64}[],
                WW_016_100pct_mean_local_cases = Float64[],
                WW_016_100pct_local_cases_samples = Vector{Float64}[],
                WW_016_100pct_detections = Int[],
                mean_latent_imports_per_sample = Float64[],
                mean_infectious_imports_per_sample = Float64[],
                mean_detectable_imports_per_sample = Float64[]
            )
        end
    else
        # Initialize empty results DataFrame
        results_df = DataFrame(
            country = String[],
            R0 = Float64[],
            gen_time = Float64[],
            mean_detection_time_from_csv = Float64[],
            max_observation_time = Float64[],
            ICU_mean_detection_time = Float64[],
            ICU_detection_times = Vector{Float64}[],
            ICU_mean_local_cases = Float64[],
            ICU_local_cases_samples = Vector{Float64}[],
            HARISS_mean_detection_time = Float64[],
            HARISS_detection_times = Vector{Float64}[],
            HARISS_mean_local_cases = Float64[],
            HARISS_local_cases_samples = Vector{Float64}[],
            HARISS_detections = Int[],
            # Base p_det = 0.08
            WW_008_10pct_mean_detection_time = Float64[],
            WW_008_10pct_detection_times = Vector{Float64}[],
            WW_008_10pct_mean_local_cases = Float64[],
            WW_008_10pct_local_cases_samples = Vector{Float64}[],
            WW_008_10pct_detections = Int[],
            WW_008_25pct_mean_detection_time = Float64[],
            WW_008_25pct_detection_times = Vector{Float64}[],
            WW_008_25pct_mean_local_cases = Float64[],
            WW_008_25pct_local_cases_samples = Vector{Float64}[],
            WW_008_25pct_detections = Int[],
            WW_008_50pct_mean_detection_time = Float64[],
            WW_008_50pct_detection_times = Vector{Float64}[],
            WW_008_50pct_mean_local_cases = Float64[],
            WW_008_50pct_local_cases_samples = Vector{Float64}[],
            WW_008_50pct_detections = Int[],
            WW_008_100pct_mean_detection_time = Float64[],
            WW_008_100pct_detection_times = Vector{Float64}[],
            WW_008_100pct_mean_local_cases = Float64[],
            WW_008_100pct_local_cases_samples = Vector{Float64}[],
            WW_008_100pct_detections = Int[],
            # Base p_det = 0.16
            WW_016_10pct_mean_detection_time = Float64[],
            WW_016_10pct_detection_times = Vector{Float64}[],
            WW_016_10pct_mean_local_cases = Float64[],
            WW_016_10pct_local_cases_samples = Vector{Float64}[],
            WW_016_10pct_detections = Int[],
            WW_016_25pct_mean_detection_time = Float64[],
            WW_016_25pct_detection_times = Vector{Float64}[],
            WW_016_25pct_mean_local_cases = Float64[],
            WW_016_25pct_local_cases_samples = Vector{Float64}[],
            WW_016_25pct_detections = Int[],
            WW_016_50pct_mean_detection_time = Float64[],
            WW_016_50pct_detection_times = Vector{Float64}[],
            WW_016_50pct_mean_local_cases = Float64[],
            WW_016_50pct_local_cases_samples = Vector{Float64}[],
            WW_016_50pct_detections = Int[],
            WW_016_100pct_mean_detection_time = Float64[],
            WW_016_100pct_detection_times = Vector{Float64}[],
            WW_016_100pct_mean_local_cases = Float64[],
            WW_016_100pct_local_cases_samples = Vector{Float64}[],
            WW_016_100pct_detections = Int[],
            mean_latent_imports_per_sample = Float64[],
            mean_infectious_imports_per_sample = Float64[],
            mean_detectable_imports_per_sample = Float64[]
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
    
    # Airport detection model:
    # - base_pdet ∈ {0.08, 0.16} (per-flight detection probability)
    # - sampling_fraction ∈ {10%, 25%, 50%, 100%} (proportion of flights tested)
    # - effective p_det = base_pdet × sampling_fraction
    base_pdets = [0.08, 0.16]
    sampling_fractions = [0.10, 0.25, 0.50, 1.0]  # 10%, 25%, 50%, 100%
    
    # Generate all combinations
    p_dets = Float64[]
    config_labels = String[]
    for base_pdet in base_pdets
        for sampling_frac in sampling_fractions
            push!(p_dets, base_pdet * sampling_frac)
            # Format: base_pdet as percentage (e.g., 0.08 → "008", 0.16 → "016")
            base_pct = Int(round(base_pdet * 100))  # 0.08 → 4, 0.16 → 16
            base_str = lpad(string(base_pct), 3, '0')  # 4 → "008", 16 → "016"
            sampling_pct = Int(round(sampling_frac * 100))  # 0.10 → 10, 0.25 → 25
            pct_str = string(sampling_pct) * "pct"
            push!(config_labels, "$(base_str)_$(pct_str)")
        end
    end
    
    println("\nAirport detection parameters:")
    println("  Base p_det values: $base_pdets")
    println("  Sampling fractions: $(sampling_fractions .* 100)%")
    println("  Total WW configurations: $(length(p_dets))")
    for (i, (label, pdet)) in enumerate(zip(config_labels, p_dets))
        println("    $label: p_det = $pdet")
    end
    println()
    
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
            mean_det_time = param_row.mean_detection_time
            
            # Calculate max observation time for this simulation
            max_obs_time = mean_det_time + extra_time
            
            println("[$global_idx/$total_combinations] Processing: $country | R0=$R0 | gen_time=$gen_time")
            
            # Get data for this combination
            country_data = filter(row -> 
                row.R0 == R0 && 
                row.generation_time == gen_time && 
                row.outbreak_country == country,
                merged_data
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
            
            try
                result = simulate_country_detection(
                    country_trimmed,
                    country,
                    num_samples,
                    R0,
                    gen_time,
                    icu_sampling_proportion,
                    p_dets,  # Pass vector of all p_det values
                    max_obs_time;
                    turnaround_time = turnaround_time,
                    n_hosp_samples_per_week = n_hosp_samples_per_week,
                    verbose = false
                )

                icu_mean_time = isempty(result.icu_detection_times) ? NaN : mean(result.icu_detection_times)
                icu_mean_cases = isempty(result.icu_local_cases_at_detection) ? NaN : mean(result.icu_local_cases_at_detection)

                hariss_mean_time = isempty(result.hariss_detection_times) ? NaN : mean(result.hariss_detection_times)
                hariss_mean_cases = isempty(result.hariss_local_cases_at_detection) ? NaN : mean(result.hariss_local_cases_at_detection)
                hariss_num_detections = length(result.hariss_detection_times)
                
                # Process results for each p_det configuration
                ww_results_by_config = Dict{String, NamedTuple}()
                
                for (i, (label, p_det)) in enumerate(zip(config_labels, p_dets))
                    ww_mean_time = isempty(result.airport_results[p_det][:detection_times]) ? NaN : 
                                   mean(result.airport_results[p_det][:detection_times])
                    ww_mean_cases = isempty(result.airport_results[p_det][:local_cases_at_detection]) ? NaN : 
                                    mean(result.airport_results[p_det][:local_cases_at_detection])
                    
                    ww_results_by_config[label] = (
                        detection_times = result.airport_results[p_det][:detection_times],
                        mean_detection_time = ww_mean_time,
                        local_cases_samples = result.airport_results[p_det][:local_cases_at_detection],
                        mean_local_cases = ww_mean_cases,
                        num_detections = length(result.airport_results[p_det][:detection_times])
                    )
                end
                
                println("[$global_idx/$total_combinations] ✓ $country: ICU=$(round(icu_mean_time, digits=2))d HARISS=$(round(hariss_mean_time, digits=2))d")

                return (
                    country = country,
                    R0 = R0,
                    gen_time = gen_time,
                    mean_detection_time_from_csv = mean_det_time,
                    max_observation_time = max_obs_time,
                    ICU_mean_detection_time = icu_mean_time,
                    ICU_detection_times = result.icu_detection_times,
                    ICU_mean_local_cases = icu_mean_cases,
                    ICU_local_cases_samples = result.icu_local_cases_at_detection,
                    HARISS_mean_detection_time = hariss_mean_time,
                    HARISS_detection_times = result.hariss_detection_times,
                    HARISS_mean_local_cases = hariss_mean_cases,
                    HARISS_local_cases_samples = result.hariss_local_cases_at_detection,
                    HARISS_detections = hariss_num_detections,
                    # Base p_det = 0.08, 10% sampling
                    WW_008_10pct_mean_detection_time = ww_results_by_config["008_10pct"].mean_detection_time,
                    WW_008_10pct_detection_times = ww_results_by_config["008_10pct"].detection_times,
                    WW_008_10pct_mean_local_cases = ww_results_by_config["008_10pct"].mean_local_cases,
                    WW_008_10pct_local_cases_samples = ww_results_by_config["008_10pct"].local_cases_samples,
                    WW_008_10pct_detections = ww_results_by_config["008_10pct"].num_detections,
                    # Base p_det = 0.08, 25% sampling
                    WW_008_25pct_mean_detection_time = ww_results_by_config["008_25pct"].mean_detection_time,
                    WW_008_25pct_detection_times = ww_results_by_config["008_25pct"].detection_times,
                    WW_008_25pct_mean_local_cases = ww_results_by_config["008_25pct"].mean_local_cases,
                    WW_008_25pct_local_cases_samples = ww_results_by_config["008_25pct"].local_cases_samples,
                    WW_008_25pct_detections = ww_results_by_config["008_25pct"].num_detections,
                    # Base p_det = 0.08, 50% sampling
                    WW_008_50pct_mean_detection_time = ww_results_by_config["008_50pct"].mean_detection_time,
                    WW_008_50pct_detection_times = ww_results_by_config["008_50pct"].detection_times,
                    WW_008_50pct_mean_local_cases = ww_results_by_config["008_50pct"].mean_local_cases,
                    WW_008_50pct_local_cases_samples = ww_results_by_config["008_50pct"].local_cases_samples,
                    WW_008_50pct_detections = ww_results_by_config["008_50pct"].num_detections,
                    # Base p_det = 0.08, 100% sampling
                    WW_008_100pct_mean_detection_time = ww_results_by_config["008_100pct"].mean_detection_time,
                    WW_008_100pct_detection_times = ww_results_by_config["008_100pct"].detection_times,
                    WW_008_100pct_mean_local_cases = ww_results_by_config["008_100pct"].mean_local_cases,
                    WW_008_100pct_local_cases_samples = ww_results_by_config["008_100pct"].local_cases_samples,
                    WW_008_100pct_detections = ww_results_by_config["008_100pct"].num_detections,
                    # Base p_det = 0.16, 10% sampling
                    WW_016_10pct_mean_detection_time = ww_results_by_config["016_10pct"].mean_detection_time,
                    WW_016_10pct_detection_times = ww_results_by_config["016_10pct"].detection_times,
                    WW_016_10pct_mean_local_cases = ww_results_by_config["016_10pct"].mean_local_cases,
                    WW_016_10pct_local_cases_samples = ww_results_by_config["016_10pct"].local_cases_samples,
                    WW_016_10pct_detections = ww_results_by_config["016_10pct"].num_detections,
                    # Base p_det = 0.16, 25% sampling
                    WW_016_25pct_mean_detection_time = ww_results_by_config["016_25pct"].mean_detection_time,
                    WW_016_25pct_detection_times = ww_results_by_config["016_25pct"].detection_times,
                    WW_016_25pct_mean_local_cases = ww_results_by_config["016_25pct"].mean_local_cases,
                    WW_016_25pct_local_cases_samples = ww_results_by_config["016_25pct"].local_cases_samples,
                    WW_016_25pct_detections = ww_results_by_config["016_25pct"].num_detections,
                    # Base p_det = 0.16, 50% sampling
                    WW_016_50pct_mean_detection_time = ww_results_by_config["016_50pct"].mean_detection_time,
                    WW_016_50pct_detection_times = ww_results_by_config["016_50pct"].detection_times,
                    WW_016_50pct_mean_local_cases = ww_results_by_config["016_50pct"].mean_local_cases,
                    WW_016_50pct_local_cases_samples = ww_results_by_config["016_50pct"].local_cases_samples,
                    WW_016_50pct_detections = ww_results_by_config["016_50pct"].num_detections,
                    # Base p_det = 0.16, 100% sampling
                    WW_016_100pct_mean_detection_time = ww_results_by_config["016_100pct"].mean_detection_time,
                    WW_016_100pct_detection_times = ww_results_by_config["016_100pct"].detection_times,
                    WW_016_100pct_mean_local_cases = ww_results_by_config["016_100pct"].mean_local_cases,
                    WW_016_100pct_local_cases_samples = ww_results_by_config["016_100pct"].local_cases_samples,
                    WW_016_100pct_detections = ww_results_by_config["016_100pct"].num_detections,
                    mean_latent_imports_per_sample = result.mean_latent_imports_per_sample,
                    mean_infectious_imports_per_sample = result.mean_infectious_imports_per_sample,
                    mean_detectable_imports_per_sample = result.mean_detectable_imports_per_sample
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
        
        # SAVE AFTER EACH BATCH using safe write function
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

# Resolve paths relative to the repository root so the script is portable
project_root = normpath(joinpath(@__DIR__, "..", ".."))
input_csv_path = "global_model/pgfgleam/all_results/global/daily_imports_sensitivity.csv"
output_csv_path = "global_model/pgfgleam/all_results/local/full_ICU_AWW_HARISS_result.csv"

results = run_simulations_from_merged_csv(
    input_csv_path;
    num_samples = 100,
    turnaround_time = 3.0,
    max_detection_time_threshold = 200.0,
    extra_time = 35.0,
    icu_sampling_proportion = 0.10,
    n_hosp_samples_per_week = NBPMscape.P.n_hosp_samples_per_week,
    output_path = output_csv_path,
    batch_size = 125
)

println("\n✓ ICU + AWW + HARISS simulations complete!")
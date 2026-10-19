#!/bin/bash
#SBATCH --job-name=hariss_sens
#SBATCH --partition=standard
#SBATCH --account=no-project
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/breddyk/AWW_and_ICU/logs/hariss_sens_%j.out
#SBATCH --error=/home/users/breddyk/AWW_and_ICU/logs/hariss_sens_%j.err

# HARISS sensitivity analysis on JASMIN LOTUS. Submit from a sci server:
#   COVID:      sbatch submit_hariss_sensitivity_lotus.sh
#   Influenza:  sbatch --export=ALL,DISEASE=influenza submit_hariss_sensitivity_lotus.sh
#   Quick test: sbatch --export=ALL,N_SAMPLES=4,COUNTRIES=Spain --time=02:00:00 submit_hariss_sensitivity_lotus.sh
# Workers = cpus-per-task - 1 (read from SLURM_CPUS_PER_TASK by the script).

REPO=/home/users/breddyk/AWW_and_ICU
mkdir -p "${REPO}/logs" "${REPO}/global_model/pgfgleam/all_results/local"

$(which julia) --project="${REPO}/local_model" -e 'using Pkg; Pkg.instantiate()'
$(which julia) --project="${REPO}/local_model" "${REPO}/local_model/NBPMscape/hariss_sensitivity.jl"

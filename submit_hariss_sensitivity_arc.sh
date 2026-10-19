#!/usr/bin/env bash
#SBATCH --job-name=hariss_sens
#SBATCH --clusters=arc
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/data/biol-epi/%u/AWW_and_ICU/logs/hariss_sens_%j.out
#SBATCH --error=/data/biol-epi/%u/AWW_and_ICU/logs/hariss_sens_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=benjamin.reddy@stats.ox.ac.uk

# HARISS sensitivity analysis (local_model/NBPMscape/hariss_sensitivity.jl).
# The script caps itself at 24 workers, hence 25 CPUs (24 workers + main).
#
#   COVID:      sbatch submit_hariss_sensitivity_arc.sh
#   Influenza:  sbatch --export=ALL,DISEASE=influenza submit_hariss_sensitivity_arc.sh
#   Quick test: sbatch --export=ALL,N_SAMPLES=2,COUNTRIES=Spain --partition=short --time=02:00:00 submit_hariss_sensitivity_arc.sh
# Other overrides: R0, GEN_TIME, COUNTRIES="Spain;Germany;United States;Japan"

module purge
module load Julia/1.12.1-linux-x86_64

REPO=/data/biol-epi/${USER}/AWW_and_ICU
mkdir -p "${REPO}/logs" "${REPO}/global_model/pgfgleam/all_results/local"

julia --project="${REPO}/local_model" -e 'using Pkg; Pkg.instantiate()'

julia --project="${REPO}/local_model" "${REPO}/local_model/NBPMscape/hariss_sensitivity.jl"

#!/usr/bin/env bash
#SBATCH --job-name=aww_icu_hariss
#SBATCH --clusters=arc
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/data/biol-epi/%u/AWW_and_ICU/logs/aww_%j.out
#SBATCH --error=/data/biol-epi/%u/AWW_and_ICU/logs/aww_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your.oxford.email@ndph.ox.ac.uk>

# ── environment ──────────────────────────────────────────────────────────────
module purge
module load Julia/1.10.0-linux-x86_64     # check with: module spider julia

REPO=/data/biol-epi/${USER}/AWW_and_ICU

mkdir -p "${REPO}/logs"
mkdir -p "${REPO}/global_model/pgfgleam/all_results/local"

# Instantiate packages on first run (no-op if already done)
julia --project="${REPO}/local_model" -e 'using Pkg; Pkg.instantiate()'

# ── run ──────────────────────────────────────────────────────────────────────
# --threads=auto uses all cpus-per-task cores for Julia threading.
# addprocs inside the script spawns 31 workers (cpus-per-task - 1).
julia --project="${REPO}/local_model" \
      --threads=auto \
      "${REPO}/local_model/NBPMscape/full_ICU_AWW_HARISS_update.jl"
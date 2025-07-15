#!/bin/bash
#SBATCH --job-name=ld_window
#SBATCH --array=0-99                # one task per window (adjust!)
#SBATCH --output=logs/ld_%A_%a.out
#SBATCH --error=logs/ld_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab

# ----------------------------------------------------------------------
# USER SETTINGS  -------------------------------------------------------
# ----------------------------------------------------------------------
# Path to your experiment‑config JSON (any model)
CONFIG_JSON="/projects/kernlab/akapoor/Infer_Demography/config_files/experiment_config_split_isolation.json"

# Simulation ID you want to process  (e.g.  "0003")
SIM_ID="0003"

# Base directory that Snakemake / your pipeline used
SIM_ROOT="/projects/kernlab/akapoor/Infer_Demography/MomentsLD/LD_stats"

# Comma‑separated r‑bin edges **MUST match** what was used upstream
RBINS="0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

# Python env that has  moments  installed
module load miniconda
conda activate snakemake-env
# ----------------------------------------------------------------------

WIN_ID=${SLURM_ARRAY_TASK_ID}                     # 0 … (array‑max)

SIM_DIR="${SIM_ROOT}/sim_${SIM_ID}"               # e.g. …/sim_0003
echo "Compute LD  |  sim=${SIM_ID}   window=${WIN_ID}"

python /projects/kernlab/akapoor/Infer_Demography/snakemake_scripts/compute_ld_window.py \
       --sim-dir      "${SIM_DIR}" \
       --window-index "${WIN_ID}" \
       --config-file  "${CONFIG_JSON}" \
       --r-bins       "${RBINS}"

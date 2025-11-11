#!/bin/bash
#SBATCH --job-name=ld_stats
#SBATCH --array=0-9999
#SBATCH --output=logs/ld_%A_%a.out
#SBATCH --error=logs/ld_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --exclusive
#SBATCH --partition=kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# Load required modules for GPU support
module load cuda/12.4.1 || echo "CUDA module not found, continuing..."
module load python3/3.11.4 || echo "Python 3.11 module not found, using system python"

# Activate conda environment with snakemake
source ~/miniforge3/etc/profile.d/conda.sh
conda activate snakemake-env

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=1     # number of (sim,window) jobs per array task - reduced for GPU memory
# ----------------------------------------------------------------------------

# -------- config & constants -----------------------------------------------
# : "${CFG_PATH:?CFG_PATH is not defined}"
# CFG="$CFG_PATH"
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_migration.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")
NUM_WINDOWS=100
TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

# -------- slice of indices handled by *this* array task --------------------
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $END -ge $TOTAL_TASKS ]] && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $START .. $END"

# -------- loop over (sim, window) pairs ------------------------------------
for IDX in $(seq "$START" "$END"); do
    SID=$(( IDX / NUM_WINDOWS ))
    WIN=$(( IDX % NUM_WINDOWS ))

    TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/LD_stats/LD_stats_window_${WIN}.pkl"
    echo "LD‑stats: SID=$SID  WIN=$WIN  →  $TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT"      \
              --nolock                  \
              --latency-wait 120        \
              --rerun-incomplete        \
              --resources gpu_mem=1     \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET" || { echo "Failed for SID=$SID WIN=$WIN"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

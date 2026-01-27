#!/bin/bash
#SBATCH --job-name=ld_stats
#SBATCH --array=0-9999
#SBATCH --output=logs/ld_%A_%a.out
#SBATCH --error=logs/ld_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=kerngpu,gpu,gpulong
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -eo pipefail

# --- Make modules available (Talapas-style) ---
module --ignore_cache purge || true

# Load the CUDA module THAT WORKED for you (use the exact name you successfully loaded)
# Example (replace with your working module):
module --ignore_cache load cuda/11.8


# --- Conda env ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate snakemake-env

# Ensure conda libs (incl nvrtc) are visible at runtime
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Optional but helpful (avoid hammering $HOME with JIT cache)
export CUPY_CACHE_DIR="/tmp/${USER}/cupy_cache_${SLURM_JOB_ID}"
mkdir -p "$CUPY_CACHE_DIR"

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=50
# ----------------------------------------------------------------------------

CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_isolation.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")
NUM_WINDOWS=100
TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $END -ge $TOTAL_TASKS ]] && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $START .. $END"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import cupy; from cupy_backends.cuda.libs import nvrtc; print('NVRTC', nvrtc.getVersion())" || true

for IDX in $(seq "$START" "$END"); do
    SID=$(( IDX / NUM_WINDOWS ))
    WIN=$(( IDX % NUM_WINDOWS ))

    TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/LD_stats/LD_stats_window_${WIN}.pkl"
    ABS_TARGET="${ROOT}/${TARGET}"

    if [[ -f "$ABS_TARGET" ]]; then
        echo "SKIP: LD-stats already exist for SID=$SID WIN=$WIN ($ABS_TARGET)"
        continue
    fi

    echo "RUN: LD-stats: SID=$SID  WIN=$WIN  →  $TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT"      \
              --nolock                  \
              --latency-wait 120        \
              --rerun-incomplete        \
              --rerun-triggers mtime    \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET"
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

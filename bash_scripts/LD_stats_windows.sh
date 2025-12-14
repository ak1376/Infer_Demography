#!/bin/bash
#SBATCH --job-name=ld_stats
#SBATCH --array=0-9999
#SBATCH --output=logs/ld_%A_%a.out
#SBATCH --error=logs/ld_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=kerngpu,gpu,gpulong
#SBATCH --account=kernlab
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=50     # number of (sim,window) jobs per array task
# ----------------------------------------------------------------------------

# -------- config & constants -----------------------------------------------
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_bottleneck.json"
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
    ABS_TARGET="${ROOT}/${TARGET}"

    # 🔍 Skip if LD_stats already exists
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

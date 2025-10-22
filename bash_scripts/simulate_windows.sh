#!/bin/bash
#SBATCH --job-name=win_sim
#SBATCH --array=0-9999
#SBATCH --output=logs/win_sim_%A_%a.out
#SBATCH --error=logs/win_sim_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

BATCH_SIZE=50

# : "${CFG_PATH:?CFG_PATH is not defined}"
# CFG="$CFG_PATH"
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_migration.json"

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws' "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")
NUM_WINDOWS=100
TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

# first launch: resubmit with proper array range
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    : "${MAX_CONCURRENT:=200}"
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY"%$MAX_CONCURRENT "$0"
    exit 0
fi

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
(( END >= TOTAL_TASKS )) && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $START .. $END"

for IDX in $(seq "$START" "$END"); do
    SID=$(( IDX / NUM_WINDOWS ))
    WIN=$(( IDX % NUM_WINDOWS ))

    # Guard: only attempt windows when simulation is finished
    SIM_DIR="$ROOT/experiments/$MODEL/simulations/$SID"
    if [[ ! -f "$SIM_DIR/.done" ]] || [[ ! -f "$SIM_DIR/sampled_params.pkl" ]]; then
        echo "[SKIP] SID=$SID not ready (.done or sampled_params missing)"
        continue
    fi

    TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/windows/window_${WIN}.vcf.gz"
    echo "→ SID=$SID WIN=$WIN  Target=$TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT" \
              --nolock \
              --rerun-incomplete \
              --allowed-rules simulate_window ld_window \
              --latency-wait 300 \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET"
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

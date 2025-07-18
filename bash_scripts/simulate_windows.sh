#!/bin/bash
#SBATCH --job-name=win_sim
#SBATCH --array=0-9999
#SBATCH --output=logs/win_sim_%A_%a.out
#SBATCH --error=logs/win_sim_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# ---------------------------------------------------------------------------
# 0. batching parameters (adjust if desired) --------------------------------
# ---------------------------------------------------------------------------
BATCH_SIZE=50          # number of (sim,window) combos this array task processes

# ---------------------------------------------------------------------------
# 1. config & derived constants --------------------------------------------
# ---------------------------------------------------------------------------
: "${CFG_PATH:?CFG_PATH is not defined}"          # exported by the driver script
CFG="$CFG_PATH"

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'         "$CFG")
MODEL=$(jq -r '.demographic_model'     "$CFG")
NUM_WINDOWS=100                         # same constant as in Snakefile

TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))   # every (sim,window) pair

# zero‑pad width for simulation folders
PAD=$(python - <<EOF
import math, sys; n=int(sys.argv[1]); print(int(math.log10(n-1))+1)
EOF
"$NUM_DRAWS")

# ---------------------------------------------------------------------------
# 2. (re)submit if script wasn't launched as an array yet -------------------
# ---------------------------------------------------------------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY"%$MAX_CONCURRENT "$0" "$@"
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. compute batch bounds for this array ID ---------------------------------
# ---------------------------------------------------------------------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$((  (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))

echo "Array task $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

mkdir -p logs

# ---------------------------------------------------------------------------
# 4. loop over indices in this batch ---------------------------------------
# ---------------------------------------------------------------------------
for TASK_ID in $(seq "$BATCH_START" "$BATCH_END"); do
    SID=$(( TASK_ID / NUM_WINDOWS ))
    WIN=$(( TASK_ID % NUM_WINDOWS ))
    PAD_SID=$(printf "%0${PAD}d" "$SID")

    TARGET="experiments/${MODEL}/inferences/sim_${PAD_SID}/MomentsLD/windows/window_${WIN}.vcf.gz"
    echo "Processing SID=$SID  WIN=$WIN  →  $TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT" \
              --nolock \
              --rerun-incomplete \
              -j "$SLURM_CPUS_PER_TASK" \
              --latency-wait 60 \
              "$TARGET"

    if [[ $? -ne 0 ]]; then
        echo "Snakemake failed for SID=$SID WIN=$WIN"
        exit 1
    fi
done

echo "Array task $SLURM_ARRAY_TASK_ID finished successfully."

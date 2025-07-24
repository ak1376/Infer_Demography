#!/bin/bash
#SBATCH --job-name=opt_momLD
#SBATCH --array=0-19
#SBATCH --output=logs/optLD_%A_%a.out
#SBATCH --error=logs/optLD_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue                               # Requeue on preemption
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# ---------------------------------------------------------------------------
# 0. batching parameters ----------------------------------------------------
# ---------------------------------------------------------------------------
BATCH_SIZE=1          # ← 5 simulations optimised per array task

# ---------------------------------------------------------------------------
# 1. paths & config ---------------------------------------------------------
# ---------------------------------------------------------------------------
: "${CFG_PATH:?CFG_PATH is not defined}"       # export this before sbatch
CFG="$CFG_PATH"

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws' "$CFG")          # total simulations
MODEL=$(jq -r '.demographic_model' "$CFG")

# zero‑pad width for sim folder names
PAD=$(python - <<EOF
import math, sys; n=int(sys.argv[1])
print(int(math.log10(n-1))+1)
EOF
"$NUM_DRAWS")

# ---------------------------------------------------------------------------
# 2. resubmit with correct --array range if first launch --------------------
# ---------------------------------------------------------------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    NUM_ARRAY=$(( (NUM_DRAWS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY"%$MAX_CONCURRENT "$0" "$@"
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. determine which sims this array task handles ---------------------------
# ---------------------------------------------------------------------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $NUM_DRAWS ]] && BATCH_END=$(( NUM_DRAWS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → sims $BATCH_START .. $BATCH_END"

# ---------------------------------------------------------------------------
# 4. loop through the sims in this batch ------------------------------------
# ---------------------------------------------------------------------------
for SID in $(seq "$BATCH_START" "$BATCH_END"); do
    PAD_SID=$(printf "%0${PAD}d" "$SID")
    TARGET="experiments/${MODEL}/inferences/sim_${PAD_SID}/MomentsLD/best_fit.pkl"

    echo "Optimising Moments‑LD for SID=$SID  →  $TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT" \
              --rerun-incomplete \
              --nolock \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET"

    if [[ $? -ne 0 ]]; then
        echo "Snakemake failed for SID=$SID"
        exit 1
    fi
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

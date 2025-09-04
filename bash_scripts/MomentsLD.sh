#!/bin/bash
#SBATCH --job-name=opt_momLD
#SBATCH --array=0-999                # <— full range, no %MAX_CONCURRENT
#SBATCH --output=logs/optLD_%A_%a.out
#SBATCH --error=logs/optLD_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. batching parameters ----------------------------------------------------
# ---------------------------------------------------------------------------
BATCH_SIZE=1

# ---------------------------------------------------------------------------
# 1. paths & config ---------------------------------------------------------
# ---------------------------------------------------------------------------
# : "${CFG_PATH:?CFG_PATH is not defined}"
# CFG="$CFG_PATH"
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_migration.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws' "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")

# ---------------------------------------------------------------------------
# 2. resubmit with correct --array range if launched without --array --------
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_ARRAY=$(( (NUM_DRAWS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
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
# 4. run Snakemake for each sim in this batch -------------------------------
# ---------------------------------------------------------------------------
for SID in $(seq "$BATCH_START" "$BATCH_END"); do
    TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/best_fit.pkl"
    echo "Optimising Moments‑LD for SID=$SID  →  $TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory "$ROOT" \
              --rerun-incomplete \
              --nolock \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET" \
              || { echo "Snakemake failed for SID=$SID"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

#!/bin/bash
#SBATCH --job-name=dadi_infer
#SBATCH --array=0-9999
#SBATCH --output=logs/dadi_%A_%a.out
#SBATCH --error=logs/dadi_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# ---------------- batching knob ----------------
BATCH_SIZE=50

# ---------------- paths & config ----------------
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

# Prefer CFG_PATH if exported; otherwise default to a specific file.
CFG="${CFG_PATH:-/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_bottleneck.json}"

# Export so the Snakefile picks it up (Snakefile should read os.environ['EXP_CFG'])
export EXP_CFG="$CFG"

# ---------------- derive run sizes from JSON ----------------
NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")
TOTAL_TASKS=$(( NUM_DRAWS * NUM_OPTIMS ))

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_DRAWS: $NUM_DRAWS  NUM_OPTIMS: $NUM_OPTIMS  TOTAL_TASKS: $TOTAL_TASKS"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"

# ---------------- first-launch helper (keeps your original behavior) ----------------
# If you run this script directly (not via sbatch) it will submit itself with the right array range.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
  echo "Submitting array 0..${NUM_ARRAY}"
  sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
  exit 0
fi

# ---------------- compute slice for this array ID ----------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

# ---------------- loop over (sid,opt) pairs in this slice ----------------
for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
  SID=$(( IDX / NUM_OPTIMS ))
  OPT=$(( IDX % NUM_OPTIMS ))

  TARGET="experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/dadi/fit_params.pkl"
  echo "dadi optimisation: SID=$SID  OPT=$OPT  →  $TARGET"

  snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --nolock \
    -j "$SLURM_CPUS_PER_TASK" \
    "$TARGET" || { echo "Snakemake failed for SID=$SID OPT=$OPT"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

#!/bin/bash
#SBATCH --job-name=dadi_infer_cpu
#SBATCH --array=0-9999
#SBATCH --output=logs/dadi_cpu_%A_%a.out
#SBATCH --error=logs/dadi_cpu_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

BATCH_SIZE=50

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

CFG="${CFG_PATH:-/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_drosophila_three_epoch.json}"
export EXP_CFG="$CFG"

# Hard-disable GPU visibility (belt + suspenders)
export CUDA_VISIBLE_DEVICES=""
export SLURM_GPUS=0

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")
TOTAL_TASKS=$(( NUM_DRAWS * NUM_OPTIMS ))

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_DRAWS: $NUM_DRAWS  NUM_OPTIMS: $NUM_OPTIMS  TOTAL_TASKS: $TOTAL_TASKS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Self-submit if launched without an array task id
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
  echo "Submitting array 0..${NUM_ARRAY}"
  sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
  exit 0
fi

# --- compute slice for this array id ---
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))
echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

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

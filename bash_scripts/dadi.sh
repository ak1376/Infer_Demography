#!/bin/bash
#SBATCH --job-name=dadi_infer_cpu
#SBATCH --array=0-9999
#SBATCH --output=logs/dadi_cpu_%A_%a.out
#SBATCH --error=logs/dadi_cpu_%A_%a.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=2
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

# ----------------------------
# Helper: decide whether canonical output is "non-empty"
# Non-empty := pickle loads AND n_files_found>0 AND best_ll is non-empty (or best_params non-empty)
# Empty example you showed has n_files_found=0 and lists empty.
# Returns:
#   0 => NON-EMPTY (skip)
#   1 => EMPTY / missing / unreadable (do not skip; allow recompute)
# ----------------------------
is_nonempty_canon() {
  local pkl="$1"

  [[ -f "$pkl" ]] || return 1

  python3 - <<'PY' "$pkl"
import pickle, sys
p = sys.argv[1]
try:
    d = pickle.load(open(p, "rb"))
except Exception:
    sys.exit(1)

# tolerate missing keys
n = d.get("n_files_found", 0)
best_ll = d.get("best_ll", [])
best_params = d.get("best_params", [])

nonempty = (n is not None and int(n) > 0) and (len(best_ll) > 0 or len(best_params) > 0)
sys.exit(0 if nonempty else 1)
PY
}

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

  # ---- canonical "sim_XX" output we use to decide whether to skip ----
  CANON_OUT="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/dadi/fit_params.pkl"

  # Skip ONLY if canonical exists AND is non-empty.
  # If canonical missing OR "empty" (like your example) OR unreadable -> recompute.
  if is_nonempty_canon "$CANON_OUT"; then
    echo "SKIP: sim_${SID} has NON-EMPTY $CANON_OUT (so skipping SID=$SID OPT=$OPT)"
    continue
  else
    if [[ -f "$CANON_OUT" ]]; then
      echo "RE-RUN: sim_${SID} has EMPTY/UNREADABLE $CANON_OUT (so running SID=$SID OPT=$OPT)"
    else
      echo "RUN: sim_${SID} missing $CANON_OUT (so running SID=$SID OPT=$OPT)"
    fi
  fi

  # ---- what this job will build (per-run output) ----
  TARGET="experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/dadi/fit_params.pkl"
  echo "→ build $TARGET"

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

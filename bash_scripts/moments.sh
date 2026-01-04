#!/bin/bash
#SBATCH --job-name=moments_infer
#SBATCH --array=0-9999
#SBATCH --output=logs/moments_%A_%a.out
#SBATCH --error=logs/moments_%A_%a.err
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

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=200   # number of (sim,opt) pairs per array element
# ----------------------------------------------------------------------------

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

CFG="${CFG_PATH:-/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_drosophila_three_epoch.json}"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")
TOTAL_TASKS=$(( NUM_DRAWS * NUM_OPTIMS ))

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_DRAWS: $NUM_DRAWS  NUM_OPTIMS: $NUM_OPTIMS  TOTAL_TASKS: $TOTAL_TASKS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# Optional: wait for required inputs from simulation stage
wait_for() {
  local timeout="$1"; shift
  local start
  start=$(date +%s)
  while :; do
    local ok=1
    for f in "$@"; do
      [[ -s "$f" ]] || { ok=0; break; }
    done
    (( ok )) && return 0
    (( $(date +%s) - start >= timeout )) && return 1
    sleep 2
  done
}

# ----------------------------
# Helper: decide whether canonical output is "non-empty"
# Non-empty := pickle loads AND n_files_found>0 AND best_ll is non-empty (or best_params non-empty)
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

n = d.get("n_files_found", 0)
best_ll = d.get("best_ll", [])
best_params = d.get("best_params", [])

try:
    n = int(n) if n is not None else 0
except Exception:
    n = 0

nonempty = (n > 0) and (len(best_ll) > 0 or len(best_params) > 0)
sys.exit(0 if nonempty else 1)
PY
}

# -------- first launch: compute proper --array range -----------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
  echo "Submitting array 0..${NUM_ARRAY}"
  sbatch --array=0-"$NUM_ARRAY"%${MAX_CONCURRENT:-100} "$0" "$@"
  exit 0
fi

# -------- slice for this array element -------------------------------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

if [[ $BATCH_START -gt $BATCH_END ]]; then
  echo "No work for array task $SLURM_ARRAY_TASK_ID (start=$BATCH_START > end=$BATCH_END)"
  exit 0
fi

# -------- loop over (sim,opt) pairs ----------------------------------------
for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
  SID=$(( IDX / NUM_OPTIMS ))
  OPT=$(( IDX % NUM_OPTIMS ))

  # ---- canonical "sim_XX" output we use to decide whether to skip ----
  CANON_OUT="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/moments/fit_params.pkl"

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
  TARGET="experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/moments/fit_params.pkl"
  echo "→ build $TARGET"

  # ---- inputs (optional but you already had these checks) ----
  SFS="$ROOT/experiments/${MODEL}/simulations/${SID}/SFS.pkl"
  PAR="$ROOT/experiments/${MODEL}/simulations/${SID}/sampled_params.pkl"

  if ! wait_for 600 "$SFS" "$PAR"; then
    echo "Timeout waiting for inputs: $SFS $PAR" >&2
    exit 1
  fi

  snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --rerun-incomplete \
    --nolock \
    --latency-wait 300 \
    --rerun-triggers mtime \
    -j "$SLURM_CPUS_PER_TASK" \
    "$TARGET" \
    || { echo "Snakemake failed for SID=$SID OPT=$OPT"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

#!/bin/bash
#SBATCH --job-name=agg_opts_both
#SBATCH --output=logs/agg_both_%A_%a.out
#SBATCH --error=logs/agg_both_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# -----------------------------
# Tunables
# -----------------------------
BATCH_SIZE="${BATCH_SIZE:-1}"      # sims per array task
SIM_RANGE="${SIM_RANGE:-4000-4999}"         # optional "5000-20000" inclusive
FORCE="${FORCE:-0}"                # 1 => force rerun even if canonical looks good
DRYRUN="${DRYRUN:-0}"              # 1 => add -n to snakemake (no execution)

# -----------------------------
# Paths & config
# -----------------------------
CFG="${CFG_PATH:-/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_isolation.json}"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")

mkdir -p logs

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_DRAWS: $NUM_DRAWS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "SIM_RANGE=${SIM_RANGE:-<unset>}"
echo "FORCE=$FORCE DRYRUN=$DRYRUN"

# ----------------------------
# Helper: canonical output is "non-empty"
# Non-empty := pickle loads AND n_files_found>0 AND (best_ll non-empty OR best_params non-empty)
# Returns:
#   0 => NON-EMPTY (good)
#   1 => EMPTY / missing / unreadable
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
best_ll = d.get("best_ll", []) or []
best_params = d.get("best_params", []) or []

try:
    n = int(n)
except Exception:
    n = 0

nonempty = (n > 0) and (len(best_ll) > 0 or len(best_params) > 0)
sys.exit(0 if nonempty else 1)
PY
}

# -----------------------------
# Parse SIM_RANGE (optional)
# -----------------------------
SIM_LO=0
SIM_HI=$(( NUM_DRAWS - 1 ))

if [[ -n "$SIM_RANGE" ]]; then
  if [[ "$SIM_RANGE" =~ ^[0-9]+-[0-9]+$ ]]; then
    SIM_LO="${SIM_RANGE%-*}"
    SIM_HI="${SIM_RANGE#*-}"
  else
    echo "ERROR: SIM_RANGE must look like '5000-20000' (got '$SIM_RANGE')"
    exit 2
  fi
  (( SIM_LO < 0 )) && SIM_LO=0
  (( SIM_HI > NUM_DRAWS-1 )) && SIM_HI=$(( NUM_DRAWS - 1 ))
  if (( SIM_LO > SIM_HI )); then
    echo "ERROR: SIM_RANGE lower bound > upper bound after clamping: ${SIM_LO}-${SIM_HI}"
    exit 2
  fi
fi

# -----------------------------
# Self-submit if launched without array id
# -----------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  START_BATCH=$(( SIM_LO / BATCH_SIZE ))
  END_BATCH=$(( SIM_HI / BATCH_SIZE ))
  echo "Submitting array ${START_BATCH}-${END_BATCH} to cover sims ${SIM_LO}-${SIM_HI} (inclusive)"
  sbatch --array="${START_BATCH}-${END_BATCH}" --export=ALL "$0" "$@"
  exit 0
fi

# -----------------------------
# Compute this task's batch slice
# -----------------------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $NUM_DRAWS ]] && BATCH_END=$(( NUM_DRAWS - 1 ))

if (( BATCH_END < SIM_LO || BATCH_START > SIM_HI )); then
  echo "Array $SLURM_ARRAY_TASK_ID covers sims $BATCH_START..$BATCH_END, outside requested $SIM_LO..$SIM_HI → nothing to do."
  exit 0
fi

RUN_START=$BATCH_START
RUN_END=$BATCH_END
(( RUN_START < SIM_LO )) && RUN_START=$SIM_LO
(( RUN_END   > SIM_HI )) && RUN_END=$SIM_HI

echo "Array $SLURM_ARRAY_TASK_ID → sims $RUN_START .. $RUN_END (batch was $BATCH_START .. $BATCH_END)"

# -----------------------------
# Snakemake args
# -----------------------------
SMK_ARGS=(
  -j "$SLURM_CPUS_PER_TASK"
  --snakefile "$SNAKEFILE"
  --directory "$ROOT"
  --rerun-incomplete
  --nolock
  --rerun-triggers mtime
  --allowed-rules aggregate_opts_dadi aggregate_opts_moments cleanup_optimization_runs
  --keep-going
)
(( DRYRUN )) && SMK_ARGS+=(-n)

# -----------------------------
# Loop over sims
# -----------------------------
for sid in $(seq "$RUN_START" "$RUN_END"); do
  echo "aggregate_opts (both engines): sid=$sid"

  CANON_MOM="$ROOT/experiments/${MODEL}/inferences/sim_${sid}/moments/fit_params.pkl"
  CANON_DADI="$ROOT/experiments/${MODEL}/inferences/sim_${sid}/dadi/fit_params.pkl"
  CLEANUP="$ROOT/experiments/${MODEL}/inferences/sim_${sid}/cleanup_done.txt"

  mom_ok=0
  dadi_ok=0
  is_nonempty_canon "$CANON_MOM"  && mom_ok=1
  is_nonempty_canon "$CANON_DADI" && dadi_ok=1
  cleanup_exists=0
  [[ -f "$CLEANUP" ]] && cleanup_exists=1

  if (( FORCE == 0 && mom_ok == 1 && dadi_ok == 1 && cleanup_exists == 1 )); then
    echo "SKIP: sim_${sid} has NON-EMPTY canonical moments+dadi pkls and cleanup_done.txt"
    continue
  fi

  if (( FORCE == 1 )); then
    echo "FORCE=1: rerunning aggregation/cleanup for sim_${sid}"
  else
    echo "RE-RUN: sim_${sid} needs work (mom_ok=$mom_ok dadi_ok=$dadi_ok cleanup_exists=$cleanup_exists)"
  fi

  # IMPORTANT: always pass explicit targets so snakemake never defaults to rule all
  TGT_MOM="experiments/${MODEL}/inferences/sim_${sid}/moments/fit_params.pkl"
  TGT_DADI="experiments/${MODEL}/inferences/sim_${sid}/dadi/fit_params.pkl"
  TGT_CLEAN="experiments/${MODEL}/inferences/sim_${sid}/cleanup_done.txt"

  echo "Targets: $TGT_MOM  $TGT_DADI  $TGT_CLEAN"

  snakemake "${SMK_ARGS[@]}" "$TGT_MOM" "$TGT_DADI" "$TGT_CLEAN" \
    || { echo "Snakemake failed for sid=$sid"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

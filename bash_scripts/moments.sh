#!/bin/bash
#SBATCH --job-name=moments_infer
#SBATCH --array=0-9999
#SBATCH --output=logs/moments_%A_%a.out
#SBATCH --error=logs/moments_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=10          # number of (sim,opt) pairs this array task handles
# ----------------------------------------------------------------------------

# -------- config -----------------------------------------------------------
: "${CFG_PATH:?CFG_PATH is not defined}"
CFG="$CFG_PATH"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")

TOTAL_TASKS=$(( NUM_DRAWS * NUM_OPTIMS ))

# Optional: small helper to wait for required inputs from simulation stage
wait_for() {
  local timeout="$1"; shift
  local start=$(date +%s)
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

# -------- first launch: compute proper --array range -----------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY"%${MAX_CONCURRENT:-100} "$0" "$@"
    exit 0
fi

# -------- slice for this array element -------------------------------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

# -------- loop over (sim,opt) pairs ----------------------------------------
for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
    SID=$(( IDX / NUM_OPTIMS ))
    OPT=$(( IDX % NUM_OPTIMS ))

    # target path must match Snakefile (no padding)
    TARGET="experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/moments/fit_params.pkl"
    echo "moments optimisation: SID=$SID  OPT=$OPT  →  $TARGET"

    # ensure inputs from simulation are visible (optional but recommended)
    SFS="experiments/${MODEL}/simulations/${SID}/SFS.pkl"
    PAR="experiments/${MODEL}/simulations/${SID}/sampled_params.pkl"
    if ! wait_for 600 "$ROOT/$SFS" "$ROOT/$PAR"; then
        echo "Timeout waiting for inputs: $SFS $PAR" >&2
        exit 1
    fi

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT" \
              --rerun-incomplete \
              --nolock \
              --latency-wait 300 \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET" \
        || { echo "Snakemake failed for SID=$SID OPT=$OPT"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

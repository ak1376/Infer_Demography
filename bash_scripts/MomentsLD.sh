#!/bin/bash
#SBATCH --job-name=opt_momLD
#SBATCH --output=logs/optLD_%A_%a.out
#SBATCH --error=logs/optLD_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose
set -euo pipefail

# ---------------------------------------------------------------------------
# One LHS/jitter-seeded MomentsLD restart per (sid,opt), same array-sizing
# pattern as moments.sh/dadi.sh (NUM_DRAWS*NUM_OPTIMS, not just NUM_DRAWS)
# so all num_optimizations restarts per sim actually run in parallel across
# the array instead of serially inside one per-sim task. Requires
# MomentsLD_prep.sh to have already built means.varcovs.pkl for every sim
# (this script's infer_momentsld/infer_momentsld_pruned targets depend on
# it but never build it themselves, so concurrent opts for the same sid
# never race on creating it). Aggregation across opts happens afterward in
# aggregate_momentsld.sh.
#
# All targets in one array task's batch are collected first and built in a
# SINGLE Snakemake call (same rationale as LD_stats_windows.sh/
# build_windows.sh: cuts redundant per-target DAG-parse/startup overhead).
# Each infer_momentsld[_pruned] job only needs threads:1, so passing
# -j $SLURM_CPUS_PER_TASK (now 8, was 1) also lets up to that many restarts
# in the batch run concurrently instead of strictly one at a time.
# NOTE: raising cpus-per-task multiplies this stage's total concurrent core
# footprint by the same factor across every array task running at once --
# check your account's QOS headroom (sacctmgr show qos
# format=Name,MaxTRESPU,MaxJobsPU) and/or lower the array throttle
# (MAX_CONCURRENT=625 instead of the 5000 default) before submitting at
# scale, since the throttle was originally sized assuming 1 core/task.
# ---------------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-50}"   # number of (sid,opt) pairs per array element

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r    '.demographic_model'   "$CFG")
TOTAL_TASKS=$(( NUM_DRAWS * NUM_OPTIMS ))

PRUNE_MODE=$(jq -r '.prune_mode // "off"' "$CFG")
case "$PRUNE_MODE" in
    fraction) PRUNE_FRACS=$(jq -r '(.prune_keep_values // [])[] | (. * 100 | round | tostring) | "thin" + .' "$CFG" 2>/dev/null || true) ;;
    count)    PRUNE_FRACS=$(jq -r '(.prune_keep_values // [])[] | tostring | "n" + .'                      "$CFG" 2>/dev/null || true) ;;
    *)        PRUNE_FRACS="" ;;
esac

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_DRAWS: $NUM_DRAWS  NUM_OPTIMS: $NUM_OPTIMS  TOTAL_TASKS: $TOTAL_TASKS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    echo "Submitting array 0..${NUM_ARRAY}"
    sbatch --array=0-"$NUM_ARRAY"%${MAX_CONCURRENT:-5000} "$0" "$@"
    exit 0
fi

BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

if [[ $BATCH_START -gt $BATCH_END ]]; then
    echo "No work for array task $SLURM_ARRAY_TASK_ID (start=$BATCH_START > end=$BATCH_END)"
    exit 0
fi

# Collect every target this array task needs, then build them all in one
# Snakemake call (see header comment for rationale).
TARGETS=()
if [[ -n "$PRUNE_FRACS" ]]; then
    ALLOWED_RULE="infer_momentsld_pruned"
else
    ALLOWED_RULE="infer_momentsld"
fi

for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
    SID=$(( IDX / NUM_OPTIMS ))
    OPT=$(( IDX % NUM_OPTIMS ))

    if [[ -n "$PRUNE_FRACS" ]]; then
        for FRAC_TAG in $PRUNE_FRACS; do
            CANON_OUT="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/pruning/${FRAC_TAG}/best_fit.pkl"
            if [[ -s "$CANON_OUT" ]]; then
                echo "[sim_${SID} ${FRAC_TAG}] already aggregated -> skipping OPT=$OPT"
                continue
            fi
            echo "QUEUE: (mixed, ${FRAC_TAG}) SID=$SID OPT=$OPT"
            TARGETS+=("experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/MomentsLD/pruning/${FRAC_TAG}/best_fit.pkl")
        done
    else
        CANON_OUT="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/best_fit.pkl"
        if [[ -s "$CANON_OUT" ]]; then
            echo "[sim_${SID}] already aggregated -> skipping OPT=$OPT"
            continue
        fi
        echo "QUEUE: SID=$SID OPT=$OPT"
        TARGETS+=("experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/MomentsLD/best_fit.pkl")
    fi
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Nothing to build for this array task (all skipped)."
else
    echo "Building ${#TARGETS[@]} targets in one Snakemake call (-j $SLURM_CPUS_PER_TASK, allowed-rule=$ALLOWED_RULE)..."
    # --keep-going: one restart failing shouldn't strand the rest of this
    # batch's otherwise-good restarts. Not wrapped in `|| true` -- a real
    # failure should still surface as this array task's exit status.
    snakemake --snakefile "$SNAKEFILE" \
              --directory "$ROOT" \
              --rerun-incomplete \
              --nolock \
              --keep-going \
              --allowed-rules "$ALLOWED_RULE" \
              -j "$SLURM_CPUS_PER_TASK" \
              "${TARGETS[@]}"
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

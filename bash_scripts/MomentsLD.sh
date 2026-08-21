#!/bin/bash
#SBATCH --job-name=opt_momLD
#SBATCH --output=logs/optLD_%A_%a.out
#SBATCH --error=logs/optLD_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
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

PRUNE_FRACS=$(jq -r '(.prune_keep_fractions // [])[] | (. * 100 | round | tostring) | "thin" + .' "$CFG" 2>/dev/null || true)

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

            TARGET="experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/MomentsLD/pruning/${FRAC_TAG}/best_fit.pkl"
            echo "Optimising Moments-LD (mixed, ${FRAC_TAG}) for SID=$SID OPT=$OPT → $TARGET"
            snakemake --snakefile "$SNAKEFILE" \
                      --directory "$ROOT" \
                      --rerun-incomplete \
                      --nolock \
                      --allowed-rules infer_momentsld_pruned \
                      -j "$SLURM_CPUS_PER_TASK" \
                      "$TARGET" \
                      || { echo "Snakemake failed for SID=$SID OPT=$OPT FRAC=$FRAC_TAG"; exit 1; }
        done
    else
        CANON_OUT="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/best_fit.pkl"
        if [[ -s "$CANON_OUT" ]]; then
            echo "[sim_${SID}] already aggregated -> skipping OPT=$OPT"
            continue
        fi

        TARGET="experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/MomentsLD/best_fit.pkl"
        echo "Optimising Moments-LD for SID=$SID OPT=$OPT → $TARGET"
        snakemake --snakefile "$SNAKEFILE" \
                  --directory "$ROOT" \
                  --rerun-incomplete \
                  --nolock \
                  --allowed-rules infer_momentsld \
                  -j "$SLURM_CPUS_PER_TASK" \
                  "$TARGET" \
                  || { echo "Snakemake failed for SID=$SID OPT=$OPT"; exit 1; }
    fi
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

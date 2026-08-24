#!/bin/bash
#SBATCH --job-name=agg_momLD
#SBATCH --output=logs/aggLD_%A_%a.out
#SBATCH --error=logs/aggLD_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose
set -euo pipefail

# ---------------------------------------------------------------------------
# Picks the top-K MomentsLD restarts per sim, once all num_optimizations
# restarts from MomentsLD.sh have finished. Split out of MomentsLD.sh so
# that script's array can be sized NUM_DRAWS*NUM_OPTIMS (parallel restarts)
# rather than NUM_DRAWS (one task doing all restarts + aggregation
# serially) -- mirrors the moments/dadi -> aggregate_moments_dadi.sh split.
# ---------------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-1}"

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws'         "$CFG")
MODEL=$(jq -r    '.demographic_model'  "$CFG")

PRUNE_MODE=$(jq -r '.prune_mode // "off"' "$CFG")
case "$PRUNE_MODE" in
    fraction) PRUNE_FRACS=$(jq -r '(.prune_keep_values // [])[] | (. * 100 | round | tostring) | "thin" + .' "$CFG" 2>/dev/null || true) ;;
    count)    PRUNE_FRACS=$(jq -r '(.prune_keep_values // [])[] | tostring | "n" + .'                      "$CFG" 2>/dev/null || true) ;;
    *)        PRUNE_FRACS="" ;;
esac

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_ARRAY=$(( (NUM_DRAWS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
    exit 0
fi

BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $NUM_DRAWS ]] && BATCH_END=$(( NUM_DRAWS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → sims $BATCH_START .. $BATCH_END"

for SID in $(seq "$BATCH_START" "$BATCH_END"); do
    if [[ -n "$PRUNE_FRACS" ]]; then
        for FRAC_TAG in $PRUNE_FRACS; do
            TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/pruning/${FRAC_TAG}/best_fit.pkl"
            echo "Aggregating Moments-LD opts (mixed, ${FRAC_TAG}) for SID=$SID → $TARGET"
            snakemake --snakefile "$SNAKEFILE" \
                      --directory "$ROOT" \
                      --rerun-incomplete \
                      --nolock \
                      --allowed-rules aggregate_opts_momentsld_pruned \
                      -j "$SLURM_CPUS_PER_TASK" \
                      "$TARGET" \
                      || { echo "Snakemake failed for SID=$SID FRAC=$FRAC_TAG"; exit 1; }
        done
    else
        TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/best_fit.pkl"
        echo "Aggregating Moments-LD opts for SID=$SID → $TARGET"
        snakemake --snakefile "$SNAKEFILE" \
                  --directory "$ROOT" \
                  --rerun-incomplete \
                  --nolock \
                  --allowed-rules aggregate_opts_momentsld \
                  -j "$SLURM_CPUS_PER_TASK" \
                  "$TARGET" \
                  || { echo "Snakemake failed for SID=$SID"; exit 1; }
    fi
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

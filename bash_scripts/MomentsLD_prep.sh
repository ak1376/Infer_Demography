#!/bin/bash
#SBATCH --job-name=prep_momLD
#SBATCH --output=logs/prepLD_%A_%a.out
#SBATCH --error=logs/prepLD_%A_%a.err
#SBATCH --time=2:00:00
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
# Builds means.varcovs.pkl (+ bootstrap sets) once per sim, BEFORE the
# multi-restart optimization stage (MomentsLD.sh) runs. Splitting this out
# means every infer_momentsld/infer_momentsld_pruned restart across the
# NUM_DRAWS*NUM_OPTIMS array in MomentsLD.sh can rely on this file already
# existing instead of racing to build it as a shared Snakemake dependency
# when multiple opts for the same sid land in concurrent array tasks.
# ---------------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-1}"

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'         "$CFG")
MODEL=$(jq -r    '.demographic_model'  "$CFG")

PRUNE_FRACS=$(jq -r '(.prune_keep_fractions // [])[] | (. * 100 | round | tostring) | "thin" + .' "$CFG" 2>/dev/null || true)

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
    TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/means.varcovs.pkl"
    echo "Aggregating LD stats for SID=$SID → $TARGET"
    snakemake --snakefile "$SNAKEFILE" \
              --directory "$ROOT" \
              --rerun-incomplete \
              --nolock \
              --allowed-rules aggregate_ld_stats \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET" \
              || { echo "Snakemake failed for SID=$SID"; exit 1; }

    if [[ -n "$PRUNE_FRACS" ]]; then
        for FRAC_TAG in $PRUNE_FRACS; do
            TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/pruning/${FRAC_TAG}/means.varcovs.pkl"
            echo "Aggregating pruned LD stats (${FRAC_TAG}) for SID=$SID → $TARGET"
            snakemake --snakefile "$SNAKEFILE" \
                      --directory "$ROOT" \
                      --rerun-incomplete \
                      --nolock \
                      --allowed-rules aggregate_ld_stats_pruned \
                      -j "$SLURM_CPUS_PER_TASK" \
                      "$TARGET" \
                      || { echo "Snakemake failed for SID=$SID FRAC=$FRAC_TAG"; exit 1; }
        done
    fi
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

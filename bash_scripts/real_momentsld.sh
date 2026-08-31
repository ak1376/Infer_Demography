#!/bin/bash
#SBATCH --job-name=real_momld
#SBATCH --output=logs/real_momld_%A_%a.out
#SBATCH --error=logs/real_momld_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage D2: one LHS/jitter-seeded MomentsLD restart per opt, same
# array-per-restart pattern as MomentsLD.sh -- every one of
# num_optimizations restarts gets its own array slot instead of being
# funneled through a single job's core count. Requires real_momentsld_prep.sh
# (means.varcovs.pkl) and real_aggregate_sfs.sh (moments best_fit, used only
# as an optimization seed) to have already finished.
#
# Rule run: infer_momentsld_real (targeted per-restart directly; the
# aggregate step is a separate job -- see real_aggregate_momentsld.sh).

set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-10}"   # opts per array element

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
export EXP_CFG="$CFG"

MODEL=$(jq -r '.demographic_model' "$CFG")
NUM_REAL_OPTIMS=$(jq -r '.num_optimizations // 3' "$CFG")

REAL_RUN_ROOT="experiments/${MODEL}/real_data_analysis/runs"

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_REAL_OPTIMS: $NUM_REAL_OPTIMS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_ARRAY=$(( (NUM_REAL_OPTIMS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    echo "Submitting array 0..${NUM_ARRAY}"
    sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
    exit 0
fi

BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $NUM_REAL_OPTIMS ]] && BATCH_END=$(( NUM_REAL_OPTIMS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → opts $BATCH_START .. $BATCH_END"

TARGETS=()
for OPT in $(seq "$BATCH_START" "$BATCH_END"); do
    TARGET="${REAL_RUN_ROOT}/run_${OPT}/inferences/MomentsLD/best_fit.pkl"
    if [[ -s "$ROOT/$TARGET" ]]; then
        echo "SKIP: OPT=$OPT (already exists: $TARGET)"
        continue
    fi
    echo "QUEUE: OPT=$OPT -> $TARGET"
    TARGETS+=("$TARGET")
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Nothing to build for this array task (all skipped)."
else
    echo "Building ${#TARGETS[@]} targets in one Snakemake call (-j $SLURM_CPUS_PER_TASK)..."
    snakemake \
        --snakefile "$SNAKEFILE" \
        --directory "$ROOT" \
        --nolock \
        --keep-going \
        --rerun-incomplete \
        --rerun-triggers mtime \
        --latency-wait 120 \
        --allowed-rules infer_momentsld_real \
        -j "$SLURM_CPUS_PER_TASK" \
        "${TARGETS[@]}"
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

#!/bin/bash
#SBATCH --job-name=real_sfs_infer
#SBATCH --output=logs/real_sfs_infer_%A_%a.out
#SBATCH --error=logs/real_sfs_infer_%A_%a.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage B1: one moments/dadi NLopt restart per (engine,opt), same array-per-
# restart pattern as moments.sh/dadi.sh -- every one of num_optimizations
# restarts, for BOTH engines, gets its own array slot instead of being
# funneled through a single job's core count. Requires real_data_prep.sh
# to have already produced the combined autosomal SFS.
#
# Rule run: infer_engine_real (targeted per-restart directly; the aggregate
# step is a separate job -- see real_aggregate_sfs.sh -- exactly like
# aggregate_moments_dadi.sh is separate from moments.sh/dadi.sh).
#
# Index space is engine*NUM_REAL_OPTIMS + opt, engine in {moments, dadi}, so
# both engines' restarts are interleaved across the same array instead of
# needing two separate scripts.

set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-10}"   # (engine,opt) pairs per array element

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
export EXP_CFG="$CFG"

MODEL=$(jq -r '.demographic_model'      "$CFG")
NUM_REAL_OPTIMS=$(jq -r '.num_optimizations // 3' "$CFG")
ENGINES=(moments dadi)
TOTAL_TASKS=$(( ${#ENGINES[@]} * NUM_REAL_OPTIMS ))

REAL_RUN_ROOT="experiments/${MODEL}/real_data_analysis/runs"

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_REAL_OPTIMS: $NUM_REAL_OPTIMS  TOTAL_TASKS: $TOTAL_TASKS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    echo "Submitting array 0..${NUM_ARRAY}"
    sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
    exit 0
fi

BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

TARGETS=()
for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
    ENGINE_I=$(( IDX / NUM_REAL_OPTIMS ))
    OPT=$(( IDX % NUM_REAL_OPTIMS ))
    ENGINE="${ENGINES[$ENGINE_I]}"

    TARGET="${REAL_RUN_ROOT}/run_${OPT}/inferences/${ENGINE}/best_fit.pkl"
    if [[ -s "$ROOT/$TARGET" ]]; then
        echo "SKIP: ENGINE=$ENGINE OPT=$OPT (already exists: $TARGET)"
        continue
    fi
    echo "QUEUE: ENGINE=$ENGINE OPT=$OPT -> $TARGET"
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
        --allowed-rules infer_engine_real \
        -j "$SLURM_CPUS_PER_TASK" \
        "${TARGETS[@]}"
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

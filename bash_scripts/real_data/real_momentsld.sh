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

# Stage D2: one LHS/jitter-seeded MomentsLD restart per opt -- or, in
# pooling_mode="individual", per (arm,opt) -- same array-per-restart pattern
# as MomentsLD.sh. Requires real_momentsld_prep.sh (means.varcovs.pkl, per-
# arm in individual mode) and real_aggregate_sfs.sh (moments best_fit, used
# only as an optimization seed, per-arm in individual mode) to have already
# finished.
#
# Rule run (pooled):     infer_momentsld_real
# Rule run (individual): infer_momentsld_real_by_arm (no cross-arm pooling
#                         anywhere in this stage)

set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-10}"   # tasks per array element

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
export EXP_CFG="$CFG"

load_real_data_config "$CFG"
NUM_REAL_OPTIMS=$(jq -r '.num_optimizations // 3' "$CFG")

if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    TOTAL_TASKS=$NUM_REAL_OPTIMS
else
    TOTAL_TASKS=$(( NUM_REAL_OPTIMS * ${#REAL_ARMS[@]} ))
fi

echo "CFG: $CFG"
echo "MODEL: $MODEL  REAL_POOLING_MODE: $REAL_POOLING_MODE  NUM_REAL_OPTIMS: $NUM_REAL_OPTIMS  TOTAL_TASKS: $TOTAL_TASKS"
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
if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    for OPT in $(seq "$BATCH_START" "$BATCH_END"); do
        TARGET="${REAL_RUN_ROOT}/run_${OPT}/inferences/${REAL_LD_ENGINE}/best_fit.pkl"
        if [[ -s "$ROOT/$TARGET" ]]; then
            echo "SKIP: OPT=$OPT (already exists: $TARGET)"
            continue
        fi
        echo "QUEUE: OPT=$OPT -> $TARGET"
        TARGETS+=("$TARGET")
    done
    ALLOWED_RULES=(infer_momentsld_real)
else
    for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
        ARM_I=$(( IDX / NUM_REAL_OPTIMS ))
        OPT=$(( IDX % NUM_REAL_OPTIMS ))
        ARM="${REAL_ARMS[$ARM_I]}"

        RUN_ROOT_ARM="$(real_data_chrom_path "$REAL_RUN_ROOT_CHROM_TMPL" "$ARM")"
        TARGET="${RUN_ROOT_ARM}/run_${OPT}/inferences/MomentsLD/best_fit.pkl"
        if [[ -s "$ROOT/$TARGET" ]]; then
            echo "SKIP: ARM=$ARM OPT=$OPT (already exists: $TARGET)"
            continue
        fi
        echo "QUEUE: ARM=$ARM OPT=$OPT -> $TARGET"
        TARGETS+=("$TARGET")
    done
    ALLOWED_RULES=(infer_momentsld_real_by_arm)
fi

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
        --allowed-rules "${ALLOWED_RULES[@]}" \
        -j "$SLURM_CPUS_PER_TASK" \
        "${TARGETS[@]}"
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

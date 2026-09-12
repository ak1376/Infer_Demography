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

# Stage B1: one moments/dadi NLopt restart per (engine,opt) -- or, in
# pooling_mode="individual", per (arm,engine,opt) -- same array-per-restart
# pattern as moments.sh/dadi.sh, every restart gets its own array slot
# instead of being funneled through a single job's core count. Requires
# real_data_prep.sh to have already produced each arm's SFS (and, in pooled
# mode, the combined autosomal SFS).
#
# Rule run (pooled):     infer_engine_real
# Rule run (individual): infer_engine_real_chrom
# (the aggregate step is a separate job -- see real_aggregate_sfs.sh --
# exactly like aggregate_moments_dadi.sh is separate from moments.sh/dadi.sh)
#
# pooled index space: engine*NUM_REAL_OPTIMS + opt, engine in {moments, dadi}.
# individual index space additionally multiplies in arm: arm and engine are
# interleaved across the same array so both stay balanced across array tasks.

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
ENGINES=(moments dadi)

if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    TOTAL_TASKS=$(( ${#ENGINES[@]} * NUM_REAL_OPTIMS ))
else
    TOTAL_TASKS=$(( ${#ENGINES[@]} * NUM_REAL_OPTIMS * ${#REAL_ARMS[@]} ))
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
    ALLOWED_RULES=(infer_engine_real)
else
    N_ARMS=${#REAL_ARMS[@]}
    PER_ARM=$(( ${#ENGINES[@]} * NUM_REAL_OPTIMS ))
    for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
        ARM_I=$(( IDX / PER_ARM ))
        REM=$(( IDX % PER_ARM ))
        ENGINE_I=$(( REM / NUM_REAL_OPTIMS ))
        OPT=$(( REM % NUM_REAL_OPTIMS ))
        ARM="${REAL_ARMS[$ARM_I]}"
        ENGINE="${ENGINES[$ENGINE_I]}"

        RUN_ROOT_ARM="$(real_data_chrom_path "$REAL_RUN_ROOT_CHROM_TMPL" "$ARM")"
        TARGET="${RUN_ROOT_ARM}/run_${OPT}/inferences/${ENGINE}/best_fit.pkl"
        if [[ -s "$ROOT/$TARGET" ]]; then
            echo "SKIP: ARM=$ARM ENGINE=$ENGINE OPT=$OPT (already exists: $TARGET)"
            continue
        fi
        echo "QUEUE: ARM=$ARM ENGINE=$ENGINE OPT=$OPT -> $TARGET"
        TARGETS+=("$TARGET")
    done
    ALLOWED_RULES=(infer_engine_real_chrom)
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

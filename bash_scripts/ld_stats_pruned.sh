#!/bin/bash
#SBATCH --job-name=ld_pruned
#SBATCH --array=0-9999
#SBATCH --output=logs/ld_pruned_%A_%a.out
#SBATCH --error=logs/ld_pruned_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=kerngpu,gpulong,gpu
#SBATCH --gres=gpu:1
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -eo pipefail

module --ignore_cache purge || true
module --ignore_cache load cuda/12.4.1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate snakemake-env

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR="/tmp/${USER}/cupy_cache_${SLURM_JOB_ID}"
mkdir -p "$CUPY_CACHE_DIR"

# ---------------------------------------------------------------------------
BATCH_SIZE=50
THIN_TAG="thin15"
KEEP_FRAC="0.15"
R_BINS="0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"
# ---------------------------------------------------------------------------

CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_migration_growth.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"

NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r    '.demographic_model' "$CFG")
NUM_WINDOWS=100
TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $END -ge $TOTAL_TASKS ]] && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $START .. $END"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import cupy; from cupy_backends.cuda.libs import nvrtc; print('NVRTC', nvrtc.getVersion())" || true

cd "$ROOT"

for IDX in $(seq "$START" "$END"); do
    SID=$(( IDX / NUM_WINDOWS ))
    WIN=$(( IDX % NUM_WINDOWS ))

    WINDOWS_DIR="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/windows"
    PRUNING_DIR="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/pruning"
    SIM_DIR="$PRUNING_DIR/${THIN_TAG}"
    TARGET="$SIM_DIR/LD_stats/LD_stats_window_${WIN}.pkl"
    PRUNED_VCF="$SIM_DIR/windows/window_${WIN}.vcf.gz"

    UNPRUNED_PKL="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/LD_stats/LD_stats_window_${WIN}.pkl"

    if [[ -f "$TARGET" ]]; then
        echo "SKIP: SID=$SID WIN=$WIN (pruned LD stats already exist)"
        continue
    fi

    if [[ -f "$UNPRUNED_PKL" ]]; then
        echo "SKIP: SID=$SID WIN=$WIN (unpruned LD stats already exist)"
        continue
    fi

    # Prune this window if not already done
    if [[ ! -f "$PRUNED_VCF" ]]; then
        echo "PRUNE: SID=$SID WIN=$WIN"
        PYTHONPATH="$ROOT" python "$ROOT/test_scripts/prune_vcf.py" \
            --vcf            "$WINDOWS_DIR/window_${WIN}.vcf.gz" \
            --out-dir        "$PRUNING_DIR"                      \
            --keep-fractions "$KEEP_FRAC"                        \
            --workers        1
    fi

    echo "RUN: SID=$SID WIN=$WIN → $TARGET"
    PYTHONPATH="$ROOT" python "$ROOT/snakemake_scripts/compute_ld_window.py" \
        --sim-dir      "$SIM_DIR" \
        --window-index "$WIN"     \
        --config-file  "$CFG"     \
        --r-bins       "$R_BINS"
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

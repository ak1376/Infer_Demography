#!/bin/bash
#SBATCH --job-name=real_ld
#SBATCH --output=logs/real_ld_%A_%a.out
#SBATCH --error=logs/real_ld_%A_%a.err
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
# NOTE: partition/gres above are only the defaults for a bare `sbatch` call.
# The self-resubmission below overrides both based on the active config's
# use_gpu_ld, mirroring LD_stats_windows.sh -- flip that one key to switch
# between GPU and CPU-only nodes, no need to edit this file.

# Stage C: split the (single-chromosome, Chr2L) diploid-recoded VCF into
# windows and compute per-window LD stats for the real-data MomentsLD
# inference (real_data_prep.sh must have already produced Chr2L's
# polarized.diploidGT.vcf.gz).
#
# Rules run: split_real_vcf_window, compute_ld_real
#
# --resources gpu=1 caps concurrent GPU-resident LD jobs at one per node,
# same fix as LD_stats_windows.sh -- without it, up to
# $SLURM_CPUS_PER_TASK compute_ld_real jobs (each threads:1) can share the
# single GPU this job was allocated and intermittently OOM.

set -eo pipefail

BATCH_SIZE="${BATCH_SIZE:-20}"

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model'    "$CFG")
NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG")
USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG")

REAL_LD_ROOT="experiments/${MODEL}/real_data_analysis/inferences/MomentsLD"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_ARRAY=$(( (NUM_WINDOWS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    if [[ "$USE_GPU_LD" == "true" ]]; then
        echo "Submitting array 0..${NUM_ARRAY} (use_gpu_ld=true -> GPU partition)"
        sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
    else
        echo "Submitting array 0..${NUM_ARRAY} (use_gpu_ld=false -> CPU-only, excluding kerngpu)"
        sbatch --array=0-"$NUM_ARRAY" --partition=kern,preempt --gres=gpu:0 "$0" "$@"
    fi
    exit 0
fi

module --ignore_cache purge || true
if [[ "$USE_GPU_LD" == "true" ]]; then
    module --ignore_cache load cuda/12.4.1
fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate snakemake-env

if [[ "$USE_GPU_LD" == "true" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export CUPY_CACHE_DIR="/tmp/${USER}/cupy_cache_${SLURM_JOB_ID}"
    mkdir -p "$CUPY_CACHE_DIR"
fi

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $END -ge $NUM_WINDOWS ]] && END=$(( NUM_WINDOWS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → windows $START .. $END  MODEL=$MODEL NUM_WINDOWS=$NUM_WINDOWS"

TARGETS=()
for I in $(seq "$START" "$END"); do
    PKL="${REAL_LD_ROOT}/LD_stats/LD_stats_window_${I}.pkl"
    if [[ -f "$ROOT/$PKL" ]]; then
        echo "SKIP: window $I exists"
        continue
    fi
    echo "QUEUE: window $I"
    TARGETS+=("$PKL")
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Nothing to build for this array task (all skipped)."
else
    echo "Building ${#TARGETS[@]} targets in one Snakemake call (-j $SLURM_CPUS_PER_TASK)..."
    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT" \
              --nolock \
              --keep-going \
              --latency-wait 120 \
              --rerun-incomplete \
              --rerun-triggers mtime \
              --resources gpu=1 \
              --allowed-rules split_real_vcf_window compute_ld_real \
              -j "$SLURM_CPUS_PER_TASK" \
              "${TARGETS[@]}" || true
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

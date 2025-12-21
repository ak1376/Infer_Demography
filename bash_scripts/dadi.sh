#!/bin/bash
#SBATCH --job-name=dadi_infer
#SBATCH --array=0-9999
#SBATCH --output=logs/dadi_%A_%a.out
#SBATCH --error=logs/dadi_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=kerngpu,gpu,gpulong
#SBATCH --account=kernlab
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

BATCH_SIZE=50

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

CFG="${CFG_PATH:-/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_drosophila_three_epoch.json}"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")
TOTAL_TASKS=$(( NUM_DRAWS * NUM_OPTIMS ))

echo "CFG: $CFG"
echo "MODEL: $MODEL  NUM_DRAWS: $NUM_DRAWS  NUM_OPTIMS: $NUM_OPTIMS  TOTAL_TASKS: $TOTAL_TASKS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
  echo "Submitting array 0..${NUM_ARRAY}"
  sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
  exit 0
fi

# --- GPU sanity ---
nvidia-smi -L || true
nvidia-smi || true

# --- Load CUDA toolkit ONCE (nvcc must exist) ---
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda 2>/dev/null || true
module list 2>/dev/null || true
echo "nvcc: $(command -v nvcc || echo NOT_FOUND)"
nvcc --version || true

# --- One cache dir per ARRAY TASK (shared across the 50 sid/opt fits in this task) ---
export PYCUDA_CACHE_DIR="/tmp/pycuda_cache_${USER}_dadi_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$PYCUDA_CACHE_DIR"
export CUDAHOSTCXX=/usr/bin/g++
export PYCUDA_DEFAULT_NVCC_OPTIONS="-O3"

echo "PYCUDA_CACHE_DIR=$PYCUDA_CACHE_DIR"
echo "CUDAHOSTCXX=$CUDAHOSTCXX"

# --- Warmup ONCE: triggers PyCUDA nvcc compilation into $PYCUDA_CACHE_DIR ---
python - <<'PY'
import os
print("[warmup] PYCUDA_CACHE_DIR =", os.environ.get("PYCUDA_CACHE_DIR"))
import dadi.cuda
print("[warmup] dadi.cuda import: OK (kernels compiled/cached if needed)")
PY

# --- compute slice for this array id ---
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))
echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
  SID=$(( IDX / NUM_OPTIMS ))
  OPT=$(( IDX % NUM_OPTIMS ))

  TARGET="experiments/${MODEL}/runs/run_${SID}_${OPT}/inferences/dadi/fit_params.pkl"
  echo "dadi optimisation: SID=$SID  OPT=$OPT  →  $TARGET"

  snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --nolock \
    -j "$SLURM_CPUS_PER_TASK" \
    "$TARGET" || { echo "Snakemake failed for SID=$SID OPT=$OPT"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

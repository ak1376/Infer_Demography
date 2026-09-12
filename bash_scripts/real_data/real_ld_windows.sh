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

# Stage C: split each configured arm's diploid-recoded VCF into windows and
# compute per-window LD stats for the real-data MomentsLD inference
# (real_data_prep.sh must have already produced each arm's
# polarized.diploidGT.vcf.gz).
#
# Rules run: real_vcf_windows (checkpoint), compute_ld_real
#
# REAL_LD_ROOT is NOT model-scoped (must match the Snakefile's REAL_LD_ROOT =
# f"{DROSO_DIR}/{REAL_LD_ENGINE}", REAL_LD_ENGINE = f"MomentsLD_{REAL_TAG}",
# REAL_TAG = "_".join(REAL_ARMS) + ("_genmap" if use_genmap)): the window
# split + per-window LD stats are pure functions of the VCF/window-size/r-bins
# for this arm-set/genmap choice, not of which demographic model you're
# fitting, so this whole (GPU-bound) stage is computed once per arm-set and
# reused across every model instead of being redone per experiment.
#
# Per-arm window counts aren't knowable from config alone -- num_windows may
# be "auto" (derived by the real_vcf_windows rule from the VCF's actual span
# at runtime), and there may be more than one arm. So this script builds the
# real_vcf_windows checkpoint directly for every configured arm BEFORE sizing
# the SLURM array (this runs synchronously in whatever shell invokes the
# script, same as the old NUM_WINDOWS-from-config lookup did, but is now a
# real computation instead of a guess -- for a very large VCF this may take a
# little while), then re-derives the identical (arm, window-index) ordering
# inside every array task (by globbing, not recomputing) to pick its slice.
#
# --resources gpu=1 caps concurrent GPU-resident LD jobs at one per node,
# same fix as LD_stats_windows.sh -- without it, up to
# $SLURM_CPUS_PER_TASK compute_ld_real jobs (each threads:1) can share the
# single GPU this job was allocated and intermittently OOM.

set -eo pipefail

BATCH_SIZE="${BATCH_SIZE:-20}"

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG")

load_real_data_config "$CFG"

# Deterministically list every (arm, window-index) pair actually present on
# disk, arms in config order and indices numerically sorted within each arm.
# Called both by the dispatcher (to size the array) and by every array task
# (to pick its slice) so both agree without passing state between them --
# safe as long as the windows/ dirs don't change in between, which holds
# since the dispatcher fully materializes them before ever calling sbatch.
list_all_targets() {
    for arm in "${REAL_ARMS[@]}"; do
        shopt -s nullglob
        local files=("$ROOT/${REAL_LD_ROOT}/${arm}/windows/"window_*.vcf.gz)
        shopt -u nullglob
        local idxs=()
        local f b
        for f in "${files[@]}"; do
            b=$(basename "$f" .vcf.gz)
            idxs+=("${b#window_}")
        done
        if [[ ${#idxs[@]} -gt 0 ]]; then
            printf '%s\n' "${idxs[@]}" | sort -n | while read -r i; do
                echo "${arm} ${i}"
            done
        fi
    done
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate snakemake-env

    for arm in "${REAL_ARMS[@]}"; do
        echo "Ensuring window split for arm=$arm..."
        snakemake --snakefile "$SNAKEFILE" --directory "$ROOT" --nolock \
            --allowed-rules real_vcf_windows \
            -j 1 \
            "${REAL_LD_ROOT}/${arm}/windows"
    done

    mapfile -t ALL_TARGETS < <(list_all_targets)
    TOTAL=${#ALL_TARGETS[@]}
    if [[ "$TOTAL" -eq 0 ]]; then
        echo "ERROR: no windows found across arms (${REAL_ARMS[*]}) under ${REAL_LD_ROOT}" >&2
        exit 1
    fi

    NUM_ARRAY=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    if [[ "$USE_GPU_LD" == "true" ]]; then
        echo "Submitting array 0..${NUM_ARRAY} (use_gpu_ld=true -> GPU partition), $TOTAL windows across ${#REAL_ARMS[@]} arm(s)"
        sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
    else
        echo "Submitting array 0..${NUM_ARRAY} (use_gpu_ld=false -> CPU-only, excluding kerngpu), $TOTAL windows across ${#REAL_ARMS[@]} arm(s)"
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

mapfile -t ALL_TARGETS < <(list_all_targets)
TOTAL=${#ALL_TARGETS[@]}

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $END -ge $TOTAL ]] && END=$(( TOTAL - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → tasks $START .. $END  TOTAL=$TOTAL"

TARGETS=()
for idx in $(seq "$START" "$END"); do
    read -r arm i <<< "${ALL_TARGETS[$idx]}"
    PKL="${REAL_LD_ROOT}/${arm}/LD_stats/LD_stats_window_${i}.pkl"
    if [[ -f "$ROOT/$PKL" ]]; then
        echo "SKIP: $arm window $i exists"
        continue
    fi
    echo "QUEUE: $arm window $i"
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
              --allowed-rules real_vcf_windows compute_ld_real \
              -j "$SLURM_CPUS_PER_TASK" \
              "${TARGETS[@]}" || true
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

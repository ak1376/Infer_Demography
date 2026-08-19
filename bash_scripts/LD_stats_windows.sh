#!/bin/bash
#SBATCH --job-name=ld_stats
#SBATCH --output=logs/ld_%A_%a.out
#SBATCH --error=logs/ld_%A_%a.err
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
# use_gpu_ld — flip that one key to switch between GPU and CPU-only nodes,
# no need to edit this file. This self-resubmission only fires when
# SLURM_ARRAY_TASK_ID is unset (i.e. this script was sbatch'd bare, without
# --array). master_script.sh submits with --array already set (see
# lib_array_size.sh), so it computes its own use_gpu_ld-driven
# --partition/--gres and passes them on that initial sbatch call instead.

set -eo pipefail

# -------- batching knobs ---------------------------------------------------
# Overridable so master_script.sh can export the exact value it used to size
# the --array range it submits this script with.
BATCH_SIZE="${BATCH_SIZE:-50}"
# ----------------------------------------------------------------------------

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

# -------- everything below is read from the config; nothing hardcoded -------
NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
MODEL=$(jq -r    '.demographic_model'   "$CFG")
NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG")
USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG")

TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

# First launch (no array id yet): compute the correct --array range from
# num_draws/num_windows and resubmit as an array, instead of a fixed huge
# range regardless of what the active config actually needs. Also picks the
# partition/gres here, driven by use_gpu_ld, since #SBATCH directives in the
# file itself can't be conditional — sbatch flags on this resubmission
# override them.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    if [[ "$USE_GPU_LD" == "true" ]]; then
        echo "Submitting array 0..${NUM_ARRAY} (use_gpu_ld=true -> GPU partition)"
        sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
    else
        echo "Submitting array 0..${NUM_ARRAY} (use_gpu_ld=false -> CPU-only, excluding kerngpu)"
        sbatch --array=0-"$NUM_ARRAY" --partition=kern,preempt --gres=gpu:0 "$0" "$@"
    fi
    exit 0
fi

# --- Make modules available (Talapas-style) ---
module --ignore_cache purge || true

if [[ "$USE_GPU_LD" == "true" ]]; then
    # pg_gpu requires CUDA 12 (cupy>=13, cuda-version=12.*)
    module --ignore_cache load cuda/12.4.1
fi

# --- Conda env ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate snakemake-env

if [[ "$USE_GPU_LD" == "true" ]]; then
    # Ensure conda libs (incl nvrtc) are visible at runtime
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

    # Optional but helpful (avoid hammering $HOME with JIT cache)
    export CUPY_CACHE_DIR="/tmp/${USER}/cupy_cache_${SLURM_JOB_ID}"
    mkdir -p "$CUPY_CACHE_DIR"
fi

# Pruning keep-fractions from config (empty => pruning disabled). We only need
# the fraction values here; r_bins / keep_frac live in the Snakefile rules that
# the Snakemake targets below invoke, so nothing about LD binning is hardcoded.
mapfile -t PRUNE_FRACS < <(jq -r '(.prune_keep_fractions // [])[]' "$CFG")
PRUNING_ENABLED=$(( ${#PRUNE_FRACS[@]} > 0 ))

# thin<NN> tag for a keep-fraction, matching src.prune_vcf._frac_tag: round(f*100), 2-digit
frac_tag() { printf "thin%02d" "$(awk "BEGIN{printf \"%.0f\", $1 * 100}")"; }

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $END -ge $TOTAL_TASKS ]] && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $START .. $END"
echo "MODEL=$MODEL NUM_DRAWS=$NUM_DRAWS NUM_WINDOWS=$NUM_WINDOWS"
if (( PRUNING_ENABLED )); then
    echo "Pruning ENABLED (keep-fractions: ${PRUNE_FRACS[*]}) — computing PRUNED LD stats only (unpruned is skipped)."
else
    echo "Pruning DISABLED — computing UNPRUNED LD stats only."
fi
if [[ "$USE_GPU_LD" == "true" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    python -c "import cupy; from cupy_backends.cuda.libs import nvrtc; print('NVRTC', nvrtc.getVersion())" || true
fi

# Collect every target this array task needs, then build them all in a
# single Snakemake invocation (see below) instead of spawning one Snakemake
# process per target. Same rationale as build_windows.sh: cuts redundant
# DAG-parse/startup overhead and the number of concurrent --nolock processes
# hitting shared .snakemake/ metadata. Note ld_window/ld_window_pruned are
# threads:4 (== $SLURM_CPUS_PER_TASK), so unlike build_windows.sh this won't
# let multiple LD-stat jobs run concurrently within one array task -- the
# win here is purely the removed per-target startup overhead.
TARGETS=()
PRUNED_VCF_CLEANUP=()   # leftover-pruned-VCF safety-net cleanup, done after the batch

for IDX in $(seq "$START" "$END"); do
    SID=$(( IDX / NUM_WINDOWS ))
    WIN=$(( IDX % NUM_WINDOWS ))

    MLD_REL="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD"
    MLD_ABS="${ROOT}/${MLD_REL}"
    RAW_VCF="${MLD_ABS}/windows/window_${WIN}.vcf.gz"

    if (( PRUNING_ENABLED )); then
        # ---- PRUNED-ONLY: config specifies pruning, so unpruned is skipped ---
        for frac in "${PRUNE_FRACS[@]}"; do
            tag=$(frac_tag "$frac")
            PRUNED_PKL="${MLD_ABS}/pruning/${tag}/LD_stats/LD_stats_window_${WIN}.pkl"
            if [[ -f "$PRUNED_PKL" ]]; then
                echo "SKIP: pruned ($tag) exists  SID=$SID WIN=$WIN"
                continue
            fi
            echo "QUEUE: pruned ($tag)  SID=$SID WIN=$WIN"
            # prune_window (temp pruned VCF) → ld_window_pruned; both inherit
            # r_bins from the Snakefile.
            TARGETS+=("${MLD_REL}/pruning/${tag}/LD_stats/LD_stats_window_${WIN}.pkl")
            # Pruned VCF is temp() and normally cleaned inside the Snakemake run;
            # queue it for an explicit leftover-cleanup pass in case
            # ld_window_pruned fails mid-way for this specific target.
            PRUNED_VCF_CLEANUP+=("${MLD_ABS}/pruning/${tag}/windows/window_${WIN}.vcf.gz")
        done
        # TEMPORARILY DISABLED (comparing prune_keep_fractions across separate
        # reruns -- keep the base window VCF so it doesn't need to be
        # resimulated/rechunked for the next fraction). Re-enable by
        # uncommenting once done comparing fractions.
        # rm -f "$RAW_VCF"
    else
        # ---- UNPRUNED-ONLY: no pruning configured ---------------------------
        UNPRUNED_PKL="${MLD_ABS}/LD_stats/LD_stats_window_${WIN}.pkl"
        if [[ -f "$UNPRUNED_PKL" ]]; then
            echo "SKIP: unpruned exists  SID=$SID WIN=$WIN"
            # TEMPORARILY DISABLED -- see comment above in the pruning branch.
            # rm -f "$RAW_VCF"
            continue
        fi
        echo "QUEUE: unpruned  SID=$SID WIN=$WIN"
        TARGETS+=("${MLD_REL}/LD_stats/LD_stats_window_${WIN}.pkl")
        # TEMPORARILY DISABLED -- see comment above in the pruning branch.
        # [[ -f "$UNPRUNED_PKL" ]] && rm -f "$RAW_VCF"
    fi
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Nothing to build for this array task (all skipped)."
else
    echo "Building ${#TARGETS[@]} targets in one Snakemake call (-j $SLURM_CPUS_PER_TASK)..."
    # --keep-going: one bad target shouldn't abort the whole batch and strand
    # the rest of this array task's otherwise-good work.
    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT"      \
              --nolock                  \
              --keep-going              \
              --latency-wait 120        \
              --rerun-incomplete        \
              --rerun-triggers mtime    \
              -j "$SLURM_CPUS_PER_TASK" \
              "${TARGETS[@]}" || true

    for leftover in "${PRUNED_VCF_CLEANUP[@]}"; do
        rm -f "$leftover"
    done
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

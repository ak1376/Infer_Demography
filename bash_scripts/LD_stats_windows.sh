#!/bin/bash
#SBATCH --job-name=ld_stats
#SBATCH --array=0-9999
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

set -eo pipefail
# --gres=gpu:1
# --- Make modules available (Talapas-style) ---
module --ignore_cache purge || true

# pg_gpu requires CUDA 12 (cupy>=13, cuda-version=12.*)
module --ignore_cache load cuda/12.4.1

# --- Conda env ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate snakemake-env

# Ensure conda libs (incl nvrtc) are visible at runtime
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Optional but helpful (avoid hammering $HOME with JIT cache)
export CUPY_CACHE_DIR="/tmp/${USER}/cupy_cache_${SLURM_JOB_ID}"
mkdir -p "$CUPY_CACHE_DIR"

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=50
# ----------------------------------------------------------------------------

CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_migration_growth.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

# -------- everything below is read from the config; nothing hardcoded -------
NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
MODEL=$(jq -r    '.demographic_model'   "$CFG")
NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG")

# Pruning keep-fractions from config (empty => pruning disabled). We only need
# the fraction values here; r_bins / keep_frac live in the Snakefile rules that
# the Snakemake targets below invoke, so nothing about LD binning is hardcoded.
mapfile -t PRUNE_FRACS < <(jq -r '(.prune_keep_fractions // [])[]' "$CFG")
PRUNING_ENABLED=$(( ${#PRUNE_FRACS[@]} > 0 ))

# thin<NN> tag for a keep-fraction, matching src.prune_vcf._frac_tag: round(f*100), 2-digit
frac_tag() { printf "thin%02d" "$(awk "BEGIN{printf \"%.0f\", $1 * 100}")"; }

TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import cupy; from cupy_backends.cuda.libs import nvrtc; print('NVRTC', nvrtc.getVersion())" || true

# Run one Snakemake target (LD-stat pkl). Returns snakemake's exit status.
run_target() {
    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT"      \
              --nolock                  \
              --latency-wait 120        \
              --rerun-incomplete        \
              --rerun-triggers mtime    \
              -j "$SLURM_CPUS_PER_TASK" \
              "$1"
}

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
            echo "RUN: pruned ($tag)  SID=$SID WIN=$WIN"
            # prune_window (temp pruned VCF) → ld_window_pruned; both inherit
            # r_bins from the Snakefile. || true: don't kill the batch on one failure.
            run_target "${MLD_REL}/pruning/${tag}/LD_stats/LD_stats_window_${WIN}.pkl" || true
            # Pruned VCF is temp() and cleaned inside the Snakemake run; remove any
            # leftover explicitly in case ld_window_pruned failed mid-way.
            rm -f "${MLD_ABS}/pruning/${tag}/windows/window_${WIN}.vcf.gz"
        done
        # Raw VCF only fed the pruning step and is no longer needed.
        rm -f "$RAW_VCF"
    else
        # ---- UNPRUNED-ONLY: no pruning configured ---------------------------
        UNPRUNED_PKL="${MLD_ABS}/LD_stats/LD_stats_window_${WIN}.pkl"
        if [[ -f "$UNPRUNED_PKL" ]]; then
            echo "SKIP: unpruned exists  SID=$SID WIN=$WIN"
            rm -f "$RAW_VCF"
            continue
        fi
        echo "RUN: unpruned  SID=$SID WIN=$WIN"
        run_target "${MLD_REL}/LD_stats/LD_stats_window_${WIN}.pkl" || true
        [[ -f "$UNPRUNED_PKL" ]] && rm -f "$RAW_VCF"
    fi
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

#!/usr/bin/env bash
#SBATCH --job-name=fim
#SBATCH --output=logs/fim_%x_%A_%a.out
#SBATCH --error=logs/fim_%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# Usage:
#   ENGINE=dadi   sbatch jobs/compute_fim_array.sh
#   ENGINE=moments sbatch jobs/compute_fim_array.sh
#   ENGINE=both    sbatch jobs/compute_fim_array.sh
#
# If you run the script directly (without --array), it will auto-resubmit
# itself with the correct array size [0 .. num_draws-1] from the config.

set -euo pipefail

# ---------------- user/config paths ----------------
ROOT="/projects/kernlab/akapoor/Infer_Demography"
CFG="$ROOT/config_files/experiment_config_drosophila_three_epoch.json"
SNAKEFILE="$ROOT/Snakefile"

# Which engine(s) to compute? dadi | moments | both
ENGINE="${ENGINE:-dadi}"

# Optional: activate your env here
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate snakemake-env

# --------------- derive metadata -------------------
NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")

# Auto-resubmit as array if not already
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0"
  echo "Re-submitted as array 0-$(( NUM_DRAWS - 1 ))."
  exit 0
fi

sid="$SLURM_ARRAY_TASK_ID"
echo "Compute FIM for sim_$sid (model=$MODEL, engine=$ENGINE)"

# Build targets per engine
build_targets() {
  local sid="$1"
  local eng="$2"
  echo "experiments/${MODEL}/inferences/sim_${sid}/fim/${eng}.fim.npy"
}

declare -a TARGETS=()
case "$ENGINE" in
  dadi)    TARGETS+=("$(build_targets "$sid" "dadi")") ;;
  moments) TARGETS+=("$(build_targets "$sid" "moments")") ;;
  both)    TARGETS+=("$(build_targets "$sid" "dadi")" "$(build_targets "$sid" "moments")") ;;
  *) echo "ERROR: ENGINE must be dadi|moments|both (got '$ENGINE')" >&2; exit 2 ;;
esac

# --------------- run snakemake ---------------------
snakemake -j "${SLURM_CPUS_PER_TASK:-2}" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  "${TARGETS[@]}"

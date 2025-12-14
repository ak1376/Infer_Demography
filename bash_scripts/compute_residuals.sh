#!/usr/bin/env bash
#SBATCH --job-name=sfsres
#SBATCH --output=logs/sfs_residuals_%x_%A_%a.out
#SBATCH --error=logs/sfs_residuals_%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# Usage examples:
#   ENGINE=moments sbatch jobs/compute_sfs_residuals_array.sh
#   ENGINE=dadi    sbatch jobs/compute_sfs_residuals_array.sh
#   ENGINE=both    sbatch jobs/compute_sfs_residuals_array.sh
#
# Optional overrides:
#   ROOT=/path/to/Infer_Demography
#   CFG=/path/to/experiment_config.json
#   SNAKEFILE=/path/to/Snakefile
#   SNAKEMAKE_OPTS="--keep-going -p"

set -euo pipefail
mkdir -p logs

# ---------------- user/config paths ----------------
ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
CFG="${CFG:-$ROOT/config_files/experiment_config_bottleneck.json}"
SNAKEFILE="${SNAKEFILE:-$ROOT/Snakefile}"

ENGINE="${ENGINE:-moments}"          # moments | dadi | both
SNAKEMAKE_OPTS="${SNAKEMAKE_OPTS:-}" # extra flags for snakemake

# Optional: activate your env
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate snakemake-env

# --------------- derive metadata -------------------
# Ensure residuals are enabled in the config
USE_RESID="$(jq -r '.use_residuals // false' "$CFG")"
if [[ "$USE_RESID" != "true" ]]; then
  echo "[INFO] use_residuals=false in $CFG — nothing to do."
  exit 0
fi

NUM_DRAWS="$(jq -r '.num_draws' "$CFG")"
MODEL="$(jq -r '.demographic_model' "$CFG")"

# Auto-resubmit as array if not already
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0"
  echo "Re-submitted as array 0-$(( NUM_DRAWS - 1 ))."
  exit 0
fi

sid="$SLURM_ARRAY_TASK_ID"
echo "Compute SFS residuals for sim_$sid (model=$MODEL, engine=$ENGINE)"

# -------- path helpers (must match Snakefile rule sfs_residuals) ----------
fit_path() {
  local sid="$1" eng="$2"
  echo "experiments/${MODEL}/inferences/sim_${sid}/${eng}/fit_params.pkl"
}
resid_target_flat() {
  local sid="$1" eng="$2"
  # NOTE: residuals live under **inferences**, not simulations
  echo "experiments/${MODEL}/inferences/sim_${sid}/sfs_residuals/${eng}/residuals_flat.npy"
}

# Build target list but only for engines whose fit exists
declare -a TARGETS=()

maybe_add() {
  local eng="$1"
  local fit_rel; fit_rel="$(fit_path "$sid" "$eng")"
  if [[ -f "$ROOT/$fit_rel" ]]; then
    TARGETS+=("$(resid_target_flat "$sid" "$eng")")
  else
    echo "SKIP sim_$sid engine=$eng (missing fit: $fit_rel)"
  fi
}

case "$ENGINE" in
  moments) maybe_add "moments" ;;
  dadi)    maybe_add "dadi" ;;
  both)    maybe_add "moments"; maybe_add "dadi" ;;
  *) echo "ERROR: ENGINE must be moments|dadi|both (got '$ENGINE')" >&2; exit 2 ;;
esac

# If nothing to do for this sid, exit cleanly
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No SFS residual targets for sim_$sid — nothing to run."
  exit 0
fi

# --------------- run snakemake ---------------------
snakemake -j "${SLURM_CPUS_PER_TASK:-1}" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  ${SNAKEMAKE_OPTS} \
  "${TARGETS[@]}"

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

# Usage examples:
#   ENGINE=moments sbatch jobs/compute_fim_array.sh
#   ENGINE=dadi    sbatch jobs/compute_fim_array.sh
#   ENGINE=both    sbatch jobs/compute_fim_array.sh
#
# Pass extra snakemake flags via SNAKEMAKE_OPTS, e.g.:
#   ENGINE=moments SNAKEMAKE_OPTS="--keep-going -p" sbatch jobs/compute_fim_array.sh

set -euo pipefail

mkdir -p logs

# ---------------- user/config paths ----------------
ROOT="/projects/kernlab/akapoor/Infer_Demography"
CFG="$ROOT/config_files/experiment_config_drosophila_three_epoch.json"
SNAKEFILE="$ROOT/Snakefile"

# Which engine(s) to compute? moments | dadi | both
ENGINE="${ENGINE:-moments}"          # default now matches your Snakefile’s FIM_ENGINES
SNAKEMAKE_OPTS="${SNAKEMAKE_OPTS:-}" # optional extra flags

# Optional: activate your env
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

# Helper: path to the *fit* that compute_fim needs (used for existence check)
fit_path() {
  local sid="$1" eng="$2"
  echo "experiments/${MODEL}/inferences/sim_${sid}/${eng}/fit_params.pkl"
}

# Helper: FIM target produced by compute_fim
fim_target() {
  local sid="$1" eng="$2"
  echo "experiments/${MODEL}/inferences/sim_${sid}/fim/${eng}.fim.npy"
}

# Build target list, but **only include engines whose fit exists** to avoid triggering infer rules
declare -a TARGETS=()

maybe_add() {
  local eng="$1"
  local fit="$(fit_path "$sid" "$eng")"
  if [[ -f "$ROOT/$fit" ]]; then
    TARGETS+=("$(fim_target "$sid" "$eng")")
  else
    echo "SKIP sim_$sid engine=$eng (missing fit: $fit)"
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
  echo "No FIM targets for sim_$sid — nothing to run."
  exit 0
fi

# --------------- run snakemake ---------------------
snakemake -j "${SLURM_CPUS_PER_TASK:-2}" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  ${SNAKEMAKE_OPTS} \
  "${TARGETS[@]}"

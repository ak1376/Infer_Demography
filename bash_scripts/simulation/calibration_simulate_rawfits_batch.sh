#!/bin/bash
#SBATCH --job-name=calib_sim_rawfits
#SBATCH --output=logs/calib_sim_rawfits_%j.out
#SBATCH --error=logs/calib_sim_rawfits_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# TEMPORARY diagnostic script (companion to the TEMPORARY model_key
# wildcard_constraints widening in the Snakefile): run calibration_simulate
# for the raw dadi/moments/momentsLD real-data fits (not a trained ML model),
# for calibration_n_replicates replicates each, all engines in parallel via
# one snakemake -j invocation (local core parallelism across 3 x
# calibration_n_replicates independent jobs) instead of 3 separate SLURM
# array jobs -- meant to run unattended as a proper sbatch submission after
# the earlier interactive srun attempt ran out of wall time.
#
# Usage: sbatch bash_scripts/simulation/calibration_simulate_rawfits_batch.sh
#   (VARIANT env var override supported, defaults to wo_FIM_wo_SFSresids)

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")
VARIANT="${VARIANT:-wo_FIM_wo_SFSresids}"

REAL_INF_ROOT="experiments/${MODEL}/real_data_analysis/inferences"
PRED_DIR="experiments/${MODEL}/real_data_analysis/prediction_${VARIANT}"

readarray -t REAL_ARMS < <(jq -r '.real_data_analysis.arms // ["Chr3L"] | .[]' "$CFG")
REAL_USE_GENMAP=$(jq -r '.real_data_analysis.use_genmap // false' "$CFG")
REAL_TAG=$(IFS=_; echo "${REAL_ARMS[*]}")
[[ "$REAL_USE_GENMAP" == "true" ]] && REAL_TAG="${REAL_TAG}_genmap"
REAL_LD_ENGINE="MomentsLD_${REAL_TAG}"

cd "$ROOT"

# Generate the three raw-fit prediction JSONs if they don't already exist
# (idempotent -- safe to resubmit this script).
declare -A FIT_SUBDIR=(
  [dadi]="dadi"
  [moments]="moments"
  [momentsLD]="$REAL_LD_ENGINE"
)
for engine in dadi moments momentsLD; do
  out_json="${PRED_DIR}/predictions_${engine}.json"
  if [[ ! -s "$out_json" ]]; then
    echo "Generating $out_json"
    python snakemake_scripts/real_fit_to_prediction_json.py \
      --best-fit-pkl "${REAL_INF_ROOT}/${FIT_SUBDIR[$engine]}/best_fit.pkl" \
      --engine "$engine" \
      --out-path "$out_json"
  fi
done

# All 3 engines' full replicate sets, run in parallel using local core
# parallelism (-j) instead of separate SLURM array jobs per engine.
TARGETS=(
  "experiments/${MODEL}/real_data_analysis/calibration_${VARIANT}/dadi/.all_reps_done"
  "experiments/${MODEL}/real_data_analysis/calibration_${VARIANT}/moments/.all_reps_done"
  "experiments/${MODEL}/real_data_analysis/calibration_${VARIANT}/momentsLD/.all_reps_done"
)

echo "Running calibration_simulate for: ${TARGETS[*]}"
snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-triggers mtime \
  --nolock \
  --keep-going \
  --rerun-incomplete \
  --allowed-rules calibration_simulate calibration_simulate_all_reps \
  -j "${SLURM_CPUS_PER_TASK:-8}" \
  -- \
  "${TARGETS[@]}"

echo "calibration_simulate_rawfits_batch finished."

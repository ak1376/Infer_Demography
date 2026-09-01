#!/bin/bash
#SBATCH --job-name=calibration_sim
#SBATCH --output=logs/calibration_sim_%A_%a.out
#SBATCH --error=logs/calibration_sim_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Model calibration check (posterior-predictive): simulate one tree sequence +
# SFS per replicate at the real-data fitted params, for one (VARIANT,
# MODEL_KEY). Each replicate runs as its own SLURM array task (mirrors
# running_simulation.sh's self-resubmitting array pattern) instead of one job
# looping over all replicates sequentially. Requires real_combine_predict.sh
# to have already produced predictions_${MODEL_KEY}.json for VARIANT.
#
# Rule run (once per array task): calibration_simulate
#
# Time/mem above are per-replicate (one msprime simulation under engine=msprime).

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")
N_REPS=$(jq -r '.calibration_n_replicates // 20' "$CFG")

# Which trained model's real-data fit to simulate from.
VARIANT="${VARIANT:-wo_FIM_wo_SFSresids}"
MODEL_KEY="${MODEL_KEY:-xgboost}"

# First launch (no array id yet): resubmit as an array sized from
# calibration_n_replicates, one task per replicate.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Submitting array 0..$((N_REPS - 1)) (calibration_n_replicates=${N_REPS} from ${CFG})"
    VARIANT="$VARIANT" MODEL_KEY="$MODEL_KEY" ROOT="$ROOT" \
        sbatch --array=0-"$((N_REPS - 1))" "$0" "$@"
    exit 0
fi

REP="$SLURM_ARRAY_TASK_ID"
TARGET="experiments/${MODEL}/real_data_analysis/calibration_${VARIANT}/${MODEL_KEY}/replicate_${REP}/SFS.pkl"

echo "MODEL=$MODEL  VARIANT=$VARIANT  MODEL_KEY=$MODEL_KEY  replicate=$REP"
echo "Target: $TARGET"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules calibration_simulate \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "$TARGET"

echo "calibration_simulate finished -> $TARGET"

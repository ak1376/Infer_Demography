#!/bin/bash
#SBATCH --job-name=calibration_sim
#SBATCH --output=logs/calibration_sim_%j.out
#SBATCH --error=logs/calibration_sim_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Model calibration check (posterior-predictive): simulate calibration_n_replicates
# tree sequences + SFS at the real-data fitted params for one (VARIANT, MODEL_KEY),
# all in this one job (see Snakefile's calibration_simulate rule / snakemake_scripts/
# calibration_simulate.py --n-replicates). Requires real_combine_predict.sh to have
# already produced predictions_${MODEL_KEY}.json for VARIANT.
#
# Rule run: calibration_simulate
#
# Time/mem above assume the default calibration_n_replicates=20 under engine=msprime;
# scale --time up if you raise calibration_n_replicates in the experiment config.

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

CALIB_DIR="experiments/${MODEL}/real_data_analysis/calibration_${VARIANT}/${MODEL_KEY}"
# Requesting just replicate_0 pulls the whole job: the Snakefile rule declares
# all calibration_n_replicates replicates as joint outputs of one invocation.
TARGET="${CALIB_DIR}/replicate_0/SFS.pkl"

echo "MODEL=$MODEL  VARIANT=$VARIANT  MODEL_KEY=$MODEL_KEY  N_REPS=$N_REPS"
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

echo "calibration_simulate finished -> ${CALIB_DIR}"

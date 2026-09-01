#!/bin/bash
#SBATCH --job-name=calibration_ppc
#SBATCH --output=logs/calibration_ppc_%j.out
#SBATCH --error=logs/calibration_ppc_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Model calibration check (posterior-predictive) plots + summary: compares
# SFS-derived pi/Tajima's D/FST and SFS shape between the real observed data
# and the calibration_simulate replicates, for one (VARIANT, MODEL_KEY).
# Requires all calibration_simulate replicates to already exist -- run
# bash_scripts/calibration_simulate.sh (as a SLURM array) first and wait for
# every array task to finish.
#
# Rule run: calibration_ppc

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")

VARIANT="${VARIANT:-wo_FIM_wo_SFSresids}"
MODEL_KEY="${MODEL_KEY:-xgboost}"

TARGET="experiments/${MODEL}/real_data_analysis/calibration_${VARIANT}/${MODEL_KEY}/ppc/calibration_ppc.png"

echo "MODEL=$MODEL  VARIANT=$VARIANT  MODEL_KEY=$MODEL_KEY"
echo "Target: $TARGET"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules calibration_ppc \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "$TARGET"

echo "calibration_ppc finished -> $TARGET"

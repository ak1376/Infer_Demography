#!/bin/bash
#SBATCH --job-name=calibration_ld_ppc
#SBATCH --output=logs/calibration_ld_ppc_%j.out
#SBATCH --error=logs/calibration_ld_ppc_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Model calibration check (posterior-predictive), LD version: does the
# theoretical LD decay curve at the fitted params reproduce the empirical LD
# decay measured directly from the real data, for one (VARIANT, MODEL_KEY)?
# Purely analytic -- no calibration_simulate replicates needed, so this can
# run immediately after predict_real_data, without waiting on the SLURM
# array tree-sequence simulations.
#
# Rule run: calibration_ld_ppc

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")

VARIANT="${VARIANT:-wo_FIM_wo_SFSresids}"
MODEL_KEY="${MODEL_KEY:-xgboost}"

TARGET="experiments/${MODEL}/real_data_analysis/calibration_${VARIANT}/${MODEL_KEY}/ppc/calibration_ld_ppc.pdf"

echo "MODEL=$MODEL  VARIANT=$VARIANT  MODEL_KEY=$MODEL_KEY"
echo "Target: $TARGET"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules calibration_ld_ppc \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "$TARGET"

echo "calibration_ld_ppc finished -> $TARGET"

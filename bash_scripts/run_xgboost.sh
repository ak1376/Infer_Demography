#!/bin/bash
#SBATCH --job-name=xgb_only
#SBATCH --output=logs/xgb_only_%j.out
#SBATCH --error=logs/xgb_only_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail
set -x  # <— show every command for debugging

CFG_PATH="/projects/kernlab/akapoor/Infer_Demography/config_files/experiment_config_split_isolation.json"
PROJECT_ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$PROJECT_ROOT/Snakefile"
CORES="${SLURM_CPUS_PER_TASK:-8}"

MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

targets=(
  "experiments/${MODEL}/modeling/xgboost/xgb_mdl_obj.pkl"
  "experiments/${MODEL}/modeling/xgboost/xgb_model_error.json"
  "experiments/${MODEL}/modeling/xgboost/xgb_model.pkl"
  "experiments/${MODEL}/modeling/xgboost/xgb_results.png"
  "experiments/${MODEL}/modeling/xgboost/xgb_feature_importances.png"
)

# echo targets to ensure the array expanded
printf 'TARGET: %s\n' "${targets[@]}"

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$PROJECT_ROOT" \
  --cores "$CORES" \
  --nolock \
  --rerun-incomplete \
  --allowed-rules xgboost make_color_scheme \
  "${targets[@]}"

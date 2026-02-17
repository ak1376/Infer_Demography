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

CFG_PATH="/projects/kernlab/akapoor/Infer_Demography/config_files/experiment_config_IM_symmetric.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# XGBoost outputs (Snakemake will pull its inputs automatically)
XGB_TARGETS=(
  "experiments/${MODEL}/modeling/xgboost/xgb_mdl_obj.pkl"
  "experiments/${MODEL}/modeling/xgboost/xgb_model_error.json"
  "experiments/${MODEL}/modeling/xgboost/xgb_model.pkl"
  "experiments/${MODEL}/modeling/xgboost/xgb_results.png"
  "experiments/${MODEL}/modeling/xgboost/xgb_feature_importances.png"
)

# show targets for sanity
printf 'XGB TARGET: %s\n' "${XGB_TARGETS[@]}"

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --cores "${SLURM_CPUS_PER_TASK}" \
  --nolock \
  --rerun-incomplete \
  --latency-wait 60 \
  --printshellcmds \
  --allowed-rules xgboost combine_features make_color_scheme \
  -- \
  "${XGB_TARGETS[@]}"

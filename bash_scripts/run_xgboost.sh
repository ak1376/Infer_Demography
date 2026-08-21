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

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG_PATH="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# Which FIM x SFS-residuals feature-set variants to run (must match
# MODELING_VARIANTS in the Snakefile)
VARIANTS=(w_FIM_w_SFSresids w_FIM_wo_SFSresids wo_FIM_w_SFSresids wo_FIM_wo_SFSresids)

# XGBoost outputs (Snakemake will pull its inputs automatically)
XGB_TARGETS=()
for variant in "${VARIANTS[@]}"; do
  XGB_TARGETS+=("experiments/${MODEL}/modeling_${variant}/xgboost/xgb_mdl_obj.pkl")
  XGB_TARGETS+=("experiments/${MODEL}/modeling_${variant}/xgboost/xgb_model_error.json")
  XGB_TARGETS+=("experiments/${MODEL}/modeling_${variant}/xgboost/xgb_model.pkl")
  XGB_TARGETS+=("experiments/${MODEL}/modeling_${variant}/xgboost/xgb_results.png")
  XGB_TARGETS+=("experiments/${MODEL}/modeling_${variant}/xgboost/xgb_feature_importances.png")
done

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

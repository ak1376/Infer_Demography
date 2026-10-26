#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --output=logs/train_models_%j.out
#SBATCH --error=logs/train_models_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# ---------------------------------------------------------------------------
# Train models (rule train_model) on every modeling variant's dataset.
#   default:                         modeling_all -- every model_key x every
#                                    variant in the experiment config
#   MODEL_KEYS="xgboost":            only those models (space-separated:
#   MODEL_KEYS="linear_ridge random_forest"   linear_<reg>, random_forest, xgboost)
# combine_features is allowed too, so missing datasets are built first.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
CFG_PATH="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/modeling.smk"
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

if [[ -z "${MODEL_KEYS:-}" ]]; then
  TARGETS=(modeling_all)
else
  # Same variant list modeling.smk uses (see rules/common.smk).
  mapfile -t VARIANTS < <(jq -r '(.modeling_variants // ["w_FIM_w_SFSresids","w_FIM_wo_SFSresids","wo_FIM_w_SFSresids","wo_FIM_wo_SFSresids"])[]' "$CFG_PATH")
  TARGETS=()
  for variant in "${VARIANTS[@]}"; do
    for key in $MODEL_KEYS; do
      TARGETS+=("experiments/${MODEL}/modeling_${variant}/${key}/mdl_obj.pkl")
    done
  done
fi
printf 'TARGET: %s\n' "${TARGETS[@]}"

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --cores "${SLURM_CPUS_PER_TASK:-8}" \
  --nolock \
  --rerun-incomplete \
  --latency-wait 60 \
  --printshellcmds \
  --allowed-rules train_model combine_features modeling_all \
  -- \
  "${TARGETS[@]}"

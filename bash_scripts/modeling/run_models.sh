#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --output=logs/train_models_%A_%a.out
#SBATCH --error=logs/train_models_%A_%a.err
#SBATCH --account=kernlab
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose
# (time / cpus / mem are set per model family by the launcher below)

# ---------------------------------------------------------------------------
# Train models (rule train_model): one SLURM array task per (variant, model).
#
# Run it as before:   sbatch bash_scripts/modeling/run_models.sh
#   (or bash bash_scripts/modeling/run_models.sh from a login node)
# The first launch (no SLURM_ARRAY_TASK_ID) is just a launcher: it submits one
# array per model family, each sized to (#variants x #models in that family)
# and with that family's own resources, then exits.
#
#   MODEL_KEYS="xgboost"                      only those models
#   MODEL_KEYS="linear_ridge random_forest"   (linear_<reg>, random_forest, xgboost)
#   VARIANTS="w_FIM_w_SFSresids"              only those variants
#                                             (default: experiment config's
#                                             modeling_variants)
#
# Datasets must already exist (aggregate_features.sh). Array tasks run ONLY
# train_model, so parallel tasks never race to rebuild a shared dataset.
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
CFG_PATH="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/modeling.smk"
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# Same defaults modeling.smk uses (see rules/common.smk).
if [[ -z "${VARIANTS:-}" ]]; then
  VARIANTS="$(jq -r '(.modeling_variants // ["w_FIM_w_SFSresids","w_FIM_wo_SFSresids","wo_FIM_w_SFSresids","wo_FIM_wo_SFSresids"]) | join(" ")' "$CFG_PATH")"
fi
MODEL_KEYS="${MODEL_KEYS:-linear_standard linear_ridge linear_lasso linear_elasticnet random_forest xgboost}"

family_of() { [[ "$1" == linear_* ]] && echo linear || echo "$1"; }

# Ordered (variant, model_key) list for one family. Launcher and array tasks
# both build it the same way, so task i always means the same pair.
pairs_for_family() {
  local fam="$1" v k
  for v in $VARIANTS; do
    for k in $MODEL_KEYS; do
      [[ "$(family_of "$k")" == "$fam" ]] && echo "$v $k"
    done
  done
  return 0
}

# ---------------------------- launcher --------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # Fail fast (before queueing anything) if a dataset is missing.
  for v in $VARIANTS; do
    ds="$ROOT/experiments/${MODEL}/modeling_${v}/datasets/normalized_train_features.pkl"
    [[ -f "$ds" ]] || { echo "Missing dataset: $ds -- run aggregate_features.sh first." >&2; exit 1; }
  done

  # Per-family resources. RF/XGB time covers a 1h Optuna cap
  # (optuna_timeout in model_config.yaml) plus the final fit and plots.
  declare -A TIME=( [linear]=01:00:00 [random_forest]=03:00:00 [xgboost]=03:00:00 )
  declare -A CPUS=( [linear]=2        [random_forest]=8        [xgboost]=8 )
  declare -A MEM=(  [linear]=8G       [random_forest]=24G      [xgboost]=16G )

  for fam in linear random_forest xgboost; do
    mapfile -t PAIRS < <(pairs_for_family "$fam")
    (( ${#PAIRS[@]} == 0 )) && continue
    printf '[%s] %d tasks:\n' "$fam" "${#PAIRS[@]}"; printf '   %s\n' "${PAIRS[@]}"
    FAMILY="$fam" VARIANTS="$VARIANTS" MODEL_KEYS="$MODEL_KEYS" ROOT="$ROOT" \
      sbatch --export=ALL --job-name="train_${fam}" \
        --array=0-"$(( ${#PAIRS[@]} - 1 ))" \
        --time="${TIME[$fam]}" --cpus-per-task="${CPUS[$fam]}" --mem="${MEM[$fam]}" \
        "$0"
  done
  exit 0
fi

# ---------------------------- array task ------------------------------------
: "${FAMILY:?FAMILY not set (submit via the launcher, not sbatch --array directly)}"
mapfile -t PAIRS < <(pairs_for_family "$FAMILY")
read -r VARIANT MODEL_KEY <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
TARGET="experiments/${MODEL}/modeling_${VARIANT}/${MODEL_KEY}/mdl_obj.pkl"
echo "FAMILY=$FAMILY  VARIANT=$VARIANT  MODEL_KEY=$MODEL_KEY"
echo "TARGET: $TARGET"

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --cores "${SLURM_CPUS_PER_TASK:-1}" \
  --nolock \
  --rerun-incomplete \
  --latency-wait 60 \
  --printshellcmds \
  --allowed-rules train_model \
  -- \
  "$TARGET"

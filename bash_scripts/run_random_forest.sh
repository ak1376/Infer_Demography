#!/bin/bash
#SBATCH --job-name=rf_only
#SBATCH --output=logs/rf_only_%j.out
#SBATCH --error=logs/rf_only_%j.err
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

# pull model name from experiment config
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# Which FIM x SFS-residuals feature-set variants to run (must match
# MODELING_VARIANTS in the Snakefile)
VARIANTS=(w_FIM_w_SFSresids w_FIM_wo_SFSresids wo_FIM_w_SFSresids wo_FIM_wo_SFSresids)

# Random Forest outputs (Snakemake will pull required inputs automatically)
RF_TARGETS=()
for variant in "${VARIANTS[@]}"; do
  RF_TARGETS+=("experiments/${MODEL}/modeling_${variant}/random_forest/random_forest_mdl_obj.pkl")
  RF_TARGETS+=("experiments/${MODEL}/modeling_${variant}/random_forest/random_forest_model_error.json")
  RF_TARGETS+=("experiments/${MODEL}/modeling_${variant}/random_forest/random_forest_model.pkl")
  RF_TARGETS+=("experiments/${MODEL}/modeling_${variant}/random_forest/random_forest_results.png")
  RF_TARGETS+=("experiments/${MODEL}/modeling_${variant}/random_forest/random_forest_feature_importances.png")
done

# sanity print
printf 'RF TARGET: %s\n' "${RF_TARGETS[@]}"

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --cores "${SLURM_CPUS_PER_TASK}" \
  --nolock \
  --rerun-incomplete \
  --latency-wait 60 \
  --printshellcmds \
  --allowed-rules random_forest combine_features make_color_scheme \
  -- \
  "${RF_TARGETS[@]}"

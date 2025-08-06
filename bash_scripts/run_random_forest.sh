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

CFG_PATH="/projects/kernlab/akapoor/Infer_Demography/config_files/experiment_config_bottleneck.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

# pull model name from experiment config
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# Random Forest outputs (Snakemake will pull required inputs automatically)
RF_TARGETS=(
  "experiments/${MODEL}/modeling/random_forest/random_forest_mdl_obj.pkl"
  "experiments/${MODEL}/modeling/random_forest/random_forest_model_error.json"
  "experiments/${MODEL}/modeling/random_forest/random_forest_model.pkl"
  "experiments/${MODEL}/modeling/random_forest/random_forest_results.png"
  "experiments/${MODEL}/modeling/random_forest/random_forest_feature_importances.png"
)

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

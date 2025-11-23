#!/bin/bash
#SBATCH --job-name=postproc_features
#SBATCH --output=logs/postprocessing_features_%j.out
#SBATCH --error=logs/postprocessing_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
set -euo pipefail
CFG_PATH="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_bottleneck.json"
# : "${CFG_PATH:?CFG_PATH is not defined}"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

DATA_TARGETS=(
  "experiments/${MODEL}/modeling/datasets/features_df.pkl"
  "experiments/${MODEL}/modeling/datasets/targets_df.pkl"
  "experiments/${MODEL}/modeling/datasets/normalized_train_features.pkl"
  "experiments/${MODEL}/modeling/datasets/normalized_train_targets.pkl"
  "experiments/${MODEL}/modeling/datasets/normalized_validation_features.pkl"
  "experiments/${MODEL}/modeling/datasets/normalized_validation_targets.pkl"
  "experiments/${MODEL}/modeling/datasets/features_scatterplot.png"
)
COLOR_TARGETS=(
  "experiments/${MODEL}/modeling/color_shades.pkl"
  "experiments/${MODEL}/modeling/main_colors.pkl"
)

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --cores "$SLURM_CPUS_PER_TASK" \
  --nolock --rerun-incomplete \
  --allowed-rules combine_features make_color_scheme \
  --forcerun combine_features make_color_scheme \
  -- \
  "${DATA_TARGETS[@]}" \
  "${COLOR_TARGETS[@]}"
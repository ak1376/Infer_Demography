#!/bin/bash
#SBATCH --job-name=linear_only
#SBATCH --output=logs/linear_only_%j.out
#SBATCH --error=logs/linear_only_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

CFG_PATH="/projects/kernlab/akapoor/Infer_Demography/config_files/experiment_config_split_isolation.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"
CORES="${SLURM_CPUS_PER_TASK:-4}"

# Model name from experiment config
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# Which linear variants to run
REGS=(standard ridge lasso elasticnet)

# Build target list for all four variants
targets=()
for reg in "${REGS[@]}"; do
  targets+=("experiments/${MODEL}/modeling/linear_${reg}/linear_mdl_obj_${reg}.pkl")
  targets+=("experiments/${MODEL}/modeling/linear_${reg}/linear_model_error_${reg}.json")
  targets+=("experiments/${MODEL}/modeling/linear_${reg}/linear_regression_model_${reg}.pkl")
  targets+=("experiments/${MODEL}/modeling/linear_${reg}/linear_results_${reg}.png")
done

# (Optional) show targets
printf 'LINEAR TARGET: %s\n' "${targets[@]}"

# Run just the needed rules: linear_regression + prerequisites
snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --cores "$CORES" \
  --nolock \
  --rerun-incomplete \
  --latency-wait 60 \
  --printshellcmds \
  --allowed-rules linear_regression combine_features make_color_scheme \
  -- \
  "${targets[@]}"

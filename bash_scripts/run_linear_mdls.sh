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

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG_PATH="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
CORES="${SLURM_CPUS_PER_TASK:-4}"

# Model name from experiment config
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# Which linear regularization types to run
REGS=(standard ridge lasso elasticnet)

# Which FIM x SFS-residuals feature-set variants to run (must match
# MODELING_VARIANTS in the Snakefile)
VARIANTS=(w_FIM_w_SFSresids w_FIM_wo_SFSresids wo_FIM_w_SFSresids wo_FIM_wo_SFSresids)

# Build target list for every variant x regularization-type combination
targets=()
for variant in "${VARIANTS[@]}"; do
  for reg in "${REGS[@]}"; do
    targets+=("experiments/${MODEL}/modeling_${variant}/linear_${reg}/linear_mdl_obj_${reg}.pkl")
    targets+=("experiments/${MODEL}/modeling_${variant}/linear_${reg}/linear_model_error_${reg}.json")
    targets+=("experiments/${MODEL}/modeling_${variant}/linear_${reg}/linear_regression_model_${reg}.pkl")
    targets+=("experiments/${MODEL}/modeling_${variant}/linear_${reg}/linear_results_${reg}.png")
  done
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

#!/bin/bash
#SBATCH --job-name=agg_opts_both
#SBATCH --output=logs/agg_both_%A_%a.out
#SBATCH --error=logs/agg_both_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

# --- paths & config ---
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_isolation.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")

# If launched without --array, resubmit with full range.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0" "$@"
  exit 0
fi

sid="$SLURM_ARRAY_TASK_ID"
echo "aggregate_opts (both engines): sid=$sid"

# Both engine outputs for this sim:
TGT_MOM="experiments/${MODEL}/inferences/sim_${sid}/moments/fit_params.pkl"
TGT_DADI="experiments/${MODEL}/inferences/sim_${sid}/dadi/fit_params.pkl"

# Only allow the aggregate rule so Snakemake won’t try to run infer_*.
snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  --allowed-rules aggregate_opts \
  --keep-going \
  "$TGT_MOM" "$TGT_DADI"

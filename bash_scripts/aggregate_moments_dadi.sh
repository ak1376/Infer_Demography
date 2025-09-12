#!/bin/bash
#SBATCH --job-name=agg_opts
#SBATCH --output=logs/agg_%A_%a.out
#SBATCH --error=logs/agg_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

# 0. paths & config
# : "${CFG_PATH:?CFG_PATH is not defined}"
# CFG="$CFG_PATH"
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_isolation.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")

# 1. if launched without --array, resubmit with correct range
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0" "$@"
    exit 0
fi

# 2. derive sid from array index (no padding)
sid="$SLURM_ARRAY_TASK_ID"
echo "aggregate_opts: sid=$sid  (folder sim_$sid)"

# 3. Snakemake target for this simulation (no padding)
TARGET="experiments/${MODEL}/inferences/sim_${sid}/moments/fit_params.pkl"

snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  "$TARGET"

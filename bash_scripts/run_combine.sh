#!/bin/bash
#SBATCH --job-name=combine_inf
#SBATCH --output=logs/combine_%A_%a.out
#SBATCH --error=logs/combine_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# 0. paths & experiment constants
# : "${CFG_PATH:?CFG_PATH is not defined}"
# CFG="$CFG_PATH"
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_drosophila_three_epoch.json"

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")

# 1. (re)submit with the correct --array range if none was provided
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0" "$@"
    exit 0
fi

# 2. decode sid from array index (no padding)
sid="$SLURM_ARRAY_TASK_ID"
echo "combine_results: sid=$sid  (folder sim_$sid)"

# 3. Snakemake target for this simulation (no padding)
TARGET="experiments/${MODEL}/inferences/sim_${sid}/all_inferences.pkl"

snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  "$TARGET"

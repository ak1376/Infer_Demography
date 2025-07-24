#!/bin/bash
#SBATCH --job-name=combine_inf
#SBATCH --output=logs/combine_%A_%a.out
#SBATCH --error=logs/combine_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue                               # Requeue on preemption
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# --------------------------------------------------------------------------
# 0. paths & experiment constants -----------------------------------------
# --------------------------------------------------------------------------
# the master script exports CFG_PATH; abort if it is not set
: "${CFG_PATH:?CFG_PATH is not defined}"
CFG="$CFG_PATH"

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")   # e.g. 10
MODEL=$(jq -r '.demographic_model'      "$CFG")   # bottleneck | split_isolation …

# How many digits to pad sid with (00, 01 …)
PAD=$(python - <<EOF
import math, sys; n=int(sys.argv[1])
print(int(math.log10(n-1))+1)
EOF
"$NUM_DRAWS")

# --------------------------------------------------------------------------
# 1. (re)submit with the correct --array range if none was provided ---------
# --------------------------------------------------------------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0" "$@"
    exit 0
fi

# --------------------------------------------------------------------------
# 2. decode sid from array index -------------------------------------------
# --------------------------------------------------------------------------
sid=$SLURM_ARRAY_TASK_ID
pad_sid=$(printf "%0${PAD}d" "$sid")
echo "combine_results: sid=$sid  (folder $pad_sid)"

# --------------------------------------------------------------------------
# 3. Snakemake target for this simulation ----------------------------------
# --------------------------------------------------------------------------
TARGET="experiments/${MODEL}/inferences/sim_${pad_sid}/all_inferences.pkl"

# This file is produced by rule `combine_results`.
# --------------------------------------------------------------------------
snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  "$TARGET"

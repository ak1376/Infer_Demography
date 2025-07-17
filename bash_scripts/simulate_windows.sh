#!/bin/bash
#SBATCH --job-name=win_sim
#SBATCH --array=0-999                            # Array range (adjust based on the number of tasks and batch size)
#SBATCH --output=logs/win_%A_%a.out
#SBATCH --error=logs/win_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# --------------------------------------------------------------------------
# 0. paths & config --------------------------------------------------------
# --------------------------------------------------------------------------
# the master script exports CFG_PATH; abort if it is not set
: "${CFG_PATH:?CFG_PATH is not defined}"
CFG="$CFG_PATH"

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'  "$CFG")        # e.g. 10
NUM_WINDOWS=100                                # hard‑coded in Snakefile
MODEL=$(jq -r '.demographic_model' "$CFG")     # bottleneck | split_isolation …

# width for zero‑padded sid (00, 01 …)
PAD=$(python - <<EOF
import math, sys
print(int(math.log10(int(sys.argv[1])-1))+1)
EOF
"$NUM_DRAWS")

# --------------------------------------------------------------------------
# 1. if launched without --array, resubmit with correct range --------------
# --------------------------------------------------------------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    sbatch --array=0-$(( NUM_DRAWS*NUM_WINDOWS - 1 )) "$0" "$@"
    exit 0
fi

# --------------------------------------------------------------------------
# 2. decode sid + win from array index -------------------------------------
# --------------------------------------------------------------------------
idx=$SLURM_ARRAY_TASK_ID
sid=$(( idx / NUM_WINDOWS ))
win=$(( idx % NUM_WINDOWS ))
pad_sid=$(printf "%0${PAD}d" "$sid")

echo "simulate_window: sid=$sid (folder $pad_sid)  win=$win"

# --------------------------------------------------------------------------
# 3. target path Snakemake must create -------------------------------------
# --------------------------------------------------------------------------
TARGET="experiments/${MODEL}/inferences/sim_${pad_sid}/MomentsLD/windows/window_${win}.vcf.gz"

# --------------------------------------------------------------------------
# 4. launch Snakemake ------------------------------------------------------
# --------------------------------------------------------------------------
snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  "$TARGET"

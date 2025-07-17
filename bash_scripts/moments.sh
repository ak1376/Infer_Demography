#!/bin/bash
#SBATCH --job-name=mom_infer
#SBATCH --array=0-19                            # Array range (adjust based on the number of tasks and batch size)
#SBATCH --output=logs/mom_%A_%a.out
#SBATCH --error=logs/mom_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# --------------------------- configuration ---------------------------------
# the master script exports CFG_PATH; abort if it is not set
: "${CFG_PATH:?CFG_PATH is not defined}"
CFG="$CFG_PATH"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")
PAD=$(python - <<EOF
import math, sys
print(int(math.log10(int(sys.argv[1])-1))+1)
EOF
"$NUM_DRAWS")

# --------------------------- array sizing ----------------------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    sbatch --array=0-$(( NUM_DRAWS*NUM_OPTIMS - 1 )) "$0"
    exit 0
fi

# --------------------------- decode index ----------------------------------
IDX=$SLURM_ARRAY_TASK_ID
sid=$(( IDX / NUM_OPTIMS ))
opt=$(( IDX % NUM_OPTIMS ))
pad_sid=$(printf "%0${PAD}d" "$sid")

echo "Moments: sid=$sid opt=$opt (folder $pad_sid)"

# --------------------------- run snakemake ---------------------------------
snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  "experiments/${MODEL}/runs/run_${pad_sid}_${opt}/inferences/moments/fit_params.pkl"

#!/bin/bash
#SBATCH --job-name=mom_infer
#SBATCH --array=0-19
#SBATCH --output=logs/mom_%A_%a.out
#SBATCH --error=logs/mom_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue                               # Requeue on preemption
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# --------------- batching parameters (tweak here) --------------------------
BATCH_SIZE=1          # number of (sim,opt) jobs per array task. This is #sims x #optimisations per sim.
# ---------------------------------------------------------------------------

# --------------- paths & config --------------------------------------------
: "${CFG_PATH:?CFG_PATH is not defined}"
CFG="$CFG_PATH"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")

TOTAL_TASKS=$(( NUM_DRAWS * NUM_OPTIMS ))

PAD=$(python - <<EOF
import math, sys; n=int(sys.argv[1]); print(int(math.log10(n-1))+1)
EOF
"$NUM_DRAWS")

# --------------- first pass: submit with proper --array --------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY"%$MAX_CONCURRENT "$0" "$@"
    exit 0
fi

#

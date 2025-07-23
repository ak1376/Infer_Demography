#!/bin/bash
#SBATCH --job-name=agg_opts
#SBATCH --output=logs/agg_%A_%a.out
#SBATCH --error=logs/agg_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue                               # Requeue on preemption
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

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")   # e.g. 10
MODEL=$(jq -r '.demographic_model'      "$CFG")   # bottleneck | split_isolation …

PAD=$(python - <<EOF
import math, sys; n=int(sys.argv[1])
print(int(math.log10(n-1))+1)
EOF
"$NUM_DRAWS")

# --------------------------------------------------------------------------
# 1. if launched without --array, resubmit with correct range --------------
# --------------------------------------------------------------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0" "$@"
    exit 0
fi

# --------------------------------------------------------------------------
# 2. derive sid from array index -------------------------------------------
# --------------------------------------------------------------------------
sid=$SLURM_ARRAY_TASK_ID
pad_sid=$(printf "%0${PAD}d" "$sid")
echo "aggregate_opts: sid=$sid  (folder $pad_sid)"

# --------------------------------------------------------------------------
# 3. Snakemake target for this simulation ----------------------------------
# --------------------------------------------------------------------------
TARGET="experiments/${MODEL}/inferences/sim_${pad_sid}/moments/fit_params.pkl"

# Running Snakemake on one of the two outputs (moments) is enough; it will
# produce both moments & dadi files declared in rule aggregate_opts.
# --------------------------------------------------------------------------
snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  "$TARGET"

#!/bin/bash
#SBATCH --job-name=opt_momLD
#SBATCH --output=logs/optLD_%A_%a.out
#SBATCH --error=logs/optLD_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

NUM_DRAWS=$(jq -r '.num_draws' "$CFG")          # e.g. 10
MODEL=$(jq -r '.demographic_model' "$CFG")      # bottleneck | split_isolation …

# zero‑padding width for sid (“00”, “01”…)
PAD=$(python - <<EOF
import math, sys; n=int(sys.argv[1])
print(int(math.log10(n-1))+1)
EOF
"$NUM_DRAWS")

# --------------------------------------------------------------------------
# 1. auto‑set --array if missing -------------------------------------------
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
echo "optimize_momentsLD: sid=$sid (folder $pad_sid)"

# --------------------------------------------------------------------------
# 3. Snakemake target for this simulation ----------------------------------
# --------------------------------------------------------------------------
TARGET="experiments/${MODEL}/inferences/sim_${pad_sid}/MomentsLD/best_fit.pkl"

# asking for best_fit.pkl triggers rule optimize_momentsld, which also
# creates means.varcovs.pkl, bootstrap_sets.pkl, and the PDF.
# --------------------------------------------------------------------------
snakemake -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --nolock \
  "$TARGET"

#!/bin/bash
#SBATCH --job-name=ld_stats
#SBATCH --array=0-1999
#SBATCH --output=logs/ld_%A_%a.out
#SBATCH --error=logs/ld_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue                               # Requeue on preemption
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=1          # number of (sim,window) jobs per array task
# ----------------------------------------------------------------------------

# -------- config & constants -----------------------------------------------
: "${CFG_PATH:?CFG_PATH is not defined}"
CFG="$CFG_PATH"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")
NUM_WINDOWS=100                          # → keep in sync with Snakefile
TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

PAD=$(python - <<EOF
import math, sys; n=int(sys.argv[1]); print(int(math.log10(n-1))+1)
EOF
"$NUM_DRAWS")

# -------- first launch: resubmit with correct --array ----------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY"%$MAX_CONCURRENT "$0" "$@"
    exit 0
fi

# -------- slice of indices handled by *this* array task --------------------
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $END -ge $TOTAL_TASKS ]] && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $START .. $END"

# -------- loop over (sim, window) pairs ------------------------------------
for IDX in $(seq "$START" "$END"); do
    SID=$(( IDX / NUM_WINDOWS ))
    WIN=$(( IDX % NUM_WINDOWS ))
    PAD_SID=$(printf "%0${PAD}d" "$SID")

    TARGET="experiments/${MODEL}/inferences/sim_${PAD_SID}/MomentsLD/LD_stats/LD_stats_window_${WIN}.pkl"
    echo "LD‑stats: SID=$SID  WIN=$WIN  →  $TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT"      \
              --rerun-incomplete        \
              --nolock                  \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET" || { echo "Failed for SID=$SID WIN=$WIN"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

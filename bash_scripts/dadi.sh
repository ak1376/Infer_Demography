#!/bin/bash
#SBATCH --job-name=dadi_infer
#SBATCH --array=0-19
#SBATCH --output=logs/dadi_%A_%a.out
#SBATCH --error=logs/dadi_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue                               # Requeue on preemption
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# -------- batching knobs ---------------------------------------------------
BATCH_SIZE=1          # number of (sim,opt) pairs this array task handles
# ----------------------------------------------------------------------------

# -------- config -----------------------------------------------------------
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

# -------- first launch: compute proper --array range -----------------------
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    sbatch --array=0-"$NUM_ARRAY"%$MAX_CONCURRENT "$0" "$@"
    exit 0
fi

# -------- slice for this array element -------------------------------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$((   (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
[[ $BATCH_END -ge $TOTAL_TASKS ]] && BATCH_END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $BATCH_START .. $BATCH_END"

# -------- loop over (sim,opt) pairs ----------------------------------------
for IDX in $(seq "$BATCH_START" "$BATCH_END"); do
    SID=$(( IDX / NUM_OPTIMS ))
    OPT=$(( IDX % NUM_OPTIMS ))
    PAD_SID=$(printf "%0${PAD}d" "$SID")

    TARGET="experiments/${MODEL}/runs/run_${PAD_SID}_${OPT}/inferences/dadi/fit_params.pkl"
    echo "dadi optimisation: SID=$SID  OPT=$OPT  →  $TARGET"

    snakemake --snakefile "$SNAKEFILE" \
              --directory  "$ROOT" \
              --rerun-incomplete \
              --nolock \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET" || { echo "Snakemake failed for SID=$SID OPT=$OPT"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

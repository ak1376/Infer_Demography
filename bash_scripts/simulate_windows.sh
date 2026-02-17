#!/bin/bash
#SBATCH --job-name=win_sim
#SBATCH --array=0-9999
#SBATCH --output=logs/win_sim_%A_%a.out
#SBATCH --error=logs/win_sim_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# --- config ---
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"
CFG="${CFG_PATH:-/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_IM_symmetric.json}"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws' "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")

NUM_WINDOWS="${NUM_WINDOWS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-200}"

TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

echo "CFG=$CFG  MODEL=$MODEL  NUM_DRAWS=$NUM_DRAWS  NUM_WINDOWS=$NUM_WINDOWS  TOTAL_TASKS=$TOTAL_TASKS"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# --- first launch: resubmit with correct array range ---
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
  echo "Submitting array 0..${NUM_ARRAY}%${MAX_CONCURRENT}"
  sbatch --array=0-"$NUM_ARRAY"%$MAX_CONCURRENT "$0" "$@"
  exit 0
fi

# --- slice work for this array task ---
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
(( END >= TOTAL_TASKS )) && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → indices $START .. $END"

for IDX in $(seq "$START" "$END"); do
  SID=$(( IDX / NUM_WINDOWS ))
  WIN=$(( IDX % NUM_WINDOWS ))

  SIM_DIR="$ROOT/experiments/$MODEL/simulations/$SID"
  [[ -f "$SIM_DIR/.done" && -f "$SIM_DIR/sampled_params.pkl" ]] || {
    echo "[SKIP] SID=$SID not ready"
    continue
  }

  LD_PKL="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/LD_stats/LD_stats_window_${WIN}.pkl"
  if [[ -s "$LD_PKL" ]]; then
    echo "[SKIP] SID=$SID WIN=$WIN (LD exists: $LD_PKL)"
    continue
  fi

  TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/windows/window_${WIN}.vcf.gz"
  echo "→ build SID=$SID WIN=$WIN  ($TARGET)"

  snakemake --snakefile "$SNAKEFILE" \
            --directory "$ROOT" \
            --nolock \
            --rerun-incomplete \
            --allowed-rules simulate_window ld_window \
            --latency-wait 300 \
            -j "$SLURM_CPUS_PER_TASK" \
            "$TARGET"
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

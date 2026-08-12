#!/bin/bash
#SBATCH --job-name=materialize_win
#SBATCH --output=logs/materialize_win_%A_%a.out
#SBATCH --error=logs/materialize_win_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# One task per simulation: writes tree_sequence.trees to one bgzipped,
# indexed whole-genome VCF (rule materialize_sim_vcf, see
# src.windowing.materialize_full_vcf). Must complete before
# build_windows.sh's chunk_window slicing stage runs for the same
# simulations (see master_script.sh's stage order) — chunk_window only
# slices an already-materialized file, it does not build materialize_sim_vcf
# itself, so those windows would fail to resolve if this hasn't run yet.

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
SNAKEFILE="$ROOT/Snakefile"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws' "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")
WINDOW_MODE=$(jq -r '.window_mode // "replicates"' "$CFG")

BATCH_SIZE="${BATCH_SIZE:-1}"
TOTAL_TASKS=$NUM_DRAWS

echo "CFG=$CFG  MODEL=$MODEL  WINDOW_MODE=$WINDOW_MODE  NUM_DRAWS=$NUM_DRAWS  TOTAL_TASKS=$TOTAL_TASKS"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

if [[ "$WINDOW_MODE" != "chunked" ]]; then
  echo "window_mode=$WINDOW_MODE (not 'chunked') — nothing to materialize, skipping."
  exit 0
fi

# --- first launch: resubmit with correct array range ---
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NUM_ARRAY=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
  echo "Submitting array 0..${NUM_ARRAY}"
  sbatch --array=0-"$NUM_ARRAY" "$0" "$@"
  exit 0
fi

# --- slice work for this array task ---
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
(( END >= TOTAL_TASKS )) && END=$(( TOTAL_TASKS - 1 ))

echo "Array $SLURM_ARRAY_TASK_ID → SID $START .. $END"

for SID in $(seq "$START" "$END"); do
  SIM_DIR="$ROOT/experiments/$MODEL/simulations/$SID"
  [[ -f "$SIM_DIR/.done" && -f "$SIM_DIR/tree_sequence.trees" ]] || {
    echo "[SKIP] SID=$SID not ready (run running_simulation.sh first)"
    continue
  }

  FULL_VCF="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/windows/full_genome.vcf.gz"
  if [[ -s "$FULL_VCF" ]]; then
    echo "[SKIP] SID=$SID (full_genome.vcf.gz already exists)"
    continue
  fi

  TARGET="experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/windows/full_genome.vcf.gz"
  echo "→ materialize SID=$SID  ($TARGET)"

  snakemake --snakefile "$SNAKEFILE" \
            --directory "$ROOT" \
            --nolock \
            --rerun-incomplete \
            --allowed-rules materialize_sim_vcf \
            --latency-wait 300 \
            -j "$SLURM_CPUS_PER_TASK" \
            "$TARGET"
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

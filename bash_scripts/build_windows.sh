#!/bin/bash
#SBATCH --job-name=build_win
#SBATCH --output=logs/build_win_%A_%a.out
#SBATCH --error=logs/build_win_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# --- config ---
ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
SNAKEFILE="$ROOT/Snakefile"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
export EXP_CFG="$CFG"

NUM_DRAWS=$(jq -r '.num_draws' "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")
WINDOW_MODE=$(jq -r '.window_mode // "replicates"' "$CFG")

NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG")
BATCH_SIZE="${BATCH_SIZE:-50}"
MAX_CONCURRENT="${MAX_CONCURRENT:-200}"

TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))

# window_mode="replicates": simulate_window_replicate independently
# re-simulates each window, so only sampled_params.pkl/.done (from an
# earlier running_simulation.sh stage) need to be ready.
# window_mode="chunked": chunk_window slices a per-simulation, already-
# materialized whole-genome VCF (bash_scripts/materialize_windows.sh must
# run first — see master_script.sh's stage order) via bcftools view -r.
# Each window is its own independent, parallel-safe job again: slicing reads
# a finished file, unlike the tree-sequence approach it replaced.
if [[ "$WINDOW_MODE" == "chunked" ]]; then
  BUILD_RULE="chunk_window"
else
  BUILD_RULE="simulate_window_replicate"
fi

echo "CFG=$CFG  MODEL=$MODEL  WINDOW_MODE=$WINDOW_MODE  NUM_DRAWS=$NUM_DRAWS  NUM_WINDOWS=$NUM_WINDOWS  TOTAL_TASKS=$TOTAL_TASKS"
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
  if [[ "$WINDOW_MODE" == "chunked" ]]; then
    FULL_VCF="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/windows/full_genome.vcf.gz"
    [[ -f "$FULL_VCF" ]] || {
      echo "[SKIP] SID=$SID full_genome.vcf.gz not ready (run materialize_windows.sh first)"
      continue
    }
  fi

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
            --allowed-rules "$BUILD_RULE" ld_window \
            --latency-wait 300 \
            -j "$SLURM_CPUS_PER_TASK" \
            "$TARGET"
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

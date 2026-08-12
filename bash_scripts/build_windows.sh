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

# window_mode="replicates": simulate_window_replicate independently
# re-simulates each window, so task granularity stays one (sim,window) pair.
# window_mode="chunked": chunk_window now loads tree_sequence.trees ONCE per
# simulation and writes every window from that single load (see Snakefile),
# instead of reloading + rescanning the whole tree sequence once per window.
# Task granularity here is therefore one whole SIMULATION (all its windows
# built in one snakemake call), not one (sim,window) pair.
if [[ "$WINDOW_MODE" == "chunked" ]]; then
  BUILD_RULE="chunk_window"
  TOTAL_TASKS=$NUM_DRAWS
else
  BUILD_RULE="simulate_window_replicate"
  TOTAL_TASKS=$(( NUM_DRAWS * NUM_WINDOWS ))
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

if [[ "$WINDOW_MODE" == "chunked" ]]; then
  # One task = one simulation's full set of windows, built in a single
  # snakemake call (one tree-sequence load) instead of NUM_WINDOWS calls.
  for SID in $(seq "$START" "$END"); do
    SIM_DIR="$ROOT/experiments/$MODEL/simulations/$SID"
    [[ -f "$SIM_DIR/.done" && -f "$SIM_DIR/sampled_params.pkl" ]] || {
      echo "[SKIP] SID=$SID not ready"
      continue
    }
    [[ -f "$SIM_DIR/tree_sequence.trees" ]] || {
      echo "[SKIP] SID=$SID tree_sequence.trees not ready (run running_simulation.sh first)"
      continue
    }

    # Skip entirely if every window's LD stats are already done — nothing
    # left needs the raw (temp) window vcfs for this sim.
    ALL_DONE=1
    for WIN in $(seq 0 $(( NUM_WINDOWS - 1 ))); do
      LD_PKL="$ROOT/experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/LD_stats/LD_stats_window_${WIN}.pkl"
      [[ -s "$LD_PKL" ]] || { ALL_DONE=0; break; }
    done
    if (( ALL_DONE )); then
      echo "[SKIP] SID=$SID (all $NUM_WINDOWS windows' LD stats already exist)"
      continue
    fi

    TARGETS=()
    for WIN in $(seq 0 $(( NUM_WINDOWS - 1 ))); do
      TARGETS+=("experiments/${MODEL}/inferences/sim_${SID}/MomentsLD/windows/window_${WIN}.vcf.gz")
    done
    echo "→ build SID=$SID  (${#TARGETS[@]} windows, one tree-sequence load)"

    snakemake --snakefile "$SNAKEFILE" \
              --directory "$ROOT" \
              --nolock \
              --rerun-incomplete \
              --allowed-rules "$BUILD_RULE" ld_window \
              --latency-wait 300 \
              -j "$SLURM_CPUS_PER_TASK" \
              "${TARGETS[@]}"
  done
else
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
              --allowed-rules "$BUILD_RULE" ld_window \
              --latency-wait 300 \
              -j "$SLURM_CPUS_PER_TASK" \
              "$TARGET"
  done
fi

echo "Array task $SLURM_ARRAY_TASK_ID finished."

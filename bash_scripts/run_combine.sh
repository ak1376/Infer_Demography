#!/bin/bash
#SBATCH --job-name=combine_inf
#SBATCH --output=logs/combine_%A_%a.out
#SBATCH --error=logs/combine_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail

# -----------------------------
# Tunables
# -----------------------------
BATCH_SIZE="${BATCH_SIZE:-2}"          # sims per array task
SIM_RANGE="${SIM_RANGE:-}"             # optional: "5000-20000"

# -----------------------------
# Paths & config
# -----------------------------
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_drosophila_three_epoch.json"
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")

mkdir -p logs

# -----------------------------
# Parse SIM_RANGE (optional)
# -----------------------------
SIM_LO=0
SIM_HI=$(( NUM_DRAWS - 1 ))

if [[ -n "$SIM_RANGE" ]]; then
  if [[ "$SIM_RANGE" =~ ^[0-9]+-[0-9]+$ ]]; then
    SIM_LO="${SIM_RANGE%-*}"
    SIM_HI="${SIM_RANGE#*-}"
  else
    echo "ERROR: SIM_RANGE must look like '5000-20000'"
    exit 2
  fi

  (( SIM_LO < 0 )) && SIM_LO=0
  (( SIM_HI > NUM_DRAWS-1 )) && SIM_HI=$(( NUM_DRAWS - 1 ))
  (( SIM_LO > SIM_HI )) && exit 2
fi

# -----------------------------
# Self-submit if not array job
# -----------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  START_BATCH=$(( SIM_LO / BATCH_SIZE ))
  END_BATCH=$(( SIM_HI / BATCH_SIZE ))

  echo "Submitting array ${START_BATCH}-${END_BATCH} for sims ${SIM_LO}-${SIM_HI}"
  sbatch --array="${START_BATCH}-${END_BATCH}" --export=ALL "$0" "$@"
  exit 0
fi

# -----------------------------
# Compute batch slice
# -----------------------------
BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
BATCH_END=$(( (SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1 ))
(( BATCH_END >= NUM_DRAWS )) && BATCH_END=$(( NUM_DRAWS - 1 ))

if (( BATCH_END < SIM_LO || BATCH_START > SIM_HI )); then
  echo "Array $SLURM_ARRAY_TASK_ID outside requested range — nothing to do."
  exit 0
fi

RUN_START=$BATCH_START
RUN_END=$BATCH_END
(( RUN_START < SIM_LO )) && RUN_START=$SIM_LO
(( RUN_END   > SIM_HI )) && RUN_END=$SIM_HI

echo "Array $SLURM_ARRAY_TASK_ID → sims $RUN_START .. $RUN_END"

# -----------------------------
# Allowed rules
# -----------------------------
ALLOWED_RULES=(
  infer_dadi
  aggregate_opts_dadi
  infer_moments
  aggregate_opts_moments
  ld_window
  optimize_momentsld
  compute_fim
  sfs_residuals
  combine_results
)

# -----------------------------
# Loop over sims in batch
# -----------------------------
for sid in $(seq "$RUN_START" "$RUN_END"); do
  echo "combine_results: sid=$sid"

  SFS_FILE="experiments/${MODEL}/simulations/${sid}/SFS.pkl"
  TARGET_COMBO="experiments/${MODEL}/inferences/sim_${sid}/all_inferences.pkl"
  TARGET_DADI="experiments/${MODEL}/inferences/sim_${sid}/dadi/fit_params.pkl"
  TARGET_MOM="experiments/${MODEL}/inferences/sim_${sid}/moments/fit_params.pkl"
  TARGET_LD="experiments/${MODEL}/inferences/sim_${sid}/MomentsLD/best_fit.pkl"

  if [[ ! -f "$SFS_FILE" ]]; then
    echo "ERROR: Missing simulation for sim_${sid}"
    continue
  fi

  snakemake --cleanup-metadata \
    "experiments/${MODEL}/simulations/${sid}/SFS.pkl" \
    "experiments/${MODEL}/simulations/${sid}/sampled_params.pkl" \
    "experiments/${MODEL}/simulations/${sid}/tree_sequence.trees" \
    "experiments/${MODEL}/simulations/${sid}/demes.png" \
    "experiments/${MODEL}/simulations/${sid}/bgs.meta.json" \
    "experiments/${MODEL}/simulations/${sid}/.done" \
    2>/dev/null || true

  # if [[ -f "$TARGET_COMBO" ]]; then
  #   echo "sim_${sid} already complete — skipping"
  #   continue
  # fi

  snakemake \
    -j "$SLURM_CPUS_PER_TASK" \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --forcerun combine_results \
    --nolock \
    --allowed-rules "${ALLOWED_RULES[@]}" \
    --keep-going \
    "$TARGET_COMBO" \
    || { echo "Snakemake failed for sid=$sid"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."
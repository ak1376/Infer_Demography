#!/bin/bash
#SBATCH --job-name=sfsres
#SBATCH --output=logs/sfs_residuals_%A_%a.out
#SBATCH --error=logs/sfs_residuals_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail
mkdir -p logs

# -----------------------------
# Tunables
# -----------------------------
BATCH_SIZE="${BATCH_SIZE:-1}"          # sims per array task
SIM_RANGE="${SIM_RANGE:-}"             # optional: "5000-20000"

# -----------------------------
# Paths & config
# -----------------------------
ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'          "$CFG")
MODEL=$(jq -r '.demographic_model'      "$CFG")

# SFS residuals are always computed for whichever engines the config lists
# (default: both moments+dadi) -- whether they're later used as a model
# feature is a separate decision made downstream in feature_extraction.py
# via --use-residuals.
mapfile -t ENGINES < <(jq -r '
  (.residual_engines // "both") as $r
  | if ($r|type) == "array" then
      ($r | map(select(. == "moments" or . == "dadi")))
      | if length==0 then ["moments","dadi"] else . end | .[]
    else
      ($r | ascii_downcase) as $v
      | if ($v == "both" or $v == "all") then ("moments","dadi") else $v end
    end
' "$CFG")

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

echo "Array $SLURM_ARRAY_TASK_ID → sims $RUN_START .. $RUN_END  engines: ${ENGINES[*]}"

# -----------------------------
# Loop over sims in batch
# -----------------------------
for sid in $(seq "$RUN_START" "$RUN_END"); do
  declare -a TARGETS=()

  for eng in "${ENGINES[@]}"; do
    fit="$ROOT/experiments/${MODEL}/inferences/sim_${sid}/${eng}/fit_params.pkl"
    if [[ -f "$fit" ]]; then
      TARGETS+=("experiments/${MODEL}/inferences/sim_${sid}/sfs_residuals/${eng}/residuals_flat.npy")
    else
      echo "SKIP sim_${sid} engine=${eng} (missing fit: $fit)"
    fi
  done

  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "No SFS residual targets for sim_${sid} — nothing to run."
    continue
  fi

  snakemake -j "${SLURM_CPUS_PER_TASK:-1}" \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --rerun-incomplete \
    --nolock \
    --allowed-rules sfs_residuals \
    --keep-going \
    "${TARGETS[@]}" \
    || { echo "Snakemake failed for sid=$sid"; exit 1; }
done

echo "Array task $SLURM_ARRAY_TASK_ID finished."

#!/bin/bash
#SBATCH --job-name=real_calib_sim
#SBATCH --output=logs/real_calib_sim_%A_%a.out
#SBATCH --error=logs/real_calib_sim_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# SFS calibration check for ONE real-data per-arm optimization run: simulate
# the arm at that run's fitted params (its sfs_fit.json), one replicate per
# SLURM array task, using the config's sample_ploidy and recombination map
# (same as every simulation) plus the observed SFS's sample sizes and
# kept-site fraction. Same self-resubmitting array pattern as
# bash_scripts/simulation/calibration_simulate.sh. Then run
# real_calibration_ppc.sh once every array task has finished.
#
# Rule run (once per array task): calibration_simulate_fit
#
# Usage (OPT is required -- which optimization run to check):
#   OPT=11 sbatch bash_scripts/real_data/real_calibration_simulate.sh
#   ARM=Chr3L ENGINE=moments OPT=11 sbatch bash_scripts/real_data/real_calibration_simulate.sh
#
# Time/mem above are per replicate (one whole-arm msprime simulation).

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

load_real_data_config "$CFG"   # sets MODEL, REAL_FIT_ROOT, ...
N_REPS=$(jq -r '.calibration_n_replicates // 20' "$CFG")

ARM="${ARM:-Chr3L}"
ENGINE="${ENGINE:-moments}"
OPT="${OPT:?set OPT to the optimization run to check, e.g. OPT=11}"

# First launch (no array id yet): resubmit as an array sized from
# calibration_n_replicates, one task per replicate.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Submitting array 0..$((N_REPS - 1)) (calibration_n_replicates=${N_REPS} from ${CFG})"
    ARM="$ARM" ENGINE="$ENGINE" OPT="$OPT" ROOT="$ROOT" \
        sbatch --array=0-"$((N_REPS - 1))" "$0" "$@"
    exit 0
fi

REP="$SLURM_ARRAY_TASK_ID"
TARGET="${REAL_FIT_ROOT}/${ARM}/calibration/${ENGINE}_run${OPT}/replicate_${REP}/SFS.pkl"

echo "MODEL=$MODEL  ARM=$ARM  ENGINE=$ENGINE  OPT=$OPT  replicate=$REP"
echo "Target: $TARGET"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules calibration_simulate_fit \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    "$TARGET"

echo "calibration_simulate_fit finished -> $TARGET"

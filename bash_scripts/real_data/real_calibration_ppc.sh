#!/bin/bash
#SBATCH --job-name=real_calib_ppc
#SBATCH --output=logs/real_calib_ppc_%j.out
#SBATCH --error=logs/real_calib_ppc_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# SFS calibration check plots + summary for ONE real-data per-arm optimization
# run: compares pi / Tajima's D / FST and SFS shape between the arm's observed
# SFS and the calibration_simulate_fit replicates. Requires every replicate
# from real_calibration_simulate.sh (same ARM/ENGINE/OPT) to have finished.
#
# Rule run: calibration_ppc_fit
#
# Usage:
#   OPT=11 sbatch bash_scripts/real_data/real_calibration_ppc.sh

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

load_real_data_config "$CFG"   # sets MODEL, REAL_FIT_ROOT, ...

ARM="${ARM:-Chr3L}"
ENGINE="${ENGINE:-moments}"
OPT="${OPT:?set OPT to the optimization run to check, e.g. OPT=11}"

TARGET="${REAL_FIT_ROOT}/${ARM}/calibration/${ENGINE}_run${OPT}/ppc/calibration_ppc.png"

echo "MODEL=$MODEL  ARM=$ARM  ENGINE=$ENGINE  OPT=$OPT"
echo "Target: $TARGET"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules calibration_ppc_fit \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    "$TARGET"

echo "calibration_ppc_fit finished -> $TARGET"

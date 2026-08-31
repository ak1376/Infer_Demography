#!/bin/bash
#SBATCH --job-name=real_momld_prep
#SBATCH --output=logs/real_momld_prep_%j.out
#SBATCH --error=logs/real_momld_prep_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage D1: build means.varcovs.pkl (+ bootstrap sets) ONCE, before the
# multi-restart MomentsLD optimization array (real_momentsld.sh) runs --
# mirrors MomentsLD_prep.sh for the simulated pipeline, so concurrent opts
# never race on building this shared file. Requires real_ld_windows.sh to
# have already produced every LD_stats_window_*.pkl.
#
# Rule run: aggregate_ld_windows_real

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")
REAL_LD_ROOT="experiments/${MODEL}/real_data_analysis/inferences/MomentsLD"

TARGET="${REAL_LD_ROOT}/means.varcovs.pkl"
echo "MODEL=$MODEL"
echo "Target: $TARGET"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules aggregate_ld_windows_real \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    "$TARGET"

echo "real_momentsld_prep finished."

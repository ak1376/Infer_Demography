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
#
# REAL_LD_ROOT is NOT model-scoped (must match the Snakefile's REAL_LD_ROOT =
# f"{DROSO_DIR}/MomentsLD") -- the aggregated means/varcovs/bootstrap are a
# pure function of the (already model-independent) per-window LD stats, so
# they're computed once and reused across every demographic model. The one
# cosmetic side effect: empirical_vs_theoretical_comparison.pdf's theoretical
# curve reflects whichever model's config was active the first time this
# target was built -- with --rerun-triggers mtime, switching MODEL later
# won't regenerate it (the file already exists and its declared inputs
# haven't changed), so it may go stale as a diagnostic plot. Delete it
# manually if you want it to reflect the current model.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

# Must match the Snakefile's DROSO_DIR / REAL_LD_ROOT constants.
DROSO_DIR="real_data_analysis/data/drosophila"
REAL_LD_ROOT="${DROSO_DIR}/MomentsLD"

TARGET="${REAL_LD_ROOT}/means.varcovs.pkl"
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

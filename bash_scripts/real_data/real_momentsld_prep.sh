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
# Rule run (pooled):     aggregate_ld_windows_real (one combined fit across
#                         every configured arm)
# Rule run (individual): aggregate_ld_windows_real_by_arm, once per arm (no
#                         cross-arm combining)
#
# REAL_LD_ROOT is NOT model-scoped -- the aggregated means/varcovs/bootstrap
# are a pure function of the (already model-independent) per-window LD
# stats, so they're computed once and reused across every demographic model.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

load_real_data_config "$CFG"

if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    TARGETS=("${REAL_LD_ROOT}/means.varcovs.pkl")
    ALLOWED_RULES=(aggregate_ld_windows_real)
else
    TARGETS=()
    for arm in "${REAL_ARMS[@]}"; do
        TARGETS+=("${REAL_LD_ROOT}/${arm}/means.varcovs.pkl")
    done
    ALLOWED_RULES=(aggregate_ld_windows_real_by_arm)
fi

echo "MODEL=$MODEL  REAL_POOLING_MODE=$REAL_POOLING_MODE"
echo "Targets: ${TARGETS[*]}"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules "${ALLOWED_RULES[@]}" \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    "${TARGETS[@]}"

echo "real_momentsld_prep finished."

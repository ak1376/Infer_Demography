#!/bin/bash
#SBATCH --job-name=real_aggregate_momld
#SBATCH --output=logs/real_aggregate_momld_%j.out
#SBATCH --error=logs/real_aggregate_momld_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage D3: pick the top-k MomentsLD restart across all num_optimizations
# real_momentsld.sh array slots. Separate job so D2's array can be pure
# per-restart parallelism (mirrors aggregate_momentsld.sh being separate
# from MomentsLD.sh for the simulated pipeline).
#
# Rule run: aggregate_opts_momentsld_real
#
# Unlike the shared REAL_LD_ROOT (windows/LD_stats/means.varcovs.pkl), this
# output IS model-scoped (experiments/{MODEL}/...): it's the fitted MomentsLD
# params under the active demographic model, not a property of the data.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")
TARGET="experiments/${MODEL}/real_data_analysis/inferences/MomentsLD/best_fit.pkl"

echo "MODEL=$MODEL"
echo "Target: $TARGET"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules aggregate_opts_momentsld_real \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    "$TARGET"

echo "real_aggregate_momentsld finished."

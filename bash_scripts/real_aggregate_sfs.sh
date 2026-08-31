#!/bin/bash
#SBATCH --job-name=real_aggregate_sfs
#SBATCH --output=logs/real_aggregate_sfs_%j.out
#SBATCH --error=logs/real_aggregate_sfs_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage B2: pick the top-k moments/dadi restart across all num_optimizations
# real_sfs_inference.sh array slots. Separate job so B1's array can be pure
# per-restart parallelism (mirrors aggregate_moments_dadi.sh being separate
# from moments.sh/dadi.sh for the simulated pipeline).
#
# Rule run: aggregate_opts_engine_real

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")
REAL_INF_ROOT="experiments/${MODEL}/real_data_analysis/inferences"

TARGETS=(
    "${REAL_INF_ROOT}/moments/best_fit.pkl"
    "${REAL_INF_ROOT}/dadi/best_fit.pkl"
)

echo "MODEL=$MODEL"
echo "Targets: ${TARGETS[*]}"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules aggregate_opts_engine_real \
    -j "${SLURM_CPUS_PER_TASK:-1}" \
    "${TARGETS[@]}"

echo "real_aggregate_sfs finished."

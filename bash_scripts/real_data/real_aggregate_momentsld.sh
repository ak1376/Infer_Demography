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
# Rule run (pooled):     aggregate_opts_momentsld_real
# Rule run (individual): aggregate_opts_momentsld_real_by_arm, once per arm
#
# Output IS model-scoped (experiments/{MODEL}/...): it's the fitted
# MomentsLD params under the active demographic model, not a property of
# the data.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

load_real_data_config "$CFG"

if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    TARGETS=("experiments/${MODEL}/real_data_analysis${TRIM_SUFFIX}/inferences/${REAL_LD_ENGINE}/best_fit.pkl")
    ALLOWED_RULES=(aggregate_opts_momentsld_real)
else
    TARGETS=()
    for arm in "${REAL_ARMS[@]}"; do
        INF_ROOT_ARM="$(real_data_chrom_path "$REAL_INF_ROOT_CHROM_TMPL" "$arm")"
        TARGETS+=("${INF_ROOT_ARM}/MomentsLD/best_fit.pkl")
    done
    ALLOWED_RULES=(aggregate_opts_momentsld_real_by_arm)
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

echo "real_aggregate_momentsld finished."

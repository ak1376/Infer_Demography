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
# Rule run (pooled):     aggregate_opts_engine_real
# Rule run (individual): aggregate_opts_engine_real_chrom, once per arm

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

load_real_data_config "$CFG"

# Same engine selection as real_sfs_inference.sh, e.g. ENGINES=moments
read -r -a ENGINE_LIST <<< "${ENGINES:-moments dadi}"
for e in "${ENGINE_LIST[@]}"; do
    [[ "$e" == moments || "$e" == dadi ]] || { echo "ENGINES must be moments and/or dadi, got: $e" >&2; exit 1; }
done

if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    TARGETS=()
    for engine in "${ENGINE_LIST[@]}"; do
        TARGETS+=("${REAL_INF_ROOT}/${engine}/best_fit.pkl")
    done
    ALLOWED_RULES=(aggregate_opts_engine_real)
else
    TARGETS=()
    for arm in "${REAL_ARMS[@]}"; do
        INF_ROOT_ARM="$(real_data_chrom_path "$REAL_INF_ROOT_CHROM_TMPL" "$arm")"
        for engine in "${ENGINE_LIST[@]}"; do
            TARGETS+=("${INF_ROOT_ARM}/${engine}/best_fit.pkl")
        done
    done
    ALLOWED_RULES=(aggregate_opts_engine_real_chrom)
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

echo "real_aggregate_sfs finished."

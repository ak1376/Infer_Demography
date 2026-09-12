#!/bin/bash
#SBATCH --job-name=real_prep
#SBATCH --output=logs/real_prep_%j.out
#SBATCH --error=logs/real_prep_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage A of the real-data (Drosophila) pipeline: polarize each configured
# arm's raw VCF against the DPGP ancestor, recode to diploid GTs (needed by
# the MomentsLD-real LD stage), and build each arm's unfolded SFS. In
# pooling_mode="pooled", also sums the autosomal arms' SFS into the combined
# SFS that the pooled moments/dadi fit uses; in "individual" mode each arm's
# own SFS is used directly (by real_sfs_inference.sh's per-chrom targets)
# and the combined SFS is skipped.
#
# Rules run: annotate_ancestral_allele, recode_polarized_to_diploid,
#            compute_unfolded_sfs, and (pooled only) combine_autosomal_sfs
# Arms are config-driven (real_data_analysis.arms), not hardcoded.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

load_real_data_config "$CFG"

TARGETS=()
for arm in "${REAL_ARMS[@]}"; do
    TARGETS+=("${DROSO_DIR}/${arm}/polarized.diploidGT.vcf.gz")
    TARGETS+=("${DROSO_DIR}/${arm}/polarized.diploidGT.vcf.gz.tbi")
    TARGETS+=("${DROSO_DIR}/${arm}/unfolded.sfs.pkl")
    TARGETS+=("${DROSO_DIR}/${arm}/unfolded.sfs.meta.json")
done

ALLOWED_RULES=(annotate_ancestral_allele recode_polarized_to_diploid compute_unfolded_sfs)
if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    TARGETS+=("${DROSO_DIR}/combined/autosomes.unfolded.sfs.pkl")
    TARGETS+=("${DROSO_DIR}/combined/autosomes.unfolded.sfs.meta.json")
    ALLOWED_RULES+=(combine_autosomal_sfs)
fi

echo "MODEL=$MODEL  REAL_ARMS=${REAL_ARMS[*]}  REAL_POOLING_MODE=$REAL_POOLING_MODE"
echo "Targets: ${TARGETS[*]}"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules "${ALLOWED_RULES[@]}" \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "${TARGETS[@]}"

echo "real_data_prep finished."

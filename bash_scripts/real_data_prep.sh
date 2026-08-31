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

# Stage A of the real-data (Drosophila) pipeline: polarize the raw per-
# chromosome VCFs against the DPGP ancestor, recode Chr2L to diploid GTs
# (needed by the MomentsLD-real LD stage), build each autosome's unfolded
# SFS, and sum them into the combined-autosome SFS that all downstream
# real-data SFS inference (moments/dadi) fits against.
#
# Rules run: annotate_ancestral_allele, recode_polarized_to_diploid,
#            compute_unfolded_sfs, combine_autosomal_sfs
# Autosomes are fixed by the Snakefile's AUTOSOMES list (currently
# Chr2L, Chr3L) -- not looped here.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")

# Must match the Snakefile's DROSO_DIR / REAL_VCF constants.
DROSO_DIR="real_data_analysis/data/drosophila"
COMBINED_SFS="${DROSO_DIR}/combined/autosomes.unfolded.sfs.pkl"
COMBINED_SFS_META="${DROSO_DIR}/combined/autosomes.unfolded.sfs.meta.json"
CHR2L_DIPLOID_VCF="${DROSO_DIR}/Chr2L/polarized.diploidGT.vcf.gz"
CHR2L_DIPLOID_TBI="${CHR2L_DIPLOID_VCF}.tbi"

echo "MODEL=$MODEL"
echo "Targets: $COMBINED_SFS $COMBINED_SFS_META $CHR2L_DIPLOID_VCF"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules annotate_ancestral_allele recode_polarized_to_diploid compute_unfolded_sfs combine_autosomal_sfs \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "$COMBINED_SFS" "$COMBINED_SFS_META" "$CHR2L_DIPLOID_VCF" "$CHR2L_DIPLOID_TBI"

echo "real_data_prep finished."

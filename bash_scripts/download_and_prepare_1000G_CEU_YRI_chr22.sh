#!/usr/bin/env bash
set -euo pipefail

########################################
# Settings — edit if needed
########################################

CHR=22                     # small chromosome
POP1="CEU"
POP2="YRI"

if [ "$#" -ge 1 ]; then
    OUTDIR="$1"
else
    OUTDIR="/sietch_colab/akapoor/Infer_Demography/test_real_data_analysis/split_migration_growth/data_chr${CHR}_${POP1}_${POP2}"
fi
mkdir -p "${OUTDIR}"

FTP_BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
VCF_BASENAME="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
PANEL_FILE="integrated_call_samples_v3.20130502.ALL.panel"

echo "Output directory: ${OUTDIR}"
cd "${OUTDIR}"

########################################
# 1. Download VCF + index + panel
########################################

if [[ ! -f "${VCF_BASENAME}" ]]; then
    echo "Downloading chr${CHR} VCF..."
    wget "${FTP_BASE}/${VCF_BASENAME}"
fi

if [[ ! -f "${VCF_BASENAME}.tbi" ]]; then
    echo "Downloading TBI index..."
    wget "${FTP_BASE}/${VCF_BASENAME}.tbi"
fi

if [[ ! -f "${PANEL_FILE}" ]]; then
    echo "Downloading panel file..."
    wget "${FTP_BASE}/${PANEL_FILE}"
fi

########################################
# 1b. Download GENCODE v19 GTF (hg19) for exon filtering
########################################

GTF_URL="ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz"
GTF_FILE="gencode.v19.annotation.gtf.gz"
EXON_BED="chr${CHR}_exons.bed"

if [[ ! -f "${GTF_FILE}" ]]; then
    echo "Downloading GENCODE v19 GTF..."
    wget "${GTF_URL}"
fi

if [[ ! -f "${EXON_BED}" ]]; then
    echo "Extracting chr${CHR} exons to BED..."
    # Extract chr22, filter for exons, convert to BED (0-based start), strip "chr" prefix from chrom name to match VCF
    zgrep "^chr${CHR}\b" "${GTF_FILE}" | \
    awk '$3=="exon" {print substr($1, 4) "\t" ($4-1) "\t" $5}' > "${EXON_BED}"
fi

########################################
# 2. Create sample lists
########################################

echo "Extracting ${POP1} and ${POP2} sample IDs..."

grep -P "\t${POP1}\t" "${PANEL_FILE}" | cut -f1 > "${POP1}.samples"
grep -P "\t${POP2}\t" "${PANEL_FILE}" | cut -f1 > "${POP2}.samples"

echo "Sample counts:"
wc -l "${POP1}.samples" "${POP2}.samples"

cat "${POP1}.samples" "${POP2}.samples" > "merged.samples"

########################################
# 3. Extract chr22 region with bcftools (removing exons)
########################################

MERGED_VCF="CEU_YRI.chr${CHR}.no_exons.vcf.gz"

echo "Extracting chromosome ${CHR} for CEU and YRI (excluding exons)..."

bcftools view \
    --samples-file merged.samples \
    --regions ${CHR} \
    --targets-file ^${EXON_BED} \
    -Oz \
    -o "${MERGED_VCF}" \
    "${VCF_BASENAME}"

########################################
# 4. Index merged VCF
########################################

echo "Indexing merged VCF..."
tabix -p vcf "${MERGED_VCF}"

########################################
# 5. Create popfile for dadi/moments
########################################

echo "Creating population file..."

POPFILE="${POP1}_${POP2}.popfile"

# Create popfile with format: sample_id<tab>population_name
awk -v pop="${POP1}" '{print $1 "\t" pop}' "${POP1}.samples" > "${POPFILE}"
awk -v pop="${POP2}" '{print $1 "\t" pop}' "${POP2}.samples" >> "${POPFILE}"

echo "Created popfile: ${POPFILE} ($(wc -l < ${POPFILE}) samples)"

echo
echo "DONE!"
echo "Merged 2-population VCF ready:"
echo "  ${OUTDIR}/${MERGED_VCF}"
echo
echo "Population file:"
echo "  ${OUTDIR}/${POPFILE}"
echo
echo "You can now build the joint 2D SFS using:"
echo
echo "  python snakemake_scripts/real_data_sfs.py \\"
echo "    --input-vcf ${OUTDIR}/${MERGED_VCF} \\"
echo "    --popfile ${OUTDIR}/${POPFILE} \\"
echo "    --config config_files/experiment_config_split_isolation.json \\"
echo "    --output-sfs ${OUTDIR}/${POP1}_${POP2}.chr${CHR}.no_exons.sfs.pkl"
echo


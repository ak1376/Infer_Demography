#!/usr/bin/env bash
set -euo pipefail

########################################
# Settings — edit if needed
########################################

CHR=1                     # small chromosome

# Three populations
POP1="YRI"
POP2="CEU"
POP3="CHB"

# Toggle: set to "true" to remove exons; "false" to keep everything
REMOVE_EXONS="true"

if [ "$#" -ge 1 ]; then
    OUTDIR="$1"
else
    OUTDIR="/sietch_colab/akapoor/Infer_Demography/test_real_data_analysis/OOA_three_pop/data_chr${CHR}_${POP1}_${POP2}_${POP3}"
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
else
    echo "VCF ${VCF_BASENAME} already exists, skipping download."
fi

if [[ ! -f "${VCF_BASENAME}.tbi" ]]; then
    echo "Downloading TBI index..."
    wget "${FTP_BASE}/${VCF_BASENAME}.tbi"
else
    echo "Index ${VCF_BASENAME}.tbi already exists, skipping download."
fi

if [[ ! -f "${PANEL_FILE}" ]]; then
    echo "Downloading panel file..."
    wget "${FTP_BASE}/${PANEL_FILE}"
else
    echo "Panel file ${PANEL_FILE} already exists, skipping download."
fi

########################################
# 1b. (Optional) Download GENCODE v19 GTF (hg19) for exon filtering
########################################

GTF_URL="ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz"
GTF_FILE="gencode.v19.annotation.gtf.gz"
EXON_BED="chr${CHR}_exons.bed"

if [[ "${REMOVE_EXONS}" == "true" ]]; then
    if [[ ! -f "${GTF_FILE}" ]]; then
        echo "Downloading GENCODE v19 GTF..."
        wget "${GTF_URL}"
    else
        echo "GTF ${GTF_FILE} already exists, skipping download."
    fi

    if [[ ! -f "${EXON_BED}" ]]; then
        echo "Extracting chr${CHR} exons to BED..."
        # Extract chr${CHR}, filter for exons, convert to BED (0-based start), strip "chr"
        zgrep "^chr${CHR}\b" "${GTF_FILE}" | \
        awk '$3=="exon" {print substr($1, 4) "\t" ($4-1) "\t" $5}' > "${EXON_BED}"
    else
        echo "Exon BED ${EXON_BED} already exists, skipping creation."
    fi
else
    echo "REMOVE_EXONS=false -> skipping GENCODE download and exon BED creation."
fi

########################################
# 2. Create sample lists (YRI, CEU, CHB)
########################################

echo "Ensuring ${POP1}, ${POP2}, and ${POP3} sample ID files exist..."

if [[ ! -f "${POP1}.samples" ]]; then
    grep -P "\t${POP1}\t" "${PANEL_FILE}" | cut -f1 > "${POP1}.samples"
fi

if [[ ! -f "${POP2}.samples" ]]; then
    grep -P "\t${POP2}\t" "${PANEL_FILE}" | cut -f1 > "${POP2}.samples"
fi

if [[ ! -f "${POP3}.samples" ]]; then
    grep -P "\t${POP3}\t" "${PANEL_FILE}" | cut -f1 > "${POP3}.samples"
fi

echo "Sample counts:"
wc -l "${POP1}.samples" "${POP2}.samples" "${POP3}.samples"

if [[ ! -f "merged.samples" ]]; then
    cat "${POP1}.samples" "${POP2}.samples" "${POP3}.samples" > "merged.samples"
else
    echo "merged.samples already exists, skipping."
fi

########################################
# 3. Extract chr region with bcftools (optionally removing exons)
########################################

# File naming: add .no_exons if we excluded exons
SUFFIX=""
if [[ "${REMOVE_EXONS}" == "true" ]]; then
    SUFFIX=".no_exons"
fi

MERGED_VCF="${POP1}_${POP2}_${POP3}.chr${CHR}${SUFFIX}.vcf.gz"

echo "Preparing merged VCF name: ${MERGED_VCF}"

EXCLUDE_ARGS=()
if [[ "${REMOVE_EXONS}" == "true" ]]; then
    echo "Excluding exons using BED: ${EXON_BED}"
    EXCLUDE_ARGS=(--targets-file "^${EXON_BED}")
else
    echo "Keeping all sites (no exon filtering)."
fi

if [[ ! -f "${MERGED_VCF}" ]]; then
    echo "Extracting chromosome ${CHR} for ${POP1}, ${POP2}, ${POP3}..."
    bcftools view \
        --samples-file merged.samples \
        --regions ${CHR} \
        "${EXCLUDE_ARGS[@]}" \
        -Oz \
        -o "${MERGED_VCF}" \
        "${VCF_BASENAME}"
else
    echo "Merged VCF ${MERGED_VCF} already exists, skipping bcftools view."
fi

########################################
# 4. Index merged VCF
########################################

if [[ -f "${MERGED_VCF}" ]]; then
    if [[ ! -f "${MERGED_VCF}.tbi" && ! -f "${MERGED_VCF}.csi" ]]; then
        echo "Indexing merged VCF..."
        tabix -p vcf "${MERGED_VCF}"
    else
        echo "Index for ${MERGED_VCF} already exists, skipping tabix."
    fi
else
    echo "WARNING: ${MERGED_VCF} not found, cannot index."
fi

########################################
# 5. Create popfile for dadi/moments
########################################

echo "Ensuring population file exists..."

POPFILE="${POP1}_${POP2}_${POP3}.popfile"

if [[ ! -f "${POPFILE}" ]]; then
    # Create popfile with format: sample_id<tab>population_name
    awk -v pop="${POP1}" '{print $1 "\t" pop}' "${POP1}.samples" > "${POPFILE}"
    awk -v pop="${POP2}" '{print $1 "\t" pop}' "${POP2}.samples" >> "${POPFILE}"
    awk -v pop="${POP3}" '{print $1 "\t" pop}' "${POP3}.samples" >> "${POPFILE}"
    echo "Created popfile: ${POPFILE} ($(wc -l < ${POPFILE}) samples)"
else
    echo "Popfile ${POPFILE} already exists, skipping creation."
fi

echo
echo "DONE!"
echo "Merged 3-population VCF ready:"
echo "  ${OUTDIR}/${MERGED_VCF}"
echo
echo "Population file:"
echo "  ${OUTDIR}/${POPFILE}"
echo
echo "You can now build the joint 3D SFS using something like:"
echo
echo "  python snakemake_scripts/real_data_sfs.py \\"
echo "    --input-vcf ${OUTDIR}/${MERGED_VCF} \\"
echo "    --popfile ${OUTDIR}/${POPFILE} \\"
echo "    --config config_files/experiment_config_OOA_three_pop.json \\"
echo "    --output-sfs ${OUTDIR}/${POP1}_${POP2}_${POP3}.chr${CHR}${SUFFIX}.sfs.pkl"

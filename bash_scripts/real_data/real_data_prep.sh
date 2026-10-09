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
# Rules run: trim_raw_vcf_region, missing_data_kept_fraction (when
#            real_data_analysis.original_vcf is set), annotate_ancestral_allele,
#            make_pseudodiploid_pairs, recode_polarized_to_diploid,
#            build_genetic_map_real, compute_unfolded_sfs, and (pooled only)
#            combine_autosomal_sfs
# Prints the settings it runs with up front and a per-arm summary of what it
# produced at the end (even if some targets failed).
# Arms are config-driven (real_data_analysis.arms), not hardcoded.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

load_real_data_config "$CFG"

HAS_ORIGINAL_VCF=$(python -c "import json,sys; print(int(bool(json.load(open(sys.argv[1])).get('real_data_analysis', {}).get('original_vcf'))))" "$CFG")

TARGETS=()
for arm in "${REAL_ARMS[@]}"; do
    TARGETS+=("${DROSO_DIR}/${arm}/polarized.diploidGT${DIPLOID_SUFFIX}.phased.vcf.gz")
    TARGETS+=("${DROSO_DIR}/${arm}/polarized.diploidGT${DIPLOID_SUFFIX}.phased.vcf.gz.tbi")
    TARGETS+=("${GENMAP_DIR}/${arm}/genetic_map.txt")
    TARGETS+=("${DROSO_DIR}/${arm}/unfolded${SFS_SUFFIX}.sfs.pkl")
    TARGETS+=("${DROSO_DIR}/${arm}/unfolded${SFS_SUFFIX}.sfs.meta.json")
    # Requested explicitly: some Snakemake versions won't build a missing input
    # of an otherwise up-to-date SFS, so the L correction would never happen.
    # Once it's (re)built, the SFS meta is older than it and gets rebuilt too.
    if [[ "$HAS_ORIGINAL_VCF" == "1" ]]; then
        TARGETS+=("${DROSO_DIR}/${arm}/missing_data_kept_fraction.json")
    fi
done

ALLOWED_RULES=(trim_raw_vcf_region missing_data_kept_fraction annotate_ancestral_allele make_pseudodiploid_pairs
               recode_polarized_to_diploid build_genetic_map_real compute_unfolded_sfs)
if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    TARGETS+=("${DROSO_DIR}/combined/autosomes.unfolded${SFS_SUFFIX}.sfs.pkl")
    TARGETS+=("${DROSO_DIR}/combined/autosomes.unfolded${SFS_SUFFIX}.sfs.meta.json")
    ALLOWED_RULES+=(combine_autosomal_sfs)
fi

echo "================ real_data_prep settings ================"
echo "config:          $CFG"
echo "model:           $MODEL"
echo "arms:            ${REAL_ARMS[*]}   (pooling_mode=$REAL_POOLING_MODE)"
echo "processed data:  $DROSO_DIR"
python - "$CFG" "${REAL_ARMS[@]}" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1])); rd = cfg.get("real_data_analysis", {})
print(f"exclude_samples: {rd.get('exclude_samples') or 'none (all flies)'}")
orig = rd.get("original_vcf")
print(f"original_vcf:    {orig or 'NOT SET -> no missing-data correction of L'}")
for arm in sys.argv[2:]:
    tr = rd.get("trim_region", {}).get(arm)
    print(f"trim_region {arm}: {f'{tr[0]:,}-{tr[1]:,}' if tr else 'none (whole arm)'}")
EOF
echo "targets:"
printf '  %s\n' "${TARGETS[@]}"
echo "=========================================================="

set +e
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
SNAKEMAKE_STATUS=$?
set -e

echo "================ real_data_prep summary ================"
python - "$ROOT" "$DROSO_DIR" "$SFS_SUFFIX" "$DIPLOID_SUFFIX" "$GENMAP_DIR" "${REAL_ARMS[@]}" <<'EOF'
import json, pickle, sys
from pathlib import Path
root, droso, sfs_sfx, dip_sfx, genmap = sys.argv[1:6]
for arm in sys.argv[6:]:
    d = Path(root) / droso / arm
    print(f"--- {arm}")
    for f in [d / "polarized.vcf.gz", d / f"polarized.diploidGT{dip_sfx}.phased.vcf.gz",
              Path(root) / genmap / arm / "genetic_map.txt",
              d / "missing_data_kept_fraction.json",
              d / f"unfolded{sfs_sfx}.sfs.pkl", d / f"unfolded{sfs_sfx}.sfs.meta.json",
              d / f"unfolded{sfs_sfx}.sfs.png"]:
        print(f"  {'OK     ' if f.exists() else 'MISSING'} {f.relative_to(root)}")
    sfs = d / f"unfolded{sfs_sfx}.sfs.pkl"
    if sfs.exists():
        with open(sfs, "rb") as fh:
            fs = pickle.load(fh)
        sizes = dict(zip(getattr(fs, "pop_ids", None) or ["pop0", "pop1"], [n - 1 for n in fs.shape]))
        print(f"  SFS: {sizes} haploid samples, {float(fs.S()):,.0f} segregating sites")
    mj = d / "missing_data_kept_fraction.json"
    if mj.exists():
        m = json.loads(mj.read_text())
        print(f"  missing-data kept fraction: {m['missing_data_kept_fraction']:.5f} "
              f"({m['variable_sites_no_missing']:,} kept / {m['variable_sites_with_missing']:,} dropped)")
    meta = d / f"unfolded{sfs_sfx}.sfs.meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        print(f"  L: region {m.get('region_length', 0):,} bp x polarization {m.get('kept_fraction') or 1:.4f} "
              f"x missing data {m.get('missing_data_kept_fraction') or 1:.4f} = {m['sequence_length']:,} bp")
EOF
echo "========================================================"

if [[ $SNAKEMAKE_STATUS -ne 0 ]]; then
    echo "real_data_prep: snakemake exited with status $SNAKEMAKE_STATUS -- see the log above / logs/real_prep_*.err"
    exit $SNAKEMAKE_STATUS
fi
echo "real_data_prep finished."

#!/usr/bin/env bash
# Shared by every bash_scripts/real_data/*.sh script and real_data_master.sh:
# derives every config-driven path/tag that must byte-for-byte match the
# Snakefile's own REAL_ARMS / REAL_TAG / REAL_LD_ENGINE / _trim_dir_suffix() /
# REAL_POOLING_MODE logic, from ONE place. Each script used to keep its own
# copy of this logic, which had already drifted (missing the trim suffix in
# most scripts, a stale "MomentsLD" instead of "MomentsLD_${REAL_TAG}" in
# real_momentsld_prep.sh, real_data_prep.sh hardcoded to Chr3L only) --
# consolidating it here is the fix, not another copy.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/lib_real_data_config.sh"
#   load_real_data_config "$CFG"
#   # now available: MODEL, REAL_ARMS (array), REAL_USE_GENMAP, REAL_TAG,
#   # REAL_LD_ENGINE, REAL_POOLING_MODE, TRIM_SUFFIX, DROSO_BASE_DIR,
#   # DROSO_DIR, REAL_RUN_ROOT, REAL_INF_ROOT, REAL_LD_ROOT,
#   # REAL_RUN_ROOT_CHROM_TMPL, REAL_INF_ROOT_CHROM_TMPL (both contain a
#   # literal "{arm}" placeholder -- resolve with real_data_chrom_path)

load_real_data_config() {
    local cfg="$1"

    MODEL=$(jq -r '.demographic_model' "$cfg")
    readarray -t REAL_ARMS < <(jq -r '.real_data_analysis.arms // ["Chr3L"] | .[]' "$cfg")
    REAL_USE_GENMAP=$(jq -r '.real_data_analysis.use_genmap // false' "$cfg")
    REAL_POOLING_MODE=$(jq -r '.real_data_analysis.pooling_mode // "pooled"' "$cfg")

    if [[ "$REAL_POOLING_MODE" != "pooled" && "$REAL_POOLING_MODE" != "individual" ]]; then
        echo "ERROR: real_data_analysis.pooling_mode must be \"pooled\" or \"individual\", got \"$REAL_POOLING_MODE\"" >&2
        exit 1
    fi

    REAL_TAG=$(IFS=_; echo "${REAL_ARMS[*]}")
    [[ "$REAL_USE_GENMAP" == "true" ]] && REAL_TAG="${REAL_TAG}_genmap"
    REAL_LD_ENGINE="MomentsLD_${REAL_TAG}"

    # Must match the Snakefile's _trim_dir_suffix(): "" when trim_region is
    # empty/absent, else "_trim_<chrom>-<start>-<end>_<chrom>-<start>-<end>_..."
    # with chroms sorted lexicographically (same order Python's sorted() on
    # dict keys gives).
    TRIM_SUFFIX=$(jq -r '
      (.real_data_analysis.trim_region // {}) as $tr
      | if ($tr | length) == 0 then ""
        else "_trim_" + ($tr | to_entries | sort_by(.key) | map("\(.key)-\(.value[0])-\(.value[1])") | join("_"))
        end
    ' "$cfg")

    DROSO_BASE_DIR="real_data_analysis/data/drosophila"
    DROSO_DIR="${DROSO_BASE_DIR}${TRIM_SUFFIX}"
    REAL_RUN_ROOT="experiments/${MODEL}/real_data_analysis${TRIM_SUFFIX}/runs"
    REAL_INF_ROOT="experiments/${MODEL}/real_data_analysis${TRIM_SUFFIX}/inferences"
    REAL_LD_ROOT="${DROSO_DIR}/${REAL_LD_ENGINE}"

    # Per-arm (pooling_mode=individual) SFS/MomentsLD run+inference roots --
    # "{arm}" is a literal placeholder, not yet substituted; use
    # real_data_chrom_path to resolve it for a specific arm.
    REAL_RUN_ROOT_CHROM_TMPL="experiments/${MODEL}/real_data_analysis${TRIM_SUFFIX}/{arm}/runs"
    REAL_INF_ROOT_CHROM_TMPL="experiments/${MODEL}/real_data_analysis${TRIM_SUFFIX}/{arm}/inferences"
}

# Substitute the literal "{arm}" placeholder in a REAL_*_CHROM_TMPL path.
real_data_chrom_path() {
    local template="$1" arm="$2"
    echo "${template//\{arm\}/$arm}"
}

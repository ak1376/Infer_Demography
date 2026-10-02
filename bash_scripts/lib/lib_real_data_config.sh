#!/usr/bin/env bash
# Shared by every bash_scripts/real_data/*.sh script and real_data_master.sh:
# reads every config-driven path from src/real_paths.py (shared with the
# Snakefile) plus REAL_ARMS / REAL_POOLING_MODE from the config. Each script used to keep its own
# copy of this logic, which had already drifted (missing the trim suffix in
# most scripts, a stale "MomentsLD" instead of "MomentsLD_${REAL_TAG}" in
# real_momentsld_prep.sh, real_data_prep.sh hardcoded to Chr3L only) --
# consolidating it here is the fix, not another copy.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/lib_real_data_config.sh"
#   load_real_data_config "$CFG"
#   # now available: MODEL, REAL_ARMS (array), REAL_USE_GENMAP,
#   # REAL_LD_ENGINE, REAL_LD_ENGINE_ARM, REAL_POOLING_MODE, DROSO_BASE_DIR,
#   # DROSO_DIR, REAL_FIT_ROOT, REAL_RUN_ROOT, REAL_INF_ROOT, REAL_LD_ROOT,
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

    # Every path comes from src/real_paths.py -- the same module the
    # Snakefile imports -- so the two can't drift apart.
    eval "$(cd "$ROOT" && python -m src.real_paths --config "$cfg" --shell)"
    REAL_RUN_ROOT_CHROM_TMPL="${REAL_RUN_ROOT_CHROM//\{chrom\}/\{arm\}}"
    REAL_INF_ROOT_CHROM_TMPL="${REAL_INF_ROOT_CHROM//\{chrom\}/\{arm\}}"
}

# Substitute the literal "{arm}" placeholder in a REAL_*_CHROM_TMPL path.
real_data_chrom_path() {
    local template="$1" arm="$2"
    echo "${template//\{arm\}/$arm}"
}

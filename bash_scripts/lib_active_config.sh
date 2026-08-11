#!/usr/bin/env bash
# Shared by every bash_scripts/*.sh: resolves the active experiment config
# path from ONE place, so switching experiments means editing a single YAML
# key instead of a hardcoded path in every sbatch script.
#
# Precedence:
#   1. $CFG_PATH, if already exported (e.g. by master_script.sh via
#      `sbatch --export=ALL`) — lets one pipeline run pin all its stages to
#      a specific config even if model_config.yaml changes mid-flight.
#   2. config_files/model_config.yaml's `active_experiment_config` key —
#      the same key the Snakefile itself reads
#      (EXP_CFG = config["active_experiment_config"]), so this is always
#      in lockstep with what `snakemake` would use.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib_active_config.sh"
#   CFG="$(resolve_cfg_path "$ROOT")"
resolve_cfg_path() {
    local repo="$1"

    if [ -n "${CFG_PATH:-}" ]; then
        echo "$CFG_PATH"
        return 0
    fi

    local yaml="$repo/config_files/model_config.yaml"
    local rel
    rel=$(grep -E '^[[:space:]]*active_experiment_config[[:space:]]*:' "$yaml" 2>/dev/null \
          | head -1 \
          | sed -E 's/^[^:]+:[[:space:]]*"?([^"#]+)"?[[:space:]]*(#.*)?$/\1/' \
          | sed -E 's/[[:space:]]+$//')

    if [ -z "$rel" ]; then
        echo "ERROR: active_experiment_config not set in $yaml (and CFG_PATH not exported)" >&2
        exit 1
    fi

    echo "$repo/$rel"
}
#!/usr/bin/env bash
# run_inference_standalone.sh
#
# One-off dadi+moments inference for a single simulation with specific
# parameters fixed from its ground-truth sampled_params.pkl.
# Results go to a custom directory, entirely separate from the Snakemake workflow.
#
# Usage (from project root):
#   bash run_inference_standalone.sh

set -euo pipefail

##############################################################################
# CONFIGURE HERE
##############################################################################
SID=42
MODEL="split_migration_growth"

# Space-separated names of parameters to fix (values pulled from ground truth)
FIXED_PARAMS="N_FR0 m_FR_CO"

# Output root — completely outside the workflow's inferences/ tree
OUT_DIR="standalone_inferences/sim_${SID}_fixed_${FIXED_PARAMS// /_}"

# How many independent optimization restarts to run
NUM_RESTARTS=3

##############################################################################
# PATHS  (should not need editing if run from project root)
##############################################################################
EXP_CFG="config_files/experiment_config_split_migration_growth.json"
SIM_BASEDIR="experiments/${MODEL}/simulations"
SFS_FILE="${SIM_BASEDIR}/${SID}/SFS.pkl"
GROUND_TRUTH="${SIM_BASEDIR}/${SID}/sampled_params.pkl"
INFER_SCRIPT="snakemake_scripts/moments_dadi_inference.py"
MODEL_PY="src.simulation:${MODEL}_model"

##############################################################################
# VALIDATE INPUTS
##############################################################################
for f in "$SFS_FILE" "$GROUND_TRUTH" "$EXP_CFG" "$INFER_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "❌  Required file not found: $f"
        exit 1
    fi
done

##############################################################################
# PATCH CONFIG — inject fixed_parameters (values resolved at runtime from
# ground truth; only the *names* need to be present in the JSON)
##############################################################################
PATCHED_CFG=$(mktemp /tmp/exp_cfg_patched_XXXXXX.json)
trap 'rm -f "$PATCHED_CFG"' EXIT

python3 - <<PYEOF
import json

cfg_path    = "${EXP_CFG}"
out_path    = "${PATCHED_CFG}"
fixed_names = "${FIXED_PARAMS}".split()

with open(cfg_path) as f:
    cfg = json.load(f)

# Values are None here — sfs_inference_runner reads the real values
# from sampled_params.pkl via --ground-truth at runtime.
cfg["fixed_parameters"] = {p: None for p in fixed_names}

with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"  fixed_parameters injected: {fixed_names}")
print(f"  patched config → {out_path}")
PYEOF

##############################################################################
# RUN RESTARTS
##############################################################################
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  SID          : ${SID}"
echo "  Fixed params : ${FIXED_PARAMS}"
echo "  Restarts     : ${NUM_RESTARTS}"
echo "  Output root  : ${OUT_DIR}"
echo "============================================================"

for restart in $(seq 1 "$NUM_RESTARTS"); do
    RESTART_OUT="${OUT_DIR}/restart_${restart}"
    mkdir -p "$RESTART_OUT"

    echo ""
    echo "── restart ${restart}/${NUM_RESTARTS} → ${RESTART_OUT}"

    PYTHONPATH="$(pwd)" python3 "$INFER_SCRIPT" \
        --mode            both          \
        --sfs-file        "$SFS_FILE"   \
        --config          "$PATCHED_CFG" \
        --model-py        "$MODEL_PY"   \
        --outdir          "$RESTART_OUT" \
        --ground-truth    "$GROUND_TRUTH" \
        --generate-profiles              \
        --verbose

    echo "✅  restart ${restart} done"
done

##############################################################################
# OPTIONAL: pick best result across restarts
##############################################################################
echo ""
echo "── aggregating best result across ${NUM_RESTARTS} restarts"

python3 - <<PYEOF
import pickle, glob, numpy as np
from pathlib import Path

out_dir = Path("${OUT_DIR}")

for mode in ("moments", "dadi"):
    pkls = sorted(out_dir.glob(f"restart_*/{mode}/best_fit.pkl"))
    if not pkls:
        print(f"  [{mode}] no results found")
        continue

    best_ll, best_result = -np.inf, None
    for p in pkls:
        d = pickle.load(open(p, "rb"))
        ll = d.get("best_ll")
        if ll is not None and float(ll) > best_ll:
            best_ll = float(ll)
            best_result = d

    agg_path = out_dir / f"best_{mode}.pkl"
    pickle.dump(best_result, open(agg_path, "wb"))
    print(f"  [{mode}] best LL={best_ll:.4f} → {agg_path}")
PYEOF

echo ""
echo "============================================================"
echo "Done. Results in: ${OUT_DIR}"
echo "============================================================"
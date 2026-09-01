#!/bin/bash
# bash_scripts/compare_real_fit_params.sh
#
# Diagnostic (not a pipeline stage, no snakemake rule): print the real-data
# fitted params side by side -- the XGBoost (or other MODEL_KEY) ML
# prediction vs. the raw moments/dadi SFS-optimizer best_fit.pkl -- so you
# can see whether a real-data miscalibration traces back to the ML
# regression step or is already present in the raw SFS fit itself.
#
# Usage:
#   bash bash_scripts/compare_real_fit_params.sh
#   VARIANT=w_FIM_w_SFSresids MODEL_KEY=random_forest bash bash_scripts/compare_real_fit_params.sh

set -euo pipefail

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"

MODEL=$(jq -r '.demographic_model' "$CFG")
VARIANT="${VARIANT:-wo_FIM_wo_SFSresids}"
MODEL_KEY="${MODEL_KEY:-xgboost}"

REAL_INF_ROOT="$ROOT/experiments/${MODEL}/real_data_analysis/inferences"
PRED_JSON="$ROOT/experiments/${MODEL}/real_data_analysis/prediction_${VARIANT}/predictions_${MODEL_KEY}.json"

echo "MODEL=$MODEL  VARIANT=$VARIANT  MODEL_KEY=$MODEL_KEY"
echo "ML prediction : $PRED_JSON"
echo "moments fit   : ${REAL_INF_ROOT}/moments/best_fit.pkl"
echo "momentsLD fit : ${REAL_INF_ROOT}/MomentsLD/best_fit.pkl"
echo "dadi fit      : ${REAL_INF_ROOT}/dadi/best_fit.pkl"
echo

PYTHONPATH="$ROOT" python3 - "$CFG" "$PRED_JSON" "${REAL_INF_ROOT}/moments/best_fit.pkl" "${REAL_INF_ROOT}/MomentsLD/best_fit.pkl" "${REAL_INF_ROOT}/dadi/best_fit.pkl" <<'PYEOF'
import json, pickle, sys

cfg_path, pred_path, moments_path, momentsld_path, dadi_path = sys.argv[1:6]

cfg = json.loads(open(cfg_path).read())
priors = cfg["priors"]
param_order = list(priors.keys())

def load_pickle(path):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None

def best_params_from_fit(blob):
    """Handle {'best_params': dict} or {'best_params': list[dict], 'best_ll': list}."""
    if blob is None:
        return None
    bp = blob.get("best_params")
    if isinstance(bp, dict):
        return {k: float(v) for k, v in bp.items()}
    if isinstance(bp, list) and bp:
        ll = blob.get("best_ll")
        idx = 0
        if isinstance(ll, list) and len(ll) == len(bp):
            idx = max(range(len(ll)), key=lambda i: ll[i])
        return {k: float(v) for k, v in bp[idx].items()}
    return None

pred_raw = json.loads(open(pred_path).read())
xgb = pred_raw.get("predictions", pred_raw)

moments_best = best_params_from_fit(load_pickle(moments_path))
momentsld_best = best_params_from_fit(load_pickle(momentsld_path))
dadi_best = best_params_from_fit(load_pickle(dadi_path))

def fmt(v):
    return f"{v:,.6g}" if v is not None else "--"

# Column order: moments, momentsLD, xgboost, dadi (dadi kept for reference).
SOURCES = [("moments", moments_best), ("momentsLD", momentsld_best),
           ("xgboost", xgb), ("dadi", dadi_best)]

col = {"param": 10, "prior_lo": 12, "prior_hi": 12}
width = 16
header = (f"{'param':<{col['param']}} {'prior_lo':>{col['prior_lo']}} {'prior_hi':>{col['prior_hi']}} "
          + " ".join(f"{label:>{width}}" for label, _ in SOURCES))
print(header)
print("-" * len(header))

for p in param_order:
    lo, hi = priors[p]
    row = f"{p:<{col['param']}} {fmt(lo):>{col['prior_lo']}} {fmt(hi):>{col['prior_hi']}} "
    row += " ".join(f"{fmt(params.get(p) if params else None):>{width}}" for _, params in SOURCES)
    print(row)

# Implied ancestral theta PER SITE = 4*N_ANC*mu (no L -- that's what's
# comparable to observed per-site pi). L only belongs in theta_total
# (expected genome-wide segregating sites), shown separately for reference.
mu = float(cfg["mutation_rate"])
L = float(cfg["sequence_length"])
print()
print(f"Implied ancestral theta (4*N_ANC*mu, mu={mu:.3g}) -- compare theta_per_site to observed pi:")
for label, params in SOURCES:
    n_anc = params.get("N_ANC") if params else None
    theta_per_site = 4 * n_anc * mu if n_anc is not None else None
    theta_total = theta_per_site * L if theta_per_site is not None else None
    print(f"  {label:<10s} N_ANC={fmt(n_anc):>16s}   theta_per_site={fmt(theta_per_site):>12s}   "
          f"theta_total(over L={L:,.0f})={fmt(theta_total)}")
PYEOF

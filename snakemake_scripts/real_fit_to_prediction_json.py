#!/usr/bin/env python3
# snakemake_scripts/real_fit_to_prediction_json.py
#
# Write a predict_real_data.py-shaped predictions_<engine>.json from a raw
# real-data best_fit.pkl (dadi/moments/momentsLD SFS/LD-optimizer fit), so
# calibration_simulate.py / calibration_ld_ppc.py can run their PPC against
# the *raw* tool fit instead of an ML-surrogate prediction -- this is how you
# tell whether a calibration miscalibration traces back to the ML model or
# is already present in the underlying dadi/moments/momentsLD fit itself.
#
# Mirrors the best_params_from_fit() logic in
# bash_scripts/real_data/compare_real_fit_params.sh exactly, so results here
# match what that diagnostic prints.
#
# Usage:
#   python snakemake_scripts/real_fit_to_prediction_json.py \
#       --best-fit-pkl experiments/<model>/real_data_analysis/inferences/moments/best_fit.pkl \
#       --engine moments \
#       --out-path experiments/<model>/real_data_analysis/prediction_<variant>/predictions_moments.json
#
# Then e.g.:
#   VARIANT=<variant> MODEL_KEY=moments bash bash_scripts/simulation/calibration_simulate.sh
#   VARIANT=<variant> MODEL_KEY=moments sbatch bash_scripts/simulation/calibration_ld_ppc.sh

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best-fit-pkl", required=True, type=Path,
                     help="Real-data best_fit.pkl for one engine (dadi/moments/momentsLD).")
    ap.add_argument("--engine", required=True,
                     help="Label written into 'model_key' and used as the MODEL_KEY "
                          "value when invoking calibration_simulate.sh/calibration_ld_ppc.sh "
                          "(e.g. dadi, moments, momentsLD).")
    ap.add_argument("--out-path", required=True, type=Path,
                     help="Output predictions_<engine>.json path (same shape "
                          "predict_real_data.py writes).")
    return ap.parse_args()


def best_params_from_fit(blob: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Handle {'best_params': dict} or {'best_params': list[dict], 'best_ll': list}."""
    if not isinstance(blob, dict):
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


def main() -> None:
    args = _parse_args()
    with open(args.best_fit_pkl, "rb") as fh:
        blob = pickle.load(fh)

    params = best_params_from_fit(blob)
    if not params:
        raise SystemExit(f"No usable best_params found in {args.best_fit_pkl}")

    payload = {
        "model_key": args.engine,
        "model_obj": str(args.best_fit_pkl),
        "predictions": params,
        "target_order": list(params.keys()),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, indent=2))

    print(f"✓ wrote {args.out_path} from {args.best_fit_pkl}")
    for k, v in params.items():
        print(f"    {k:28s} {v:.6g}")


if __name__ == "__main__":
    main()

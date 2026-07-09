#!/usr/bin/env python3
# snakemake_scripts/predict_real_data.py
#
# Push the real-data feature row (built by build_real_prediction_dataset.py)
# through a trained model object and write physical-unit predictions.
#
# Works with the *_mdl_obj.pkl objects produced by random_forest.py,
# xgboost_evaluation.py and linear_evaluation.py. They all store the fitted
# estimator under "model"; RF/XGB also store "feature_names"/"target_order",
# while linear stores only "param_names" (feature order falls back to the
# training features template).

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-obj", required=True, type=Path,
                    help="Trained *_mdl_obj.pkl (RF / XGB / linear).")
    ap.add_argument("--real-features", required=True, type=Path,
                    help="Normalized real features pkl (real_features_df.pkl).")
    ap.add_argument("--train-features", required=True, type=Path,
                    help="Training features_df.pkl (feature-order fallback for linear).")
    ap.add_argument("--config", required=True, type=Path,
                    help="Experiment config JSON (priors, for de-normalization).")
    ap.add_argument("--out-prefix", required=True, type=Path,
                    help="Output path prefix; writes <prefix>.json and <prefix>.csv")
    ap.add_argument("--model-key", default="", help="Label recorded in outputs.")
    return ap.parse_args()


def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    args = _parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT))
    import feature_extraction_helpers as fx  # noqa: E402

    cfg = json.loads(args.config.read_text())
    mu, sigma = fx.prior_stats(cfg["priors"])

    obj = _load_pickle(args.model_obj)
    if not isinstance(obj, dict) or "model" not in obj:
        raise SystemExit(f"{args.model_obj} is not a recognized model object "
                         f"(expected a dict with a 'model' key).")
    model = obj["model"]

    feat_order = obj.get("feature_names")
    if not feat_order:
        feat_order = list(_load_pickle(args.train_features).columns)
    targ_order = obj.get("target_order") or obj.get("param_names")
    if not targ_order:
        raise SystemExit(f"{args.model_obj} has no target/param name list.")

    X = _load_pickle(args.real_features)
    if not isinstance(X, pd.DataFrame):
        raise SystemExit("real-features must be a DataFrame.")

    # exact column order the estimator was trained on
    missing = [c for c in feat_order if c not in X.columns]
    if missing:
        raise SystemExit(
            f"Real feature frame is missing {len(missing)} model columns "
            f"e.g. {missing[:8]}. Rebuild the real dataset against this model's "
            f"training template."
        )
    X = X.loc[:, feat_order]
    if X.isna().any().any():
        bad = [c for c in feat_order if X[c].isna().any()]
        raise SystemExit(f"Real features contain NaN in: {bad[:8]}")

    pred_norm = np.asarray(model.predict(X.values))
    if pred_norm.ndim == 1:
        pred_norm = pred_norm.reshape(1, -1)

    # de-normalize back to physical units (bgs_target_coverage_frac has no
    # prior, so it was never normalized in training -> leave as-is)
    rows = []
    for j, p in enumerate(targ_order):
        base = fx.base_param(p)
        norm_val = float(pred_norm[0, j])
        if base in mu:
            phys_val = norm_val * sigma[base] + mu[base]
        else:
            phys_val = norm_val
        rows.append({"parameter": p, "prediction": phys_val,
                     "prediction_normalized": norm_val})

    df = pd.DataFrame(rows).set_index("parameter")

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_suffix(".csv"))

    payload = {
        "model_key": args.model_key,
        "model_obj": str(args.model_obj),
        "predictions": {p: float(df.at[p, "prediction"]) for p in df.index},
        "predictions_normalized": {
            p: float(df.at[p, "prediction_normalized"]) for p in df.index
        },
        "target_order": list(targ_order),
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    print(f"✓ predictions ({args.model_key or args.model_obj.name}) → "
          f"{out_prefix.with_suffix('.json')}")
    for p in df.index:
        print(f"    {p:28s} {df.at[p, 'prediction']:.6g}")


if __name__ == "__main__":
    main()

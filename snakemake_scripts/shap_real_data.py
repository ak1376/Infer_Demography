#!/usr/bin/env python3
# snakemake_scripts/shap_real_data.py
#
# Explain a trained model's prediction on the real-data feature row (built by
# build_real_prediction_dataset.py) with SHAP, one explainer per target
# parameter.
#
# Works with the same *_mdl_obj.pkl objects predict_real_data.py consumes
# (dict with a "model" key). RF/XGB store "model" as a
# sklearn.multioutput.MultiOutputRegressor -- each target gets its own
# single-output estimator (model.estimators_[j]), explained with
# shap.TreeExplainer. Linear objects store a single multi-output estimator
# (LinearRegression/Ridge/Lasso/ElasticNet) -- each target is explained via a
# thin single-output wrapper with shap.Explainer's model-agnostic Permutation
# algorithm (TreeExplainer construction fails fast on non-tree models, which
# is how the per-target branch below picks the right algorithm without
# hardcoding estimator class names).
#
# SHAP values are computed in the same normalized (z-scored) feature/target
# space the model was trained and predicts in -- not de-normalized physical
# units.

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.multioutput import MultiOutputRegressor


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-obj", required=True, type=Path,
                    help="Trained *_mdl_obj.pkl (RF / XGB / linear).")
    ap.add_argument("--real-features", required=True, type=Path,
                    help="Normalized real features pkl (real_features_df.pkl).")
    ap.add_argument("--background-features", required=True, type=Path,
                    help="Normalized training features pkl "
                         "(normalized_train_features.pkl) used as the SHAP "
                         "background/reference distribution.")
    ap.add_argument("--train-features", required=True, type=Path,
                    help="Training features_df.pkl (feature-order fallback "
                         "when the model object has no feature_names).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-key", default="", help="Label recorded in outputs.")
    ap.add_argument("--background-size", type=int, default=100,
                    help="Max background rows sampled for the explainer.")
    ap.add_argument("--top-k", type=int, default=15,
                    help="Top-|SHAP| features kept per target in the summary/plot.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


class _SingleOutputWrapper:
    """Adapts one column of a multi-output estimator's predict() to a
    single-output callable, so a non-multi-output-wrapped model (e.g. the
    linear_* mdl_obj's estimator, which predicts all targets at once) can be
    explained one target at a time like the RF/XGB per-target estimators."""

    def __init__(self, base_estimator, target_idx: int):
        self.base_estimator = base_estimator
        self.target_idx = target_idx

    def predict(self, X):
        pred = np.asarray(self.base_estimator.predict(X))
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred[:, self.target_idx]


def _align(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise SystemExit(f"{name} must be a DataFrame.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{name} is missing {len(missing)} model columns, "
            f"e.g. {missing[:8]}. Rebuild it against this model's training template."
        )
    out = df.loc[:, columns]
    if out.isna().any().any():
        bad = [c for c in columns if out[c].isna().any()]
        raise SystemExit(f"{name} contains NaN in: {bad[:8]}")
    return out


def _make_explainer(estimator, background: np.ndarray, feature_names: list[str]):
    try:
        return shap.TreeExplainer(estimator, data=background), "tree"
    except Exception:
        return (
            shap.Explainer(estimator.predict, background, feature_names=feature_names),
            "generic",
        )


def main() -> None:
    args = _parse_args()
    rng = np.random.RandomState(args.seed)

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

    X_real = _align(_load_pickle(args.real_features), feat_order, "real-features")
    background_df = _align(
        _load_pickle(args.background_features), feat_order, "background-features"
    )
    if len(background_df) > args.background_size:
        background_df = background_df.sample(
            n=args.background_size, random_state=args.seed
        )
    background = background_df.to_numpy(dtype=float)
    real_arr = X_real.to_numpy(dtype=float)
    real_row = real_arr[0]

    if isinstance(model, MultiOutputRegressor):
        estimators = list(model.estimators_)
    else:
        estimators = [_SingleOutputWrapper(model, j) for j in range(len(targ_order))]
    if len(estimators) != len(targ_order):
        raise SystemExit(
            f"Got {len(estimators)} per-target estimators but "
            f"{len(targ_order)} target names -- can't pair them up."
        )

    shap_rows = []
    long_records = []
    summary = {
        "model_key": args.model_key,
        "model_obj": str(args.model_obj),
        "feature_names": feat_order,
        "target_order": list(targ_order),
        "background_rows_used": int(len(background_df)),
        "targets": {},
    }

    for j, (target_name, estimator) in enumerate(zip(targ_order, estimators)):
        explainer, kind = _make_explainer(estimator, background, feat_order)
        explanation = explainer(real_arr)
        values = np.asarray(explanation.values)
        row = values[0] if values.ndim == 2 else np.asarray(values).reshape(-1)
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        prediction = float(base_value + row.sum())

        shap_rows.append(row)
        for feat_name, shap_val, feat_val in zip(feat_order, row, real_row):
            long_records.append({
                "target": target_name,
                "feature": feat_name,
                "shap_value": float(shap_val),
                "feature_value_normalized": float(feat_val),
            })

        order = np.argsort(-np.abs(row))[: args.top_k]
        summary["targets"][target_name] = {
            "explainer": kind,
            "expected_value": base_value,
            "prediction": prediction,
            "top_features": [
                {
                    "feature": feat_order[i],
                    "shap_value": float(row[i]),
                    "feature_value_normalized": float(real_row[i]),
                }
                for i in order
            ],
        }
        print(f"[INFO] {target_name}: explainer={kind} "
              f"expected={base_value:.4f} prediction={prediction:.4f}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.model_key}" if args.model_key else ""

    shap_wide = pd.DataFrame(shap_rows, index=list(targ_order), columns=feat_order)
    shap_wide.to_pickle(out_dir / f"shap_values{tag}.pkl")

    long_df = pd.DataFrame.from_records(long_records)
    long_df.to_csv(out_dir / f"shap_values{tag}.csv", index=False)

    (out_dir / f"shap_summary{tag}.json").write_text(json.dumps(summary, indent=2))

    n_targets = len(targ_order)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).flatten()
    for j, target_name in enumerate(targ_order):
        row = shap_wide.loc[target_name].to_numpy()
        order = np.argsort(np.abs(row))[-args.top_k:]
        vals = row[order]
        names = [feat_order[i] for i in order]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

        ax = axes[j]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (normalized units)")
        ax.set_title(target_name, fontsize=10)
    for ax in axes[n_targets:]:
        ax.axis("off")
    fig.suptitle(
        f"SHAP explanation of real-data prediction -- {args.model_key}", fontsize=13
    )
    fig.savefig(out_dir / f"shap_summary{tag}.png", dpi=200)
    plt.close(fig)

    print(f"[INFO] SHAP analysis complete. Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()

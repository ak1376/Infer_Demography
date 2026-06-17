#!/usr/bin/env python3
"""
prepare_sfs_splits.py

Split and normalize the SFS features/targets DataFrames for modeling.
When --split-indices is provided, reuses the same train/tune/val sim IDs
as the existing inference-based dataset for a fair comparison.

Outputs the same normalized pkl format expected by linear_evaluation.py,
random_forest.py, and xgboost_evaluation.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",      required=True, help="sfs_features_df.pkl")
    ap.add_argument("--targets",       required=True, help="sfs_targets_df.pkl")
    ap.add_argument("--out-dir",       required=True)
    ap.add_argument("--split-indices", default=None,
                    help="split_indices.json from existing pipeline (reuses same sim IDs)")
    ap.add_argument("--train-frac",    type=float, default=0.70)
    ap.add_argument("--tune-frac",     type=float, default=0.15)
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_pickle(args.features)
    y = pd.read_pickle(args.targets)

    # normalize index to str for consistent lookup
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)

    if args.split_indices is not None:
        # reuse existing split — intersect with available SFS sims
        split_json = json.loads(Path(args.split_indices).read_text())
        available = set(X.index)

        def _ids(key):
            return [str(i) for i in split_json[key] if str(i) in available]

        train_ids = _ids("train_idx")
        tune_ids  = _ids("tune_idx")
        val_ids   = _ids("val_idx")

        n_missing = sum(
            len(split_json[k]) - len(_ids(k))
            for k in ("train_idx", "tune_idx", "val_idx")
        )
        if n_missing:
            print(f"  NOTE: {n_missing} sim IDs from split_indices.json not in SFS dataset (skipped)")
    else:
        # fresh random split
        from sklearn.model_selection import train_test_split
        ids = list(X.index)
        val_frac = 1.0 - args.train_frac - args.tune_frac
        train_ids, tv_ids = train_test_split(ids, test_size=(1.0 - args.train_frac), random_state=args.seed)
        tune_of_tv = args.tune_frac / (args.tune_frac + val_frac)
        tune_ids, val_ids = train_test_split(tv_ids, test_size=(1.0 - tune_of_tv), random_state=args.seed)

    X_train, y_train = X.loc[train_ids], y.loc[train_ids]
    X_tune,  y_tune  = X.loc[tune_ids],  y.loc[tune_ids]
    X_val,   y_val   = X.loc[val_ids],   y.loc[val_ids]

    # fit scalers on train only
    scaler_X = StandardScaler().fit(X_train.values)
    scaler_y = StandardScaler().fit(y_train.values)

    def _scale(scaler, df):
        return pd.DataFrame(scaler.transform(df.values), columns=df.columns, index=df.index)

    splits = {
        "normalized_train_features": _scale(scaler_X, X_train),
        "normalized_train_targets":  _scale(scaler_y, y_train),
        "normalized_tune_features":  _scale(scaler_X, X_tune),
        "normalized_tune_targets":   _scale(scaler_y, y_tune),
        "normalized_val_features":   _scale(scaler_X, X_val),
        "normalized_val_targets":    _scale(scaler_y, y_val),
    }

    for name, df in splits.items():
        df.to_pickle(out_dir / f"{name}.pkl")

    meta = {
        "n_train":      len(train_ids),
        "n_tune":       len(tune_ids),
        "n_val":        len(val_ids),
        "n_features":   X_train.shape[1],
        "n_targets":    y_train.shape[1],
        "reused_split": args.split_indices is not None,
        "target_cols":  list(y_train.columns),
    }
    (out_dir / "sfs_splits_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"✅ train={len(train_ids)}  tune={len(tune_ids)}  val={len(val_ids)}")
    print(f"✅ reused_split={args.split_indices is not None}")
    print(f"✅ splits written to {out_dir}")


if __name__ == "__main__":
    main()

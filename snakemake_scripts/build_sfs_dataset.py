#!/usr/bin/env python3
"""
build_sfs_dataset.py

Build a features/targets dataset where:
  features = flattened observed SFS (interior entries, excluding fixed classes)
  targets  = sampled demographic (+ BGS) parameters

Discovers all available simulations at runtime — skips any with missing files.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _sfs_to_feature_vector(sfs) -> np.ndarray:
    """
    Convert a moments.Spectrum (masked array) to a 1-D feature vector.

    Excludes the fixed/monomorphic boundary entries:
      1-D SFS: drops index 0 and index n  → length n-1
      2-D SFS: drops row 0, row n1, col 0, col n2  → (n1-1) x (n2-1)
    Masked entries within the interior are set to 0.
    """
    arr = np.asarray(sfs, dtype=float)  # fills masked → 0

    if arr.ndim == 1:
        return arr[1:-1].copy()
    elif arr.ndim == 2:
        return arr[1:-1, 1:-1].flatten()
    else:
        # higher-dim: strip first/last slice on every axis then flatten
        slices = tuple(slice(1, -1) for _ in range(arr.ndim))
        return arr[slices].flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir",    required=True, help="Base simulations directory")
    ap.add_argument("--config",     required=True, help="Experiment config JSON")
    ap.add_argument("--out-dir",    required=True, help="Output directory for dataset pkl files")
    ap.add_argument("--min-sims",   type=int, default=10,
                    help="Minimum number of valid simulations required")
    args = ap.parse_args()

    sim_dir = Path(args.sim_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text())
    param_order = list(cfg["parameter_order"])

    # ── discover available simulations ──────────────────────────────────────
    sfs_paths    = sorted(sim_dir.glob("*/SFS.pkl"))
    param_paths  = {p.parent.name: p for p in sim_dir.glob("*/sampled_params.pkl")}

    feature_rows, target_rows, sim_ids = [], [], []

    for sfs_path in sfs_paths:
        sid = sfs_path.parent.name
        par_path = param_paths.get(sid)
        if par_path is None:
            print(f"  SKIP {sid}: no sampled_params.pkl")
            continue

        try:
            sfs = pickle.loads(sfs_path.read_bytes())
            params = pickle.loads(par_path.read_bytes())
        except Exception as e:
            print(f"  SKIP {sid}: load error — {e}")
            continue

        feat = _sfs_to_feature_vector(sfs)
        if not np.all(np.isfinite(feat)):
            print(f"  SKIP {sid}: non-finite SFS entries")
            continue

        # build target vector in canonical parameter_order
        try:
            target = np.array([float(params[p]) for p in param_order])
        except KeyError as e:
            print(f"  SKIP {sid}: missing parameter {e}")
            continue

        feature_rows.append(feat)
        target_rows.append(target)
        sim_ids.append(sid)

    n = len(sim_ids)
    print(f"Loaded {n} simulations ({len(sfs_paths) - n} skipped)")

    if n < args.min_sims:
        raise RuntimeError(
            f"Only {n} valid simulations found (need >= {args.min_sims}). Aborting."
        )

    # ── build DataFrames ────────────────────────────────────────────────────
    n_feat = len(feature_rows[0])
    feat_cols = [f"sfs_{i}" for i in range(n_feat)]

    features_df = pd.DataFrame(
        np.stack(feature_rows), columns=feat_cols, index=sim_ids
    )
    features_df.index.name = "sim_id"

    targets_df = pd.DataFrame(
        np.stack(target_rows), columns=param_order, index=sim_ids
    )
    targets_df.index.name = "sim_id"

    # ── save ────────────────────────────────────────────────────────────────
    feat_out   = out_dir / "sfs_features_df.pkl"
    target_out = out_dir / "sfs_targets_df.pkl"
    meta_out   = out_dir / "sfs_dataset_meta.json"

    features_df.to_pickle(feat_out)
    targets_df.to_pickle(target_out)

    meta = {
        "n_sims":        n,
        "n_features":    n_feat,
        "parameters":    param_order,
        "sfs_interior_shape": list(np.array(feature_rows[0]).shape),
        "sim_ids_sample": sim_ids[:5],
    }
    meta_out.write_text(json.dumps(meta, indent=2))

    print(f"✅ features  → {feat_out}  shape={features_df.shape}")
    print(f"✅ targets   → {target_out}  shape={targets_df.shape}")
    print(f"✅ meta      → {meta_out}")


if __name__ == "__main__":
    main()

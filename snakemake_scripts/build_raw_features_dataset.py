#!/usr/bin/env python3
"""
build_raw_features_dataset.py

Build a features/targets dataset where:
  features = [flattened observed SFS interior  |  flattened MomentsLD means (all bins + H)]
  targets  = sampled demographic parameters

Discovers available simulations at runtime — skips any with missing files.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _sfs_interior(sfs) -> np.ndarray:
    arr = np.asarray(sfs, dtype=float)
    if arr.ndim == 1:
        return arr[1:-1].copy()
    elif arr.ndim == 2:
        return arr[1:-1, 1:-1].flatten()
    else:
        slices = tuple(slice(1, -1) for _ in range(arr.ndim))
        return arr[slices].flatten()


def _ld_means_vector(mv_path: Path) -> np.ndarray:
    with open(mv_path, "rb") as f:
        dat = pickle.load(f)
    return np.concatenate(dat["means"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir",       required=True, help="Base simulations directory")
    ap.add_argument("--inference-dir", required=True, help="Inferences root (contains sim_*/MomentsLD/)")
    ap.add_argument("--config",        required=True, help="Experiment config JSON")
    ap.add_argument("--out-dir",       required=True, help="Output directory for dataset pkl files")
    ap.add_argument("--min-sims",      type=int, default=10)
    args = ap.parse_args()

    sim_dir  = Path(args.sim_dir)
    inf_dir  = Path(args.inference_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text())
    param_order_cfg = list(cfg["parameter_order"])

    sfs_paths   = sorted(sim_dir.glob("*/SFS.pkl"))
    param_paths = {p.parent.name: p for p in sim_dir.glob("*/sampled_params.pkl")}

    # Resolve param_order against what's actually in sampled_params
    param_order = None
    for p in sorted(param_paths.values()):
        try:
            sample = pickle.loads(p.read_bytes())
            param_order = [k for k in param_order_cfg if k in sample]
            dropped = [k for k in param_order_cfg if k not in sample]
            if dropped:
                print(f"  NOTE: dropping {dropped} from param_order (not in sampled_params)")
            break
        except Exception:
            continue
    if param_order is None:
        raise RuntimeError("Could not load any sampled_params.pkl to determine parameter order")

    feature_rows, target_rows, sim_ids = [], [], []
    n_sfs_feat = n_ld_feat = None

    for sfs_path in sfs_paths:
        sid = sfs_path.parent.name
        par_path = param_paths.get(sid)
        mv_path  = inf_dir / f"sim_{sid}" / "MomentsLD" / "means.varcovs.pkl"

        if par_path is None:
            print(f"  SKIP {sid}: no sampled_params.pkl")
            continue
        if not mv_path.exists():
            print(f"  SKIP {sid}: no means.varcovs.pkl at {mv_path}")
            continue

        try:
            sfs    = pickle.loads(sfs_path.read_bytes())
            params = pickle.loads(par_path.read_bytes())
            ld_vec = _ld_means_vector(mv_path)
        except Exception as e:
            print(f"  SKIP {sid}: load error — {e}")
            continue

        sfs_vec = _sfs_interior(sfs)

        if not np.all(np.isfinite(sfs_vec)):
            print(f"  SKIP {sid}: non-finite SFS entries")
            continue
        if not np.all(np.isfinite(ld_vec)):
            print(f"  SKIP {sid}: non-finite LD means entries")
            continue

        try:
            target = np.array([float(params[p]) for p in param_order])
        except KeyError as e:
            print(f"  SKIP {sid}: missing parameter {e}")
            continue

        feature_rows.append(np.concatenate([sfs_vec, ld_vec]))
        target_rows.append(target)
        sim_ids.append(sid)

        if n_sfs_feat is None:
            n_sfs_feat = len(sfs_vec)
            n_ld_feat  = len(ld_vec)

    n = len(sim_ids)
    print(f"Loaded {n} simulations ({len(sfs_paths) - n} skipped)")
    print(f"  SFS features : {n_sfs_feat}")
    print(f"  LD  features : {n_ld_feat}")
    print(f"  Total features: {n_sfs_feat + n_ld_feat}")

    if n < args.min_sims:
        raise RuntimeError(
            f"Only {n} valid simulations found (need >= {args.min_sims}). Aborting."
        )

    sfs_cols  = [f"sfs_{i}"  for i in range(n_sfs_feat)]
    ld_cols   = [f"ld_{i}"   for i in range(n_ld_feat)]
    feat_cols = sfs_cols + ld_cols

    features_df = pd.DataFrame(
        np.stack(feature_rows), columns=feat_cols, index=sim_ids
    )
    features_df.index.name = "sim_id"

    targets_df = pd.DataFrame(
        np.stack(target_rows), columns=param_order, index=sim_ids
    )
    targets_df.index.name = "sim_id"

    feat_out   = out_dir / "raw_features_df.pkl"
    target_out = out_dir / "raw_targets_df.pkl"
    meta_out   = out_dir / "raw_dataset_meta.json"

    features_df.to_pickle(feat_out)
    targets_df.to_pickle(target_out)

    meta = {
        "n_sims":         n,
        "n_sfs_features": n_sfs_feat,
        "n_ld_features":  n_ld_feat,
        "n_total_features": n_sfs_feat + n_ld_feat,
        "parameters":     param_order,
        "sim_ids_sample": sim_ids[:5],
    }
    meta_out.write_text(json.dumps(meta, indent=2))

    print(f"✅ features → {feat_out}  shape={features_df.shape}")
    print(f"✅ targets  → {target_out}  shape={targets_df.shape}")
    print(f"✅ meta     → {meta_out}")


if __name__ == "__main__":
    main()

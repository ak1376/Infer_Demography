#!/usr/bin/env python3
"""
Aggregate fitted parameters + ground‑truth parameters across all simulations,
optionally normalise them by the prior distribution, split into train / validation
sets, and create replicate‑specific columns when best_params contains a list.

Outputs (written to --out-dir):
  features*.npy / targets*.npy  +   features*.pkl / targets*.pkl
(see docstring in the original script for the full list)
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, pickle
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def param_dicts(tool: str, blob: dict) -> list[dict]:
    """Return a *list* of {param: value} dicts (1 per replicate)."""
    if tool.lower() == "momentsld":
        # preferred single‑rep format
        if "opt_params" in blob:
            return [blob["opt_params"]]
    # fall back to generic {"best_params": …}
    bp = blob.get("best_params")
    if isinstance(bp, list):
        return bp
    return [bp]


def prior_stats(priors: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, float]]:
    mu, sigma = {}, {}
    for p, (lo, hi) in priors.items():
        mu[p]    = (lo + hi) / 2
        sigma[p] = (hi - lo) / np.sqrt(12)
    return mu, sigma


def normalise_df(df: pd.DataFrame, mu: dict[str, float], sigma: dict[str, float],
                 feature_cols_have_prefix: bool) -> pd.DataFrame:
    """
    Apply (x - mu) / sigma column‑wise, correctly handling _rep_ suffixes.
    """
    out = df.copy()
    for col in out.columns:
        key = col
        if feature_cols_have_prefix:
            key = key.split("_", 1)[1]             # drop tool_
        key = key.split("_rep_", 1)[0]              # drop _rep_N if present
        out[col] = (out[col] - mu[key]) / sigma[key]
    return out


# --------------------------------------------------------------------------- #
def main(cfg_path: Path, out_dir: Path) -> None:
    cfg        = json.loads(cfg_path.read_text())
    model      = cfg["demographic_model"]
    n_sims     = int(cfg["num_draws"])
    train_pct  = float(cfg.get("training_percentage", 0.8))
    rng        = np.random.default_rng(cfg.get("seed", 42))

    sim_basedir   = Path(f"experiments/{model}/simulations")
    infer_basedir = Path(f"experiments/{model}/inferences")

    feature_rows, target_rows, index = [], [], []

    for sid in range(n_sims):
        inf_pickle   = infer_basedir / f"sim_{sid}/all_inferences.pkl"
        truth_pickle = sim_basedir   / f"{sid}/sampled_params.pkl"

        data   = pickle.load(inf_pickle.open("rb"))
        truth  = pickle.load(truth_pickle.open("rb"))

        row = {}
        for tool in ("moments", "dadi", "momentsLD"):
            if tool in data:
                for rep_idx, pdict in enumerate(param_dicts(tool, data[tool])):
                    for k, v in pdict.items():
                        if tool.lower() == "momentsld":
                            col = f"{tool}_{k}"
                        else:
                            col = f"{tool}_{k}_rep_{rep_idx}"
                        row[col] = v
        feature_rows.append(row)
        target_rows.append(truth)
        index.append(sid)

    feat_df = pd.DataFrame(feature_rows, index=index).sort_index(axis=1)
    targ_df = pd.DataFrame(target_rows,  index=index).sort_index(axis=1)

    # ---------- normalise -------------------------------------------------- #
    mu, sigma = prior_stats(cfg["priors"])
    feat_norm_df = normalise_df(feat_df, mu, sigma, feature_cols_have_prefix=True)
    targ_norm_df = normalise_df(targ_df, mu, sigma, feature_cols_have_prefix=False)

    # ---------- train / val split ------------------------------------------ #
    perm      = rng.permutation(n_sims)
    n_train   = int(round(train_pct * n_sims))
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    split = lambda df, idx: df.iloc[idx].to_numpy(float)

    arrays = {
        "features.npy"           : feat_df.to_numpy(float),
        "targets.npy"            : targ_df.to_numpy(float),
        "features_norm.npy"      : feat_norm_df.to_numpy(float),
        "targets_norm.npy"       : targ_norm_df.to_numpy(float),
        "features_train.npy"     : split(feat_df,  train_idx),
        "targets_train.npy"      : split(targ_df,  train_idx),
        "features_val.npy"       : split(feat_df,  val_idx),
        "targets_val.npy"        : split(targ_df,  val_idx),
        "features_train_norm.npy": split(feat_norm_df, train_idx),
        "targets_train_norm.npy" : split(targ_norm_df, train_idx),
        "features_val_norm.npy"  : split(feat_norm_df, val_idx),
        "targets_val_norm.npy"   : split(targ_norm_df, val_idx),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, arr in arrays.items():
        np.save(out_dir / fname, arr)

    # DataFrame pickles (full)
    feat_df.to_pickle(out_dir / "features_df.pkl")
    targ_df.to_pickle(out_dir / "targets_df.pkl")
    feat_norm_df.to_pickle(out_dir / "features_norm_df.pkl")
    targ_norm_df.to_pickle(out_dir / "targets_norm_df.pkl")

    # OPTIONAL: train / val DataFrames (uncomment if you need them)
    feat_df.iloc[train_idx].to_pickle(out_dir / "features_train_df.pkl")
    targ_df.iloc[train_idx].to_pickle(out_dir / "targets_train_df.pkl")
    feat_df.iloc[val_idx].to_pickle(out_dir / "features_val_df.pkl")
    targ_df.iloc[val_idx].to_pickle(out_dir / "targets_val_df.pkl")
    feat_norm_df.iloc[train_idx].to_pickle(out_dir / "features_train_norm_df.pkl")
    targ_norm_df.iloc[train_idx].to_pickle(out_dir / "targets_train_norm_df.pkl")
    feat_norm_df.iloc[val_idx].to_pickle(out_dir / "features_val_norm_df.pkl")
    targ_norm_df.iloc[val_idx].to_pickle(out_dir / "targets_val_norm_df.pkl")

    print(f"✓ wrote {len(arrays)} .npy files + 4 DataFrames → {out_dir}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--experiment-config", required=True, type=Path)
    pa.add_argument("--out-dir", required=True, type=Path)
    args = pa.parse_args()
    main(args.experiment_config, args.out_dir)

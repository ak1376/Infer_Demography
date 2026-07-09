#!/usr/bin/env python3
# snakemake_scripts/build_real_prediction_dataset.py
#
# Build a single-row feature frame from the REAL-data dadi / moments / MomentsLD
# fits, formatted *identically* to the training features_df so it can be pushed
# through a trained model.
#
# The column convention mirrors src/feature_extraction_helpers.build_feature_target_tables:
#   dadi_{param}_rep_{i}, moments_{param}_rep_{i}, momentsLD_{param}
# Values are z-score normalized by the uniform-prior stats (same as training).

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
    ap.add_argument("--config", required=True, type=Path,
                    help="Experiment config JSON (for priors).")
    ap.add_argument("--real-inf-dir", required=True, type=Path,
                    help="REAL inferences root containing moments/, dadi/, MomentsLD/ best_fit.pkl")
    ap.add_argument("--train-features", required=True, type=Path,
                    help="Training features_df.pkl used as the exact column template.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-missing", action="store_true",
                    help="Fill missing/NaN feature columns with 0 (normalized-space) "
                         "instead of erroring. Off by default.")
    return ap.parse_args()


def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _momentsld_param_dict(ld_blob: dict) -> dict | None:
    """
    Real MomentsLD best_fit stores its physical estimate under
    best_params_abs + param_order (best_params is often None).
    Return a {param: value} dict, or None if unavailable.
    """
    if not isinstance(ld_blob, dict):
        return None
    bp = ld_blob.get("best_params")
    if isinstance(bp, dict) and bp:
        return bp
    absd = ld_blob.get("best_params_abs")
    if isinstance(absd, dict) and absd:
        order = ld_blob.get("param_order") or list(absd.keys())
        return {p: absd[p] for p in order if p in absd}
    return None


def main() -> None:
    args = _parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT))
    import feature_extraction_helpers as fx  # noqa: E402

    cfg = json.loads(args.config.read_text())
    priors = cfg["priors"]
    mu, sigma = fx.prior_stats(priors)

    real = args.real_inf_dir
    # ---- assemble a data blob shaped like all_inferences.pkl -------------
    data: dict = {}

    mom_path = real / "moments" / "best_fit.pkl"
    dadi_path = real / "dadi" / "best_fit.pkl"
    ld_path = real / "MomentsLD" / "best_fit.pkl"

    if mom_path.exists():
        data["moments"] = _load_pickle(mom_path)
    if dadi_path.exists():
        data["dadi"] = _load_pickle(dadi_path)
    if ld_path.exists():
        ld_bp = _momentsld_param_dict(_load_pickle(ld_path))
        if ld_bp is not None:
            data["momentsLD"] = {"best_params": ld_bp}

    # ---- build the single feature row (same logic as training) -----------
    row: dict[str, float] = {}
    for tool in fx.TOOLS_DEFAULT:  # ("dadi", "moments", "momentsLD")
        blob = data.get(tool)
        if blob is None:
            continue
        for rep_idx, pdict in enumerate(fx.param_dicts(tool, blob)):
            if not isinstance(pdict, dict):
                continue
            for k, v in pdict.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                col = (
                    f"{tool}_{k}"
                    if tool.lower() == "momentsld"
                    else f"{tool}_{k}_rep_{rep_idx}"
                )
                row[col] = float(v)

    raw_df = pd.DataFrame([row], index=["real"]).sort_index(axis=1)

    # ---- align to the exact training feature columns ---------------------
    train_cols = list(_load_pickle(args.train_features).columns)
    aligned = raw_df.reindex(columns=train_cols)

    extra = [c for c in raw_df.columns if c not in train_cols]
    missing = [c for c in train_cols if pd.isna(aligned.at["real", c])]

    if extra:
        print(f"[warn] {len(extra)} real feature columns not in the model "
              f"template were dropped (first few): {extra[:8]}")

    if missing and not args.allow_missing:
        # Group missing columns for a readable error.
        by_tool: dict[str, list[str]] = {}
        for c in missing:
            by_tool.setdefault(c.split("_", 1)[0], []).append(c)
        summary = "\n".join(f"    {t}: {len(cs)} cols e.g. {cs[:4]}"
                            for t, cs in sorted(by_tool.items()))
        raise SystemExit(
            f"[build_real_prediction_dataset] {len(missing)} of {len(train_cols)} "
            f"model feature columns are missing/NaN for the real sample:\n{summary}\n"
            f"Most commonly this means the real moments/dadi fits have fewer "
            f"replicates than the model expects. Re-run the real moments/dadi "
            f"inference with enough optimizations (real_top_k) to produce every "
            f"rep_* slot, or pass --allow-missing to zero-fill (not recommended)."
        )

    # ---- normalize (z-score by prior stats), same as training -----------
    norm = fx.normalise_df(aligned, mu, sigma)

    if missing and args.allow_missing:
        norm = norm.fillna(0.0)  # 0 == prior mean in normalized space
        print(f"[warn] zero-filled {len(missing)} missing columns "
              f"(--allow-missing): {missing[:8]}")

    # ---- write outputs ---------------------------------------------------
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "real_features_df.pkl", "wb") as fh:
        pickle.dump(norm, fh)
    with open(out / "real_features_raw_df.pkl", "wb") as fh:
        pickle.dump(aligned, fh)

    meta = {
        "n_features": int(len(train_cols)),
        "tools_present": sorted(k for k in data.keys()),
        "n_missing_columns": int(len(missing)),
        "missing_columns": missing,
        "n_extra_columns_dropped": int(len(extra)),
        "extra_columns_dropped": extra,
        "train_features_template": str(args.train_features),
        "real_inf_dir": str(real),
        "normalized": True,
        "normalization": "z-score by uniform-prior mean/std (matches training)",
    }
    (out / "real_dataset_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"✓ wrote real prediction dataset ({norm.shape[1]} features) → {out}")
    print(f"  tools present: {meta['tools_present']}")


if __name__ == "__main__":
    main()

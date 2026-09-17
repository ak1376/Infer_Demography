#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/sfs_fit_one_start.py
"""
Thin wrapper called by Snakemake rule `sfs_fit_one_start`.

Runs ONE LHS-seeded moments SFS fit (one optimizer start) for one arm + model
and writes just that start's result.

Heavy lifting lives in:
  godambe_correction_LRT/src/sfs_fit_one_start.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import moments

from sfs_fit_one_start import run_one_sfs_start


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True,
                    help="demographic model name, e.g. split_migration_growth "
                         "(resolved to src.demes_models:<model>_model)")
    ap.add_argument("--sfs", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--pop-ids", default="CO,FR")
    ap.add_argument("--opt-index", type=int, required=True)
    ap.add_argument("--n-starts", type=int, required=True,
                    help="total starts across all jobs (sizes the shared LHS grid)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    param_order = list(cfg["parameter_order"])

    with open(args.sfs, "rb") as f:
        sfs = pickle.load(f)
    sfs = moments.Spectrum(sfs)
    sfs.pop_ids = [s.strip() for s in args.pop_ids.split(",")]

    result = run_one_sfs_start(
        arm=args.arm, model=args.model, sfs=sfs, cfg=cfg,
        opt_index=args.opt_index, n_starts=args.n_starts, param_order=param_order,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)

    print(f"[{args.arm}/{args.model}] start {args.opt_index}: "
          f"ll={result['ll_hat']:.6f} -> {args.out}")


if __name__ == "__main__":
    main()

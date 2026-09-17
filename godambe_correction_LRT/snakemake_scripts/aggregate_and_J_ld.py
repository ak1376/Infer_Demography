#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/aggregate_and_J_ld.py

"""
Thin wrapper called by Snakemake rule `j_at_blocksize`.

Aggregate the per-tile LD stats for one (arm, block size) and compute J at p0.
Writes {blocksize, J, var, mean, n_windows}.

Heavy lifting lives in:
  godambe_correction_LRT/src/aggregate_and_J_ld.py
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aggregate_and_J_ld import aggregate_and_compute_J


def main():
    ap = argparse.ArgumentParser(description="Aggregate LD tiles + compute J at p0.")
    ap.add_argument("--ld-stats-dir", type=Path, required=True)
    ap.add_argument("--null-fit", type=Path, required=True)
    ap.add_argument("--blocksize", type=int, required=True)
    ap.add_argument("--r-bins", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    r_bins = [float(x) for x in args.r_bins.split(",")]
    with open(args.null_fit, "rb") as f:
        null_params = pickle.load(f)["best_params"]

    nn, J, scores = aggregate_and_compute_J(
        args.ld_stats_dir, null_params, r_bins, n_boot=args.n_boot, seed=args.seed)

    row = {"blocksize": args.blocksize, "J": J, "var": float(scores.var()),
           "mean": float(scores.mean()), "n_windows": nn}
    with open(args.out, "wb") as f:
        pickle.dump(row, f)
    print(f"blocksize {args.blocksize}: J={J:.4g} var={row['var']:.4g} "
          f"n_windows={nn} -> {args.out}")


if __name__ == "__main__":
    main()

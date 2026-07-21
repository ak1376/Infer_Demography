#!/usr/bin/env python3
# godambe_correction_LRT/scripts/aggregate_and_J_ld.py

"""
Aggregate the per-tile LD stats (from the parallel per-window jobs) for one
(arm, block size), then compute J at p0 (the SIMPLE-model fit, embedded in the
complex model). Writes {blocksize, J, var, mean, n_windows}.

This is the tail of the old monolithic arm_J_at_blocksize.py, split out so the
tiling and per-window LD stats can be separate Snakemake jobs.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import moments

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_ld import average_ld_structure
import compute_J_ld as cj


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

    windows = {}
    for p in args.ld_stats_dir.glob("LD_stats_window_*.pkl"):
        s = pickle.load(p.open("rb"))
        if isinstance(s, dict) and s.get("empty"):
            continue                                  # skip empty-window sentinels
        windows[int(p.stem.split("_")[-1])] = s
    if not windows:
        raise RuntimeError(f"No (non-empty) LD stats in {args.ld_stats_dir}")
    win_list = list(windows.values())
    ld_names, h_names = win_list[0]["stats"]
    nn = len(win_list)

    np.random.seed(args.seed)
    mv = moments.LD.Parsing.bootstrap_data(windows)
    rng = np.random.default_rng(args.seed)
    all_boot = [
        average_ld_structure([win_list[i] for i in rng.integers(0, nn, size=nn)],
                             ld_names, h_names)
        for _ in range(args.n_boot)
    ]

    cj.R_BINS = np.array([float(x) for x in args.r_bins.split(",")])
    with open(args.null_fit, "rb") as f:
        null_params = pickle.load(f)["best_params"]
    J, scores = cj.compute_J(all_boot, mv["varcovs"], null_params)

    row = {"blocksize": args.blocksize, "J": J, "var": float(scores.var()),
           "mean": float(scores.mean()), "n_windows": nn}
    with open(args.out, "wb") as f:
        pickle.dump(row, f)
    print(f"blocksize {args.blocksize}: J={J:.4g} var={row['var']:.4g} "
          f"n_windows={nn} -> {args.out}")


if __name__ == "__main__":
    main()

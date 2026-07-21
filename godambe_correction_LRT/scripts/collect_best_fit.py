#!/usr/bin/env python3
# godambe_correction_LRT/scripts/collect_best_fit.py

"""Pick the best (highest-LL) start across the per-start fit pkls -> best_fit.pkl
(the p0). Also saves all starts for convergence inspection."""

import argparse
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Collect best LHS start -> best_fit.pkl")
    ap.add_argument("--starts", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="best_fit.pkl")
    args = ap.parse_args()

    results = []
    for p in args.starts:
        with open(p, "rb") as f:
            results.append(pickle.load(f))
    results.sort(key=lambda r: r["best_lls"], reverse=True)
    best = results[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(best, f)
    with open(args.out.parent / "all_starts.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"best LL={best['best_lls']:.4f} (start {best['opt']}, "
          f"{len(results)} starts) -> {args.out}")
    for p, v in best["best_params"].items():
        print(f"  {p:8s} = {v:.4g}")


if __name__ == "__main__":
    main()

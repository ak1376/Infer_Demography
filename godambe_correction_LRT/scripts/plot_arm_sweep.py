#!/usr/bin/env python3
# godambe_correction_LRT/scripts/plot_arm_sweep.py

"""Plot J (var of scores) vs non-overlapping block size for one arm, from the
per-blocksize J pkls written by arm_J_at_blocksize.py.

Flat plateau -> J well-defined at that arm's block size.
Monotonic climb / no plateau -> blocks not big enough (slow-recomb regions).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Plot J vs block length for one arm.")
    ap.add_argument("--js", nargs="+", type=Path, required=True,
                    help="per-blocksize J pkls")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", type=str, default="")
    args = ap.parse_args()

    rows = [pickle.load(p.open("rb")) for p in args.js]
    rows.sort(key=lambda r: r["blocksize"])
    sizes_kb = [r["blocksize"] / 1000 for r in rows]
    var = [r["var"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(sizes_kb, var, "o-", lw=1.5)
    for r, xk in zip(rows, sizes_kb):
        ax.annotate(f"n={r['n_windows']}", (xk, r["var"]),
                    fontsize=7, textcoords="offset points", xytext=(0, 6))
    ax.set_xscale("log")
    ax.set_xlabel("non-overlapping block length (kb)")
    ax.set_ylabel("var(scores)  (J, block-size-sensitive part)")
    ax.set_title(f"J vs block length — {args.arm}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

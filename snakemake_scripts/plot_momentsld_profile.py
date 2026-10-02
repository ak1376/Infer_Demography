#!/usr/bin/env python3
"""
Collect a MomentsLD profile likelihood: one parameter pinned at each grid value,
all other parameters re-optimized from several starts. For every grid value the
best start is kept.

Writes <out>.tsv (grid value, best LL, drop from the peak, every re-optimized
parameter) and <out>.png (profile LL vs the pinned value; dashed line = the
1.92-unit drop giving an approximate 95% interval; x marks the unrestricted
best fit; grey band = the prior range for that parameter).

Usage:
  python snakemake_scripts/plot_momentsld_profile.py --param T \
      --fits .../profiles/<engine>/T/pt*/start*/best_fit.pkl \
      --overall-best .../inferences/<engine>/best_fit.pkl \
      --config config_files/experiment_config_*.json --out .../profiles/<engine>/T/profile_T
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _best(d):
    bp, ll = d["best_params"], d["best_ll"]
    if isinstance(bp, list):
        return dict(bp[0]), float(np.ravel(ll)[0])
    return dict(bp), float(np.ravel(ll)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", required=True)
    ap.add_argument("--fits", nargs="+", required=True)
    ap.add_argument("--overall-best", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_pt = defaultdict(list)
    for f in args.fits:
        k = int(re.search(r"/pt(\d+)/", f).group(1))
        p, ll = _best(pickle.load(open(f, "rb")))
        by_pt[k].append((ll, p, f))
    rows = []
    for k in sorted(by_pt):
        ll, p, f = max(by_pt[k], key=lambda t: t[0])
        rows.append((p[args.param], ll, p, len(by_pt[k]), f))
    names = list(rows[0][2])
    peak = max(r[1] for r in rows)
    ov_p, ov_ll = _best(pickle.load(open(args.overall_best, "rb")))
    top = max(peak, ov_ll)

    out = Path(args.out)
    with open(f"{out}.tsv", "w") as fh:
        fh.write("\t".join([args.param + "_fixed", "best_ll", "drop_from_peak", "n_starts"] + [f"fit_{n}" for n in names]) + "\n")
        for v, ll, p, n, _ in rows:
            fh.write("\t".join([f"{v:.6g}", f"{ll:.4f}", f"{top - ll:.4f}", str(n)] + [f"{p[x]:.6g}" for x in names]) + "\n")

    cfg = json.load(open(args.config))
    lo, hi = cfg.get("priors_real_data_analysis_absolute", cfg["priors"])[args.param]   # same box MomentsLD_real_data.py searched
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axvspan(lo, hi, color="grey", alpha=0.10, label="prior range")
    ax.plot(x, y, "o-", color="tab:blue", label="profile (others re-optimized)")
    ax.axhline(top - 1.92, color="tab:red", ls="--", lw=1, label="peak − 1.92 (≈95% interval)")
    ax.plot([ov_p[args.param]], [ov_ll], "x", color="black", ms=10, mew=2, label="unrestricted best fit")
    ax.set_xscale("log")
    ax.set_xlabel(f"{args.param} (pinned)"); ax.set_ylabel("composite log-likelihood")
    inside = x[y >= top - 1.92]
    ci = f"{inside.min():.3g} – {inside.max():.3g}" if len(inside) else "n/a"
    ax.set_title((args.title + "\n" if args.title else "") + f"profile of {args.param}: points within 1.92 of peak span {ci}", fontsize=10)
    finite = y[np.isfinite(y)]
    ax.set_ylim(max(finite.min(), top - 40), top + 3)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=130)
    print(f"-> {out}.tsv\n-> {out}.png\npeak LL {peak:.3f}; unrestricted best {ov_ll:.3f}; ~95% range of {args.param}: {ci}")


if __name__ == "__main__":
    main()

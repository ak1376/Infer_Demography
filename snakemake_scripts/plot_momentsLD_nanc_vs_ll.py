#!/usr/bin/env python3
# snakemake_scripts/plot_momentsLD_nanc_vs_ll.py
#
# Diagnostic: N_ANC vs. log-likelihood across all MomentsLD per-run best fits
# under real_data_analysis/runs/run_*/inferences/MomentsLD/best_fit.pkl.

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_RUN_RE = re.compile(r"^run_(\d+)$")


def _load_pickle(path: Path):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (FileNotFoundError, EOFError):
        return None


def nanc_and_ll(blob) -> Optional[Tuple[float, float]]:
    if blob is None:
        return None
    ll = blob.get("best_ll")
    bp = blob.get("best_params")
    if isinstance(ll, list):
        idx = max(range(len(ll)), key=lambda i: ll[i])
        ll = ll[idx]
        bp = bp[idx] if isinstance(bp, list) else bp
    if not isinstance(bp, dict) or "N_ANC" not in bp or ll is None:
        return None
    return float(bp["N_ANC"]), float(ll)


def load_runs(runs_dir: Path):
    rows = []
    for d in sorted(runs_dir.iterdir(), key=lambda p: (
        int(m.group(1)) if (m := _RUN_RE.match(p.name)) else -1
    )):
        m = _RUN_RE.match(d.name)
        if not m:
            continue
        p = d / "inferences" / "MomentsLD" / "best_fit.pkl"
        result = nanc_and_ll(_load_pickle(p))
        if result is not None:
            n_anc, ll = result
            rows.append((int(m.group(1)), n_anc, ll))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True, type=Path,
                     help="real_data_analysis/runs directory (contains run_0, run_1, ...).")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = load_runs(args.runs_dir)
    print(f"momentsLD: {len(rows)} runs loaded from {args.runs_dir}")
    if not rows:
        raise SystemExit("No MomentsLD best_fit.pkl files found.")

    run_ids, n_anc, ll = zip(*rows)
    n_anc = np.array(n_anc)
    ll = np.array(ll)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # LL is negative, so plot -LL (positive) on a log y-axis -- lower is better.
    neg_ll = -ll

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(n_anc, neg_ll, s=30, color="tab:blue", edgecolor="k", linewidth=0.3)
    ax.set_xlabel("N_ANC")
    ax.set_ylabel("-Log-likelihood (lower is better)")
    ax.set_yscale("log")
    ax.set_title(f"MomentsLD: N_ANC vs. LL across {len(rows)} runs")

    out_png = args.out_dir / "momentsLD_nanc_vs_ll.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# godambe_correction_LRT/src/plot_ld_window_diagnostics.py

"""
Two diagnostic plots for the momentsLD real-data window/bootstrap pipeline:

  (A) per-window decay curves: each of the N raw (non-overlapping) windows,
      normalized individually the same way the real pipeline normalizes
      (bootstrap_ld.average_ld_structure on a single window), overlaid --
      reveals outlier windows, whether the curve actually decays across the
      r range used, and data-density consistency across windows.

  (B) bootstrap replicate spread: all bootstrap_sets.pkl replicates overlaid
      against the true aggregate mean curve -- reveals whether the current
      number of windows gives a stable estimate (narrow band) or not (wide,
      ragged band), and whether a few outlier replicates (usually driven by
      resampling an outlier window multiple times) dominate the spread.

Usage:
  python godambe_correction_LRT/src/plot_ld_window_diagnostics.py \
      --ld-stats-dir godambe_correction_LRT/real_arms/Chr3L/momentsld/LD_stats \
      --means-varcovs godambe_correction_LRT/real_arms/Chr3L/momentsld/means.varcovs.pkl \
      --bootstrap-sets godambe_correction_LRT/real_arms/Chr3L/momentsld/bootstrap_sets.pkl \
      --out-prefix godambe_correction_LRT/real_arms/Chr3L/momentsld/diagnostics
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_ld import load_windows, average_ld_structure, NORM_LD, NORM_H  # noqa: E402

# Representative two-point stats: within-CO, within-FR, cross CO-FR.
FOCUS_LD_STATS = ["DD_0_0", "DD_1_1", "DD_0_1"]


def bin_midpoints(bins):
    return np.array([(lo + hi) / 2.0 for lo, hi in bins])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ld-stats-dir", required=True, type=Path)
    ap.add_argument("--means-varcovs", required=True, type=Path)
    ap.add_argument("--bootstrap-sets", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True, type=Path)
    args = ap.parse_args()

    windows = load_windows(args.ld_stats_dir)
    win_list = list(windows.values())
    ld_names, h_names = win_list[0]["stats"]
    bins = win_list[0]["bins"]
    x = bin_midpoints(bins)
    focus_idx = [ld_names.index(s) for s in FOCUS_LD_STATS]

    with open(args.means_varcovs, "rb") as f:
        mv = pickle.load(f)
    with open(args.bootstrap_sets, "rb") as f:
        all_boot = pickle.load(f)

    # ---- Plot A: per-window normalized curves -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, stat, k in zip(axes, FOCUS_LD_STATS, focus_idx):
        per_window_curves = []
        for wid, w in windows.items():
            normed = average_ld_structure([w], ld_names, h_names)
            curve = np.array([normed[b][k] for b in range(len(bins))])
            per_window_curves.append(curve)
            ax.plot(x, curve, color="tab:blue", alpha=0.25, lw=1)
        mean_curve = np.array([mv["means"][b][k] for b in range(len(bins))])
        ax.plot(x, mean_curve, color="black", lw=2.5, label=f"aggregate mean ({len(win_list)} windows)")
        ax.set_xscale("log")
        ax.set_xlabel("recombination distance r (bin midpoint)")
        ax.set_ylabel(f"{stat} / {NORM_LD}")
        ax.set_title(f"{stat}: {len(per_window_curves)} windows overlaid")
        ax.legend(fontsize=8)
    fig.suptitle("(A) Per-window normalized decay curves -- outliers, shape, density", y=1.02)
    fig.tight_layout()
    outA = args.out_prefix.with_name(args.out_prefix.name + "_per_window.png")
    fig.savefig(outA, dpi=150, bbox_inches="tight")
    print(f"wrote {outA}")

    # ---- Plot B: bootstrap replicate spread --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, stat, k in zip(axes, FOCUS_LD_STATS, focus_idx):
        for rep in all_boot:
            curve = np.array([rep[b][k] for b in range(len(bins))])
            ax.plot(x, curve, color="tab:orange", alpha=0.05, lw=1)
        mean_curve = np.array([mv["means"][b][k] for b in range(len(bins))])
        ax.plot(x, mean_curve, color="black", lw=2.5, label=f"aggregate mean ({len(win_list)} windows)")
        ax.set_xscale("log")
        ax.set_xlabel("recombination distance r (bin midpoint)")
        ax.set_ylabel(f"{stat} / {NORM_LD}")
        ax.set_title(f"{stat}: {len(all_boot)} bootstrap replicates")
        ax.legend(fontsize=8)
    fig.suptitle(f"(B) Bootstrap replicate spread -- is {len(win_list)} windows enough?", y=1.02)
    fig.tight_layout()
    outB = args.out_prefix.with_name(args.out_prefix.name + "_bootstrap_spread.png")
    fig.savefig(outB, dpi=150, bbox_inches="tight")
    print(f"wrote {outB}")


if __name__ == "__main__":
    main()

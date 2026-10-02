#!/usr/bin/env python3
# godambe_correction_LRT/src/plot_pooled_ld_stats.py

"""
Plot the EMPIRICAL LD-decay curves (mean +/- sqrt(var) from means.varcovs.pkl,
overlaid with the bootstrap replicate spread from bootstrap_sets.pkl) for the
three representative two-point stats (within-pop0, within-pop1, cross), plus
the heterozygosity stats. No fitted model needed -- this is a look at the
aggregated data itself, before/instead of any moments-LD fit.

Usage:
  python godambe_correction_LRT/src/plot_pooled_ld_stats.py \
      --mv godambe_correction_LRT/real_arms/NoBreakpoints_Chr2L_Chr2R_Chr3L_Chr3R/momentsld/means.varcovs.pkl \
      --boots godambe_correction_LRT/real_arms/NoBreakpoints_Chr2L_Chr2R_Chr3L_Chr3R/momentsld/bootstrap_sets.pkl \
      --ld-stats-dir godambe_correction_LRT/real_arms/NoBreakpoints_Chr2L_Chr2R_Chr3L_Chr3R/momentsld/LD_stats \
      --title "NoBreakpoints_Chr2L_Chr2R_Chr3L_Chr3R (3Mb breakpoint buffer, 100kb blocks)" \
      --out godambe_correction_LRT/real_arms/NoBreakpoints_Chr2L_Chr2R_Chr3L_Chr3R/momentsld/pooled_ld_stats.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOCUS_LD_STATS = ["DD_0_0", "DD_1_1", "DD_0_1"]  # within-CO, within-FR, cross CO-FR
POP_LABELS = {"DD_0_0": "within-CO", "DD_1_1": "within-FR", "DD_0_1": "cross CO-FR"}


def bin_midpoints(edges):
    return np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mv", required=True, type=Path)
    ap.add_argument("--boots", required=True, type=Path)
    ap.add_argument("--ld-stats-dir", required=True, type=Path,
                     help="Any dir of LD_stats_window_*.pkl for this pool, just to read stat names.")
    ap.add_argument("--r-bins", required=True,
                     help="The EXACT r_bins string used to compute this mv/boots (e.g. cj.R_BINS or "
                          "EXCL_BREAKPOINTS_R_BINS) -- must match, or bins silently mismatch/truncate.")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with args.mv.open("rb") as f:
        mv = pickle.load(f)
    with args.boots.open("rb") as f:
        boots = pickle.load(f)

    sample_pkl = sorted(args.ld_stats_dir.glob("LD_stats_window_*.pkl"))[0]
    with sample_pkl.open("rb") as f:
        ld_names, h_names = pickle.load(f)["stats"]

    r_edges = np.array([float(x) for x in args.r_bins.split(",")])
    mids = bin_midpoints(r_edges)
    n_rbins = len(mids)
    n_rbin_arrays = len(mv["means"]) - 1  # last array is H, r-bin-independent
    if n_rbins != n_rbin_arrays:
        raise SystemExit(
            f"--r-bins implies {n_rbins} bins but mv has {n_rbin_arrays} r-bin arrays -- "
            f"wrong --r-bins for this mv/boots pair."
        )

    fig, axes = plt.subplots(1, len(FOCUS_LD_STATS) + 1, figsize=(4.5 * (len(FOCUS_LD_STATS) + 1), 4.2))

    for ax, stat in zip(axes[:-1], FOCUS_LD_STATS):
        idx = ld_names.index(stat)
        means = np.array([mv["means"][k][idx] for k in range(n_rbins)])
        errs = np.array([np.sqrt(mv["varcovs"][k][idx, idx]) for k in range(n_rbins)])

        for rep in boots:
            rep_vals = [rep[k][idx] for k in range(n_rbins)]
            ax.plot(mids, rep_vals, color="tab:blue", alpha=0.03, linewidth=1, zorder=1)

        ax.errorbar(mids, means, yerr=errs, fmt="o-", color="black", capsize=3,
                    linewidth=1.5, markersize=5, zorder=2, label="mean +/- SE (varcov)")
        ax.set_xscale("log")
        ax.set_xlabel("recombination distance r")
        ax.set_ylabel(stat)
        ax.set_title(POP_LABELS.get(stat, stat))
        ax.legend(fontsize=8)

    ax = axes[-1]
    h_pos = np.arange(len(h_names))
    h_means = np.array(mv["means"][-1])
    h_errs = np.array([np.sqrt(mv["varcovs"][-1][i, i]) for i in range(len(h_names))])
    for rep in boots:
        ax.scatter(h_pos, rep[-1], color="tab:blue", alpha=0.03, zorder=1)
    ax.errorbar(h_pos, h_means, yerr=h_errs, fmt="o", color="black", capsize=3, zorder=2)
    ax.set_xticks(h_pos)
    ax.set_xticklabels(h_names, rotation=45, ha="right")
    ax.set_ylabel("H")
    ax.set_title("heterozygosity")

    fig.suptitle(args.title or str(args.mv.parent))
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"done -> {args.out}")
    print(f"n_bootstrap_replicates = {len(boots)}")
    print(f"n_windows_pooled = {len(list(args.ld_stats_dir.glob('LD_stats_window_*.pkl')))}")


if __name__ == "__main__":
    main()

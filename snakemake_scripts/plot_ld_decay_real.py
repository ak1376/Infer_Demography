#!/usr/bin/env python3
"""
Plot real-data LD decay from the per-window LD_stats pickles, in the same
layout as empirical_vs_theoretical_comparison.pdf (moments.LD.Plotting.
plot_ld_curves_comp) but data only -- no model curve.

Windows are aggregated exactly as aggregate_ld_windows_real_by_arm does
(moments.LD.Parsing.bootstrap_data: means normalized by population 0's pi2,
varcovs from the across-window bootstrap). Each panel shows the mean (dashed)
and a +/- 1.96 SE band, as plot_ld_curves_comp draws the data.

Usage:
  python snakemake_scripts/plot_ld_decay_real.py \
      --ld-stats-dir <REAL_LD_ROOT>/Chr3L/LD_stats \
      --r-bins "0,1e-06,...,0.001" \
      --out <REAL_LD_ROOT>/Chr3L/ld_decay.pdf
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
import moments.LD.Plotting as ldplot
import numpy as np

# Same panels and labels as create_comparison_plot in src/MomentsLD_inference.py
STATS_TO_PLOT = [["DD_0_0"], ["DD_0_1"], ["DD_1_1"],
                 ["Dz_0_0_0"], ["Dz_0_1_1"], ["Dz_1_1_1"],
                 ["pi2_0_0_1_1"], ["pi2_0_1_0_1"], ["pi2_1_1_1_1"]]
LABELS = [[r"$D_0^2$"], [r"$D_0 D_1$"], [r"$D_1^2$"],
          [r"$Dz_{0,0,0}$"], [r"$Dz_{0,1,1}$"], [r"$Dz_{1,1,1}$"],
          [r"$\pi_{2;0,0,1,1}$"], [r"$\pi_{2;0,1,0,1}$"], [r"$\pi_{2;1,1,1,1}$"]]
ROWS, COLS = 3, 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ld-stats-dir", required=True, type=Path)
    ap.add_argument("--r-bins", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ld = {}
    for f in sorted(args.ld_stats_dir.glob("LD_stats_window_*.pkl"), key=lambda f: int(f.stem.split("_")[-1])):
        with open(f, "rb") as fh:
            ld[int(f.stem.split("_")[-1])] = pickle.load(fh)
    stat_names = next(iter(ld.values()))["stats"][0]

    mv = moments.LD.Parsing.bootstrap_data(ld)
    ms, vcs = mv["means"][:-1], mv["varcovs"][:-1]          # last entry = heterozygosity

    rs = np.array([float(x) for x in args.r_bins.split(",")])
    rs_to_plot = (rs[:-1] + rs[1:]) / 2

    fig = plt.figure(figsize=(6, 4), dpi=300)
    for i, (stats, label) in enumerate(zip(STATS_TO_PLOT, LABELS)):
        ax = plt.subplot(ROWS, COLS, i + 1)
        neg_vals = False
        for stat, lab in zip(stats, label):
            k = stat_names.index(stat)
            y = np.array([ms[j][k] for j in range(len(rs_to_plot))])
            err = np.array([vcs[j][k][k] ** 0.5 * 1.96 for j in range(len(rs_to_plot))])
            neg_vals |= bool(np.any(y <= 0))
            lo = y - err
            if not (stat.startswith("pi2") or neg_vals):
                lo = np.maximum(lo, y * 0.05)       # keep the band drawable on a log axis
            ax.set_prop_cycle(None)
            ax.fill_between(rs_to_plot, lo, y + err, alpha=0.25)
            ax.set_prop_cycle(None)
            ax.plot(rs_to_plot, y, "--", label=lab)
        ax.set_xscale("log")
        if not (stats[0].startswith("pi2") or neg_vals):
            ax.set_yscale("log")
        if i >= len(STATS_TO_PLOT) - COLS:
            ax.set_xlabel("$r$")
        ax.legend(frameon=False, fontsize=6)
        if i % COLS == 0:
            ax.set_ylabel("Statistic")

    with matplotlib.rc_context(ldplot.FONT_SETTINGS):
        fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"Saved -> {args.out}  ({len(ld)} windows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compare_ld_decay_autosomes.py

Overlay empirical LD decay curves across autosomes, one panel per LD statistic.

Each input is a moments.LD ``means.varcovs.pkl`` produced by
``aggregate_ld_statistics`` for a single chromosome. That dict holds:
    bins     : list of (r_lo, r_hi) recombination-bin edges (n_bins entries)
    stats    : (ld_stat_names[15], H_stat_names[3])
    means    : list of length n_bins + 1; means[b] is the (pi2-normalised)
               vector of LD statistics in bin b; means[-1] is heterozygosity
    varcovs  : matching list of bootstrap covariance matrices

For each LD statistic we plot its value vs. the recombination-bin midpoint,
overlaying one curve per chromosome with bootstrap standard-error bars.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--means",
        required=True,
        nargs="+",
        type=Path,
        help="means.varcovs.pkl files, one per chromosome (order matches --labels)",
    )
    p.add_argument(
        "--labels",
        required=True,
        nargs="+",
        help="Chromosome labels, parallel to --means",
    )
    p.add_argument("--out-pdf", required=True, type=Path)
    return p.parse_args()


def pretty_label(stat, pops):
    """Turn a raw moments stat name (e.g. 'pi2_0_1_1_1') into a readable label
    with population names substituted for the numeric indices."""
    parts = stat.split("_")
    kind = parts[0]
    try:
        names = [pops[int(i)] for i in parts[1:]]
    except (ValueError, IndexError):
        return stat
    if kind == "DD":
        a, b = names
        return rf"$D_{{{a}}}^2$" if a == b else rf"$D_{{{a}}}\,D_{{{b}}}$"
    if kind == "Dz":
        return rf"$Dz_{{{','.join(names)}}}$"
    if kind == "pi2":
        return rf"$\pi_2^{{{','.join(names)}}}$"
    if kind == "H":
        return rf"$H_{{{','.join(names)}}}$"
    return stat


def bin_midpoints(bins):
    """Geometric midpoint of each recombination bin (handles a 0 lower edge)."""
    mids = []
    for lo, hi in bins:
        if lo <= 0:
            mids.append(hi / 2.0)
        else:
            mids.append(np.sqrt(lo * hi))
    return np.asarray(mids, dtype=float)


def curve_for_stat(mv, stat_index):
    """Return (y, yerr) across bins for one LD-statistic column."""
    n_bins = len(mv["bins"])
    y = np.array([mv["means"][b][stat_index] for b in range(n_bins)], dtype=float)
    yerr = np.array(
        [np.sqrt(max(mv["varcovs"][b][stat_index, stat_index], 0.0)) for b in range(n_bins)],
        dtype=float,
    )
    return y, yerr


def main():
    args = parse_args()
    if len(args.means) != len(args.labels):
        raise SystemExit(
            f"--means ({len(args.means)}) and --labels ({len(args.labels)}) "
            "must have the same length"
        )

    data = {}
    for label, path in zip(args.labels, args.means):
        with open(path, "rb") as fh:
            data[label] = pickle.load(fh)

    # Use the first chromosome to define the statistic list / bins; sanity-check
    # the rest share the same layout.
    ref = data[args.labels[0]]
    ld_stat_names = list(ref["stats"][0])
    pop_names = list(ref["pops"])
    n_bins = len(ref["bins"])
    for label, mv in data.items():
        if list(mv["stats"][0]) != ld_stat_names:
            raise SystemExit(f"{label}: LD statistic names differ from reference")
        if len(mv["bins"]) != n_bins:
            raise SystemExit(f"{label}: number of r-bins differs from reference")

    mids = bin_midpoints(ref["bins"])

    n_stats = len(ld_stat_names)
    ncols = 3
    nrows = int(np.ceil(n_stats / ncols))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(args.labels), 1)))

    with PdfPages(args.out_pdf) as pdf:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 3.0 * nrows),
            squeeze=False,
            layout="constrained",
        )
        axes_flat = axes.flatten()

        for s, stat in enumerate(ld_stat_names):
            ax = axes_flat[s]
            for c, label in enumerate(args.labels):
                y, yerr = curve_for_stat(data[label], s)
                ax.errorbar(
                    mids,
                    y,
                    yerr=yerr,
                    marker="o",
                    ms=3,
                    lw=1.2,
                    capsize=2,
                    color=colors[c],
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_title(pretty_label(stat, pop_names), fontsize=11)
            ax.tick_params(labelsize=7)
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if s % ncols == 0:
                ax.set_ylabel("normalised value", fontsize=8)
            if s >= n_stats - ncols:
                ax.set_xlabel("recombination distance r", fontsize=8)

        # Hide any unused panels
        for k in range(n_stats, len(axes_flat)):
            axes_flat[k].set_visible(False)

        # constrained layout reserves space for an "outside" legend and the
        # suptitle, so neither overlaps the panels or each other.
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=len(args.labels),
            fontsize=10,
            frameon=False,
        )
        fig.suptitle("LD decay across autosomes", fontsize=14)
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

    print(f"✓ wrote {args.out_pdf}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot a real-data 2-population unfolded SFS: each population's own (marginal)
spectrum against the neutral constant-size expectation (proportional to 1/i),
plus the joint SFS as a log-scaled heatmap.

Usage:
  python snakemake_scripts/plot_unfolded_sfs.py \
      --sfs  real_data_analysis/data/drosophila_trimmed/Chr3L/unfolded.sfs.pkl \
      --meta real_data_analysis/data/drosophila_trimmed/Chr3L/unfolded.sfs.meta.json \
      --out  real_data_analysis/data/drosophila_trimmed/Chr3L/unfolded.sfs.png
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

# Palette (dataviz reference instance, light mode)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POP_COLORS = ["#2a78d6", "#eb6834"]      # categorical slots 1-2, validated (CVD dE 24.7)
BLUES = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_marginal(ax, counts, pop, color, other):
    n = len(counts) - 1
    i = np.arange(1, n)
    seg = counts[1:n]
    prop = seg / seg.sum()
    neutral = (1 / i) / np.sum(1 / i)
    ax.bar(i, prop, width=0.72, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.plot(i, neutral, color=INK_2, linestyle="--", linewidth=1.5, marker="o",
            markersize=4, markerfacecolor=SURFACE, zorder=3)
    ax.annotate("neutral, constant size (∝ 1/i)", xy=(i[1], neutral[1]),
                xytext=(14, 10), textcoords="offset points", ha="left",
                fontsize=8.5, color=INK_2)
    style(ax)
    ax.set_xticks(i)
    ax.set_xlabel(f"derived allele count in {pop} (of {n})", color=INK_2, fontsize=10)
    ax.set_ylabel("share of segregating sites", color=INK_2, fontsize=10)
    ax.set_title(f"{pop}: {int(seg.sum()):,} segregating sites", loc="left",
                 color=INK, fontsize=11, fontweight="bold", pad=20)
    ax.text(0, 1.015, f"not shown: {int(counts[0]):,} absent, {int(counts[n]):,} fixed in {pop} "
                      f"(segregating only in {other})",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sfs", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.sfs, "rb") as fh:
        fs = pickle.load(fh)
    meta = json.loads(args.meta.read_text())
    pops = list(getattr(fs, "pop_ids", None) or ["pop0", "pop1"])
    data = np.asarray(fs.data, dtype=float)
    L = float(meta["sequence_length"])

    fig = plt.figure(figsize=(15, 4.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.32)
    for k in range(2):
        marginal = np.asarray(fs.marginalize([1 - k]).data, dtype=float)
        plot_marginal(fig.add_subplot(gs[0, k]), marginal, pops[k], POP_COLORS[k], pops[1 - k])

    # Joint SFS: rows = pop 0 derived count, columns = pop 1 derived count.
    ax = fig.add_subplot(gs[0, 2])
    joint = np.ma.masked_where(data <= 0, data)
    joint[0, 0] = np.ma.masked                       # monomorphic corners carry no information
    joint[-1, -1] = np.ma.masked
    cmap = BLUES.copy()
    cmap.set_bad(SURFACE)
    im = ax.imshow(joint, origin="lower", cmap=cmap, aspect="auto",
                   norm=LogNorm(vmin=max(1, joint.min()), vmax=joint.max()))
    ax.set_xlabel(f"derived allele count in {pops[1]}", color=INK_2, fontsize=10)
    ax.set_ylabel(f"derived allele count in {pops[0]}", color=INK_2, fontsize=10)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("joint SFS (sites per cell, log scale)", loc="left", color=INK,
                 fontsize=11, fontweight="bold", pad=20)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8.5)
    cb.outline.set_visible(False)

    chrom = meta.get("chrom", "")
    fst = float(fs.Fst()) if hasattr(fs, "Fst") else float("nan")
    thetas = [float(fs.marginalize([1 - k]).Watterson_theta()) / L for k in range(2)]
    fig.suptitle(
        f"{chrom} unfolded SFS  ·  {int(fs.S()):,} segregating sites  ·  Fst = {fst:.3f}  ·  "
        f"Watterson θ/bp: {pops[0]} {thetas[0]:.2e}, {pops[1]} {thetas[1]:.2e}  "
        f"(L = {L / 1e6:.2f} Mb effective of {meta.get('region_length', L) / 1e6:.2f} Mb)",
        x=0.01, ha="left", y=1.02, color=INK, fontsize=11.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

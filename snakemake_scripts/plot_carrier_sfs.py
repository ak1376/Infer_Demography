#!/usr/bin/env python3
"""
Plot one population's unfolded SFS without and with a set of samples (e.g.
the In(3L)P carriers FR217 and FR361), side by side:

  without : the population minus the carriers
  with    : every sample in the population

Each panel shows the share of segregating sites per derived-allele count
against its own neutral constant-size expectation (proportional to 1/i); the
sample sizes differ, so each panel has its own x-axis.

Usage:
  python snakemake_scripts/plot_carrier_sfs.py \
      --input-vcf real_data_analysis/data/drosophila_trimmed/Chr3L/polarized.vcf.gz \
      --popfile   real_data_analysis/data/drosophila/popfile.txt \
      --pop FR --carriers FR217,FR361 \
      --out real_data_analysis/data/drosophila_trimmed/Chr3L/FR_sfs_with_without_carriers.png
"""
import argparse
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette (dataviz reference instance, light mode); slots 1-2 validated (CVD dE 24.7)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WITHOUT_COLOR, WITH_COLOR = "#2a78d6", "#eb6834"


def derived_matrix(vcf: Path, samples):
    """(n_sites, len(samples)) 0/1 derived-allele matrix from a polarized
    haploid VCF (AA INFO field); sites with missing GT or no usable AA skipped."""
    rows = []
    with gzip.open(vcf, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                col = {s: i for i, s in enumerate(fields)}
                missing = [s for s in samples if s not in col]
                if missing:
                    raise SystemExit(f"samples not in VCF: {missing}")
                idx = [col[s] for s in samples]
                continue
            aa = next((t[3:] for t in fields[7].split(";") if t.startswith("AA=")), None)
            if aa == fields[3]:
                flip = False
            elif aa == fields[4]:
                flip = True
            else:
                continue
            gts = [fields[i] for i in idx]
            if "." in gts:
                continue
            g = [int(x) for x in gts]
            rows.append([1 - x for x in g] if flip else g)
    return np.array(rows, dtype=np.uint8)


def sfs_share(counts, n):
    """Share of segregating sites (1..n-1) at each derived count."""
    hist = np.bincount(counts, minlength=n + 1)[1:n].astype(float)
    return hist / hist.sum(), int(hist.sum())


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input-vcf", required=True, type=Path)
    ap.add_argument("--popfile", required=True, type=Path)
    ap.add_argument("--pop", required=True)
    ap.add_argument("--carriers", required=True, help="comma-separated sample IDs")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    carriers = [s for s in args.carriers.split(",") if s]
    pop_samples = [l.split()[0] for l in open(args.popfile)
                   if len(l.split()) >= 2 and l.split()[1] == args.pop]
    missing = [c for c in carriers if c not in pop_samples]
    if missing:
        raise SystemExit(f"carriers not in {args.pop}: {missing}")
    others = [s for s in pop_samples if s not in carriers]
    k, n = len(carriers), len(others)

    X = derived_matrix(args.input_vcf, carriers + others)
    print(f"{X.shape[0]:,} polarized sites, {k} carriers + {n} non-carriers")

    who = " & ".join(carriers)
    panels = [
        (f"without {who}", f"{n} non-carriers", X[:, k:], WITHOUT_COLOR),
        (f"with {who}", f"all {n + k} {args.pop} flies", X, WITH_COLOR),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.12})
    for ax, (title, sub, G, color) in zip(axes, panels):
        m = G.shape[1]
        share, segregating = sfs_share(G.sum(axis=1), m)
        i = np.arange(1, m)
        neutral = (1 / i) / np.sum(1 / i)
        ax.bar(i, share, width=0.72, color=color, edgecolor=SURFACE, linewidth=2,
               label="observed", zorder=2)
        ax.plot(i, neutral, color=INK_2, linestyle="--", linewidth=1.5, marker="o", markersize=5,
                markerfacecolor=SURFACE, label="expected if constant size, no selection (∝ 1/i)", zorder=3)
        style(ax)
        ax.set_xticks(i)
        ax.set_xlabel(f"number of the {m} flies carrying the new (derived) allele",
                      color=INK_2, fontsize=10.5)
        ax.set_title(f"{title}  ({sub})", loc="left", color=INK, fontsize=12,
                     fontweight="bold", pad=22)
        ax.text(0, 1.02, f"{segregating:,} SNPs", transform=ax.transAxes,
                color=MUTED, fontsize=9.5, va="bottom")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    axes[0].set_ylabel("share of SNPs", color=INK_2, fontsize=10.5)

    fig.suptitle(
        f"{args.pop} site frequency spectrum, {args.input_vcf.parent.name}: without vs. with {who}",
        x=0.01, ha="left", y=1.04, color=INK, fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Zoom in on ONE validated-blocks boundary and show the raw SNP x SNP r2
matrix around it, per population -- lets you visually check whether a
boundary the pipeline called "good" or "bad" actually looks that way (a
visible drop in correlation right at the dashed line vs. correlation that
clearly bleeds across it).

Unlike plot_ld_blocks_zoom.py (which zooms to an entire 300kb Stage-1 chunk
and marks every block boundary inside it), this zooms tightly to a window
of +/- --margin bp around ONE chosen boundary, so that one boundary is easy
to see clearly rather than one of many in a big chunk.

If --csv (the validated_blocks_boundary_detail.csv from
compute_validated_blocks.py) is given, annotates the plot with that
boundary's actual cross_r2/floor/ratio/bad values for each population.

Usage:
  python godambe_correction_LRT/src/plot_boundary_zoom.py \
      --vcf real_data_analysis/data/drosophila_trim_Chr3L-447386-18392988/Chr3L/polarized.diploidGT.vcf.gz \
      --popfile real_data_analysis/data/drosophila/popfile.txt --arm Chr3L \
      --boundary-bp 16592569 --margin 26381 \
      --csv godambe_correction_LRT/real_arms/Chr3L/validated_blocks_boundary_detail.csv \
      --out godambe_correction_LRT/real_arms/Chr3L/boundary_zoom_bad.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pg_gpu import HaplotypeMatrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_validated_blocks import read_popfile, polymorphic_mask, pairwise_r2_cpu


def load_boundary_row(csv_path: Path, boundary_bp: int, pop: str) -> dict | None:
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["pop"] == pop and int(row["boundary_bp"]) == boundary_bp:
                return row
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--boundary-bp", type=int, required=True, help="exact block-boundary position to zoom on")
    ap.add_argument("--margin", type=int, default=26_381, help="bp shown on each side of the boundary")
    ap.add_argument("--csv", type=Path, default=None, help="validated_blocks_boundary_detail.csv, for annotation")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    co_samples, fr_samples = read_popfile(args.popfile)
    region_start = args.boundary_bp - args.margin
    region_end = args.boundary_bp + args.margin
    region = f"{args.arm}:{region_start}-{region_end}"
    print(f"region: {region}")

    hm_co = HaplotypeMatrix.from_vcf(args.vcf, region=region, samples=co_samples)
    hm_fr = HaplotypeMatrix.from_vcf(args.vcf, region=region, samples=fr_samples)
    positions = hm_co.positions

    poly_co = polymorphic_mask(hm_co.haplotypes)
    poly_fr = polymorphic_mask(hm_fr.haplotypes)

    r2_co = pairwise_r2_cpu(hm_co.haplotypes)
    r2_fr = pairwise_r2_cpu(hm_fr.haplotypes)

    positions_co = positions[poly_co]
    positions_fr = positions[poly_fr]
    r2_co = r2_co[np.ix_(poly_co, poly_co)]
    r2_fr = r2_fr[np.ix_(poly_fr, poly_fr)]
    print(f"CO: {len(positions_co)} polymorphic SNPs, FR: {len(positions_fr)} polymorphic SNPs")

    bidx_co = int(np.searchsorted(positions_co, args.boundary_bp))
    bidx_fr = int(np.searchsorted(positions_fr, args.boundary_bp))

    mask_co = np.triu(np.ones(r2_co.shape, dtype=bool), k=1)
    mask_fr = np.triu(np.ones(r2_fr.shape, dtype=bool), k=1)
    co_masked = np.where(mask_co, r2_co, np.nan)
    fr_masked = np.where(mask_fr, r2_fr, np.nan)
    vmax = np.nanpercentile(np.concatenate([co_masked[mask_co], fr_masked[mask_fr]]), 95)

    cmap = plt.cm.Greys.copy()
    cmap.set_bad(color="#eeeeee")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, masked, label, bidx, n_snps, pop in zip(
            axes, [co_masked, fr_masked], ["CO", "FR"], [bidx_co, bidx_fr],
            [len(positions_co), len(positions_fr)], ["CO", "FR"]):
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, interpolation="none")
        ax.axvline(bidx - 0.5, color="#d62728", lw=1.2, ls="--", alpha=0.9)
        ax.axhline(bidx - 0.5, color="#d62728", lw=1.2, ls="--", alpha=0.9)
        title = f"{label}  ({n_snps} SNPs)"
        if args.csv:
            row = load_boundary_row(args.csv, args.boundary_bp, pop)
            if row:
                tag = "BAD" if row["bad"] == "True" else "ok"
                title += f"\ncross_r2={float(row['cross_r2']):.3f}  floor={float(row['floor']):.3f}  ratio={float(row['ratio']):.2f}  [{tag}]"
        ax.set_title(title)
        ax.set_xlabel("SNP index")
    axes[0].set_ylabel("SNP index")
    fig.suptitle(f"{args.arm}:{region_start}-{region_end} -- boundary @ {args.boundary_bp:,} (dashed red)")
    fig.colorbar(axes[1].images[0], ax=axes, label="r2", shrink=0.7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

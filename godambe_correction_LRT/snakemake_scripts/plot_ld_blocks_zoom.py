#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/plot_ld_blocks_zoom.py
"""
Thin wrapper called by Snakemake rule `ld_chunk_zoom`.

Visualizes the validated block boundaries directly on top of the raw
SNP x SNP r^2 matrix, per population (CO/FR), for ONE chunk.

Heavy lifting lives in:
  godambe_correction_LRT/src/plot_ld_blocks_zoom.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plot_ld_blocks_zoom import compute_zoom_data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True, help="raw, UNMASKED diploidGT VCF for one arm")
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--blocks-bed", required=True, help="validated_blocks.bed")
    ap.add_argument("--chunk-index", type=int, default=0,
                     help="which 300kb (--chunk-size) chunk, 0-based from the arm's start")
    ap.add_argument("--chunk-size", type=int, default=300_000,
                     help="must match compute_validated_blocks.py's --chunk-size (default 300kb)")
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    data = compute_zoom_data(
        args.vcf, args.popfile, args.arm, args.blocks_bed,
        args.chunk_index, args.chunk_size,
    )
    region = data["region"]
    r2_co, r2_fr = data["r2_co"], data["r2_fr"]
    positions_co, positions_fr = data["positions_co"], data["positions_fr"]
    bidx_co, bidx_fr = data["bidx_co"], data["bidx_fr"]

    mask_co = np.triu(np.ones(r2_co.shape, dtype=bool), k=1)
    mask_fr = np.triu(np.ones(r2_fr.shape, dtype=bool), k=1)
    co_masked = np.where(mask_co, r2_co, np.nan)
    fr_masked = np.where(mask_fr, r2_fr, np.nan)

    vmax = np.nanpercentile(np.concatenate([co_masked[mask_co], fr_masked[mask_fr]]), 95)

    # black & white: white = uncorrelated, black = r2=vmax -- easiest scheme
    # for spotting block structure by eye. Masked (upper-triangle) cells get a
    # faint gray so they're distinguishable from real near-zero (white) cells.
    cmap = plt.cm.Greys.copy()
    cmap.set_bad(color="#eeeeee")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, masked, label, bidx, n_snps in zip(
            axes, [co_masked, fr_masked], ["CO", "FR"], [bidx_co, bidx_fr],
            [len(positions_co), len(positions_fr)]):
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, interpolation="none")
        for b in bidx:
            ax.axvline(b - 0.5, color="#d62728", lw=0.8, ls="--", alpha=0.8)
            ax.axhline(b - 0.5, color="#d62728", lw=0.8, ls="--", alpha=0.8)
        ax.set_title(f"{label}  ({n_snps} SNPs)")
        ax.set_xlabel("SNP index")
    axes[0].set_ylabel("SNP index")
    fig.suptitle(f"{args.arm} r2, chunk {args.chunk_index} ({region}) -- CO vs FR\n"
                 f"dashed red lines = validated block-tiling boundaries")
    fig.colorbar(axes[1].images[0], ax=axes, label="r2", shrink=0.7)
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()

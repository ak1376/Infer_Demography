#!/usr/bin/env python3
# godambe_correction_LRT/scripts/plot_ld_blocks_zoom.py
"""
Visualize the validated block boundaries (validated_blocks.bed) directly on
top of the raw SNP x SNP r^2 matrix, per population (CO/FR), for ONE 300kb
chunk -- the same physical unit and chunk boundaries compute_validated_blocks.py
used to compute that chunk's floor/crossing distance. Answers: does the block
tiling (computed from the aggregate decay curve) line up with anything visible
in this chunk's raw matrix?

Monomorphic-within-population sites are dropped (same filtering as
compute_validated_blocks.py) since pairwise_r2() silently returns 0 (not NaN)
for these, which would otherwise show up as fake "decorrelation" at those
sites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pg_gpu import HaplotypeMatrix


def read_popfile(popfile_path):
    co_samples, fr_samples = [], []
    with open(popfile_path) as f:
        for line in f:
            sample, pop = line.split()
            if pop == "CO":
                co_samples.append(sample)
            elif pop == "FR":
                fr_samples.append(sample)
    return co_samples, fr_samples


def read_bed(bed_path):
    blocks = []
    with open(bed_path) as f:
        for line in f:
            chrom, start, end = line.split()
            blocks.append((chrom, int(start), int(end)))
    return blocks


def polymorphic_mask(haplotypes):
    freq = haplotypes.mean(axis=0)
    return (freq > 0) & (freq < 1)


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

    co_samples, fr_samples = read_popfile(args.popfile)
    blocks = read_bed(args.blocks_bed)
    chrom_start = blocks[0][1]  # bed's first block starts at the arm's usable start

    chunk_start = chrom_start + args.chunk_index * args.chunk_size
    chunk_end = chunk_start + args.chunk_size
    region = f"{args.arm}:{chunk_start}-{chunk_end}"

    # block-tiling boundaries that fall strictly inside this chunk
    boundaries_bp = [b_end for (_, b_start, b_end) in blocks
                      if chunk_start < b_end < chunk_end]
    print(f"region: {region} (chunk {args.chunk_index}, {args.chunk_size / 1e3:.0f} kb)")
    print(f"block-tiling boundaries inside this chunk (bp): {boundaries_bp}")

    hm_co = HaplotypeMatrix.from_vcf(args.vcf, region=region, samples=co_samples)
    hm_fr = HaplotypeMatrix.from_vcf(args.vcf, region=region, samples=fr_samples)
    positions = hm_co.positions

    # capture haplotypes BEFORE pairwise_r2() -- it transfers to GPU internally,
    # after which .haplotypes returns a cupy array instead of numpy
    poly_co = polymorphic_mask(hm_co.haplotypes)
    poly_fr = polymorphic_mask(hm_fr.haplotypes)

    r2_co = hm_co.pairwise_r2().get()
    r2_fr = hm_fr.pairwise_r2().get()

    positions_co = positions[poly_co]
    positions_fr = positions[poly_fr]
    r2_co = r2_co[np.ix_(poly_co, poly_co)]
    r2_fr = r2_fr[np.ix_(poly_fr, poly_fr)]

    print(f"CO: {len(positions_co)} polymorphic SNPs, FR: {len(positions_fr)} polymorphic SNPs")

    bidx_co = [int(np.searchsorted(positions_co, b)) for b in boundaries_bp]
    bidx_fr = [int(np.searchsorted(positions_fr, b)) for b in boundaries_bp]

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

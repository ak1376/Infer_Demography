#!/usr/bin/env python3
"""
Empirically check how sensitive Stage 1 of compute_validated_blocks.py (the
per-chunk LD-decay floor/crossing-distance estimate) is to the --chunk-size
choice.

For a single starting position, computes the binned-median-r2 decay curve
(and its derived floor + crossing distance) using several different chunk
sizes that all share that same start, and overlays them so you can see
whether the curve / floor / crossing distance actually change with chunk
size or are already stable by some size.

Reuses compute_validated_blocks.py's exact Stage 1 functions -- this is not
a reimplementation, it calls the same code with different --chunk-size-like
window widths.

Usage:
  python godambe_correction_LRT/src/chunk_size_sensitivity.py \
      --vcf real_data_analysis/data/drosophila/Chr3L/polarized.diploidGT.vcf.gz \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --arm Chr3L --start 5000000 \
      --chunk-sizes 100000,300000,600000,1200000 \
      --out godambe_correction_LRT/real_arms/Chr3L/chunk_size_sensitivity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pg_gpu import HaplotypeMatrix

from compute_validated_blocks import (
    read_popfile,
    polymorphic_mask,
    binned_median_r2,
    compute_floor,
    find_crossing_distance,
)


def _select_best_gpu() -> None:
    """Pick the CUDA device with the most free memory (same logic as
    src/LD_stats.py::_select_best_gpu) -- avoids landing on a device someone
    else on this shared host is already hammering."""
    import cupy as cp

    best_gpu, max_free_mem = 0, 0
    for gpu_id in range(cp.cuda.runtime.getDeviceCount()):
        cp.cuda.Device(gpu_id).use()
        free_mem, _total = cp.cuda.runtime.memGetInfo()
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_gpu = gpu_id
    cp.cuda.Device(best_gpu).use()
    name = cp.cuda.runtime.getDeviceProperties(best_gpu)["name"].decode()
    print(f"Using GPU {best_gpu} ({name}) with {max_free_mem / 1e9:.1f}GB free memory")


def run_one_chunk_size(vcf, arm, samples, start, chunk_size, floor_cutoff_bp, tolerance, n_bins):
    region = f"{arm}:{start}-{start + chunk_size}"
    hm = HaplotypeMatrix.from_vcf(vcf, region=region, samples=samples)
    positions = hm.positions
    if len(positions) < 2:
        return None
    poly = polymorphic_mask(hm.haplotypes)
    r2 = hm.pairwise_r2().get()
    positions = positions[poly]
    r2 = r2[np.ix_(poly, poly)]
    if len(positions) < 2:
        return None
    edges, medians = binned_median_r2(r2, positions, n_bins)
    floor = compute_floor(medians, edges, floor_cutoff_bp)
    crossing = find_crossing_distance(medians, edges, floor, tolerance)
    return dict(edges=edges, medians=medians, floor=floor, crossing=crossing, n_snps=len(positions))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--start", type=int, required=True, help="shared chunk start position (bp)")
    ap.add_argument("--chunk-sizes", required=True, help="comma-separated bp, e.g. 100000,300000,600000,1200000")
    ap.add_argument("--floor-cutoff-bp", type=int, default=20_000)
    ap.add_argument("--tolerance", type=float, default=1.10)
    ap.add_argument("--n-bins", type=int, default=60)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    _select_best_gpu()
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    co_samples, fr_samples = read_popfile(args.popfile)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, pop, samples in [(axes[0], "CO", co_samples), (axes[1], "FR", fr_samples)]:
        for cs in chunk_sizes:
            r = run_one_chunk_size(args.vcf, args.arm, samples, args.start, cs,
                                    args.floor_cutoff_bp, args.tolerance, args.n_bins)
            if r is None:
                print(f"{pop} chunk_size={cs:,}: too few polymorphic sites, skipped")
                continue
            label = f"{cs:,} bp (n_snps={r['n_snps']}, floor={r['floor']:.4f}, crossing={r['crossing']})"
            ax.plot(r["edges"], r["medians"], marker="o", markersize=2, linewidth=1, label=label)
            ax.axhline(r["floor"], linestyle=":", linewidth=0.7, alpha=0.5)
            print(f"{pop} chunk_size={cs:,}: n_snps={r['n_snps']}, floor={r['floor']:.4f}, "
                  f"crossing_distance={r['crossing']}")
        ax.set_xscale("log")
        ax.set_xlabel("Distance between SNP pair (bp)")
        ax.set_title(f"{pop}, chunk start={args.start:,}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Median r2 per distance bin")
    fig.suptitle(f"{args.arm}: LD-decay curve vs. chunk size, all starting at {args.start:,}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

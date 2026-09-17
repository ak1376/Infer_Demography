#!/usr/bin/env python3
# godambe_correction_LRT/src/plot_ld_blocks_zoom.py
"""
Data/computation for visualizing the validated block boundaries
(validated_blocks.bed) against the raw SNP x SNP r^2 matrix, per population
(CO/FR), for one chunk of an arm's VCF.

Monomorphic-within-population sites are dropped (same filtering as
src/compute_validated_blocks.py) since pairwise_r2() silently returns 0 (not
NaN) for these, which would otherwise show up as fake "decorrelation" at
those sites.

CLI wrapper: godambe_correction_LRT/snakemake_scripts/plot_ld_blocks_zoom.py
"""

from __future__ import annotations

import numpy as np
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


def compute_zoom_data(vcf, popfile, arm, blocks_bed, chunk_index, chunk_size):
    """Compute the CO/FR r^2 matrices (polymorphic sites only) and the
    block-tiling boundary indices for one chunk of an arm's VCF.

    Returns a dict: region, boundaries_bp, r2_co, r2_fr, positions_co,
    positions_fr, bidx_co, bidx_fr.
    """
    co_samples, fr_samples = read_popfile(popfile)
    blocks = read_bed(blocks_bed)
    chrom_start = blocks[0][1]  # bed's first block starts at the arm's usable start

    chunk_start = chrom_start + chunk_index * chunk_size
    chunk_end = chunk_start + chunk_size
    region = f"{arm}:{chunk_start}-{chunk_end}"

    # block-tiling boundaries that fall strictly inside this chunk
    boundaries_bp = [b_end for (_, b_start, b_end) in blocks
                      if chunk_start < b_end < chunk_end]
    print(f"region: {region} (chunk {chunk_index}, {chunk_size / 1e3:.0f} kb)")
    print(f"block-tiling boundaries inside this chunk (bp): {boundaries_bp}")

    hm_co = HaplotypeMatrix.from_vcf(vcf, region=region, samples=co_samples)
    hm_fr = HaplotypeMatrix.from_vcf(vcf, region=region, samples=fr_samples)
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

    return dict(
        region=region, boundaries_bp=boundaries_bp,
        r2_co=r2_co, r2_fr=r2_fr,
        positions_co=positions_co, positions_fr=positions_fr,
        bidx_co=bidx_co, bidx_fr=bidx_fr,
    )

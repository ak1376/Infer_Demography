#!/usr/bin/env python3
# godambe_correction_LRT/src/compute_validated_blocks.py
"""
Find a validated bootstrap block size for one chromosome arm.

Productionized version of the method built interactively in building_bridges.py.
For each population separately (never pooled -- pooling two differentiated
populations manufactures fake LD via the Wahlund effect), and per chunk of the
arm:
  1. Compute the exact SNP x SNP r^2 matrix (pg_gpu).
  2. Drop sites that are monomorphic WITHIN that population -- pairwise_r2()
     silently returns 0 (not NaN) for these, which otherwise contaminates
     every distance bin with meaningless zeros.
  3. Bin pairs by physical distance, take the median r^2 per bin.
  4. Floor = mean of the bins comfortably past any real decay.
  5. Crossing distance = smallest distance where the curve first gets within
     `tolerance` of that floor.

Combine across chunks via `percentile` (not the max -- a single noisy chunk
can otherwise dominate), take the max across populations (the bootstrap needs
one size safe for both), then VALIDATE it directly: tile the arm with the
chosen size and check that real adjacent blocks are actually near their floor.

Chunks that never reach their floor at all (typically centromere-proximal,
recombination-suppressed regions) mark the end of the usable arm; nothing
past the first such chunk is included in the final tiling.

CLI wrapper: godambe_correction_LRT/snakemake_scripts/compute_validated_blocks.py
"""

from __future__ import annotations

import numpy as np
from pg_gpu import HaplotypeMatrix


# --------------------------------------------------------------------------
# Core statistics (identical logic to building_bridges.py)
# --------------------------------------------------------------------------

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


def binned_median_r2(r2_matrix, positions, n_bins):
    """median r2 per log-spaced distance bin, computed directly from an
    already-materialized r2 matrix (no pg_gpu round-trip needed)."""
    n = r2_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    dist = positions[iu[1]] - positions[iu[0]]
    r2_vals = r2_matrix[iu]
    bins = np.logspace(0, np.log10(dist.max()), n_bins + 1)
    edges = bins[1:]
    bin_idx = np.clip(np.digitize(dist, bins) - 1, 0, n_bins - 1)
    medians = np.full(n_bins, np.nan)
    for k in range(n_bins):
        vals = r2_vals[bin_idx == k]
        if len(vals):
            medians[k] = np.median(vals)
    return edges, medians


def compute_floor(medians, edges, cutoff_bp):
    far = edges > cutoff_bp
    return float(np.mean(medians[far]))


def find_crossing_distance(medians, edges, floor, tolerance):
    """Smallest distance at which median r2 first drops to <= tolerance * floor."""
    for edge, m in zip(edges, medians):
        if m <= tolerance * floor:
            return float(edge)
    return None  # never reached the floor within the tested distance range


def polymorphic_mask(haplotypes):
    """Sites polymorphic WITHIN this set of haplotypes -- excludes sites that
    are only variable because of a *different* population's samples."""
    freq = haplotypes.mean(axis=0)
    return (freq > 0) & (freq < 1)


# --------------------------------------------------------------------------
# Per-chunk processing
# --------------------------------------------------------------------------

def process_chunk(vcf, popfile, co_samples, fr_samples, arm, chunk_start, chunk_end,
                   floor_cutoff_bp, tolerance, n_bins):
    """One chunk's floor/crossing-distance for CO and FR. Returns None if the
    chunk has too few polymorphic-within-population sites for either."""
    region = f"{arm}:{chunk_start}-{chunk_end}"
    try:
        hm_co = HaplotypeMatrix.from_vcf(vcf, region=region, samples=co_samples)
        hm_fr = HaplotypeMatrix.from_vcf(vcf, region=region, samples=fr_samples)
    except ValueError:
        return None

    positions = hm_co.positions
    if len(positions) < 2:
        return None

    # capture haplotypes (for the monomorphic-site mask) BEFORE pairwise_r2() --
    # pairwise_r2() transfers the HaplotypeMatrix to GPU internally, after which
    # .haplotypes returns a cupy array instead of numpy
    poly_co = polymorphic_mask(hm_co.haplotypes)
    poly_fr = polymorphic_mask(hm_fr.haplotypes)

    r2_co = hm_co.pairwise_r2().get()
    r2_fr = hm_fr.pairwise_r2().get()

    positions_co = positions[poly_co]
    positions_fr = positions[poly_fr]
    r2_co = r2_co[np.ix_(poly_co, poly_co)]
    r2_fr = r2_fr[np.ix_(poly_fr, poly_fr)]

    if len(positions_co) < 2 or len(positions_fr) < 2:
        return None

    edges_co, med_co = binned_median_r2(r2_co, positions_co, n_bins)
    edges_fr, med_fr = binned_median_r2(r2_fr, positions_fr, n_bins)

    floor_co = compute_floor(med_co, edges_co, floor_cutoff_bp)
    floor_fr = compute_floor(med_fr, edges_fr, floor_cutoff_bp)
    block_co = find_crossing_distance(med_co, edges_co, floor_co, tolerance)
    block_fr = find_crossing_distance(med_fr, edges_fr, floor_fr, tolerance)

    return dict(region=region, chunk_start=chunk_start, chunk_end=chunk_end,
                floor_co=floor_co, floor_fr=floor_fr, block_co=block_co, block_fr=block_fr)


def find_floor_for_position(chunk_results, pos):
    """The floor from whichever original process_chunk neighborhood contains
    this bp position -- floors vary locally, so a boundary's ratio is judged
    against its own neighborhood's background level."""
    for c in chunk_results:
        if c["chunk_start"] <= pos < c["chunk_end"]:
            return c["floor_co"], c["floor_fr"]
    return None, None


def validate_blocks_direct(vcf, co_samples, fr_samples, arm, blocks, chunk_results, group_size):
    """Cross-r2 for every REAL adjacent block-pair in the literal, final
    tiling (the same blocks written to validated_blocks.bed) -- unlike the
    old chunk-local re-tiling, this checks the EXACT boundaries that get
    bootstrapped, not a same-size approximation that drifts out of phase with
    them. Processes `group_size`+1 consecutive blocks per VCF fetch,
    overlapping by one block, so every real boundary is checked exactly once
    at its exact bp position.
    """
    detail_rows = []
    i = 0
    n = len(blocks)
    while i < n - 1:
        window = blocks[i: i + group_size + 1]
        if len(window) < 2:
            break
        region_start, region_end = window[0][0], window[-1][1]
        region = f"{arm}:{region_start}-{region_end}"
        hm_co = HaplotypeMatrix.from_vcf(vcf, region=region, samples=co_samples)
        hm_fr = HaplotypeMatrix.from_vcf(vcf, region=region, samples=fr_samples)

        positions = hm_co.positions
        # capture haplotypes BEFORE pairwise_r2() -- see process_chunk's comment
        poly_co = polymorphic_mask(hm_co.haplotypes)
        poly_fr = polymorphic_mask(hm_fr.haplotypes)

        r2_co = hm_co.pairwise_r2().get()
        r2_fr = hm_fr.pairwise_r2().get()

        positions_co = positions[poly_co]
        positions_fr = positions[poly_fr]
        r2_co = r2_co[np.ix_(poly_co, poly_co)]
        r2_fr = r2_fr[np.ix_(poly_fr, poly_fr)]

        # assign each SNP to its REAL block index, using the literal known
        # boundaries -- not a re-derived local tiling
        boundaries = [b[1] for b in window[:-1]]
        block_idx_co = np.searchsorted(boundaries, positions_co, side="right")
        block_idx_fr = np.searchsorted(boundaries, positions_fr, side="right")

        for k in range(len(window) - 1):
            boundary_bp = window[k][1]
            floor_co, floor_fr = find_floor_for_position(chunk_results, boundary_bp)

            idx_i_co = np.where(block_idx_co == k)[0]
            idx_j_co = np.where(block_idx_co == k + 1)[0]
            if floor_co is not None and len(idx_i_co) and len(idx_j_co):
                cr = float(np.median(r2_co[np.ix_(idx_i_co, idx_j_co)]))
                detail_rows.append(dict(pop="CO", block_index=i + k, boundary_bp=boundary_bp,
                                         block_i=list(window[k]), block_j=list(window[k + 1]),
                                         cross_r2=cr, floor=floor_co, ratio=cr / floor_co))

            idx_i_fr = np.where(block_idx_fr == k)[0]
            idx_j_fr = np.where(block_idx_fr == k + 1)[0]
            if floor_fr is not None and len(idx_i_fr) and len(idx_j_fr):
                cr = float(np.median(r2_fr[np.ix_(idx_i_fr, idx_j_fr)]))
                detail_rows.append(dict(pop="FR", block_index=i + k, boundary_bp=boundary_bp,
                                         block_i=list(window[k]), block_j=list(window[k + 1]),
                                         cross_r2=cr, floor=floor_fr, ratio=cr / floor_fr))

        i += group_size
    return detail_rows


# --------------------------------------------------------------------------
# Orchestration (everything except argparse + writing bed/report/plots)
# --------------------------------------------------------------------------

def find_validated_blocks(vcf, popfile, arm, chunk_size, floor_cutoff_bp,
                          tolerance, n_bins, percentile, validate_group_size):
    """Run the full validated-block-size analysis for one arm.

    Returns a dict with everything the CLI wrapper needs to write
    validated_blocks.bed / the JSON report / the two diagnostic plots:
      chrom_start, chrom_end, usable_chrom_end, chunk_results, blocks,
      block_co_all, block_fr_all, p_co, p_fr, final_block_size_bp,
      detail_rows, co_rows, fr_rows,
      total_pairs_co, total_bad_co, pct_bad_co,
      total_pairs_fr, total_bad_fr, pct_bad_fr,
      worst_co, worst_fr.
    """
    co_samples, fr_samples = read_popfile(popfile)
    print(f"{arm}: {len(co_samples)} CO samples, {len(fr_samples)} FR samples")

    full_hm = HaplotypeMatrix.from_vcf(vcf)
    chrom_start = int(full_hm.positions.min())
    chrom_end = int(full_hm.positions.max())
    del full_hm
    print(f"{arm} spans {chrom_start:,}-{chrom_end:,} "
          f"({(chrom_end - chrom_start) / 1e6:.1f} Mb)")

    # ---- per-chunk floor/crossing-distance ----
    chunk_results = []
    for chunk_start in range(chrom_start, chrom_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, chrom_end)
        if chunk_end - chunk_start < 2 * floor_cutoff_bp:
            continue  # too small a leftover chunk to reliably estimate a floor

        result = process_chunk(vcf, popfile, co_samples, fr_samples, arm,
                                chunk_start, chunk_end, floor_cutoff_bp,
                                tolerance, n_bins)
        if result is None:
            continue
        chunk_results.append(result)
        print(f"  {result['region']}: block_co={result['block_co']}, "
              f"block_fr={result['block_fr']}")

    block_co_all = np.array([c["block_co"] for c in chunk_results if c["block_co"] is not None])
    block_fr_all = np.array([c["block_fr"] for c in chunk_results if c["block_fr"] is not None])

    # ---- usable range: stop at the first chunk that never reached its floor
    # in EITHER population (typically centromere-proximal, no real decay) ----
    usable_chrom_end = chrom_end
    for c in chunk_results:
        if c["block_co"] is None and c["block_fr"] is None:
            usable_chrom_end = c["chunk_start"]
            break

    # ---- combine across chunks: percentile per population, then max ----
    p_co = float(np.percentile(block_co_all, percentile))
    p_fr = float(np.percentile(block_fr_all, percentile))
    final_block_size_bp = int(round(max(p_co, p_fr)))
    print(f"\nCO p{percentile:.0f} = {p_co:,.0f} bp, FR p{percentile:.0f} = {p_fr:,.0f} bp")
    print(f"final block size = {final_block_size_bp:,} bp")
    print(f"usable range: {arm}:{chrom_start:,}-{usable_chrom_end:,} "
          f"(excluding {(chrom_end - usable_chrom_end) / 1e6:.1f} Mb at the end)")

    # ---- the literal, final block list (same one written to --out-bed below) ----
    block_starts = list(range(chrom_start, usable_chrom_end, final_block_size_bp))
    blocks = [(start, min(start + final_block_size_bp, usable_chrom_end))
              for start in block_starts]

    # ---- validate: check the EXACT real adjacent blocks in `blocks`, not a
    # re-derived local approximation. detail_rows: one entry per real
    # adjacent block-pair, so we can see not just the pass/fail count but
    # HOW FAR above the floor the failures are. ----
    detail_rows = validate_blocks_direct(vcf, co_samples, fr_samples, arm,
                                          blocks, chunk_results, validate_group_size)

    co_rows = [r for r in detail_rows if r["pop"] == "CO"]
    fr_rows = [r for r in detail_rows if r["pop"] == "FR"]
    total_pairs_co, total_pairs_fr = len(co_rows), len(fr_rows)
    total_bad_co = sum(1 for r in co_rows if r["ratio"] > tolerance)
    total_bad_fr = sum(1 for r in fr_rows if r["ratio"] > tolerance)

    pct_bad_co = 100 * total_bad_co / total_pairs_co if total_pairs_co else float("nan")
    pct_bad_fr = 100 * total_bad_fr / total_pairs_fr if total_pairs_fr else float("nan")
    print(f"\nvalidation: CO {total_bad_co}/{total_pairs_co} adjacent pairs still above "
          f"{tolerance}x floor ({pct_bad_co:.1f}%)")
    print(f"validation: FR {total_bad_fr}/{total_pairs_fr} adjacent pairs still above "
          f"{tolerance}x floor ({pct_bad_fr:.1f}%)")

    worst_co = sorted(co_rows, key=lambda r: -r["ratio"])[:10]
    worst_fr = sorted(fr_rows, key=lambda r: -r["ratio"])[:10]
    print("\nworst CO offenders (block_i, block_j, cross-r2/floor ratio):")
    for r in worst_co:
        print(f"  block {r['block_index']}: {r['block_i']} <-> {r['block_j']}  ratio={r['ratio']:.2f}")
    print("worst FR offenders (block_i, block_j, cross-r2/floor ratio):")
    for r in worst_fr:
        print(f"  block {r['block_index']}: {r['block_i']} <-> {r['block_j']}  ratio={r['ratio']:.2f}")

    return dict(
        chrom_start=chrom_start, chrom_end=chrom_end, usable_chrom_end=usable_chrom_end,
        chunk_results=chunk_results, blocks=blocks,
        block_co_all=block_co_all, block_fr_all=block_fr_all,
        p_co=p_co, p_fr=p_fr, final_block_size_bp=final_block_size_bp,
        detail_rows=detail_rows, co_rows=co_rows, fr_rows=fr_rows,
        total_pairs_co=total_pairs_co, total_bad_co=total_bad_co, pct_bad_co=pct_bad_co,
        total_pairs_fr=total_pairs_fr, total_bad_fr=total_bad_fr, pct_bad_fr=pct_bad_fr,
        worst_co=worst_co, worst_fr=worst_fr,
    )

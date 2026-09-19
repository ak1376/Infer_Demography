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

import subprocess

import numpy as np
from pg_gpu import HaplotypeMatrix


def select_best_gpu() -> None:
    """Pick the CUDA device with the most free memory (same logic as
    src/LD_stats.py::_select_best_gpu) -- without this, cupy silently
    defaults to device 0, which may be heavily used by someone else on this
    shared host while other devices sit idle."""
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


def binned_median_r2(r2_matrix, positions, n_bins, max_pairs=2_000_000, seed=0):
    """median r2 per log-spaced distance bin, computed directly from an
    already-materialized r2 matrix (no pg_gpu round-trip needed).

    A dense chunk can have tens of millions of SNP pairs (n_snps choose 2).
    Using every single one doesn't change a bin's median in any meaningful
    way -- once a bin has a few thousand pairs, more samples just cost CPU
    time without changing the estimate -- but the pair count is NOT uniform
    across bins: within a fixed-width chunk, short distances have far more
    possible pairs than long ones (log-spaced bins only partly compensate),
    so the bins beyond floor_cutoff_bp -- exactly the ones the floor
    estimate depends on -- are naturally the sparsest to begin with.

    A single global uniform subsample would thin every bin by the same
    fraction, so those already-sparse far bins (the ones that matter most
    for a robust floor) would get hit hardest. Instead, cap pairs PER BIN
    (max_pairs / n_bins each): sparse far bins, usually already under that
    cap, keep essentially all their pairs; only the over-represented near
    bins get thinned. Distances/bin assignment are computed for every pair
    first (cheap, vectorized) -- only the expensive r2-value gather + median
    step operates on the capped-per-bin subset."""
    n = r2_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    dist = positions[iu[1]] - positions[iu[0]]
    bins = np.logspace(0, np.log10(dist.max()), n_bins + 1)
    edges = bins[1:]
    bin_idx = np.clip(np.digitize(dist, bins) - 1, 0, n_bins - 1)

    max_pairs_per_bin = max(1, max_pairs // n_bins)
    rng = np.random.default_rng(seed)
    keep_parts = []
    for k in range(n_bins):
        idx_k = np.where(bin_idx == k)[0]
        if len(idx_k) > max_pairs_per_bin:
            idx_k = rng.choice(idx_k, size=max_pairs_per_bin, replace=False)
        keep_parts.append(idx_k)
    keep = np.concatenate(keep_parts)

    iu = (iu[0][keep], iu[1][keep])
    bin_idx = bin_idx[keep]
    r2_vals = r2_matrix[iu]

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


def pairwise_r2_cpu(haplotypes):
    """Pure NumPy reimplementation of pg_gpu.HaplotypeMatrix.pairwise_r2('r2')
    (via its _pairwise_ld_core) -- identical math, no GPU/cupy involved.

    haplotypes: (n_samples, n_snps) array, negative = missing, 0/1 otherwise.

    On this host, GPU calls carry large, unpredictable CUDA driver/
    synchronization overhead (seconds to many minutes) that has nothing to
    do with the actual FLOP count -- and the actual FLOP count here is tiny
    regardless (the matmuls below have inner dimension = sample count, ~10).
    A plain BLAS-backed NumPy call sidesteps that overhead entirely."""
    hap = np.asarray(haplotypes, dtype=np.float64)
    valid_mask = (hap >= 0).astype(np.float64)
    hap_clean = np.where(hap >= 0, hap, 0.0)
    n_valid = valid_mask.sum(axis=0)
    p = np.where(n_valid > 0, hap_clean.sum(axis=0) / n_valid, 0.0)
    joint_n = valid_mask.T @ valid_mask
    joint_11 = hap_clean.T @ hap_clean
    p_AB = np.where(joint_n > 0, joint_11 / joint_n, 0.0)
    D = p_AB - np.outer(p, p)
    denom_squared = np.outer(p * (1 - p), p * (1 - p))
    r2 = np.where(denom_squared > 0, (D ** 2) / denom_squared, 0.0)
    np.fill_diagonal(r2, 0)
    return r2


# --------------------------------------------------------------------------
# Per-chunk processing
# --------------------------------------------------------------------------

def process_chunk(vcf, popfile, co_samples, fr_samples, arm, chunk_start, chunk_end,
                   floor_cutoff_bp, tolerance, n_bins, max_pairs=2_000_000, verbose_timing=False):
    """One chunk's floor/crossing-distance for CO and FR. Returns None if the
    chunk has too few polymorphic-within-population sites for either."""
    import time
    t0 = time.time()
    region = f"{arm}:{chunk_start}-{chunk_end}"
    try:
        hm_co = HaplotypeMatrix.from_vcf(vcf, region=region, samples=co_samples)
        hm_fr = HaplotypeMatrix.from_vcf(vcf, region=region, samples=fr_samples)
    except ValueError:
        return None

    positions = hm_co.positions
    if len(positions) < 2:
        return None
    if verbose_timing:
        print(f"  [{region}] from_vcf: {time.time()-t0:.2f}s, n_snps={len(positions)}")

    poly_co = polymorphic_mask(hm_co.haplotypes)
    poly_fr = polymorphic_mask(hm_fr.haplotypes)

    t1 = time.time()
    r2_co = pairwise_r2_cpu(hm_co.haplotypes)
    t2 = time.time()
    r2_fr = pairwise_r2_cpu(hm_fr.haplotypes)
    t3 = time.time()
    if verbose_timing:
        print(f"  [{region}] pairwise_r2 (CPU): CO={t2-t1:.2f}s, FR={t3-t2:.2f}s")

    positions_co = positions[poly_co]
    positions_fr = positions[poly_fr]
    r2_co = r2_co[np.ix_(poly_co, poly_co)]
    r2_fr = r2_fr[np.ix_(poly_fr, poly_fr)]

    if len(positions_co) < 2 or len(positions_fr) < 2:
        return None

    t4 = time.time()
    edges_co, med_co = binned_median_r2(r2_co, positions_co, n_bins, max_pairs=max_pairs)
    t5 = time.time()
    edges_fr, med_fr = binned_median_r2(r2_fr, positions_fr, n_bins, max_pairs=max_pairs)
    t6 = time.time()
    if verbose_timing:
        print(f"  [{region}] binned_median_r2: CO={t5-t4:.2f}s (n_pairs={len(positions_co)*(len(positions_co)-1)//2:,}), "
              f"FR={t6-t5:.2f}s (n_pairs={len(positions_fr)*(len(positions_fr)-1)//2:,})")

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
        poly_co = polymorphic_mask(hm_co.haplotypes)
        poly_fr = polymorphic_mask(hm_fr.haplotypes)

        r2_co = pairwise_r2_cpu(hm_co.haplotypes)
        r2_fr = pairwise_r2_cpu(hm_fr.haplotypes)

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
# Chunk enumeration (cheap: VCF span only, no r2) -- lets Stage 1 be
# parallelized across chunks as separate Snakemake jobs instead of one
# sequential Python loop.
# --------------------------------------------------------------------------

def get_chrom_span(vcf) -> tuple[int, int]:
    """(min POS, max POS) via bcftools -- no GPU/pg_gpu involved."""
    out = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", str(vcf)],
        stdout=subprocess.PIPE, check=True, text=True,
    )
    positions = [int(p) for p in out.stdout.split()]
    return min(positions), max(positions)


def get_chunk_bounds(vcf, chunk_size, floor_cutoff_bp):
    """(chrom_start, chrom_end, [(chunk_start, chunk_end), ...]) -- the exact
    same chunk enumeration find_validated_blocks used to do inline, including
    the "too small a leftover chunk" skip, but with no per-chunk r2 work."""
    chrom_start, chrom_end = get_chrom_span(vcf)
    chunks = []
    for chunk_start in range(chrom_start, chrom_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, chrom_end)
        if chunk_end - chunk_start < 2 * floor_cutoff_bp:
            continue  # too small a leftover chunk to reliably estimate a floor
        chunks.append((chunk_start, chunk_end))
    return chrom_start, chrom_end, chunks


# --------------------------------------------------------------------------
# Stage 2 + 3: combine per-chunk results into one block size, then validate
# against the real tiling. Takes already-computed chunk_results so this can
# run as a single aggregation step after Stage 1's chunks have been computed
# in parallel (e.g. as separate Snakemake jobs).
# --------------------------------------------------------------------------

def combine_and_validate(vcf, popfile, arm, chrom_start, chrom_end, chunk_results,
                          tolerance, percentile, validate_group_size):
    co_samples, fr_samples = read_popfile(popfile)

    block_co_all = np.array([c["block_co"] for c in chunk_results if c["block_co"] is not None])
    block_fr_all = np.array([c["block_fr"] for c in chunk_results if c["block_fr"] is not None])

    # ---- usable range: stop at the first chunk that never reached its floor
    # in EITHER population (typically centromere-proximal, no real decay) ----
    usable_chrom_end = chrom_end
    for c in sorted(chunk_results, key=lambda c: c["chunk_start"]):
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
        blocks=blocks, block_co_all=block_co_all, block_fr_all=block_fr_all,
        p_co=p_co, p_fr=p_fr, final_block_size_bp=final_block_size_bp,
        detail_rows=detail_rows, co_rows=co_rows, fr_rows=fr_rows,
        total_pairs_co=total_pairs_co, total_bad_co=total_bad_co, pct_bad_co=pct_bad_co,
        total_pairs_fr=total_pairs_fr, total_bad_fr=total_bad_fr, pct_bad_fr=pct_bad_fr,
        worst_co=worst_co, worst_fr=worst_fr,
    )


# --------------------------------------------------------------------------
# Orchestration: single-process convenience path (Stage 1 runs sequentially
# here). The parallel Snakemake path instead calls get_chunk_bounds,
# process_chunk (per chunk, in separate jobs), and combine_and_validate
# directly -- see rules chunk_bounds / process_one_chunk / validated_blocks.
# --------------------------------------------------------------------------

def find_validated_blocks(vcf, popfile, arm, chunk_size, floor_cutoff_bp,
                          tolerance, n_bins, percentile, validate_group_size):
    """Run the full validated-block-size analysis for one arm, sequentially.

    Returns the same dict as combine_and_validate, plus chunk_results.
    """
    co_samples, fr_samples = read_popfile(popfile)
    print(f"{arm}: {len(co_samples)} CO samples, {len(fr_samples)} FR samples")

    chrom_start, chrom_end, chunk_bounds = get_chunk_bounds(vcf, chunk_size, floor_cutoff_bp)
    print(f"{arm} spans {chrom_start:,}-{chrom_end:,} "
          f"({(chrom_end - chrom_start) / 1e6:.1f} Mb)")

    chunk_results = []
    for chunk_start, chunk_end in chunk_bounds:
        result = process_chunk(vcf, popfile, co_samples, fr_samples, arm,
                                chunk_start, chunk_end, floor_cutoff_bp,
                                tolerance, n_bins)
        if result is None:
            continue
        chunk_results.append(result)
        print(f"  {result['region']}: block_co={result['block_co']}, "
              f"block_fr={result['block_fr']}")

    result = combine_and_validate(vcf, popfile, arm, chrom_start, chrom_end,
                                   chunk_results, tolerance, percentile, validate_group_size)
    result["chunk_results"] = chunk_results
    return result

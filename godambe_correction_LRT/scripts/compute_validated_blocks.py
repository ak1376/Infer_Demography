#!/usr/bin/env python3
# godambe_correction_LRT/scripts/compute_validated_blocks.py
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
     `--tolerance` of that floor.

Combine across chunks via `--percentile` (not the max -- a single noisy chunk
can otherwise dominate), take the max across populations (the bootstrap needs
one size safe for both), then VALIDATE it directly: tile the arm with the
chosen size and check that real adjacent blocks are actually near their floor.

Chunks that never reach their floor at all (typically centromere-proximal,
recombination-suppressed regions) mark the end of the usable arm; nothing
past the first such chunk is included in the final tiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True, help="raw, UNMASKED diploidGT VCF for one arm")
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--arm", required=True, help="chromosome name, e.g. Chr3L")
    ap.add_argument("--chunk-size", type=int, default=300_000)
    ap.add_argument("--floor-cutoff-bp", type=int, default=20_000,
                     help="distance beyond which the decay curve is trusted to be flat")
    ap.add_argument("--tolerance", type=float, default=1.10,
                     help="'close enough to floor' = within this multiple of the floor")
    ap.add_argument("--n-bins", type=int, default=60)
    ap.add_argument("--percentile", type=float, default=99,
                     help="percentile (per population) used to combine per-chunk crossing "
                          "distances into the final block size; validated default is 99")
    ap.add_argument("--out-bed", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-histogram", required=True)
    ap.add_argument("--out-crossr2-check", required=True,
                     help="histogram of adjacent-block cross-r2 / floor ratios, i.e. how far "
                          "above the background floor 'bad' block-pairs actually are")
    ap.add_argument("--validate-group-size", type=int, default=4,
                     help="consecutive real blocks per VCF fetch during validation "
                          "(group_size+1 blocks per window, overlapping by 1, so every "
                          "real boundary is checked exactly once)")
    args = ap.parse_args()

    co_samples, fr_samples = read_popfile(args.popfile)
    print(f"{args.arm}: {len(co_samples)} CO samples, {len(fr_samples)} FR samples")

    full_hm = HaplotypeMatrix.from_vcf(args.vcf)
    chrom_start = int(full_hm.positions.min())
    chrom_end = int(full_hm.positions.max())
    del full_hm
    print(f"{args.arm} spans {chrom_start:,}-{chrom_end:,} "
          f"({(chrom_end - chrom_start) / 1e6:.1f} Mb)")

    # ---- per-chunk floor/crossing-distance ----
    chunk_results = []
    for chunk_start in range(chrom_start, chrom_end, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, chrom_end)
        if chunk_end - chunk_start < 2 * args.floor_cutoff_bp:
            continue  # too small a leftover chunk to reliably estimate a floor

        result = process_chunk(args.vcf, args.popfile, co_samples, fr_samples, args.arm,
                                chunk_start, chunk_end, args.floor_cutoff_bp,
                                args.tolerance, args.n_bins)
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
    p_co = float(np.percentile(block_co_all, args.percentile))
    p_fr = float(np.percentile(block_fr_all, args.percentile))
    final_block_size_bp = int(round(max(p_co, p_fr)))
    print(f"\nCO p{args.percentile:.0f} = {p_co:,.0f} bp, FR p{args.percentile:.0f} = {p_fr:,.0f} bp")
    print(f"final block size = {final_block_size_bp:,} bp")
    print(f"usable range: {args.arm}:{chrom_start:,}-{usable_chrom_end:,} "
          f"(excluding {(chrom_end - usable_chrom_end) / 1e6:.1f} Mb at the end)")

    # ---- the literal, final block list (same one written to --out-bed below) ----
    block_starts = list(range(chrom_start, usable_chrom_end, final_block_size_bp))
    blocks = [(start, min(start + final_block_size_bp, usable_chrom_end))
              for start in block_starts]

    # ---- validate: check the EXACT real adjacent blocks in `blocks`, not a
    # re-derived local approximation. detail_rows: one entry per real
    # adjacent block-pair, so we can see not just the pass/fail count but
    # HOW FAR above the floor the failures are. ----
    detail_rows = validate_blocks_direct(args.vcf, co_samples, fr_samples, args.arm,
                                          blocks, chunk_results, args.validate_group_size)

    co_rows = [r for r in detail_rows if r["pop"] == "CO"]
    fr_rows = [r for r in detail_rows if r["pop"] == "FR"]
    total_pairs_co, total_pairs_fr = len(co_rows), len(fr_rows)
    total_bad_co = sum(1 for r in co_rows if r["ratio"] > args.tolerance)
    total_bad_fr = sum(1 for r in fr_rows if r["ratio"] > args.tolerance)

    pct_bad_co = 100 * total_bad_co / total_pairs_co if total_pairs_co else float("nan")
    pct_bad_fr = 100 * total_bad_fr / total_pairs_fr if total_pairs_fr else float("nan")
    print(f"\nvalidation: CO {total_bad_co}/{total_pairs_co} adjacent pairs still above "
          f"{args.tolerance}x floor ({pct_bad_co:.1f}%)")
    print(f"validation: FR {total_bad_fr}/{total_pairs_fr} adjacent pairs still above "
          f"{args.tolerance}x floor ({pct_bad_fr:.1f}%)")

    # ---- save the block BED file (the exact `blocks` list just validated) ----
    Path(args.out_bed).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_bed, "w") as f:
        for start, end in blocks:
            f.write(f"{args.arm}\t{start}\t{end}\n")
    print(f"\nsaved {len(blocks)} blocks to {args.out_bed}")

    # worst-offending block-pairs per population (highest ratio above floor) --
    # candidates for excluding that specific block boundary from the tiling
    worst_co = sorted(co_rows, key=lambda r: -r["ratio"])[:10]
    worst_fr = sorted(fr_rows, key=lambda r: -r["ratio"])[:10]
    print("\nworst CO offenders (block_i, block_j, cross-r2/floor ratio):")
    for r in worst_co:
        print(f"  block {r['block_index']}: {r['block_i']} <-> {r['block_j']}  ratio={r['ratio']:.2f}")
    print("worst FR offenders (block_i, block_j, cross-r2/floor ratio):")
    for r in worst_fr:
        print(f"  block {r['block_index']}: {r['block_i']} <-> {r['block_j']}  ratio={r['ratio']:.2f}")

    # ---- save the JSON report ----
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        arm=args.arm, chrom_start=chrom_start, chrom_end=chrom_end,
        usable_chrom_end=usable_chrom_end, n_blocks=len(blocks),
        final_block_size_bp=final_block_size_bp, percentile=args.percentile,
        co_percentile_bp=p_co, fr_percentile_bp=p_fr,
        n_chunks=len(chunk_results),
        validation=dict(
            co_pairs=total_pairs_co, co_bad=total_bad_co, co_pct_bad=pct_bad_co,
            fr_pairs=total_pairs_fr, fr_bad=total_bad_fr, fr_pct_bad=pct_bad_fr,
            worst_co=worst_co, worst_fr=worst_fr,
        ),
    )
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved report to {args.out_report}")

    # ---- histogram of per-chunk crossing distances ----
    log_bins = np.logspace(0, np.log10(max(block_co_all.max(), block_fr_all.max())), 30)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].hist(block_co_all, bins=log_bins, color="#2a78d6", edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_title(f"CO ({len(block_co_all)} chunks)")
    axes[0].set_xlabel("crossing distance (bp, log scale)")
    axes[0].set_ylabel("number of chunks")
    axes[1].hist(block_fr_all, bins=log_bins, color="#eb6834", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_title(f"FR ({len(block_fr_all)} chunks)")
    axes[1].set_xlabel("crossing distance (bp, log scale)")
    fig.suptitle(f"Per-chunk crossing distances across {args.arm} -- CO vs FR")
    fig.tight_layout()
    Path(args.out_histogram).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_histogram, dpi=150)
    print(f"saved histogram to {args.out_histogram}")

    # ---- cross-r2/floor ratio check: how far above the floor are the
    # "bad" adjacent block-pairs, not just how many of them there are ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rows, label, color in zip(axes, [co_rows, fr_rows], ["CO", "FR"],
                                       ["#2a78d6", "#eb6834"]):
        ratios = np.array([r["ratio"] for r in rows], dtype=float)
        n_nonfinite = int(np.sum(~np.isfinite(ratios)))
        print(f"{label} ratio stats: n={len(ratios)}, nonfinite={n_nonfinite}, "
              f"min={np.nanmin(ratios[np.isfinite(ratios)]):.4g}, "
              f"max={np.nanmax(ratios[np.isfinite(ratios)]):.4g}")
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        good = ratios[ratios <= args.tolerance]
        bad = ratios[ratios > args.tolerance]
        lo = max(float(ratios.min()) * 0.9, 1e-3)
        hi = max(float(ratios.max()) * 1.1, lo * 1.5)
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)
        ax.hist(good, bins=bins, color=color, alpha=0.85, edgecolor="white",
                label=f"OK ({len(good)})")
        ax.hist(bad, bins=bins, color="#d62728", alpha=0.85, edgecolor="white",
                label=f"bad ({len(bad)})")
        ax.axvline(1.0, color="black", ls="-", lw=1.2, label="floor (ratio=1)")
        ax.axvline(args.tolerance, color="black", ls="--", lw=1.2,
                    label=f"tolerance ({args.tolerance}x)")
        ax.set_xscale("log")
        ax.set_xlabel("adjacent-block cross-r2 / that chunk's floor")
        ax.set_title(f"{label}: {len(bad)}/{len(ratios)} pairs bad "
                     f"({100 * len(bad) / len(ratios):.1f}%)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count (adjacent block pairs)")
    fig.suptitle(f"{args.arm}: how far above the background floor are 'bad' block pairs?")
    fig.tight_layout()
    Path(args.out_crossr2_check).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_crossr2_check, dpi=150)
    print(f"saved cross-r2 check to {args.out_crossr2_check}")


if __name__ == "__main__":
    main()

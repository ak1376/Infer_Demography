#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/compute_validated_blocks.py
"""
Thin wrapper called by Snakemake rule `validated_blocks` -- the AGGREGATION
step. Stage 1 (per-chunk floor/crossing-distance) now runs as separate,
parallel Snakemake jobs (checkpoint chunk_bounds + rule process_one_chunk);
this script consumes their already-computed result JSONs, does Stage 2
(percentile combine) + Stage 3 (validate against the real tiling), and
writes the block BED, a JSON report, and two diagnostic plots. See the
module docstring in src/compute_validated_blocks.py for the method.

Heavy lifting lives in:
  godambe_correction_LRT/src/compute_validated_blocks.py
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from compute_validated_blocks import get_chrom_span, combine_and_validate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True, help="raw, UNMASKED diploidGT VCF for one arm")
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--arm", required=True, help="chromosome name, e.g. Chr3L")
    ap.add_argument("--chunk-results", required=True, nargs="+", type=Path,
                     help="per-chunk result JSONs from rule process_one_chunk")
    ap.add_argument("--tolerance", type=float, default=1.10,
                     help="'close enough to floor' = within this multiple of the floor")
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
    ap.add_argument("--out-boundary-detail", default=None,
                     help="optional CSV of EVERY boundary's pop/position/cross_r2/floor/ratio "
                          "(not just the top-10 worst offenders in the JSON report) -- lets you "
                          "plot where bad boundaries fall along the chromosome")
    args = ap.parse_args()

    chrom_start, chrom_end = get_chrom_span(args.vcf)
    chunk_results = []
    for p in args.chunk_results:
        r = json.loads(Path(p).read_text())
        if r.get("empty"):
            continue
        chunk_results.append(r)
    print(f"{args.arm} spans {chrom_start:,}-{chrom_end:,}; "
          f"{len(chunk_results)}/{len(args.chunk_results)} chunks usable")

    result = combine_and_validate(
        vcf=args.vcf, popfile=args.popfile, arm=args.arm,
        chrom_start=chrom_start, chrom_end=chrom_end, chunk_results=chunk_results,
        tolerance=args.tolerance, percentile=args.percentile,
        validate_group_size=args.validate_group_size,
    )

    usable_chrom_end = result["usable_chrom_end"]
    blocks = result["blocks"]
    block_co_all = result["block_co_all"]
    block_fr_all = result["block_fr_all"]
    p_co, p_fr = result["p_co"], result["p_fr"]
    final_block_size_bp = result["final_block_size_bp"]
    co_rows, fr_rows = result["co_rows"], result["fr_rows"]
    total_pairs_co, total_bad_co, pct_bad_co = (
        result["total_pairs_co"], result["total_bad_co"], result["pct_bad_co"])
    total_pairs_fr, total_bad_fr, pct_bad_fr = (
        result["total_pairs_fr"], result["total_bad_fr"], result["pct_bad_fr"])
    worst_co, worst_fr = result["worst_co"], result["worst_fr"]

    if args.out_boundary_detail:
        import csv
        Path(args.out_boundary_detail).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_boundary_detail, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pop", "block_index", "boundary_bp", "cross_r2", "floor", "ratio", "bad"])
            for r in co_rows + fr_rows:
                w.writerow([r["pop"], r["block_index"], r["boundary_bp"],
                            r["cross_r2"], r["floor"], r["ratio"], r["ratio"] > args.tolerance])
        print(f"saved all {len(co_rows) + len(fr_rows)} boundary details to {args.out_boundary_detail}")

    # ---- save the block BED file (the exact `blocks` list just validated) ----
    Path(args.out_bed).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_bed, "w") as f:
        for start, end in blocks:
            f.write(f"{args.arm}\t{start}\t{end}\n")
    print(f"\nsaved {len(blocks)} blocks to {args.out_bed}")

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

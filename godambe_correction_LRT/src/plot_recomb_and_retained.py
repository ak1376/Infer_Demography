#!/usr/bin/env python3
"""
QC visualization for the centromere/telomere trimming decision.

Top panel : local Comeron recombination rate (cM/Mb) across the FULL,
            untrimmed chromosome arm.
Bottom panel: per-population segregating-SNP density (fixed windows) across
            that same full, untrimmed span (see snp_density_by_window.py).

Both panels shade the region actually RETAINED after trimming (the bounds
used by trim_chromosome_extents.py) so it's visually obvious whether the
excluded centromere/telomere-proximal flanks correspond to suppressed
recombination (and/or altered SNP density) relative to the retained region.

Read-only on the VCF and the Comeron xlsx.

Usage:
  python godambe_correction_LRT/src/plot_recomb_and_retained.py \
      --chrom Chr3L \
      --xlsx real_data_analysis/data/drosophila/recombination_maps/Comeron_100kb_R5_R6.xlsx \
      --vcf real_data_analysis/data/drosophila/Chr3L/polarized.diploidGT.vcf.gz \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --out godambe_correction_LRT/real_arms/Chr3L/recomb_and_retained.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "snakemake_scripts"))
from build_genetic_map import load_sheet_grid, find_block  # noqa: E402

from trim_chromosome_extents import KNOWN_EXTENTS  # noqa: E402
from snp_density_by_window import (  # noqa: E402
    parse_popfile,
    vcf_samples,
    detect_ploidy,
    segregating_positions,
    window_counts,
    all_positions,
)


def load_recomb_rate(xlsx: Path, chrom: str, sheet_substr: str = "R5") -> tuple[list[int], list[float]]:
    """Raw (midpoint_bp, rate_cM_per_Mb) pairs for chrom, sorted by position --
    same parse as build_genetic_map.py, but returning the rate itself rather
    than the cumulative map."""
    grid = load_sheet_grid(xlsx, sheet_substr)
    mid_col, rate_col, data_row0 = find_block(grid, chrom)
    max_row = max(rn for _, rn in grid)
    windows = []
    for rn in range(data_row0, max_row + 1):
        m = grid.get((mid_col, rn))
        r = grid.get((rate_col, rn))
        if m is None or r is None or str(m).strip() == "":
            continue
        windows.append((int(float(m)), float(r)))
    windows.sort()
    mids = [w[0] for w in windows]
    rates = [w[1] for w in windows]
    return mids, rates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chrom", required=True, help="e.g. Chr3L (must match both the VCF and the Comeron sheet)")
    ap.add_argument("--xlsx", required=True, type=Path, help="Comeron_100kb_R5_R6.xlsx")
    ap.add_argument("--vcf", required=True, type=Path, help="FULL, untrimmed chromosome VCF. Never modified.")
    ap.add_argument("--popfile", required=True, type=Path)
    ap.add_argument("--window-size", type=int, default=100_000, help="SNP-density window size in bp (default 100kb)")
    ap.add_argument("--retain-start", type=int, default=None, help="Default: KNOWN_EXTENTS[--chrom]")
    ap.add_argument("--retain-end", type=int, default=None, help="Default: KNOWN_EXTENTS[--chrom]")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.retain_start is not None and args.retain_end is not None:
        retain_start, retain_end = args.retain_start, args.retain_end
    else:
        if args.chrom not in KNOWN_EXTENTS:
            raise SystemExit(f"No known retain bounds for {args.chrom}; pass --retain-start/--retain-end.")
        retain_start, retain_end = KNOWN_EXTENTS[args.chrom]

    mids, rates = load_recomb_rate(args.xlsx, args.chrom)

    pops = parse_popfile(args.popfile)
    ploidy = detect_ploidy(args.vcf)
    present_in_vcf = set(vcf_samples(args.vcf))
    pos_all = all_positions(args.vcf)
    vcf_start, vcf_end = int(pos_all.min()), int(pos_all.max())

    fig, (ax_rate, ax_snp) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax_rate.plot(mids, rates, color="black", linewidth=1)
    ax_rate.axvspan(retain_start, retain_end, color="tab:green", alpha=0.15, label="retained (post-trim)")
    ax_rate.set_ylabel("Recombination rate\n(cM/Mb, Comeron)")
    ax_rate.set_title(f"{args.chrom}: recombination rate & SNP density, retained region shaded")
    ax_rate.legend(loc="upper right")

    for pop, samples in pops.items():
        present = [s for s in samples if s in present_in_vcf]
        if len(present) < 2:
            print(f"skipping {pop}: fewer than 2 samples present ({len(present)})")
            continue
        pos = segregating_positions(args.vcf, present, ploidy)
        win_starts, counts = window_counts(pos, vcf_start, vcf_end, args.window_size)
        ax_snp.plot(win_starts, counts, linewidth=1, label=f"{pop} (n={len(present)})")
    ax_snp.axvspan(retain_start, retain_end, color="tab:green", alpha=0.15)
    ax_snp.set_xlabel("Genomic position (bp)")
    ax_snp.set_ylabel(f"Segregating SNPs\nper {args.window_size:,} bp window")
    ax_snp.legend(loc="upper right")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

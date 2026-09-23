#!/usr/bin/env python3
"""
Plot local (Comeron) recombination rate across a chromosome arm, shading the
span of any cosmopolitan/endemic inversion(s) on that arm, plus the
centromere/telomere-proximal flanks (the region outside KNOWN_EXTENTS, i.e.
the part trim_chromosome_extents.py drops when trimming to the recombining
interval).

Breakpoints below are in FlyBase Release 5 (dm3) coordinates -- same
coordinate system as this project's Drosophila VCFs and the Comeron_100kb
map's "R5" sheet (see build_genetic_map.py). Source: published inversion
breakpoint table (Corbett-Detig & Hartl-style cosmopolitan/endemic
inversion list); "In(1)" inversions are on the X chromosome.

Usage:
  python godambe_correction_LRT/src/plot_inversion_recomb.py \
      --chrom Chr3L \
      --xlsx real_data_analysis/data/drosophila/recombination_maps/Comeron_100kb_R5_R6.xlsx \
      --out godambe_correction_LRT/real_arms/Chr3L/recomb_inversion.png
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

# name -> (chrom, distal_bp, proximal_bp), R5/dm3 coordinates.
INVERSIONS = {
    "In(2L)t": ("Chr2L", 2225744, 13154180),
    "In(2R)NS": ("Chr2R", 16163839, 11278659),
    "In(3R)K": ("Chr3R", 21966092, 7576289),
    "In(3R)Mo": ("Chr3R", 24857019, 17232639),
    "In(3R)P": ("Chr3R", 20569732, 12257931),
    "In(3L)P": ("Chr3L", 3173046, 16301941),
    "In(1)A": ("ChrX", 13519769, 19473361),
    "In(1)Be": ("ChrX", 17722945, 19487744),
}

# Full arm length, R5/dm3 (from this project's VCF/FASTA headers -- see
# drosophila_data/data/*.vcf.gz ##contig lines).
CHROM_LENGTH = {
    "Chr2L": 23011544,
    "Chr2R": 21146708,
    "Chr3L": 24543557,
    "Chr3R": 27905053,
    "ChrX": 22422827,
}


def _fmt(bp: int) -> str:
    return f"{bp:,} bp ({bp / 1e6:.2f} Mb)"


def print_summary(chrom: str) -> None:
    """One aligned row per quantity: full arm, centromere/telomere flanks,
    usable (C/T-excluded) region, and each inversion on this arm."""
    length = CHROM_LENGTH.get(chrom)
    retain = KNOWN_EXTENTS.get(chrom)
    arm_invs = {name: bp[1:] for name, bp in INVERSIONS.items() if bp[0] == chrom}

    if length is None or retain is None:
        return

    start, end = retain
    cen, tel = start, length - end
    usable = end - start

    rows = [
        ("full chromosome arm", length),
        ("centromere-proximal flank", cen),
        ("telomere-proximal flank", tel),
        ("usable region (C/T excluded)", usable),
    ]
    for name, (distal, proximal) in arm_invs.items():
        lo, hi = sorted((distal, proximal))
        rows.append((f"{name} inversion", hi - lo))

    label_w = max(len(label) for label, _ in rows)
    print(f"\n=== {chrom} size summary (R5/dm3) ===")
    for label, bp in rows:
        print(f"{label:<{label_w}} : {bp:>11,} bp  {bp / 1e6:>6.2f} Mb  {bp / length:>5.1%} of arm")

    for name, (distal, proximal) in arm_invs.items():
        lo, hi = sorted((distal, proximal))
        if lo >= start and hi <= end:
            print(f"\n{name} lies entirely within the usable region, so usable-with-inversion = {usable / 1e6:.2f} Mb (same as above).")
            print(f"usable region excluding {name} instead: {(usable - (hi - lo)) / 1e6:.2f} Mb")
        else:
            print(f"\nWARNING: {name} is NOT fully inside the usable region -- check coordinates.")
    print()


def compute_blocks(chrom: str, block_size: int, near_window: int):
    """Tile [0, arm_length) into block_size bins. Each bin is classified:
      'dropped'   - touches the centromere- or telomere-proximal flank at all
      'near'      - fully usable AND within near_window bp of an inversion
                    breakpoint on this arm (any overlap with [bp-w, bp+w])
      'far'       - fully usable, not near any breakpoint
    Returns a list of (start, end, category) tuples covering the whole arm."""
    length = CHROM_LENGTH[chrom]
    retain_start, retain_end = KNOWN_EXTENTS[chrom]
    breakpoints = []
    for name, (c, distal, proximal) in INVERSIONS.items():
        if c == chrom:
            breakpoints.append((name, "distal", distal))
            breakpoints.append((name, "proximal", proximal))

    n = length // block_size + (1 if length % block_size else 0)
    blocks = []
    for i in range(n):
        s, e = i * block_size, min((i + 1) * block_size, length)
        if s < retain_start or e > retain_end:
            cat = "dropped"
        else:
            near = any(s < bp + near_window and e > bp - near_window for _, _, bp in breakpoints)
            cat = "near" if near else "far"
        blocks.append((s, e, cat))
    return blocks, breakpoints


def load_recomb_rate(xlsx: Path, chrom: str, sheet_substr: str = "R5") -> tuple[list[int], list[float]]:
    """Raw (midpoint_bp, rate_cM_per_Mb) pairs for chrom, sorted by position."""
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
    ap.add_argument("--chrom", required=True, help="e.g. Chr3L (must match the Comeron sheet's arm naming)")
    ap.add_argument(
        "--xlsx",
        type=Path,
        default=Path("real_data_analysis/data/drosophila/recombination_maps/Comeron_100kb_R5_R6.xlsx"),
    )
    ap.add_argument("--sheet-substr", default="R5", help="R5 (dm3) matches this project's VCF coordinates")
    ap.add_argument(
        "--no-centromere-telomere",
        action="store_true",
        help="Skip shading the centromere/telomere-proximal flanks (KNOWN_EXTENTS complement).",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the a/b/c size breakdown to stdout.",
    )
    ap.add_argument(
        "--show-blocks",
        action="store_true",
        help="Add a bottom panel tiling the arm into --block-size bins, colored "
        "dropped (touches C/T flank) / near (within --near-window of a "
        "breakpoint) / far.",
    )
    ap.add_argument("--block-size", type=int, default=1_000_000)
    ap.add_argument("--near-window", type=int, default=3_000_000)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.no_summary:
        print_summary(args.chrom)

    mids, rates = load_recomb_rate(args.xlsx, args.chrom, args.sheet_substr)

    if args.show_blocks:
        fig, (ax, ax_blk) = plt.subplots(
            2, 1, figsize=(13, 5.5), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(13, 4))

    if not args.no_centromere_telomere:
        if args.chrom not in KNOWN_EXTENTS:
            raise SystemExit(f"No known centromere/telomere bounds for {args.chrom}; pass --no-centromere-telomere.")
        retain_start, retain_end = KNOWN_EXTENTS[args.chrom]
        arm_start = 0
        arm_end = CHROM_LENGTH.get(args.chrom, max(mids))
        ax.axvspan(arm_start, retain_start, color="gray", alpha=0.25, label="centromere/telomere-proximal (trimmed)")
        ax.axvspan(retain_end, arm_end, color="gray", alpha=0.25)

    ax.plot(mids, rates, color="black", linewidth=1)

    arm_inversions = {name: bp for name, bp in INVERSIONS.items() if bp[0] == args.chrom}
    colors = plt.cm.tab10.colors
    for i, (name, (_, distal, proximal)) in enumerate(arm_inversions.items()):
        lo, hi = sorted((distal, proximal))
        ax.axvspan(lo, hi, color=colors[i % len(colors)], alpha=0.2, label=name)

    ax.set_xlim(0, CHROM_LENGTH.get(args.chrom, max(mids)))
    ax.set_ylabel("Recombination rate\n(cM/Mb, Comeron)")
    ax.set_title(f"{args.chrom}: recombination rate, inversion span(s) shaded")
    if arm_inversions or not args.no_centromere_telomere:
        ax.legend(loc="upper right")

    if args.show_blocks:
        blocks, breakpoints = compute_blocks(args.chrom, args.block_size, args.near_window)
        cat_color = {"dropped": "gray", "far": "white", "near": "tab:orange"}
        cat_label = {
            "dropped": "dropped (touches centromere/telomere)",
            "far": f"kept, >{args.near_window / 1e6:.0f}Mb from a breakpoint",
            "near": f"kept, within {args.near_window / 1e6:.0f}Mb of a breakpoint",
        }
        seen = set()
        for s, e, cat in blocks:
            label = cat_label[cat] if cat not in seen else None
            seen.add(cat)
            ax_blk.axvspan(s, e, color=cat_color[cat], alpha=0.9 if cat != "far" else 1.0, label=label)
            ax_blk.axvline(s, color="black", linewidth=0.4)
        ax_blk.axvline(blocks[-1][1], color="black", linewidth=0.4)
        for _, _, bp in breakpoints:
            ax_blk.axvline(bp, color="red", linewidth=1, linestyle="--")
        ax_blk.set_yticks([])
        ax_blk.set_ylim(0, 1)
        ax_blk.set_xlabel("Genomic position (bp, R5/dm3)")
        ax_blk.set_ylabel(f"{args.block_size / 1e6:g}Mb\nblocks", fontsize=9)
        ax_blk.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=8, frameon=False)
    else:
        ax.set_xlabel("Genomic position (bp, R5/dm3)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

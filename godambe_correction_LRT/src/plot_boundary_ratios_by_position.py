#!/usr/bin/env python3
"""
Plot every validated-blocks boundary's cross-r2/floor ratio against its
genomic position, one panel per population -- lets you see whether "bad"
boundaries (ratio > tolerance) are scattered evenly along the chromosome or
concentrated in a specific region.

Input: the --out-boundary-detail CSV from
godambe_correction_LRT/snakemake_scripts/compute_validated_blocks.py
(columns: pop, block_index, boundary_bp, cross_r2, floor, ratio, bad).

Usage:
  python godambe_correction_LRT/src/plot_boundary_ratios_by_position.py \
      --csv godambe_correction_LRT/real_arms/Chr3L/validated_blocks_boundary_detail.csv \
      --tolerance 1.1 \
      --out godambe_correction_LRT/real_arms/Chr3L/boundary_ratios_by_position.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--tolerance", type=float, default=1.10)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    pops = sorted(set(r["pop"] for r in rows))

    fig, axes = plt.subplots(len(pops), 1, figsize=(14, 4 * len(pops)), sharex=True)
    if len(pops) == 1:
        axes = [axes]

    for ax, pop in zip(axes, pops):
        pop_rows = [r for r in rows if r["pop"] == pop]
        pos = [int(r["boundary_bp"]) for r in pop_rows]
        ratio = [float(r["ratio"]) for r in pop_rows]
        bad = [r["bad"] == "True" for r in pop_rows]

        good_pos = [p for p, b in zip(pos, bad) if not b]
        good_ratio = [r for r, b in zip(ratio, bad) if not b]
        bad_pos = [p for p, b in zip(pos, bad) if b]
        bad_ratio = [r for r, b in zip(ratio, bad) if b]

        ax.scatter(good_pos, good_ratio, s=10, color="tab:blue", label=f"OK ({len(good_pos)})")
        ax.scatter(bad_pos, bad_ratio, s=16, color="tab:red", label=f"bad ({len(bad_pos)})")
        ax.axhline(1.0, color="black", linewidth=1, label="floor (ratio=1)")
        ax.axhline(args.tolerance, color="black", linestyle="--", linewidth=1, label=f"tolerance ({args.tolerance}x)")
        ax.set_yscale("log")
        ax.set_ylabel("cross-r2 / floor")
        ax.set_title(f"{pop}: {len(bad_pos)}/{len(pos)} boundaries bad ({100*len(bad_pos)/len(pos):.1f}%)")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Genomic position (bp)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

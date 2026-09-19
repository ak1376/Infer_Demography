#!/usr/bin/env python3
"""
Cleaner view of where "bad" validated-blocks boundaries concentrate: bins
the chromosome into fixed-width windows and plots the FRACTION of boundaries
that are bad within each window, one line per population -- a smoothed
density instead of a noisy per-boundary scatter. Second panel overlays the
Comeron recombination rate for direct visual comparison.

Input: the --out-boundary-detail CSV from
godambe_correction_LRT/snakemake_scripts/compute_validated_blocks.py.

Usage:
  python godambe_correction_LRT/src/plot_bad_boundary_density.py \
      --csv godambe_correction_LRT/real_arms/Chr3L/validated_blocks_boundary_detail.csv \
      --chrom Chr3L \
      --xlsx real_data_analysis/data/drosophila/recombination_maps/Comeron_100kb_R5_R6.xlsx \
      --window-size 1000000 \
      --out godambe_correction_LRT/real_arms/Chr3L/bad_boundary_density.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "snakemake_scripts"))
from build_genetic_map import load_sheet_grid, find_block  # noqa: E402


def load_recomb_rate(xlsx: Path, chrom: str, sheet_substr: str = "R5"):
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
    return [w[0] for w in windows], [w[1] for w in windows]


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def binned_bad_fraction(positions, bad, start, end, window_size):
    edges = np.arange(start, end + window_size, window_size)
    n_bins = len(edges) - 1
    total = np.zeros(n_bins)
    n_bad = np.zeros(n_bins)
    idx = np.clip(np.digitize(positions, edges) - 1, 0, n_bins - 1)
    for i, b in zip(idx, bad):
        total[i] += 1
        n_bad[i] += int(b)
    frac = np.where(total > 0, n_bad / np.maximum(total, 1), np.nan)
    return edges[:-1], frac, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--window-size", type=int, default=1_000_000)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    pops = sorted(set(r["pop"] for r in rows))
    all_pos = [int(r["boundary_bp"]) for r in rows]
    start, end = min(all_pos), max(all_pos)

    mids, rates = load_recomb_rate(args.xlsx, args.chrom)

    fig, (ax_bad, ax_rate) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    for pop, color in zip(pops, ["tab:blue", "tab:orange"]):
        pop_rows = [r for r in rows if r["pop"] == pop]
        pos = [int(r["boundary_bp"]) for r in pop_rows]
        bad = [r["bad"] == "True" for r in pop_rows]
        win_starts, frac, total = binned_bad_fraction(pos, bad, start, end, args.window_size)
        ax_bad.plot(win_starts, frac, marker="o", markersize=3, linewidth=1.3,
                    color=color, label=f"{pop} (n={len(pos)} boundaries)")

    ax_bad.set_ylabel(f"Fraction bad boundaries\nper {args.window_size:,} bp window")
    ax_bad.set_ylim(-0.02, 1.02)
    ax_bad.set_title(f"{args.chrom}: where do validated-blocks boundary failures concentrate?")
    ax_bad.legend(loc="upper left", fontsize=9)
    ax_bad.axhline(0, color="gray", linewidth=0.5)

    ax_rate.plot(mids, rates, color="black", linewidth=1)
    ax_rate.set_ylabel("Recombination rate\n(cM/Mb, Comeron)")
    ax_rate.set_xlabel("Genomic position (bp)")
    ax_rate.set_xlim(start, end)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

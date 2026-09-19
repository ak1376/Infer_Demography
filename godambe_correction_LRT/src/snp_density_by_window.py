#!/usr/bin/env python3
"""
Windowed SNP-density QC plot for a chromosome VCF (typically one already
trimmed to drop centromere/telomere-proximal sequence, e.g. via
src/trim_chromosome_extents.py).

Bins the VCF's region into fixed-size, non-overlapping windows and counts,
separately per population, how many sites are segregating (both alleles
present -- not fixed ref or fixed alt) within that population's own samples
in each window. Plots genomic position (x) vs. SNP count (y), one line per
population on a shared panel.

Read-only on the input VCF.

Usage:
  python src/snp_density_by_window.py \
      --vcf real_data_analysis/data/drosophila_trim_Chr3L-447386-18392988/Chr3L/polarized.diploidGT.vcf.gz \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --window-size 100000 \
      --out godambe_correction_LRT/real_arms/Chr3L/snp_density_by_window.png
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_popfile(popfile_path: Path) -> dict[str, list[str]]:
    """<sample> <population> whitespace-separated file -> {pop: [samples]}."""
    pops: dict[str, list[str]] = {}
    for line in Path(popfile_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        sample, pop = parts[0], parts[1]
        pops.setdefault(pop, []).append(sample)
    return pops


def vcf_samples(vcf: Path) -> list[str]:
    out = subprocess.run(
        ["bcftools", "query", "-l", str(vcf)], stdout=subprocess.PIPE, check=True, text=True
    )
    return out.stdout.split()


def detect_ploidy(vcf: Path) -> int:
    """1 for haploid GT ("0"/"1"), 2 for diploid GT ("0/0", "1|1", ...)."""
    out = subprocess.run(
        ["bcftools", "query", "-f", "[%GT\n]", str(vcf)],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    first = next((l for l in out.stdout.splitlines() if l.strip()), None)
    if first is None:
        raise RuntimeError(f"No genotypes found in {vcf}")
    return 2 if ("/" in first or "|" in first) else 1


def all_positions(vcf: Path) -> np.ndarray:
    out = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", str(vcf)],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    return np.array([int(p) for p in out.stdout.split()], dtype=np.int64)


def segregating_positions(vcf: Path, samples: list[str], ploidy: int) -> np.ndarray:
    """POS of sites where `samples` are polymorphic (both alleles present),
    read-only against vcf -- subsets and filters via a pipe, writes nothing."""
    max_ac = ploidy * len(samples) - 1
    view = subprocess.run(
        ["bcftools", "view", "-s", ",".join(samples), "-c", "1", "-C", str(max_ac), str(vcf)],
        stdout=subprocess.PIPE,
        check=True,
    )
    pos_out = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n"],
        input=view.stdout,
        stdout=subprocess.PIPE,
        check=True,
    )
    return np.array([int(p) for p in pos_out.stdout.decode().split()], dtype=np.int64)


def window_counts(
    positions: np.ndarray, start: int, end: int, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-size, non-overlapping windows spanning [start, end]. Returns
    (window_start_positions, counts)."""
    edges = np.arange(start, end + window_size, window_size)
    counts, _ = np.histogram(positions, bins=edges)
    return edges[:-1], counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--vcf", required=True, type=Path, help="Chromosome VCF (bgzipped, tabix-indexed). Never modified."
    )
    ap.add_argument(
        "--popfile", required=True, type=Path, help="<sample> <population> whitespace-separated file"
    )
    ap.add_argument("--window-size", type=int, default=100_000, help="Window size in bp (default 100kb)")
    ap.add_argument("--out", required=True, type=Path, help="Output plot path (.png/.pdf)")
    ap.add_argument("--start", type=int, default=None, help="Override region start (default: VCF's own min POS)")
    ap.add_argument("--end", type=int, default=None, help="Override region end (default: VCF's own max POS)")
    args = ap.parse_args()

    pops = parse_popfile(args.popfile)
    ploidy = detect_ploidy(args.vcf)
    present_in_vcf = set(vcf_samples(args.vcf))

    if args.start is None or args.end is None:
        pos_all = all_positions(args.vcf)
        start = args.start if args.start is not None else int(pos_all.min())
        end = args.end if args.end is not None else int(pos_all.max())
    else:
        start, end = args.start, args.end

    fig, ax = plt.subplots(figsize=(12, 4))
    for pop, samples in pops.items():
        present = [s for s in samples if s in present_in_vcf]
        if len(present) < 2:
            print(f"skipping {pop}: fewer than 2 samples present in {args.vcf} ({len(present)})")
            continue
        pos = segregating_positions(args.vcf, present, ploidy)
        win_starts, counts = window_counts(pos, start, end, args.window_size)
        print(f"{pop}: n={len(present)} samples, {len(pos)} segregating sites total")
        ax.plot(win_starts, counts, marker="o", markersize=2, linewidth=1, label=f"{pop} (n={len(present)})")

    ax.set_xlabel("Genomic position (bp)")
    ax.set_ylabel(f"Segregating SNPs per {args.window_size:,} bp window")
    ax.set_title(Path(args.vcf).name)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

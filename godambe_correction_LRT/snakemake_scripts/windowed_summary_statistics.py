#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/windowed_summary_statistics.py

"""
Standalone diagnostic (not called by any Snakemake rule): windowed summary
statistics + between-block autocorrelation for the real Drosophila data.
See the module docstring in src/windowed_summary_statistics.py for the
pipeline.

Usage
-----
    python windowed_summary_statistics.py --n-windows 100
    python windowed_summary_statistics.py --n-windows 500 --max-lag 30

Heavy lifting lives in:
  godambe_correction_LRT/src/windowed_summary_statistics.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from windowed_summary_statistics import (
    load_allele_counts, make_windows, windowed_stats, autocorrelation,
    plot_stats_along_genome, plot_autocorrelation, sweep_window_counts,
)

ROOT = Path("/sietch_colab/akapoor/Infer_Demography")
REAL_VCF = str(ROOT / "real_data_analysis/data/drosophila/Chr2L/polarized.vcf.gz")
POPFILE = str(ROOT / "real_data_analysis/data/drosophila/popfile.txt")
OUT_DIR = ROOT / "figures"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-windows", type=int, default=100,
                   help="Number of equal-width non-overlapping windows.")
    p.add_argument("--max-lag", type=int, default=20,
                   help="Maximum lag (in windows) for the autocorrelation.")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep over several window counts and report the "
                        "smallest count (largest blocks) whose lag-1 "
                        "autocorrelation is inside the white-noise band for "
                        "every statistic.")
    p.add_argument("--sweep-counts", type=int, nargs="+",
                   default=[10, 20, 30, 50, 75, 100, 150, 200, 300, 500],
                   help="Candidate window counts for --sweep (ascending).")
    p.add_argument("--vcf", default=REAL_VCF)
    p.add_argument("--popfile", default=POPFILE)
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Step 1: load
    print("Loading allele counts from VCF ...")
    pos, ac, pop_names, n_samples = load_allele_counts(args.vcf, args.popfile)
    print(f"  {len(pos):,} sites, populations: {pop_names}")
    for pop in pop_names:
        print(f"    {pop}: {n_samples[pop]} haploid samples")

    if args.sweep:
        counts = sorted(set(args.sweep_counts))
        sweep_window_counts(
            pos, ac, pop_names, counts,
            OUT_DIR / "windowed_stats_acf_sweep.png",
        )
        return

    # Step 2: windows
    max_lag = min(args.max_lag, args.n_windows - 1)
    windows, centers = make_windows(pos, args.n_windows)
    span = int(pos.max() - pos.min())
    print(f"  {args.n_windows} windows over {span:,} bp "
          f"(~{span / args.n_windows:,.0f} bp/window)")

    # Step 3: per-window stats
    print("Computing windowed statistics ...")
    stats = windowed_stats(pos, ac, pop_names, windows)

    # Report how many windows are usable (finite) per statistic
    for pop in pop_names:
        n_pi = np.isfinite(stats["pi"][pop]).sum()
        n_d = np.isfinite(stats["tajima_d"][pop]).sum()
        print(f"    {pop}: pi finite in {n_pi}/{args.n_windows}, "
              f"Tajima's D finite in {n_d}/{args.n_windows} windows")
    if "fst" in stats:
        n_f = np.isfinite(stats["fst"]).sum()
        print(f"    Fst finite in {n_f}/{args.n_windows} windows")

    # Step 4: plot stats along the genome
    plot_stats_along_genome(
        centers, stats, pop_names, args.n_windows,
        OUT_DIR / f"windowed_stats_n{args.n_windows}.png",
    )

    # Step 5: autocorrelation
    plot_autocorrelation(
        stats, pop_names, args.n_windows, max_lag,
        OUT_DIR / f"windowed_stats_autocorr_n{args.n_windows}.png",
    )

    # Print lag-1 autocorrelation as a quick numeric summary
    print("\nLag-1 autocorrelation (want ~0 for independent blocks):")
    for pop in pop_names:
        _, acf_pi = autocorrelation(stats["pi"][pop], max_lag)
        _, acf_d = autocorrelation(stats["tajima_d"][pop], max_lag)
        print(f"    {pop}: pi={acf_pi[0]:+.3f}   Tajima's D={acf_d[0]:+.3f}")
    if "fst" in stats:
        _, acf_f = autocorrelation(stats["fst"], max_lag)
        print(f"    Fst: {acf_f[0]:+.3f}")
    b = 1.96 / np.sqrt(args.n_windows)
    print(f"    (white-noise 95% band at this window count: +/-{b:.3f})")


if __name__ == "__main__":
    main()

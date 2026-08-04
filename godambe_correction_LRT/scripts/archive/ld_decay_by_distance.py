#!/usr/bin/env python3
# godambe_correction_LRT/scripts/ld_decay_by_distance.py
"""
Empirical LD decay (r^2 vs. physical distance) for ONE VCF, used to pick a
bootstrap block size that's actually LD-independent.

Generalizes model_selection_helper_scripts/bootstrap_window_size.py (which
hardcoded a fixed list of arms) into a one-arm-at-a-time CLI script, so a
Snakemake rule can wildcard it per arm.

Steps (same as bootstrap_window_size.py):
  1. Load SNP positions + haploid genotypes per population from the VCF.
  2. Sweep candidate window counts -> candidate block sizes (bp) for this arm.
  3. Sample random SNP pairs, log-uniform in distance from 1 bp out to a few
     multiples of the largest candidate block size.
  4. Compute r^2 per pair per population, bin-average by distance.
  5. Background floor = mean r^2 beyond the largest candidate block.
  6. For each candidate block size and "closeness" threshold (ratio of r^2 to
     the floor), report whether that block size is small enough to still be
     at/below threshold * floor.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_popfile(popfile_path):
    sample_to_pop, pop_order = {}, []
    with open(popfile_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sample, pop = parts[0], parts[1]
            sample_to_pop[sample] = pop
            if pop not in pop_order:
                pop_order.append(pop)
    return pop_order, sample_to_pop


def load_positions_and_genotypes(vcf_path, popfile_path):
    """Returns (pos, geno, pop_names) where pos is a sorted (n_sites,) array
    of SNP positions and geno[pop] is an (n_sites, n_pop_samples) int8 array
    of 0/1 haploid allele calls, extracted in bulk via bcftools."""
    pop_names, sample_to_pop = _parse_popfile(popfile_path)

    header = subprocess.check_output(
        f"bcftools view -h '{vcf_path}' | tail -n 1", shell=True
    ).decode()
    vcf_samples = header.rstrip("\n").split("\t")[9:]
    sample_indices = {pop: [] for pop in pop_names}
    for i, s in enumerate(vcf_samples):
        pop = sample_to_pop.get(s)
        if pop is not None:
            sample_indices[pop].append(i)

    raw = subprocess.check_output(
        f"bcftools query -f '%POS[\\t%GT]\\n' '{vcf_path}'", shell=True
    ).decode()
    rows = [line.split("\t") for line in raw.splitlines() if line]
    pos = np.array([int(r[0]) for r in rows], dtype=np.int64)
    all_gt = np.array([[int(x) for x in r[1:]] for r in rows], dtype=np.int8)

    geno = {pop: all_gt[:, idx] for pop, idx in sample_indices.items()}
    return pos, geno, pop_names


def r_squared(geno_pop, i_idx, j_idx):
    """r^2 between each pair of SNPs (i_idx[k], j_idx[k]), computed from
    haploid allele-indicator vectors for one population's samples."""
    gi = geno_pop[i_idx].astype(np.float64)
    gj = geno_pop[j_idx].astype(np.float64)

    p1 = gi.mean(axis=1)
    p2 = gj.mean(axis=1)
    p11 = (gi * gj).mean(axis=1)

    num = (p11 - p1 * p2) ** 2
    den = p1 * (1 - p1) * p2 * (1 - p2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def bin_mean_r2(diff, r2, bins):
    """Average r^2 within each log-spaced distance bin, ignoring NaNs."""
    n_bins = len(bins) - 1
    bin_idx = np.clip(np.digitize(diff, bins) - 1, 0, n_bins - 1)
    centers = np.sqrt(bins[:-1] * bins[1:])

    mean_r2 = np.full(n_bins, np.nan)
    for k in range(n_bins):
        vals = r2[bin_idx == k]
        if len(vals):
            mean_r2[k] = np.nanmean(vals)
    return centers, mean_r2


def window_size_for_target(centers, mean_r2, target):
    """Smallest distance (bp) at which the LD-decay curve first drops to <= target."""
    valid = ~np.isnan(mean_r2)
    c = centers[valid]
    r = mean_r2[valid]
    for k in range(1, len(c)):
        if r[k] <= target:
            if r[k - 1] <= target or r[k - 1] == r[k]:
                return c[k]
            frac = (r[k - 1] - target) / (r[k - 1] - r[k])
            log_c = np.log(c[k - 1]) + frac * (np.log(c[k]) - np.log(c[k - 1]))
            return float(np.exp(log_c))
    return np.nan


def analyze(vcf_path, popfile_path, n_pairs, n_bins, seed,
            num_windows_min, num_windows_max, num_windows_steps):
    pos, geno, pop_names = load_positions_and_genotypes(vcf_path, popfile_path)
    print(f"loaded {len(pos):,} sites, populations: {pop_names}")
    for pop in pop_names:
        print(f"    {pop}: {geno[pop].shape[1]} samples")

    span = int(pos[-1] - pos[0])

    candidate_windows = np.unique(np.round(
        np.geomspace(num_windows_min, num_windows_max, num_windows_steps)
    ).astype(int))
    candidate_window_sizes = span / candidate_windows
    largest_window_size = np.max(candidate_window_sizes)
    max_value = 3 * largest_window_size

    rng = np.random.default_rng(seed)
    n_sites = len(pos)
    anchors = rng.integers(0, n_sites, size=n_pairs)
    target_dist = np.exp(rng.uniform(np.log(1.0), np.log(max_value), size=n_pairs))
    target_dist_absolute = pos[anchors] + target_dist
    partners = np.clip(np.searchsorted(pos, target_dist_absolute), 0, n_sites - 1)
    valid = partners != anchors
    anchors, partners = anchors[valid], partners[valid]
    diff = np.abs(pos[anchors] - pos[partners])

    r2 = {pop: r_squared(geno[pop], anchors, partners) for pop in pop_names}
    bins = np.logspace(0, np.log10(max_value), n_bins + 1)
    centers = None
    mean_r2 = {}
    for pop in pop_names:
        centers, mean_r2[pop] = bin_mean_r2(diff, r2[pop], bins)

    floor = {pop: np.nanmean(mean_r2[pop][centers > largest_window_size]) for pop in pop_names}
    for pop in pop_names:
        print(f"    floor[{pop}] = {floor[pop]:.4g}")

    return {
        "span": span,
        "pop_names": pop_names,
        "centers": centers,
        "mean_r2": mean_r2,
        "floor": floor,
        "largest_window_size": largest_window_size,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", required=True, type=Path)
    ap.add_argument("--popfile", required=True, type=Path)
    ap.add_argument("--label", default=None, help="plot title / row label (defaults to VCF stem)")
    ap.add_argument("--out-png", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--n-pairs", type=int, default=300_000)
    ap.add_argument("--n-bins", type=int, default=100)
    ap.add_argument("--seed", type=int, default=295)
    ap.add_argument("--num-windows-min", type=int, default=10)
    ap.add_argument("--num-windows-max", type=int, default=1000)
    ap.add_argument("--num-windows-steps", type=int, default=50)
    ap.add_argument("--ratio-thresholds", default="1.05,1.1,1.2,1.5,2.0")
    ap.add_argument("--highlight-threshold", type=float, default=1.1,
                     help="which threshold's block size to draw on the plot")
    args = ap.parse_args()

    label = args.label or args.vcf.stem
    thresholds = [float(t) for t in args.ratio_thresholds.split(",")]

    res = analyze(
        args.vcf, args.popfile, args.n_pairs, args.n_bins, args.seed,
        args.num_windows_min, args.num_windows_max, args.num_windows_steps,
    )
    pop_names = res["pop_names"]

    # window_bp[pop][t] = distance (bp) where mean r^2 first drops to <= t * floor
    window_bp = {pop: {} for pop in pop_names}
    for pop in pop_names:
        for t in thresholds:
            window_bp[pop][t] = window_size_for_target(
                res["centers"], res["mean_r2"][pop], t * res["floor"][pop]
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("arm,population,ratio_threshold,block_size_bp,floor_r2\n")
        for pop in pop_names:
            for t in thresholds:
                f.write(f"{label},{pop},{t},{window_bp[pop][t]},{res['floor'][pop]}\n")
    print(f"wrote {args.out_csv}")

    fig, axes = plt.subplots(1, len(pop_names), figsize=(7 * len(pop_names), 5), squeeze=False)
    axes = axes.flatten()
    for ax, pop in zip(axes, pop_names):
        ax.plot(res["centers"], res["mean_r2"][pop], marker="o", ms=3, lw=1.2)
        ax.axhline(res["floor"][pop], ls=":", alpha=0.5, label=f"floor = {res['floor'][pop]:.3g}")
        gbp = window_bp[pop].get(args.highlight_threshold, np.nan)
        if np.isfinite(gbp):
            ax.axvline(gbp, color="black", ls="--", lw=2,
                       label=f"block @ {args.highlight_threshold:.2f}x floor = {gbp:,.0f} bp")
        ax.set_xscale("log")
        ax.set_xlabel("Distance (bp)")
        ax.set_ylabel(r"mean $r^2$")
        ax.set_title(f"{label} — {pop}")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"LD decay vs. physical distance — {label}", fontsize=13)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()

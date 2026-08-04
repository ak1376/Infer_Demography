# ld_decay_for_bootstrap_window.py
"""
Empirical LD decay (r^2 vs physical distance) on the real Chr2L data, used to
pick a window size for the SFS bootstrap in bootstrapping_for_LRT.py.

Rather than checking one num_windows value at a time, this sweeps a range of
candidate num_windows values (--num-windows-min/--num-windows-max) and picks
the smallest window (largest num_windows) that still looks LD-independent:

  1. Sample random SNP pairs once, computing the LD decay curve (mean r^2 vs
     distance) out to a few multiples of the LARGEST candidate window size
     (the one from --num-windows-min). This single curve is reused for every
     candidate -- no resampling per candidate.
  2. Define one fixed background floor = mean r^2 beyond that largest
     candidate window -- a distance guaranteed to be background for every
     candidate in the sweep, not just whichever one is being tested.
  3. For every candidate num_windows, read r^2 at that window's size off the
     same curve and compute ratio = r^2_at_window / floor. ratio ~= 1 means
     windows of that size are effectively LD-independent; ratio >> 1 means
     they're still correlated (bootstrap SEs would be understated).
  4. Recommend the largest num_windows (smallest window) whose ratio is
     still <= --ratio-threshold.

These are haploid genotypes (one allele call per sample, no "/" or "|" in
GT), so r^2 is computed directly from allele-indicator vectors per
population -- no phasing ambiguity, unlike unphased diploid LD.

All-pairs LD is infeasible here (~500k SNPs on Chr2L), so pairs are sampled:
for each of many random anchor SNPs, pick a random target distance
(log-uniform up to the max distance plotted) and use the SNP nearest that
offset as the partner. This covers the full distance range without
materializing anything close to the full pairwise matrix.
"""

import argparse
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

real_vcf = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/Chr2L/polarized.vcf.gz"
popfile = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/popfile.txt"


def _parse_popfile(popfile_path):
    sample_to_pop = {}
    pop_order = []
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


def _load_positions_and_genotypes(vcf_path, popfile_path):
    """Returns (pos, geno, pop_names) where geno[pop] is an
    (n_sites, n_pop_samples) int8 array of 0/1 allele calls, using bcftools
    query for a fast bulk extract."""
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


def _sample_pairs(pos, max_dist, n_pairs, rng):
    """Random (anchor, partner) site-index pairs spanning ~1bp to max_dist."""
    n_sites = len(pos)
    anchors = rng.integers(0, n_sites, size=n_pairs)
    # Pick a target distance between SNPs drawn log-uniformly 
    target_dist = np.exp(rng.uniform(np.log(1.0), np.log(max_dist), size=n_pairs))
    # Convert the random distance into an absolute target genomic position: anchor's position plus the sampled offset. 
    target_pos = pos[anchors] + target_dist
    partners = np.clip(np.searchsorted(pos, target_pos), 0, n_sites - 1)
    valid = partners != anchors
    return anchors[valid], partners[valid]


def _r_squared(geno_pop, i_idx, j_idx):
    """Exact r^2 from haploid allele-indicator vectors (no phasing needed)."""
    gi = geno_pop[i_idx].astype(np.float64)
    gj = geno_pop[j_idx].astype(np.float64)
    p1 = gi.mean(axis=1)
    p2 = gj.mean(axis=1)
    p11 = (gi * gj).mean(axis=1)
    num = (p11 - p1 * p2) ** 2
    den = p1 * (1 - p1) * p2 * (1 - p2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _decay_curve(pos, geno_pop, max_dist, n_pairs, n_bins, rng):
    """Mean r^2 vs physical distance, log-binned out to max_dist."""
    i_idx, j_idx = _sample_pairs(pos, max_dist, n_pairs, rng)
    dist = np.abs(pos[j_idx] - pos[i_idx]).astype(np.float64)
    r2 = _r_squared(geno_pop, i_idx, j_idx)

    bins = np.logspace(0, np.log10(max_dist), n_bins + 1)
    bin_idx = np.clip(np.digitize(dist, bins) - 1, 0, n_bins - 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    mean_r2 = np.full(n_bins, np.nan)
    for k in range(n_bins):
        vals = r2[bin_idx == k]
        if len(vals):
            mean_r2[k] = np.nanmean(vals)
    return centers, mean_r2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-windows-min", type=int, default=10,
                    help="smallest num_windows candidate (-> LARGEST window size; "
                         "sets max_dist and the fixed background floor)")
    p.add_argument("--num-windows-max", type=int, default=1000,
                    help="largest num_windows candidate (-> smallest window size) to sweep")
    p.add_argument("--num-windows-steps", type=int, default=15,
                    help="number of log-spaced num_windows candidates between min and max")
    p.add_argument("--max-dist-mult", type=float, default=3.0,
                    help="sample the decay curve out to this many multiples of the "
                         "LARGEST candidate window size, leaving room beyond it to "
                         "estimate a fixed background floor")
    p.add_argument("--ratio-threshold", type=float, default=1.1,
                    help="recommend the largest num_windows (smallest window) whose "
                         "r^2-at-window / floor ratio is at or below this value")
    p.add_argument("--n-pairs", type=int, default=300_000,
                    help="random SNP pairs sampled per population, for the one shared decay curve")
    p.add_argument("--n-bins", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out",
                    default="/sietch_colab/akapoor/Infer_Demography/figures/ld_decay_window_size_sweep.png")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    pos, geno, pop_names = _load_positions_and_genotypes(real_vcf, popfile)
    chrom_start, chrom_end = pos[0], pos[-1]
    chrom_len = chrom_end - chrom_start

    num_windows_candidates = np.unique(np.round(
        np.geomspace(args.num_windows_min, args.num_windows_max, args.num_windows_steps)
    ).astype(int))
    window_sizes = chrom_len / num_windows_candidates
    window_size_max = chrom_len / args.num_windows_min
    max_dist = args.max_dist_mult * window_size_max

    print(f"Loaded {len(pos):,} sites, populations: {pop_names}")
    print(f"Chromosome span: {chrom_len:,.0f} bp")
    print(f"Sweeping {len(num_windows_candidates)} num_windows candidates in "
          f"[{args.num_windows_min}, {args.num_windows_max}]; curve sampled out to "
          f"{max_dist:,.0f} bp; fixed floor = mean r^2 beyond {window_size_max:,.0f} bp "
          f"(the largest candidate window)")

    fig, (ax_curve, ax_ratio) = plt.subplots(1, 2, figsize=(14, 5))
    recommended = {}

    for pop in pop_names:
        centers, mean_r2 = _decay_curve(
            pos, geno[pop], max_dist, args.n_pairs, args.n_bins, rng)
        floor = float(np.nanmean(mean_r2[centers > window_size_max]))
        r2_at_windows = np.interp(window_sizes, centers, mean_r2)
        ratio = r2_at_windows / floor

        passing = num_windows_candidates[ratio <= args.ratio_threshold]
        recommended[pop] = int(passing.max()) if len(passing) else None
        print(f"  {pop}: fixed floor = {floor:.4f}; recommended num_windows = "
              f"{recommended[pop] if recommended[pop] is not None else 'NONE (no candidate passed threshold)'}")

        line, = ax_curve.plot(centers, mean_r2, marker="o", ms=2, label=pop)
        ax_curve.axhline(floor, color=line.get_color(), ls=":", alpha=0.4)
        ax_ratio.plot(num_windows_candidates, ratio, marker="o", ms=4,
                      color=line.get_color(), label=pop)

    ax_curve.axvline(window_size_max, color="black", ls="--",
                      label=f"largest window (num_windows={args.num_windows_min})")
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("Distance (bp)")
    ax_curve.set_ylabel(r"mean $r^2$")
    ax_curve.set_title("LD decay (single sampled curve, shared across sweep)")
    ax_curve.legend(fontsize=8)

    ax_ratio.axhline(1.0, color="gray", ls="-", lw=1, alpha=0.6)
    ax_ratio.axhline(args.ratio_threshold, color="gray", ls="--", lw=1, alpha=0.6,
                      label=f"threshold={args.ratio_threshold}")
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("num_windows")
    ax_ratio.set_ylabel(r"$r^2$(window_size) / fixed floor")
    ax_ratio.set_title("Excess-correlation ratio vs. candidate window count")
    ax_ratio.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")

    valid = [v for v in recommended.values() if v is not None]
    if valid:
        print(f"\nConservative recommendation (safe for all populations): "
              f"num_windows = {min(valid)}")


if __name__ == "__main__":
    main()

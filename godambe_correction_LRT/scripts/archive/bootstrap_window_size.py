# bootstrap_window_size.py

"""
Because genomic data is linked, regular AIC is misleading for model comparison. What I want to do is a bootstrap windowed analysis 
to estimate the variability of the curvature of the likelihood surface. To do this, I need to do boostraps on windows of the real
data. However, how do we determine the optimal window size? We need to preserve linkage structure within blocks but different
blocks need to be as independent as possible. This is important because we want to accurately estimate the variability of the 
likelihood surface accurately, and we do this by having multiple independent replicates (just like in regular boostrapping).

We will do this in the following steps: 
    1. Load the data. I want to get the SNP positions 
    2. Sweep over number of windows (ex: 10 nonoverlapping windows, 100 nonoverlapping windows, etc.)
        2.1. Compute the window size w.r.t. number of non overlapping windows. 
    3. Determine the largest window size and define a max_dist value which will allow us to determine noise floor (because even really far apart SNPs have some level of linkage between them, albeit small). 
    4. Randomly sample a set number of SNP pairs. We want an equal number of short and long distances between them. The distances should go from 1 bp to max_dist. 
        4.1. Compute R2 value for each pair and bin by distance. 
        4.2. Average the R2 value for each bin
    5. Compute the background floor per population. Average the mean R2 values over bins whose distance is greater than window_size_max. 
    6. For each candidate window size, look up R2 from the curve
    7. Compute the ratio for each candidate window size. That is: ratio = mean R2 / fixed floor. 
    8. Combine across populations. We would be doing the above for CO and FR. But I want to have a window size that produces approx independent windows 
"""

# Autosomes to analyse. Keep this in sync with AUTOSOMES in the Snakefile
# (Chr3R is currently dropped because of the In(3R)Payne inversion signal).
DROSO_DIR = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila"
AUTOSOMES = ["Chr2L", "Chr3L"]
popfile = f"{DROSO_DIR}/popfile.txt"


def polarized_vcf(chrom):
    """Per-chromosome polarized (haploid + AA) VCF, mirroring the Snakefile helper."""
    return f"{DROSO_DIR}/{chrom}/polarized.vcf.gz"

import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_popfile(popfile_path):
    """Returns (pop_order, sample_to_pop) from a whitespace-separated
    <sample> <population> file."""
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
    haploid allele-indicator vectors for one population's samples.

    geno_pop: (n_sites, n_pop_samples) 0/1 array for one population.
    i_idx, j_idx: equal-length arrays of SNP row-indices into geno_pop.
    Returns an array of r^2 values, one per pair.
    """
    gi = geno_pop[i_idx].astype(np.float64)   # (n_pairs, n_pop_samples)
    gj = geno_pop[j_idx].astype(np.float64)

    p1 = gi.mean(axis=1)          # allele freq of SNP i, per pair
    p2 = gj.mean(axis=1)          # allele freq of SNP j, per pair
    p11 = (gi * gj).mean(axis=1)  # freq of the co-occurring allele pair

    num = (p11 - p1 * p2) ** 2
    den = p1 * (1 - p1) * p2 * (1 - p2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def bin_mean_r2(diff, r2, bins):
    """Average r^2 within each log-spaced distance bin, ignoring NaNs.

    diff: (n_pairs,) physical distance per pair.
    r2: (n_pairs,) r^2 per pair (may contain NaN).
    bins: (n_bins + 1,) bin edges.
    Returns (centers, mean_r2): one value per bin.
    """
    n_bins = len(bins) - 1
    # which bin each pair's distance falls into
    bin_idx = np.clip(np.digitize(diff, bins) - 1, 0, n_bins - 1)
    centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean of each bin's edges

    mean_r2 = np.full(n_bins, np.nan)
    for k in range(n_bins):
        vals = r2[bin_idx == k]
        if len(vals):
            mean_r2[k] = np.nanmean(vals)
    return centers, mean_r2


def window_size_for_target(centers, mean_r2, target):
    """Smallest distance (bp) at which the LD-decay curve first drops to <= target.

    Walks the (centers, mean_r2) curve from short to long distance and returns the
    first crossing, linearly interpolated in log-distance between the two bracketing
    bins. NaN bins (empty) are skipped. Returns np.nan if the curve never reaches
    the target within the sampled distance range.
    """
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


def analyze_chrom(chrom, popfile_path, n_pairs=300_000, n_bins=100, seed=295):
    """Run steps 1-5 for a single autosome.

    Returns a dict with the LD-decay curve (centers, mean_r2 per pop), the
    per-population background floor, the arm length, and the largest candidate
    window (the distance beyond which the floor is averaged). Bins/max_value are
    arm-specific (they scale with arm length), so window sizes are read off each
    arm's own curve and reported back in bp.
    """
    # Step 1: load positions + genotypes for this arm.
    pos, geno, pop_names = load_positions_and_genotypes(polarized_vcf(chrom), popfile_path)
    print(f"[{chrom}] loaded {len(pos):,} sites, populations: {pop_names}")
    for pop in pop_names:
        print(f"    {pop}: {geno[pop].shape[1]} samples")

    chr_length = int(pos[-1] - pos[0])

    # Step 2: candidate window counts -> window sizes (bp) for this arm.
    candidate_windows = np.unique(np.round(np.geomspace(10, 1000, 50)).astype(int))
    candidate_window_sizes = chr_length / candidate_windows

    # Step 3: largest candidate window; sample pairs out to 3x that distance so we
    # can see the curve flatten into its floor.
    largest_window_size = np.max(candidate_window_sizes)
    max_value = 3 * largest_window_size

    # Step 4: sample SNP pairs uniformly in log-distance from 1 bp to max_value.
    rng = np.random.default_rng(seed)
    n_sites = len(pos)
    anchors = rng.integers(0, n_sites, size=n_pairs)
    target_dist = np.exp(rng.uniform(np.log(1.0), np.log(max_value), size=n_pairs))
    target_dist_absolute = pos[anchors] + target_dist
    partners = np.clip(np.searchsorted(pos, target_dist_absolute), 0, n_sites - 1)
    valid = partners != anchors
    anchors, partners = anchors[valid], partners[valid]
    diff = np.abs(pos[anchors] - pos[partners])

    # Step 4.1/4.2: r^2 per pair per population, then bin-average.
    r2 = {pop: r_squared(geno[pop], anchors, partners) for pop in pop_names}
    bins = np.logspace(0, np.log10(max_value), n_bins + 1)
    centers = None
    mean_r2 = {}
    for pop in pop_names:
        centers, mean_r2[pop] = bin_mean_r2(diff, r2[pop], bins)

    # Step 5: per-population background floor = mean r^2 beyond the largest window.
    floor = {pop: np.nanmean(mean_r2[pop][centers > largest_window_size]) for pop in pop_names}
    for pop in pop_names:
        print(f"    floor[{pop}] = {floor[pop]:.4g}")

    return {
        "chrom": chrom,
        "chr_length": chr_length,
        "pop_names": pop_names,
        "centers": centers,
        "mean_r2": mean_r2,
        "floor": floor,
        "largest_window_size": largest_window_size,
    }


if __name__ == "__main__":

    ratio_thresholds = [1.05, 1.1, 1.2, 1.5, 2.0]

    # Steps 1-5 for every autosome.
    results = {chrom: analyze_chrom(chrom, popfile) for chrom in AUTOSOMES}
    pop_names = results[AUTOSOMES[0]]["pop_names"]

    # Step 6: window size per (arm, population, threshold), read off each arm's curve.
    #   window_bp[chrom][pop][t] = distance where mean r^2 first drops to <= t * floor.
    window_bp = {c: {pop: {} for pop in pop_names} for c in AUTOSOMES}
    for c in AUTOSOMES:
        res = results[c]
        for pop in pop_names:
            for t in ratio_thresholds:
                window_bp[c][pop][t] = window_size_for_target(
                    res["centers"], res["mean_r2"][pop], t * res["floor"][pop]
                )

    # ---- Table: window size (bp) per (arm, pop) and the cross-arm-cross-pop max ----
    col_keys = [(c, pop) for c in AUTOSOMES for pop in pop_names]
    header = f"{'closeness':>10}" + "".join(f"{c+'_'+pop:>14}" for c, pop in col_keys) + f"{'GENOME MAX':>14}"
    print("\nWindow size (bp) needed to reach each 'closeness x floor' level:")
    print(header)
    genome_bp = {}
    for t in ratio_thresholds:
        vals = [window_bp[c][pop][t] for c, pop in col_keys]
        gmax = np.nanmax(vals)          # binding constraint = slowest-decaying (arm, pop)
        genome_bp[t] = gmax
        row = f"{t:>10.2f}" + "".join(f"{v:>14,.0f}" for v in vals) + f"{gmax:>14,.0f}"
        print(row)

    # Identify which (arm, pop) sets the genome-wide block size at each threshold.
    print("\nBinding (slowest-decaying) arm/pop at each closeness level:")
    for t in ratio_thresholds:
        vals = {(c, pop): window_bp[c][pop][t] for c, pop in col_keys}
        finite = {k: v for k, v in vals.items() if np.isfinite(v)}
        if finite:
            k = max(finite, key=finite.get)
            print(f"  closeness {t:.2f}: block ~{finite[k]:,.0f} bp  (set by {k[0]} {k[1]})")
        else:
            print(f"  closeness {t:.2f}: curve never reached this level within sampled range")

    # Step 7: convert the GENOME-WIDE block size to non-overlapping window counts PER
    # ARM (each arm has its own length), and sum for the total number of bootstrap
    # blocks (the number of resampling units the Godambe variance estimate relies on).
    print("\nGenome-wide block size -> number of independent blocks (resampling units):")
    for t in ratio_thresholds:
        gbp = genome_bp[t]
        if not (np.isfinite(gbp) and gbp > 0):
            print(f"  closeness {t:.2f}: not reached within sampled range")
            continue
        per_arm = {c: int(results[c]["chr_length"] // gbp) for c in AUTOSOMES}
        total = sum(per_arm.values())
        per_arm_str = ", ".join(f"{c}:{per_arm[c]}" for c in AUTOSOMES)
        print(f"  closeness {t:.2f}: block ~{gbp:,.0f} bp -> blocks/arm [{per_arm_str}] -> TOTAL {total}")

    # ---- Overlay plot: one panel per population, one decay curve per arm ----
    arm_colors = plt.cm.tab10(np.linspace(0, 1, max(len(AUTOSOMES), 1)))
    fig, axes = plt.subplots(1, len(pop_names), figsize=(7 * len(pop_names), 5), squeeze=False)
    axes = axes.flatten()

    # Highlight one operating point on the plot (1.1x floor is a reasonable default).
    highlight_t = 1.1

    for ax, pop in zip(axes, pop_names):
        for ci, c in enumerate(AUTOSOMES):
            res = results[c]
            ax.plot(res["centers"], res["mean_r2"][pop], marker="o", ms=3, lw=1.2,
                    color=arm_colors[ci], label=c)
            ax.axhline(res["floor"][pop], color=arm_colors[ci], ls=":", alpha=0.5)
        gbp = genome_bp.get(highlight_t, np.nan)
        if np.isfinite(gbp):
            ax.axvline(gbp, color="black", ls="--", lw=2,
                       label=f"genome block @ {highlight_t:.2f}x floor\n= {gbp:,.0f} bp")
        ax.set_xscale("log")
        ax.set_xlabel("Distance (bp) = candidate block size")
        ax.set_ylabel(r"mean $r^2$")
        ax.set_title(f"LD decay — {pop}")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("Per-autosome LD decay and cross-arm block size", fontsize=13)
    fig.tight_layout()
    out_path = "/sietch_colab/akapoor/Infer_Demography/figures/bootstrap_window_size.png"
    fig.savefig(out_path, dpi=150)
    print(f'\nWrote {out_path}')





        

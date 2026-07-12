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

real_vcf = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/Chr2L/polarized.vcf.gz"
popfile = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/popfile.txt"

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


if __name__ == "__main__":

    # Step 1: Load the data.
    pos, geno, pop_names = load_positions_and_genotypes(real_vcf, popfile)
    print(f"Loaded {len(pos):,} sites, populations: {pop_names}")
    for pop in pop_names:
        print(f"  {pop}: {geno[pop].shape[1]} samples")
    
    chr_length = int(pos[-1] - pos[0])

    # Step 2: define the number of windows and calculate the window sizes
    candidate_windows = np.unique(np.round(np.geomspace(10, 1000, 50)).astype(int))
    candidate_window_sizes = chr_length / candidate_windows

    # Step 3: Determine the largest window size
    largest_window_size = np.max(candidate_window_sizes)

    max_value = 3*largest_window_size

    # Step 4: Randomly sample a set number of SNP pairs 
    seed = 295
    rng = np.random.default_rng(seed)   # create a reproducible random number generator
    n_sites = len(pos)
    n_pairs = 300_000                    # however many pairs you want to sample
    anchors = rng.integers(0, n_sites, size=n_pairs)
    print(f'Length of Anchors: {len(anchors)}')

    # Now what we do is we assign a distance to each anchor SNP. SNP B that is that distance away from SNP A will be a pair. 
    # Doing this in log space because that will allow us to uniformly pick distances across multiple scales of magnitude
    target_dist = np.exp(rng.uniform(np.log(1.0), np.log(max_value), size=n_pairs))
    # print(f'Target Dist: {target_dist}')
    target_dist_absolute = pos[anchors] + target_dist

    # Now find the partners 
    partners = np.searchsorted(pos, target_dist_absolute)
    partners = np.clip(partners, 0, n_sites - 1)

    # Edge case where the partner is the SNP itself
    valid = partners != anchors
    anchors, partners = anchors[valid], partners[valid]

    # print(f'Length of anchors after filter: {len(anchors)}')

    # print(f'Anchors: {anchors}')
    # print(f'Partners: {partners}')

    diff = np.abs(pos[anchors] - pos[partners])
    # print(f'Max Dist: {np.max(diff)}')
    # print(f'Min Dist: {np.min(diff)}')
    # print(f'Mean Dist: {np.mean(diff)}')

    # Now compute the R2 value for each pair and for each population (CO vs FR)
    r2_CO = r_squared(geno["CO"], anchors, partners)
    r2_FR = r_squared(geno["FR"], anchors, partners)

    print(f'R2_CO: {r2_CO}')
    print(f'R2_FR: {r2_FR}')

    # print(f'CO NaN count: {np.sum(np.isnan(r2_CO))} / {len(r2_CO)} ({100*np.mean(np.isnan(r2_CO)):.1f}%)')
    # print(f'FR NaN count: {np.sum(np.isnan(r2_FR))} / {len(r2_FR)} ({100*np.mean(np.isnan(r2_FR)):.1f}%)')

    # short = diff < np.median(diff)
    # far = diff >= np.median(diff)

    # print(f'CO NaN rate, short pairs: {100*np.mean(np.isnan(r2_CO[short])):.1f}%')
    # print(f'CO NaN rate, far pairs:   {100*np.mean(np.isnan(r2_CO[far])):.1f}%')
    # print(f'FR NaN rate, short pairs: {100*np.mean(np.isnan(r2_FR[short])):.1f}%')
    # print(f'FR NaN rate, far pairs:   {100*np.mean(np.isnan(r2_FR[far])):.1f}%')

    # Now let's bin these pairs. We will find the SNP pairs that fall within each bin and then average the R2 value
    n_bins = 100
    bins = np.logspace(0, np.log10(max_value), n_bins + 1)
    # print(bins)

    centers, mean_r2_CO = bin_mean_r2(diff, r2_CO, bins)
    _, mean_r2_FR = bin_mean_r2(diff, r2_FR, bins)

    print(f'Bin centers: {centers}')
    print(f'Mean r2 CO: {mean_r2_CO}')
    print(f'Mean r2 FR: {mean_r2_FR}')

    # Step 5: fixed background floor = mean r2 for bins beyond the LARGEST candidate
    # window (largest_window_size). At these distances LD has flattened out, so this
    # is the residual "independent" correlation level for each population.
    floor_CO = np.nanmean(mean_r2_CO[centers > largest_window_size])
    floor_FR = np.nanmean(mean_r2_FR[centers > largest_window_size])

    print(f'Floor CO: {floor_CO}')
    print(f'Floor FR: {floor_FR}')

    # Step 6: read candidate window sizes straight off the decay curve. For a set of
    # "closeness to floor" thresholds, find the distance where r^2 has decayed to
    # <= threshold * floor. That distance IS the window size (SNPs a window apart are
    # then within `threshold` of background). A larger threshold -> curve crosses
    # sooner -> smaller (less conservative) window.
    ratio_thresholds = [1.05, 1.1, 1.2, 1.5, 2.0]

    print("\nCandidate window sizes (bp), read off the LD decay curve:")
    print(f"{'closeness':>10} {'CO (bp)':>16} {'FR (bp)':>16} {'safe-for-both':>16}")
    candidate_bp = {}
    for t in ratio_thresholds:
        w_CO = window_size_for_target(centers, mean_r2_CO, t * floor_CO)
        w_FR = window_size_for_target(centers, mean_r2_FR, t * floor_FR)
        # Safe for both populations = the LARGER window (whichever pop decays slower
        # needs more distance to reach background).
        w_both = np.nanmax([w_CO, w_FR])
        candidate_bp[t] = (w_CO, w_FR, w_both)
        print(f"{t:>10.2f} {w_CO:>16,.0f} {w_FR:>16,.0f} {w_both:>16,.0f}")

    # Step 7: convert each safe-for-both window size to a number of windows.
    print("\nAs number of non-overlapping windows (num_windows = chr_length / window_size):")
    for t in ratio_thresholds:
        w_both = candidate_bp[t][2]
        if np.isfinite(w_both) and w_both > 0:
            print(f"  closeness {t:.2f}: window ~{w_both:,.0f} bp -> ~{chr_length / w_both:,.0f} windows")
        else:
            print(f"  closeness {t:.2f}: curve never reached this level within sampled range")

    # Plot: single LD decay curve, with each population's floor and the safe-for-both
    # candidate window sizes marked as vertical lines.
    fig, ax_curve = plt.subplots(figsize=(8, 5))

    ax_curve.plot(centers, mean_r2_CO, marker="o", ms=3, label="CO", color="tab:blue")
    ax_curve.axhline(floor_CO, color="tab:blue", ls=":", alpha=0.5)
    ax_curve.plot(centers, mean_r2_FR, marker="o", ms=3, label="FR", color="tab:orange")
    ax_curve.axhline(floor_FR, color="tab:orange", ls=":", alpha=0.5)

    cmap = plt.get_cmap("Greens")
    for i, t in enumerate(ratio_thresholds):
        w_both = candidate_bp[t][2]
        if np.isfinite(w_both):
            ax_curve.axvline(w_both, color=cmap(0.4 + 0.5 * i / len(ratio_thresholds)),
                             ls="--", lw=1, label=f"{t:.2f}x floor -> {w_both:,.0f} bp")

    # Mark specific block counts as window sizes (window = chr_length / n_blocks).
    # 50 blocks = large/very independent but few replicates; ~283 blocks = smallest
    # window that is still ~independent (1.05x floor, safe for both populations).
    for n_blocks_mark, color in [(50, "crimson"), (283, "black")]:
        w = chr_length / n_blocks_mark
        ax_curve.axvline(w, color=color, ls="-", lw=2, alpha=0.8,
                         label=f"{n_blocks_mark} blocks = {w:,.0f} bp")

    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("Distance (bp) = window size")
    ax_curve.set_ylabel(r"mean $r^2$")
    ax_curve.set_title("LD decay curve with candidate window sizes")
    ax_curve.legend(fontsize=8)

    fig.tight_layout()
    out_path = "/sietch_colab/akapoor/Infer_Demography/figures/bootstrap_window_size.png"
    fig.savefig(out_path, dpi=150)
    print(f'Wrote {out_path}')





        

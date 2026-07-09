# bootstrapping_for_LRT.py

"""
Godambe-adjusted likelihood-ratio test: does CO have exponential growth?

H0 (simple model): CO constant post-split, only FR grows.
H1 (complex/"both" model): both CO and FR grow.

Composite-likelihood LRT statistics are inflated by linkage, so we correct the
raw statistic with the Godambe factor adjust = H / J:
  * H is the observed information for the tested parameter (empirical SFS only).
  * J is the variance of the score under block-bootstrap resampling, which is
    what carries the linkage information.
D_adj = adjust * D ~ chi^2_1 under H0.

Block size sensitivity: BLOCK_COUNTS defines several non-overlapping block
counts to test. Larger blocks are safer for LD independence but give fewer
bootstrap units; smaller blocks give more bootstrap units but may violate
independence. We report the Godambe-adjusted LRT across the LD-justified range.
"""

import gzip
import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
from tqdm import tqdm
from src.demes_models import split_migration_growth_both_model

MAX_WORKERS = 25
BLOCK_COUNTS = [20, 30, 40, 50, 75, 100, 150, 200, 283]
NUM_BOOT_REPS = 1000
RNG_SEED = 0

real_vcf = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/Chr2L.polarized.vcf.gz"
popfile = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/popfile.txt"
unfolded_sfs_path = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/drosophila.unfolded.sfs.pkl"
simple_fit_path = "/sietch_colab/akapoor/Infer_Demography/experiments/split_migration_growth/real_data_analysis/inferences/moments/best_fit.pkl"
complex_fit_path = "/sietch_colab/akapoor/Infer_Demography/experiments/split_migration_growth_both/real_data_analysis/inferences/moments/best_fit.pkl"
out_dir = "/sietch_colab/akapoor/Infer_Demography/figures"
summary_csv_path = os.path.join(out_dir, "godambe_block_sensitivity.csv")
sensitivity_plot_path = os.path.join(out_dir, "godambe_block_sensitivity.png")
hist_template = os.path.join(out_dir, "J_bootstrap_hist_{num_blocks}_blocks.png")


# ---- Complex model in the growth_CO parameterization -----------------------
# The null "no CO growth" (N_CO0 == N_CO1) isn't a single coordinate of the
# (N_CO0, N_CO1) vector, so we swap N_CO0 for growth_CO = log(N_CO0 / N_CO1):
# growth_CO == 0  <=>  N_CO0 == N_CO1, i.e. the null is exactly growth_CO = 0.
param_names = [
    "N_ANC",
    "growth_CO",  # log(N_CO0 / N_CO1); == 0 under H0
    "N_CO1",
    "N_FR0",
    "N_FR1",
    "T",
    "m_CO_FR",
    "m_FR_CO",
]
growth_idx = param_names.index("growth_CO")


def split_migration_growth_both_sfs(p, ns):
    """Expected SFS for the complex model, converting growth_CO back to
    N_CO0 = N_CO1 * exp(growth_CO) for the demes graph."""
    sampled = dict(zip(param_names, p))
    growth_CO = sampled.pop("growth_CO")
    sampled["N_CO0"] = sampled["N_CO1"] * np.exp(growth_CO)
    graph = split_migration_growth_both_model(sampled)
    return moments.Spectrum.from_demes(graph, sampled_demes=["CO", "FR"], sample_sizes=ns)


# ---- Null point p0 and fixed theta -----------------------------------------
with open(simple_fit_path, "rb") as f:
    simple_fit = pickle.load(f)
simple_params = simple_fit["best_params"][0]

# H0-consistent point in the 8-param growth_CO order (growth_CO = 0).
p0 = [
    simple_params["N_ANC"],
    0.0,                     # growth_CO
    simple_params["N_CO"],   # N_CO1
    simple_params["N_FR0"],
    simple_params["N_FR1"],
    simple_params["T"],
    simple_params["m_CO_FR"],
    simple_params["m_FR_CO"],
]

with open(unfolded_sfs_path, "rb") as f:
    data_sfs = pickle.load(f)

# multinom theta: fix theta at the optimal scaling for p0, then hold it fixed.
model = split_migration_growth_both_sfs(p0, data_sfs.sample_sizes)
theta_opt = moments.Inference.optimal_sfs_scaling(model, data_sfs)
p0_theta = np.array(list(p0) + [theta_opt], dtype=float)


# ---- H: observed information for growth_CO (empirical SFS only) -------------
def _ll_growth(growth_vec, data):
    full = p0_theta.copy()
    full[growth_idx] = growth_vec[0]
    fs = full[-1] * split_migration_growth_both_sfs(full[:-1], data.sample_sizes)
    return moments.Inference.ll(fs, data)


H = -moments.Godambe.get_hess(_ll_growth, [p0_theta[growth_idx]], eps=0.01, args=[data_sfs])
H_growth = float(H[0, 0])
print("H (growth_CO):", H_growth)


# ---- Per-chunk SFS (block bootstrap units) ---------------------------------
def _get_vcf_bounds(vcf_path):
    first = int(subprocess.check_output(
        f"bcftools query -f '%POS\n' '{vcf_path}' | head -n 1", shell=True).strip())
    last = int(subprocess.check_output(
        f"bcftools query -f '%POS\n' '{vcf_path}' | tail -n 1", shell=True).strip())
    return first, last


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


def _get_vcf_sample_indices(vcf_path, popfile_path):
    pop_names, sample_to_pop = _parse_popfile(popfile_path)
    sample_indices = {pop: [] for pop in pop_names}
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                for i, s in enumerate(line.rstrip("\n").split("\t")[9:]):
                    pop = sample_to_pop.get(s)
                    if pop is not None:
                        sample_indices[pop].append(i)
                break
    return pop_names, sample_indices


pop_names, sample_indices = _get_vcf_sample_indices(real_vcf, popfile)
sample_sizes = [len(sample_indices[pop]) for pop in pop_names]
first_pos, last_pos = _get_vcf_bounds(real_vcf)


def _parse_chunk(interval):
    # `interval` in parse_vcf is 1-indexed and half-open, so consecutive
    # (start, end) pairs tile the chromosome with no overlap.
    try:
        return moments.Parsing.parse_vcf(
            real_vcf, pop_file=popfile, use_AA=True, ploidy=1, interval=interval)
    except ValueError:
        # No SNPs in this interval -> all-zero SFS instead of crashing.
        return moments.Spectrum(np.zeros([n + 1 for n in sample_sizes]))


def get_chunk_spectra(num_chunks):
    """Per-chunk unfolded SFS for `num_chunks` non-overlapping blocks tiling the
    chromosome. Cached to disk (keyed by num_chunks) so re-runs skip the parse."""
    cache = f"/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/chunk_spectra_{num_chunks}.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    bounds = np.linspace(first_pos, last_pos + 1, num_chunks + 1).astype(int)
    chunks = [(bounds[i], bounds[i + 1]) for i in range(num_chunks)]
    with ProcessPoolExecutor(max_workers=min(num_chunks, MAX_WORKERS)) as pool:
        chunk_spectra = list(tqdm(pool.map(_parse_chunk, chunks), total=num_chunks,
                                  desc=f"parse {num_chunks} blocks"))
    with open(cache, "wb") as f:
        pickle.dump(chunk_spectra, f)
    return chunk_spectra


# ---- J: score variance under block-bootstrap resampling --------------------
# For a single nested parameter, bootstrap b contributes J_b = U_b**2, where U_b
# is the score (gradient of the composite ll w.r.t. growth_CO at growth_CO = 0)
# on bootstrap b's SFS. J = mean_b(J_b). theta is held at theta_opt throughout,
# and the two null-point model spectra don't depend on the bootstrap, so build
# them once. This reproduces moments' finite-difference score, so J matches
# LRT_adjust's internal J.
score_eps = 0.01
ns = list(data_sfs.sample_sizes)
m0 = theta_opt * split_migration_growth_both_sfs(p0_theta[:-1], ns)   # growth_CO = 0
p_eps = p0_theta[:-1].copy()
p_eps[growth_idx] = score_eps
mp = theta_opt * split_migration_growth_both_sfs(p_eps, ns)           # growth_CO = +eps


def bootstrap_J(num_chunks, rng):
    """Per-bootstrap J_b: resample num_chunks blocks with replacement, sum to one
    bootstrap SFS, and square its growth_CO score."""
    chunk_spectra = get_chunk_spectra(num_chunks)
    idx = np.arange(num_chunks)
    J_boot = np.empty(NUM_BOOT_REPS)
    for b in range(NUM_BOOT_REPS):
        sampled = rng.choice(idx, size=num_chunks, replace=True)
        boot = moments.Spectrum(sum(chunk_spectra[i] for i in sampled))
        score_b = (moments.Inference.ll(mp, boot) - moments.Inference.ll(m0, boot)) / score_eps
        J_boot[b] = score_b ** 2
    return J_boot




def check_chunk_reconstruction(num_chunks):
    """Sanity check: the non-overlapping chunk SFSs should reconstruct the
    empirical SFS used in the likelihood. If this fails badly, then the
    bootstrap data are not matched to data_sfs."""
    chunk_spectra = get_chunk_spectra(num_chunks)
    chunk_sum = moments.Spectrum(sum(chunk_spectra))

    snp_diff = float(chunk_sum.S() - data_sfs.S())
    max_abs_diff = float(np.max(np.abs(chunk_sum - data_sfs)))
    sample_sizes_match = list(chunk_sum.sample_sizes) == list(data_sfs.sample_sizes)

    return {
        "chunk_sum_snps": float(chunk_sum.S()),
        "data_sfs_snps": float(data_sfs.S()),
        "snp_diff": snp_diff,
        "max_abs_sfs_diff": max_abs_diff,
        "sample_sizes_match": sample_sizes_match,
    }


def summarize_block_count(num_chunks, seed):
    """Run the Godambe correction for one block count and return one row for
    the sensitivity table, plus the vector of per-bootstrap J values."""
    rng = np.random.default_rng(seed)
    J_boot = bootstrap_J(num_chunks, rng)
    mean_J = float(J_boot.mean())
    sd_J = float(J_boot.std(ddof=1))

    adjust = H_growth / mean_J
    D_adj = adjust * D
    p_adj = float(moments.Godambe.sum_chi2_ppf(D_adj, weights=(0, 1)))
    window_bp = (last_pos - first_pos + 1) / num_chunks

    recon = check_chunk_reconstruction(num_chunks)

    row = {
        "num_blocks": int(num_chunks),
        "window_bp": float(window_bp),
        "window_kb": float(window_bp / 1e3),
        "H": float(H_growth),
        "mean_J": mean_J,
        "sd_J_boot": sd_J,
        "adjust_H_over_J": float(adjust),
        "raw_D": float(D),
        "D_adj": float(D_adj),
        "p_raw": float(p_raw),
        "p_adj": p_adj,
        **recon,
    }
    return row, J_boot


# ---- Godambe-adjusted LRT across candidate block sizes ---------------------
# Raw D = 2*(ll_complex - ll_simple) from the two joint fits. This part is
# block-size independent. Only J, H/J, D_adj, and p_adj change with block size.
with open(complex_fit_path, "rb") as f:
    complex_fit = pickle.load(f)
D = 2 * (complex_fit["best_ll"][0] - simple_fit["best_ll"][0])
p_raw = float(moments.Godambe.sum_chi2_ppf(D, weights=(0, 1)))

os.makedirs(out_dir, exist_ok=True)

results = []
J_by_block = {}
print(f"\nRaw LRT: D = {D:.6g}; unadjusted p = {p_raw:.6g}")
print(f"H (growth_CO): {H_growth:.6g}\n")

for num_chunks in BLOCK_COUNTS:
    # Use a different, reproducible seed for each block count. This avoids
    # accidentally reusing the same bootstrap-index stream for all block counts.
    seed = RNG_SEED + int(num_chunks)
    row, J_boot = summarize_block_count(num_chunks, seed)
    results.append(row)
    J_by_block[num_chunks] = J_boot

    print(
        f"{num_chunks:>4} blocks  "
        f"window={row['window_kb']:>8.1f} kb  "
        f"mean_J={row['mean_J']:>12.3g}  "
        f"H/J={row['adjust_H_over_J']:>10.5g}  "
        f"D_adj={row['D_adj']:>10.5g}  "
        f"p_adj={row['p_adj']:>10.5g}  "
        f"SFS maxdiff={row['max_abs_sfs_diff']:.3g}"
    )

# Write CSV summary without requiring pandas.
fieldnames = list(results[0].keys())
with open(summary_csv_path, "w") as f:
    f.write(",".join(fieldnames) + "\n")
    for row in results:
        f.write(",".join(str(row[k]) for k in fieldnames) + "\n")
print(f"\nwrote {summary_csv_path}")


# ---- Figure 1: sensitivity of adjusted p-value across block counts ---------
block_counts = np.array([r["num_blocks"] for r in results])
window_kb = np.array([r["window_kb"] for r in results])
p_adj_vals = np.array([r["p_adj"] for r in results])
adjust_vals = np.array([r["adjust_H_over_J"] for r in results])
D_adj_vals = np.array([r["D_adj"] for r in results])
mean_J_vals = np.array([r["mean_J"] for r in results])

fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax1.plot(block_counts, p_adj_vals, marker="o")
ax1.axhline(0.05, ls="--", lw=1, label="p = 0.05")
ax1.set_xlabel("Number of non-overlapping bootstrap blocks")
ax1.set_ylabel("Godambe-adjusted p-value")
ax1.set_title("Block-size sensitivity of Godambe-adjusted LRT")
ax1.set_xticks(block_counts)
for x, y, kb in zip(block_counts, p_adj_vals, window_kb):
    ax1.annotate(f"{kb:.0f} kb", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax1.legend(fontsize=8)
fig.tight_layout()
fig.savefig(sensitivity_plot_path, dpi=150)
print(f"wrote {sensitivity_plot_path}")


# ---- Figure 2: mean J and H/J across block counts --------------------------
# This helps diagnose WHY p-values change: usually because J changes.
J_plot_path = os.path.join(out_dir, "godambe_J_and_adjust_by_block_count.png")
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(block_counts, mean_J_vals, marker="o", label="mean J")
ax.set_xlabel("Number of non-overlapping bootstrap blocks")
ax.set_ylabel("mean J")
ax.set_title("Score-variance estimate J across block sizes")
ax.set_xticks(block_counts)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(J_plot_path, dpi=150)
print(f"wrote {J_plot_path}")

adjust_plot_path = os.path.join(out_dir, "godambe_adjust_by_block_count.png")
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(block_counts, adjust_vals, marker="o", label="H/J")
ax.set_xlabel("Number of non-overlapping bootstrap blocks")
ax.set_ylabel("Adjustment factor H/J")
ax.set_title("Godambe adjustment factor across block sizes")
ax.set_xticks(block_counts)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(adjust_plot_path, dpi=150)
print(f"wrote {adjust_plot_path}")


# ---- Optional: per-block-count J histograms --------------------------------
for num_chunks in BLOCK_COUNTS:
    J_boot = J_by_block[num_chunks]
    row = next(r for r in results if r["num_blocks"] == num_chunks)
    hist_path = hist_template.format(num_blocks=num_chunks)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(J_boot, bins=30, alpha=0.85, edgecolor="white")
    ax.axvline(row["mean_J"], lw=1.8, ls="--", label=f"mean J = {row['mean_J']:,.0f}")
    ax.axvline(H_growth, lw=2.0, label=f"H = {H_growth:,.0f}")
    ax.set_xlabel(r"per-bootstrap $J_b = (\partial_{\mathrm{growth\_CO}}\,\ell)^2$")
    ax.set_ylabel("count")
    ax.set_title(
        f"{num_chunks} blocks (~{row['window_kb']:.0f} kb)   "
        f"H/J = {row['adjust_H_over_J']:.4g}   p = {row['p_adj']:.3g}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

print("wrote per-block J histograms")

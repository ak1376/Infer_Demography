# bootstrapping_for_LRT.py

'''
I want to create bootstrapped replicates of the SFS for the real data. This is how I'm going to do it
    1. Split Chr2L into non-overlapping chunks and compute the SFS for each
    2. Randomly select len(chunks) chunks with replacement
    3. Sum their SFS to get one bootstrap replicate
    4. Repeat 200 times
'''

import gzip
import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor

import allel
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
from tqdm import tqdm
from src.demes_models import split_migration_growth_both_model

MAX_WORKERS = 25
NUM_CHUNKS = 250
num_boot_reps = 200

# The real, haploid, AA-polarized VCF (NOT the diploid-recoded copy used for
# MomentsLD windows -- that copy just duplicates each haploid call and adds
# no information; see popfile for the CO/FR sample assignment).
real_vcf = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/Chr2L.polarized.vcf.gz"
popfile = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/popfile.txt"
unfolded_sfs_path = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/drosophila.unfolded.sfs.pkl"

# Best-fit results from the real-data moments runs (H0 = simple: CO constant,
# only FR grows; H1 = complex/"both": both CO and FR grow).
simple_fit_path = "/sietch_colab/akapoor/Infer_Demography/experiments/split_migration_growth/real_data_analysis/inferences/moments/best_fit.pkl"
complex_fit_path = "/sietch_colab/akapoor/Infer_Demography/experiments/split_migration_growth_both/real_data_analysis/inferences/moments/best_fit.pkl"


# Step 1: Reparameterize the complex model's parameters to use a growth factor for CO: growth_CO = log(N_CO0 / N_CO1)
# The full "both" model is parameterized by the two CO endpoint sizes (N_CO0 at
# the split, N_CO1 at present). The LRT null is "no CO growth", i.e.
# N_CO0 == N_CO1 -- but that constraint isn't a single coordinate of the
# (N_CO0, N_CO1) vector, so Godambe's nested-model machinery (which fixes one
# coordinate to its p0 value) can't express it. Swapping N_CO0 for
# growth_CO = log(N_CO0 / N_CO1) fixes that: growth_CO == 0 <=> N_CO0 == N_CO1,
# so the null is exactly the growth_CO coordinate held at 0.

param_names = [
    "N_ANC",
    "growth_CO",  # log(N_CO0 / N_CO1); == 0 under H0 (CO constant post-split)
    "N_CO1",
    "N_FR0",
    "N_FR1",
    "T",
    "m_CO_FR",
    "m_FR_CO",
]


def split_migration_growth_both_sfs(p, ns):
    """Expected SFS for the complex model in the growth_CO parameterization.

    Converts growth_CO back to the explicit split-time size the demes model
    wants (N_CO0 = N_CO1 * exp(growth_CO)) and returns the moments.Spectrum.
    """
    sampled = dict(zip(param_names, p))

    growth_CO = sampled.pop("growth_CO")
    sampled["N_CO0"] = sampled["N_CO1"] * np.exp(growth_CO)

    graph = split_migration_growth_both_model(sampled)

    return moments.Spectrum.from_demes(
        graph,
        sampled_demes=["CO", "FR"],
        sample_sizes=ns,
    )

# Step 2: H (observed-information / sensitivity) for growth_CO only.
# Mirrors LRT_adjust's internal H with nested_indices=[growth_CO], but returns
# H itself. Depends only on the empirical SFS -- no bootstraps required.

growth_idx = param_names.index("growth_CO")

with open(simple_fit_path, "rb") as f:
    simple_fit = pickle.load(f)
simple_params = simple_fit["best_params"][0]

# Null point, in the 8-param growth_CO order the SFS wrapper expects.
# growth_CO = 0  <=>  N_CO0 == N_CO1  (CO constant post-split, i.e. H0).
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

# multinom theta handling, identical to LRT_adjust: fix theta at the optimal
# scaling for p0, then hold it fixed as a trailing parameter while we vary only
# growth_CO.
model = split_migration_growth_both_sfs(p0, data_sfs.sample_sizes)
theta_opt = moments.Inference.optimal_sfs_scaling(model, data_sfs)
p0_theta = np.array(list(p0) + [theta_opt], dtype=float)


def _ll_growth(growth_vec, data):
    # composite log-likelihood as a function of growth_CO alone
    full = p0_theta.copy()
    full[growth_idx] = growth_vec[0]
    fs = full[-1] * split_migration_growth_both_sfs(full[:-1], data.sample_sizes)
    return moments.Inference.ll(fs, data)


H = -moments.Godambe.get_hess(
    _ll_growth, [p0_theta[growth_idx]], eps=0.01, args=[data_sfs]
)
H_growth = float(H[0, 0])
print("H (growth_CO):", H_growth)

def _get_vcf_bounds(vcf_path):
    first = int(subprocess.check_output(
        f"bcftools query -f '%POS\n' '{vcf_path}' | head -n 1", shell=True
    ).strip())
    last = int(subprocess.check_output(
        f"bcftools query -f '%POS\n' '{vcf_path}' | tail -n 1", shell=True
    ).strip())
    return first, last


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


def _get_vcf_sample_indices(vcf_path, popfile_path):
    """Map each population to the column indices of its samples in the VCF."""
    pop_names, sample_to_pop = _parse_popfile(popfile_path)
    sample_indices = {pop: [] for pop in pop_names}
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                vcf_samples = line.rstrip("\n").split("\t")[9:]
                for i, s in enumerate(vcf_samples):
                    pop = sample_to_pop.get(s)
                    if pop is not None:
                        sample_indices[pop].append(i)
                break
    return pop_names, sample_indices


pop_names, sample_indices = _get_vcf_sample_indices(real_vcf, popfile)
sample_sizes = [len(sample_indices[pop]) for pop in pop_names]

# Full VCF span, shared across all block sizes. `interval` in
# moments.Parsing.parse_vcf is 1-indexed and half-open, so consecutive
# (start, end) pairs tile the chromosome with no overlap.
first_pos, last_pos = _get_vcf_bounds(real_vcf)


def _parse_chunk(interval):
    # Each chunk is an independent, non-overlapping stretch of the genome --
    # embarrassingly parallel, so farm it out across processes.
    try:
        return moments.Parsing.parse_vcf(
            real_vcf, pop_file=popfile, use_AA=True, ploidy=1, interval=interval
        )
    except ValueError:
        # No SNPs fall in this interval (e.g. a heterochromatic/repetitive
        # gap in variant calling, more likely to occur as chunks get
        # smaller) -- contributes an all-zero SFS instead of crashing.
        return moments.Spectrum(np.zeros([n + 1 for n in sample_sizes]))


def get_chunk_spectra(num_chunks):
    """Per-chunk (unfolded) SFS for `num_chunks` non-overlapping blocks tiling
    the chromosome -- each a moments.Spectrum polarized via the VCF's AA field.
    Cached to disk (keyed by num_chunks) so re-runs skip the VCF parse."""
    chunk_bounds = np.linspace(first_pos, last_pos + 1, num_chunks + 1).astype(int)
    chunks = [(chunk_bounds[i], chunk_bounds[i + 1]) for i in range(num_chunks)]

    chunk_cache = f"/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/chunk_spectra_{num_chunks}.pkl"
    if os.path.exists(chunk_cache):
        with open(chunk_cache, "rb") as f:
            return pickle.load(f)

    with ProcessPoolExecutor(max_workers=min(num_chunks, MAX_WORKERS)) as pool:
        chunk_spectra = list(
            tqdm(pool.map(_parse_chunk, chunks), total=num_chunks,
                 desc=f"parse {num_chunks} blocks")
        )
    with open(chunk_cache, "wb") as f:
        pickle.dump(chunk_spectra, f)
    return chunk_spectra


# # ---- Per-window summary statistics: pi, Tajima's D, FST ----
# # Reuses the same non-overlapping `chunks` windows as the SFS bootstrap
# # above. Pi, Tajima's D, and Hudson's FST are all invariant to allele
# # polarization, so (unlike the SFS parsing) we don't need the AA field here
# # -- just per-sample REF/ALT calls.

# def _get_vcf_chrom(vcf_path):
#     return subprocess.check_output(
#         f"bcftools query -f '%CHROM\n' '{vcf_path}' | head -n 1", shell=True
#     ).decode().strip()


# vcf_chrom = _get_vcf_chrom(real_vcf)


# def _compute_window_stats(interval):
#     """Compute pi, Tajima's D (per population) and Hudson's FST for one window."""
#     start, end = interval
#     region = f"{vcf_chrom}:{start}-{end - 1}"
#     raw = subprocess.check_output(
#         f"bcftools view -H -r '{region}' '{real_vcf}'", shell=True
#     ).decode()

#     ac = {pop: [] for pop in pop_names}
#     for line in raw.splitlines():
#         fields = line.split("\t")
#         gts = fields[9:]
#         for pop in pop_names:
#             alt_count = sum(int(gts[i]) for i in sample_indices[pop])
#             n = len(sample_indices[pop])
#             ac[pop].append((n - alt_count, alt_count))

#     n_sites = len(ac[pop_names[0]])
#     window_len = end - start

#     result = {
#         "start": start, "end": end, "mid": (start + end) / 2, "n_sites": n_sites,
#     }

#     ac_arrays = {}
#     for pop in pop_names:
#         ac_pop = np.array(ac[pop], dtype=np.int32)
#         ac_arrays[pop] = ac_pop
#         if n_sites:
#             alt = ac_pop[:, 1]
#             n = len(sample_indices[pop])
#             pi = float(np.sum(2.0 * alt * (n - alt) / (n * (n - 1))) / window_len)
#             taj_d = float(allel.tajima_d(ac_pop))
#         else:
#             pi, taj_d = np.nan, np.nan
#         result[f"pi_{pop}"] = pi
#         result[f"tajima_d_{pop}"] = taj_d

#     if n_sites and len(pop_names) >= 2:
#         num, den = allel.hudson_fst(ac_arrays[pop_names[0]], ac_arrays[pop_names[1]])
#         den_sum = np.nansum(den)
#         fst = float(np.nansum(num) / den_sum) if den_sum > 0 else np.nan
#     else:
#         fst = np.nan
#     result["fst"] = fst

#     return result


# with ProcessPoolExecutor(max_workers=min(NUM_CHUNKS, MAX_WORKERS)) as pool:
#     window_stats = list(tqdm(pool.map(_compute_window_stats, chunks), total=NUM_CHUNKS))

# window_stats_df = pd.DataFrame(window_stats)
# window_stats_csv = "/sietch_colab/akapoor/Infer_Demography/figures/window_summary_stats.csv"
# window_stats_df.to_csv(window_stats_csv, index=False)
# print(f"Wrote per-window summary statistics to {window_stats_csv}")


# def _plot_window_stat(df, cols, ylabel, title, out_path):
#     plt.figure(figsize=(10, 4))
#     for col in cols:
#         plt.plot(df["mid"], df[col], marker="o", label=col)
#     plt.xlabel(f"Position on {vcf_chrom} (bp)")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     if len(cols) > 1:
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()


# _plot_window_stat(
#     window_stats_df, [f"pi_{pop}" for pop in pop_names],
#     "pi (per site)", "Nucleotide diversity across windows",
#     "/sietch_colab/akapoor/Infer_Demography/figures/window_pi.png",
# )
# _plot_window_stat(
#     window_stats_df, [f"tajima_d_{pop}" for pop in pop_names],
#     "Tajima's D", "Tajima's D across windows",
#     "/sietch_colab/akapoor/Infer_Demography/figures/window_tajimas_d.png",
# )
# _plot_window_stat(
#     window_stats_df, ["fst"],
#     "FST (Hudson)", "FST across windows",
#     "/sietch_colab/akapoor/Infer_Demography/figures/window_fst.png",
# )


# ---- Score model spectra (block-size independent) ----
# For a single nested parameter, bootstrap b contributes J_b = U_b**2, where
# U_b is the score: the gradient of the composite log-likelihood w.r.t.
# growth_CO at the null (growth_CO = 0), on bootstrap b's SFS. The Godambe J is
# mean_b(J_b), and the adjustment factor is H / J.
#
# theta is held fixed at theta_opt (the real-data optimum) across all
# bootstraps, exactly as LRT_adjust does. The two model spectra for the
# one-sided score at growth_CO = 0 depend on neither the bootstrap nor the
# block size, so build them once. This reproduces moments' _get_grad finite
# difference at a zero-valued parameter, so J matches LRT_adjust's internal J.
score_eps = 0.01
ns = list(data_sfs.sample_sizes)
m0 = theta_opt * split_migration_growth_both_sfs(p0_theta[:-1], ns)   # growth_CO = 0
p_eps = p0_theta[:-1].copy()
p_eps[growth_idx] = score_eps
mp = theta_opt * split_migration_growth_both_sfs(p_eps, ns)           # growth_CO = +eps


def bootstrap_J(num_chunks, rng):
    """Per-bootstrap J_b for `num_chunks`-block bootstraps. H is block-size
    independent; only J (the linkage-driven score variance) changes with block
    size, so this is the whole block-size dependence of the LRT adjustment."""
    chunk_spectra = get_chunk_spectra(num_chunks)
    idx = np.arange(num_chunks)
    J_boot = np.empty(num_boot_reps)
    for b in range(num_boot_reps):
        # resample num_chunks blocks with replacement, sum to one bootstrap SFS
        sampled = rng.choice(idx, size=num_chunks, replace=True)
        boot = moments.Spectrum(sum(chunk_spectra[i] for i in sampled))
        score_b = (moments.Inference.ll(mp, boot) - moments.Inference.ll(m0, boot)) / score_eps
        J_boot[b] = score_b ** 2
    return J_boot


# ---- Block-size sweep ----
# Number of blocks tiling the chromosome; block length (bp) = span / n_blocks.
# Smaller blocks capture less linkage, so they under-estimate J (over-optimistic
# adjustment). J should stabilize once blocks exceed the LD decay scale -- that
# plateau is the defensible block size.
BLOCK_NUM_CHUNKS = [50, 100, 250, 500, 1000]
span_bp = last_pos - first_pos + 1
rng = np.random.default_rng(0)

J_by_blocks = {nc: bootstrap_J(nc, rng) for nc in BLOCK_NUM_CHUNKS}

# ---- Godambe-adjusted LRT: chi^2 statistic and p-value per block size ----
# Raw statistic D = 2*(ll_complex - ll_simple) comes from the two joint fits and
# is block-size independent. The Godambe first-order correction scales it:
# D_adj = adjust * D, and under H0 D_adj ~ chi^2_1 (one nested parameter,
# growth_CO, whose null value 0 is interior -- CO may grow or shrink -- so no
# boundary mixture). Only `adjust` (= H/J) changes with block size.
with open(complex_fit_path, "rb") as f:
    complex_fit = pickle.load(f)
ll_simple = simple_fit["best_ll"][0]
ll_complex = complex_fit["best_ll"][0]
D = 2 * (ll_complex - ll_simple)
p_raw = moments.Godambe.sum_chi2_ppf(D, weights=(0, 1))
print(f"\nraw LRT: D = {D:.4g}   unadjusted p (chi2_1) = {p_raw:.4g}")

print(
    f"\n{'n_blocks':>9} {'block_kb':>9} {'mean_J':>14} {'adjust=H/J':>11}"
    f" {'D_adj(chi2)':>12} {'p_value':>10}"
)
adjust_by_blocks = {}
p_by_blocks = {}
for nc in BLOCK_NUM_CHUNKS:
    Jm = float(J_by_blocks[nc].mean())
    adj = H_growth / Jm
    D_adj = adj * D
    p = float(moments.Godambe.sum_chi2_ppf(D_adj, weights=(0, 1)))
    adjust_by_blocks[nc] = adj
    p_by_blocks[nc] = p
    print(
        f"{nc:>9} {span_bp / nc / 1e3:>9.1f} {Jm:>14.1f} {adj:>11.5f}"
        f" {D_adj:>12.4g} {p:>10.4g}"
    )

# ---- Figure 1: per-bootstrap J histograms, one panel per block size ----
n = len(BLOCK_NUM_CHUNKS)
fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n))
axes = np.atleast_1d(axes)
for ax, nc in zip(axes, BLOCK_NUM_CHUNKS):
    Jb = J_by_blocks[nc]
    Jm = float(Jb.mean())
    ax.hist(Jb, bins=30, color="#4c72b0", alpha=0.85, edgecolor="white")
    ax.axvline(Jm, color="black", lw=1.8, ls="--", label=f"mean J = {Jm:,.0f}")
    ax.axvline(H_growth, color="#c44e52", lw=2.0, label=f"H = {H_growth:,.0f}")
    ax.set_ylabel("count")
    ax.set_title(
        f"{nc} blocks (~{span_bp / nc / 1e3:.0f} kb)   H/J = {H_growth / Jm:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
axes[-1].set_xlabel(r"per-bootstrap $J_b = (\partial_{\mathrm{growth\_CO}}\,\ell)^2$")
fig.suptitle(f"Per-bootstrap J vs block size ({num_boot_reps} bootstraps)")
fig.tight_layout()
grid_path = "/sietch_colab/akapoor/Infer_Demography/figures/J_bootstrap_hist_by_blocksize.png"
fig.savefig(grid_path, dpi=150)
print(f"wrote {grid_path}")

# ---- Figure 2: mean J and adjustment factor vs block size ----
blocks = np.array(BLOCK_NUM_CHUNKS)
block_kb = span_bp / blocks / 1e3
mean_Js = np.array([J_by_blocks[nc].mean() for nc in BLOCK_NUM_CHUNKS])
adjusts = np.array([adjust_by_blocks[nc] for nc in BLOCK_NUM_CHUNKS])
p_vals = np.array([p_by_blocks[nc] for nc in BLOCK_NUM_CHUNKS])

fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 4.5))
axL.plot(block_kb, mean_Js, "o-", color="#4c72b0")
axL.axhline(H_growth, color="#c44e52", lw=2.0, label=f"H = {H_growth:,.0f}")
axL.set_xscale("log")
axL.set_xlabel("block size (kb, log scale)")
axL.set_ylabel("mean J")
axL.set_title("mean J vs block size")
axL.legend()
axM.plot(block_kb, adjusts, "o-", color="#55a868")
axM.set_xscale("log")
axM.set_xlabel("block size (kb, log scale)")
axM.set_ylabel("adjustment factor H/J")
axM.set_title("Godambe adjustment vs block size")
axR.plot(block_kb, p_vals, "o-", color="#8172b3")
axR.axhline(0.05, color="gray", lw=1.2, ls=":", label="p = 0.05")
axR.set_xscale("log")
axR.set_xlabel("block size (kb, log scale)")
axR.set_ylabel("Godambe-adjusted p-value")
axR.set_title(f"adjusted LRT p-value (raw D = {D:.1f})")
axR.legend()
fig.tight_layout()
sweep_path = "/sietch_colab/akapoor/Infer_Demography/figures/J_adjustment_vs_blocksize.png"
fig.savefig(sweep_path, dpi=150)
print(f"wrote {sweep_path}")


# ---- Figure 3: per-window J score along the chromosome ----
# Rather than resampling windows (the bootstrap above), look at the raw
# per-window contribution: for each non-overlapping window w, the score U_w is
# the gradient of the composite log-likelihood w.r.t. growth_CO at the null
# (growth_CO = 0), evaluated on that single window's SFS, and J_w = U_w**2.
# Same m0/mp/score_eps as bootstrap_J, so J_w is directly comparable across
# windows and shows which stretches of the chromosome drive the score variance.
def per_window_J(num_chunks):
    chunk_spectra = get_chunk_spectra(num_chunks)
    J_w = np.empty(num_chunks)
    for w, cs in enumerate(chunk_spectra):
        win = moments.Spectrum(cs)
        score_w = (moments.Inference.ll(mp, win) - moments.Inference.ll(m0, win)) / score_eps
        J_w[w] = score_w ** 2
    bounds = np.linspace(first_pos, last_pos + 1, num_chunks + 1).astype(int)
    mids = (bounds[:-1] + bounds[1:]) / 2.0
    return mids, J_w


fig, ax = plt.subplots(figsize=(11, 5))
for nc in BLOCK_NUM_CHUNKS:
    mids, J_w = per_window_J(nc)
    ax.plot(mids / 1e6, J_w, marker="o", ms=3, lw=1,
            label=f"{nc} blocks (~{span_bp / nc / 1e3:.0f} kb)")
ax.set_xlabel(f"Position on chromosome (Mb)")
ax.set_ylabel(r"per-window $J_w = (\partial_{\mathrm{growth\_CO}}\,\ell)^2$")
ax.set_yscale("log")
ax.set_title("Per-window J score along the chromosome")
ax.legend(fontsize=8, title="block size")
fig.tight_layout()
window_J_path = "/sietch_colab/akapoor/Infer_Demography/figures/J_per_window.png"
fig.savefig(window_J_path, dpi=150)
print(f"wrote {window_J_path}")


# ---- Figure 4: autocorrelation of the per-window score ----
# The Godambe J is the variance of the summed per-window score under block
# resampling; the bootstrap blocks are only valid independent replicates if
# their scores are uncorrelated. So treat the per-window score U_w as a spatial
# series along the chromosome and compute its autocorrelation vs lag. Unlike the
# per-window J figure, the constant theta offset in U_w cancels under the ACF's
# mean-centering, so the raw score is fine here. The lag at which the ACF decays
# into the white-noise band is the between-window dependence length; block sizes
# above it give approximately independent bootstrap blocks.
def per_window_score(num_chunks):
    chunk_spectra = get_chunk_spectra(num_chunks)
    U = np.empty(num_chunks)
    for w, cs in enumerate(chunk_spectra):
        win = moments.Spectrum(cs)
        U[w] = (moments.Inference.ll(mp, win) - moments.Inference.ll(m0, win)) / score_eps
    return U


def acf(x, max_lag):
    """Sample autocorrelation of 1-D series x for lags 0..max_lag."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.sum(x * x)
    out = np.full(max_lag + 1, np.nan)
    if denom > 0:
        for k in range(max_lag + 1):
            out[k] = np.sum(x[: len(x) - k] * x[k:]) / denom
    return out


MAX_LAG_KB = 1000.0
fig, ax = plt.subplots(figsize=(9, 5))
for nc in BLOCK_NUM_CHUNKS:
    U = per_window_score(nc)
    block_bp = span_bp / nc
    max_lag = int(min(nc - 1, MAX_LAG_KB * 1e3 / block_bp))
    r = acf(U, max_lag)
    lag_kb = np.arange(max_lag + 1) * block_bp / 1e3
    ax.plot(lag_kb, r, marker=".", ms=5, lw=1,
            label=f"{nc} blocks (~{block_bp / 1e3:.0f} kb)")
# 95% white-noise band for the finest series (largest n -> tightest band)
n_max = max(BLOCK_NUM_CHUNKS)
ci = 1.96 / np.sqrt(n_max)
ax.axhspan(-ci, ci, color="gray", alpha=0.15,
           label=f"95% white-noise band (n={n_max})")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("genomic lag between windows (kb)")
ax.set_ylabel("autocorrelation of per-window score")
ax.set_title("Per-window score ACF vs genomic distance")
ax.legend(fontsize=8, title="block size")
fig.tight_layout()
acf_path = "/sietch_colab/akapoor/Infer_Demography/figures/score_acf.png"
fig.savefig(acf_path, dpi=150)
print(f"wrote {acf_path}")


# # Now that I have the bootstraps, I need to wrap my split migration growth both model in a format that will be recognizable by LRT_adjust in Godambe module

# param_names = [
#     "N_ANC",
#     "N_CO0",
#     "N_CO1",
#     "N_FR0",
#     "N_FR1",
#     "T",
#     "m_CO_FR",
#     "m_FR_CO",
# ]

# def split_migration_growth_both_sfs(p, ns):
#     sampled = dict(zip(param_names, p))

#     graph = split_migration_growth_both_model(sampled)

#     fs = moments.Spectrum.from_demes(
#         graph,
#         sampled_demes=["CO", "FR"],
#         sample_sizes=ns,
#     )

#     return fs


# # ---- Likelihood-ratio test: does CO actually have exponential growth? ----
# # H0 (nested/simple model): no growth in CO, i.e. N_CO0 == N_CO1. That
# # collapses one degree of freedom relative to the full "both" model, so
# # N_CO0 is the "extra" parameter that only exists in the complex model.
# nested_indices = [param_names.index("N_CO0")]

# # p0 = best fit of the null model (split_migration_growth: CO constant,
# # only FR grows), written out in the 8-param order the complex model
# # expects, with N_CO0 == N_CO1 == the null model's single N_CO. This is
# # the real H0-consistent point.
# with open(simple_fit_path, "rb") as f:
#     simple_fit = pickle.load(f)
# with open(complex_fit_path, "rb") as f:
#     complex_fit = pickle.load(f)

# simple_params = simple_fit["best_params"][0]
# ll_simple = simple_fit["best_ll"][0]
# ll_complex = complex_fit["best_ll"][0]

# p0 = [
#     simple_params["N_ANC"],
#     simple_params["N_CO"],  # N_CO0 (== N_CO1 under H0)
#     simple_params["N_CO"],  # N_CO1
#     simple_params["N_FR0"],
#     simple_params["N_FR1"],
#     simple_params["T"],
#     simple_params["m_CO_FR"],
#     simple_params["m_FR_CO"],
# ]

# # The real, non-windowed, whole-chromosome empirical SFS -- already cached
# # by compute_unfolded_sfs.py from the same haploid VCF used above.
# with open(unfolded_sfs_path, "rb") as f:
#     data_sfs = pickle.load(f)

# adjust = moments.Godambe.LRT_adjust(
#     split_migration_growth_both_sfs,
#     boot_spectra,
#     p0,
#     data_sfs,
#     nested_indices,
#     multinom=True,
# )
# print("LRT adjustment factor:", adjust)

# D = 2 * (ll_complex - ll_simple)
# D_adjusted = adjust * D
# # weights=(0, 1) -> standard chi^2 with 1 d.o.f. (N_CO0 isn't at a boundary
# # of its feasible range here, so no boundary mixture correction needed).
# p_value = moments.Godambe.sum_chi2_ppf(D_adjusted, weights=(0, 1))
# print(f"raw D = {D:.4g}")
# print(f"D_adjusted = {D_adjusted:.4g}")
# print(f"p-value = {p_value:.4g}")

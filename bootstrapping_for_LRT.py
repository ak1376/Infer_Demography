# bootstrapping_for_LRT.py

'''
I want to create bootstrapped replicates of the SFS for the real data. This is how I'm going to do it
    1. Split Chr2L into non-overlapping chunks and compute the SFS for each
    2. Randomly select len(chunks) chunks with replacement
    3. Sum their SFS to get one bootstrap replicate
    4. Repeat 200 times
'''

import gzip
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

# Non-overlapping chunk boundaries covering the full VCF span. `interval` in
# moments.Parsing.parse_vcf is 1-indexed and half-open, so consecutive
# (start, end) pairs here exactly tile the chromosome with no overlap.
first_pos, last_pos = _get_vcf_bounds(real_vcf)
chunk_bounds = np.linspace(first_pos, last_pos + 1, NUM_CHUNKS + 1).astype(int)
chunks = [(chunk_bounds[i], chunk_bounds[i + 1]) for i in range(NUM_CHUNKS)]

indices = np.arange(NUM_CHUNKS)


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


# Compute the (unfolded) SFS for each chunk once up front -- each is a
# moments.Spectrum, polarized using the VCF's AA (ancestral allele) field.
with ProcessPoolExecutor(max_workers=min(NUM_CHUNKS, MAX_WORKERS)) as pool:
    chunk_spectra = list(tqdm(pool.map(_parse_chunk, chunks), total=NUM_CHUNKS))


# ---- Per-window summary statistics: pi, Tajima's D, FST ----
# Reuses the same non-overlapping `chunks` windows as the SFS bootstrap
# above. Pi, Tajima's D, and Hudson's FST are all invariant to allele
# polarization, so (unlike the SFS parsing) we don't need the AA field here
# -- just per-sample REF/ALT calls.

def _get_vcf_chrom(vcf_path):
    return subprocess.check_output(
        f"bcftools query -f '%CHROM\n' '{vcf_path}' | head -n 1", shell=True
    ).decode().strip()


vcf_chrom = _get_vcf_chrom(real_vcf)


def _compute_window_stats(interval):
    """Compute pi, Tajima's D (per population) and Hudson's FST for one window."""
    start, end = interval
    region = f"{vcf_chrom}:{start}-{end - 1}"
    raw = subprocess.check_output(
        f"bcftools view -H -r '{region}' '{real_vcf}'", shell=True
    ).decode()

    ac = {pop: [] for pop in pop_names}
    for line in raw.splitlines():
        fields = line.split("\t")
        gts = fields[9:]
        for pop in pop_names:
            alt_count = sum(int(gts[i]) for i in sample_indices[pop])
            n = len(sample_indices[pop])
            ac[pop].append((n - alt_count, alt_count))

    n_sites = len(ac[pop_names[0]])
    window_len = end - start

    result = {
        "start": start, "end": end, "mid": (start + end) / 2, "n_sites": n_sites,
    }

    ac_arrays = {}
    for pop in pop_names:
        ac_pop = np.array(ac[pop], dtype=np.int32)
        ac_arrays[pop] = ac_pop
        if n_sites:
            alt = ac_pop[:, 1]
            n = len(sample_indices[pop])
            pi = float(np.sum(2.0 * alt * (n - alt) / (n * (n - 1))) / window_len)
            taj_d = float(allel.tajima_d(ac_pop))
        else:
            pi, taj_d = np.nan, np.nan
        result[f"pi_{pop}"] = pi
        result[f"tajima_d_{pop}"] = taj_d

    if n_sites and len(pop_names) >= 2:
        num, den = allel.hudson_fst(ac_arrays[pop_names[0]], ac_arrays[pop_names[1]])
        den_sum = np.nansum(den)
        fst = float(np.nansum(num) / den_sum) if den_sum > 0 else np.nan
    else:
        fst = np.nan
    result["fst"] = fst

    return result


with ProcessPoolExecutor(max_workers=min(NUM_CHUNKS, MAX_WORKERS)) as pool:
    window_stats = list(tqdm(pool.map(_compute_window_stats, chunks), total=NUM_CHUNKS))

window_stats_df = pd.DataFrame(window_stats)
window_stats_csv = "/sietch_colab/akapoor/Infer_Demography/figures/window_summary_stats.csv"
window_stats_df.to_csv(window_stats_csv, index=False)
print(f"Wrote per-window summary statistics to {window_stats_csv}")


def _plot_window_stat(df, cols, ylabel, title, out_path):
    plt.figure(figsize=(10, 4))
    for col in cols:
        plt.plot(df["mid"], df[col], marker="o", label=col)
    plt.xlabel(f"Position on {vcf_chrom} (bp)")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(cols) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


_plot_window_stat(
    window_stats_df, [f"pi_{pop}" for pop in pop_names],
    "pi (per site)", "Nucleotide diversity across windows",
    "/sietch_colab/akapoor/Infer_Demography/figures/window_pi.png",
)
_plot_window_stat(
    window_stats_df, [f"tajima_d_{pop}" for pop in pop_names],
    "Tajima's D", "Tajima's D across windows",
    "/sietch_colab/akapoor/Infer_Demography/figures/window_tajimas_d.png",
)
_plot_window_stat(
    window_stats_df, ["fst"],
    "FST (Hudson)", "FST across windows",
    "/sietch_colab/akapoor/Infer_Demography/figures/window_fst.png",
)


boot_spectra = []
for boot_idx in tqdm(np.arange(num_boot_reps)):
    # Randomly select NUM_CHUNKS chunks with replacement
    sampled_indices = np.random.choice(indices, size=len(indices), replace=True)
    # Sum the cached per-chunk spectra for the sampled chunks to get this
    # bootstrap replicate's SFS (still a moments.Spectrum)
    boot_sfs = sum(chunk_spectra[i] for i in sampled_indices)
    boot_spectra.append(boot_sfs)


# Now that I have the bootstraps, I need to wrap my split migration growth both model in a format that will be recognizable by LRT_adjust in Godambe module

param_names = [
    "N_ANC",
    "N_CO0",
    "N_CO1",
    "N_FR0",
    "N_FR1",
    "T",
    "m_CO_FR",
    "m_FR_CO",
]

def split_migration_growth_both_sfs(p, ns):
    sampled = dict(zip(param_names, p))

    graph = split_migration_growth_both_model(sampled)

    fs = moments.Spectrum.from_demes(
        graph,
        sampled_demes=["CO", "FR"],
        sample_sizes=ns,
    )

    return fs


# ---- Likelihood-ratio test: does CO actually have exponential growth? ----
# H0 (nested/simple model): no growth in CO, i.e. N_CO0 == N_CO1. That
# collapses one degree of freedom relative to the full "both" model, so
# N_CO0 is the "extra" parameter that only exists in the complex model.
nested_indices = [param_names.index("N_CO0")]

# p0 = best fit of the null model (split_migration_growth: CO constant,
# only FR grows), written out in the 8-param order the complex model
# expects, with N_CO0 == N_CO1 == the null model's single N_CO. This is
# the real H0-consistent point.
with open(simple_fit_path, "rb") as f:
    simple_fit = pickle.load(f)
with open(complex_fit_path, "rb") as f:
    complex_fit = pickle.load(f)

simple_params = simple_fit["best_params"][0]
ll_simple = simple_fit["best_ll"][0]
ll_complex = complex_fit["best_ll"][0]

p0 = [
    simple_params["N_ANC"],
    simple_params["N_CO"],  # N_CO0 (== N_CO1 under H0)
    simple_params["N_CO"],  # N_CO1
    simple_params["N_FR0"],
    simple_params["N_FR1"],
    simple_params["T"],
    simple_params["m_CO_FR"],
    simple_params["m_FR_CO"],
]

# The real, non-windowed, whole-chromosome empirical SFS -- already cached
# by compute_unfolded_sfs.py from the same haploid VCF used above.
with open(unfolded_sfs_path, "rb") as f:
    data_sfs = pickle.load(f)

adjust = moments.Godambe.LRT_adjust(
    split_migration_growth_both_sfs,
    boot_spectra,
    p0,
    data_sfs,
    nested_indices,
    multinom=True,
)
print("LRT adjustment factor:", adjust)

D = 2 * (ll_complex - ll_simple)
D_adjusted = adjust * D
# weights=(0, 1) -> standard chi^2 with 1 d.o.f. (N_CO0 isn't at a boundary
# of its feasible range here, so no boundary mixture correction needed).
p_value = moments.Godambe.sum_chi2_ppf(D_adjusted, weights=(0, 1))
print(f"raw D = {D:.4g}")
print(f"D_adjusted = {D_adjusted:.4g}")
print(f"p-value = {p_value:.4g}")

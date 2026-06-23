"""
Neutral PPC for Drosophila — SFS statistics.

Loads pre-computed SFS from neutral simulations, projects each one down to
the observed sample sizes, computes Pi, Tajima's D, FST, and SFS shapes,
then compares to observed Chr2L data via violin plots.

Uses the polarized (AA-annotated) haploid VCF so both observed and simulated
SFS are unfolded (derived-allele oriented).

Usage:
    python neutral_ppc_drosophila.py --model split_migration_growth
    python neutral_ppc_drosophila.py --model drosophila_three_epoch
"""

from pathlib import Path
import argparse
import pickle
import gzip
from typing import Dict, List

import numpy as np
import numpy.ma as ma
import allel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------
ROOT = Path('/sietch_colab/akapoor/Infer_Demography')

MODEL_CONFIGS = {
    "split_migration_growth": {
        "sim_dir":    ROOT / "experiments/split_migration_growth/simulations",
        "pop_labels": ["CO", "FR"],
        "out_name":   "neutral_ppc_sfs_split_migration_growth.png",
    },
    "drosophila_three_epoch": {
        # simulations_ld_ppc uses current param names; the old 20k sims are
        # fine for the SFS PPC since they only need SFS.pkl (no param dict lookup)
        "sim_dir":    ROOT / "experiments/drosophila_three_epoch/simulations",
        "pop_labels": ["CO", "FR"],
        "out_name":   "neutral_ppc_sfs_drosophila_three_epoch.png",
    },
}

REAL_VCF = str(ROOT / 'real_data_analysis/data/drosophila/Chr2L.polarized.vcf.gz')
POPFILE  = str(ROOT / 'real_data_analysis/data/drosophila/popfile.txt')
OUT_DIR  = ROOT / 'model_calibration_drosophila_model'
OUT_DIR.mkdir(exist_ok=True)

GENOME_LENGTH = 24_000_000  # bp

# ---------------------------------------------------------------------------
# Stats from SFS
# ---------------------------------------------------------------------------

def _fill(arr) -> np.ndarray:
    """Return plain float64 array with masked entries set to 0."""
    if isinstance(arr, ma.MaskedArray):
        return ma.filled(arr, 0).astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def pi_from_1d_sfs(sfs_1d, L: float = GENOME_LENGTH) -> float:
    """Per-site nucleotide diversity from unfolded 1D SFS counts."""
    c = _fill(sfs_1d)
    n = len(c) - 1
    if n < 2:
        return np.nan
    i = np.arange(n + 1, dtype=np.float64)
    w = 2.0 * i * (n - i) / (n * (n - 1))
    return float(np.dot(w, c) / L)


def tajima_d_from_1d_sfs(sfs_1d) -> float:
    """Tajima's D from unfolded 1D SFS counts (Tajima 1989)."""
    c = _fill(sfs_1d)
    n = len(c) - 1
    if n < 4:
        return np.nan
    seg = c[1:n]
    S = seg.sum()
    if S == 0:
        return np.nan

    idx = np.arange(1, n, dtype=np.float64)
    theta_pi = float(np.dot(2.0 * idx * (n - idx) / (n * (n - 1)), seg))

    a1 = np.sum(1.0 / idx)
    a2 = np.sum(1.0 / idx**2)
    theta_W = S / a1

    b1 = (n + 1) / (3.0 * (n - 1))
    b2 = 2.0 * (n**2 + n + 3) / (9.0 * n * (n - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)

    var = e1 * S + e2 * S * (S - 1)
    if var <= 0:
        return np.nan
    return float((theta_pi - theta_W) / np.sqrt(var))


def fst_from_2d_sfs(sfs_2d) -> float:
    """Hudson's FST from unfolded 2D SFS counts."""
    counts = _fill(sfs_2d)
    n1 = counts.shape[0] - 1
    n2 = counts.shape[1] - 1
    I, J = np.meshgrid(np.arange(n1 + 1), np.arange(n2 + 1), indexing='ij')

    with np.errstate(divide='ignore', invalid='ignore'):
        p1 = I / n1
        p2 = J / n2
        num = (p1 - p2)**2 - p1*(1-p1)/(n1-1) - p2*(1-p2)/(n2-1)
        den = p1*(1-p2) + p2*(1-p1)

    counts[0, 0] = 0.0
    counts[n1, n2] = 0.0
    counts[n1, 0] = 0.0
    counts[0, n2] = 0.0

    total_num = float(np.nansum(num * counts))
    total_den = float(np.nansum(den * counts))
    return total_num / total_den if total_den > 0 else np.nan


def stats_from_sfs(sfs) -> Dict[str, float]:
    m1 = sfs.marginalize([1])   # pop dim-0 marginal
    m2 = sfs.marginalize([0])   # pop dim-1 marginal
    return {
        'pi_pop1':       pi_from_1d_sfs(m1),
        'pi_pop2':       pi_from_1d_sfs(m2),
        'tajima_d_pop1': tajima_d_from_1d_sfs(m1),
        'tajima_d_pop2': tajima_d_from_1d_sfs(m2),
        'fst':           fst_from_2d_sfs(sfs),
    }


# ---------------------------------------------------------------------------
# Load and project simulations (unfolded)
# ---------------------------------------------------------------------------

def _strip_corners(sfs):
    """Zero out fixed/absent corners so simulated SFS matches VCF-derived observed SFS."""
    s = sfs.copy()
    if s.ndim == 1:
        s[0] = 0
        s[-1] = 0
    elif s.ndim == 2:
        s[0, 0] = 0
        s[-1, -1] = 0
        s[-1, 0] = 0
        s[0, -1] = 0
    return s


def load_sim_sfs(sim_dir: Path, target_sizes: List[int]) -> List:
    """Load and project simulated SFS to target_sizes. Keep unfolded."""
    sfs_list = []
    dirs = sorted([d for d in sim_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name))
    for d in dirs:
        p = d / 'SFS.pkl'
        if p.exists():
            with open(p, 'rb') as f:
                sfs = pickle.load(f)
            sfs_list.append(_strip_corners(sfs.project(target_sizes)))
    print(f"Loaded and projected to {target_sizes}: {len(sfs_list)} simulation SFS files")
    return sfs_list


# ---------------------------------------------------------------------------
# Observed stats from polarized haploid VCF
# ---------------------------------------------------------------------------

def _parse_popfile(popfile: str):
    """Returns (pop_names, sample_to_pop dict)."""
    sample_to_pop = {}
    pop_order = []
    with open(popfile) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sample, pop = parts[0], parts[1]
            sample_to_pop[sample] = pop
            if pop not in pop_order:
                pop_order.append(pop)
    return pop_order, sample_to_pop


def compute_obs_stats(vcf_path: str, popfile: str, L: float = GENOME_LENGTH):
    """
    Read the polarized haploid VCF (AA INFO field) and compute unfolded
    summary stats and SFS for each population.

    Returns: scalar_stats, sfs_co, sfs_fr, sfs_2d, n_co, n_fr
    """
    pop_names, sample_to_pop = _parse_popfile(popfile)

    opener = gzip.open if vcf_path.endswith('.gz') else open

    # First pass: get sample order from header
    sample_indices = {pop: [] for pop in pop_names}
    vcf_samples = []
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                vcf_samples = line.rstrip('\n').split('\t')[9:]
                for i, s in enumerate(vcf_samples):
                    pop = sample_to_pop.get(s)
                    if pop is not None:
                        sample_indices[pop].append(i)
                break

    co_idx = sample_indices['CO']
    fr_idx = sample_indices['FR']
    n_co, n_fr = len(co_idx), len(fr_idx)
    print(f"  CO: {n_co} haploid samples,  FR: {n_fr} haploid samples")

    # Accumulate derived allele counts and positions
    dac_co_list, dac_fr_list, pos_list = [], [], []

    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            pos  = int(fields[1])
            ref  = fields[3]
            alt  = fields[4]
            info = fields[7]
            gts  = fields[9:]

            # Parse AA
            aa = None
            for token in info.split(';'):
                if token.startswith('AA='):
                    aa = token[3:]
                    break
            if aa is None:
                continue

            flip = (aa == alt)   # True -> REF is derived, count GT=0 as derived

            alt_co = sum(int(gts[i]) for i in co_idx)
            alt_fr = sum(int(gts[i]) for i in fr_idx)

            dac_co = (n_co - alt_co) if flip else alt_co
            dac_fr = (n_fr - alt_fr) if flip else alt_fr

            dac_co_list.append(dac_co)
            dac_fr_list.append(dac_fr)
            pos_list.append(pos)

    dac_co = np.array(dac_co_list, dtype=np.int32)
    dac_fr = np.array(dac_fr_list, dtype=np.int32)
    pos    = np.array(pos_list,    dtype=np.int32)

    print(f"  Polarized sites: {len(dac_co):,}")

    # Allele count arrays for allel.tajima_d / allel.hudson_fst
    ac_co = np.stack([n_co - dac_co, dac_co], axis=1).astype(np.int32)
    ac_fr = np.stack([n_fr - dac_fr, dac_fr], axis=1).astype(np.int32)

    # Pi (from SFS formula)
    theta_pi_co = float(np.sum(2.0 * dac_co * (n_co - dac_co) / (n_co * (n_co - 1))))
    theta_pi_fr = float(np.sum(2.0 * dac_fr * (n_fr - dac_fr) / (n_fr * (n_fr - 1))))

    # Tajima's D
    taj_co = float(allel.tajima_d(ac_co, pos))
    taj_fr = float(allel.tajima_d(ac_fr, pos))

    # FST (Hudson)
    num, den = allel.hudson_fst(ac_co, ac_fr)
    fst = float(np.nansum(num) / np.nansum(den))

    scalar_stats = {
        'pi_pop1':       theta_pi_co / L,
        'pi_pop2':       theta_pi_fr / L,
        'tajima_d_pop1': taj_co,
        'tajima_d_pop2': taj_fr,
        'fst':           fst,
    }

    print(f"  pi_CO={scalar_stats['pi_pop1']:.4g}  pi_FR={scalar_stats['pi_pop2']:.4g}  "
          f"Taj_D_CO={taj_co:.4g}  Taj_D_FR={taj_fr:.4g}  FST={fst:.4g}")

    # Unfolded 1D SFS
    sfs_co = np.zeros(n_co + 1)
    sfs_fr = np.zeros(n_fr + 1)
    for v in dac_co:
        if 0 < v < n_co:
            sfs_co[v] += 1
    for v in dac_fr:
        if 0 < v < n_fr:
            sfs_fr[v] += 1

    # Unfolded 2D SFS
    sfs_2d = np.zeros((n_co + 1, n_fr + 1))
    for ci, fj in zip(dac_co, dac_fr):
        if (0 < ci < n_co) or (0 < fj < n_fr):
            sfs_2d[ci, fj] += 1

    return scalar_stats, sfs_co, sfs_fr, sfs_2d, n_co, n_fr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _norm_sfs(s: np.ndarray) -> np.ndarray:
    total = s[1:-1].sum()
    return s / total if total > 0 else s


def plot_results(
    sim_stats: List[Dict],
    obs_stats: Dict,
    sim_sfs_1: np.ndarray,
    sim_sfs_2: np.ndarray,
    sim_sfs_2d: np.ndarray,
    obs_sfs_1: np.ndarray,
    obs_sfs_2: np.ndarray,
    obs_sfs_2d: np.ndarray,
    pop_labels: List[str],
    model: str,
    out_path: Path,
) -> None:
    p1, p2 = pop_labels

    fig = plt.figure(figsize=(22, 15))
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.50, wspace=0.38)

    # ---- Row 0: scalar violin plots ----------------------------------------
    scalar_keys   = ['pi_pop1', 'pi_pop2', 'tajima_d_pop1', 'tajima_d_pop2', 'fst']
    scalar_labels = [f'π ({p1})', f'π ({p2})', f"Tajima's D ({p1})", f"Tajima's D ({p2})", r'$F_{ST}$']

    for col, (key, label) in enumerate(zip(scalar_keys, scalar_labels)):
        ax = fig.add_subplot(gs[0, col])
        vals = [s[key] for s in sim_stats if np.isfinite(s.get(key, np.nan))]
        ax.violinplot(vals, positions=[0], showmedians=True)
        ax.scatter([0], [obs_stats[key]], color='red', zorder=5, s=45, label='observed')
        ax.set_xticks([])
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)

    # ---- Row 1: 1D SFS comparison ------------------------------------------
    for col, (sim_mat, obs_sfs, pop) in enumerate([
        (sim_sfs_1, obs_sfs_1, p1),
        (sim_sfs_2, obs_sfs_2, p2),
    ]):
        ax = fig.add_subplot(gs[1, col*2 : col*2+2])
        norm_sim = np.array([_norm_sfs(s) for s in sim_mat])
        lo = np.percentile(norm_sim, 5,  axis=0)
        hi = np.percentile(norm_sim, 95, axis=0)
        mn = np.mean(norm_sim, axis=0)

        bins = np.arange(len(mn))
        ax.fill_between(bins[1:-1], lo[1:-1], hi[1:-1], alpha=0.3, color='steelblue', label='5–95% CI (sim)')
        ax.plot(bins[1:-1], mn[1:-1], color='steelblue', linewidth=1.5, label='mean sim')
        ax.plot(bins[1:-1], _norm_sfs(obs_sfs)[1:-1], color='red', linewidth=1.5, label='observed')

        ax.set_xlabel('Derived allele count', fontsize=10)
        ax.set_ylabel('Proportion of sites', fontsize=10)
        ax.set_title(f'1D SFS (unfolded) — {pop}', fontsize=11)
        ax.legend(fontsize=8)

    # ---- Row 2: 2D SFS heatmaps --------------------------------------------
    mean_2d = np.mean(sim_sfs_2d, axis=0)

    ax_sim = fig.add_subplot(gs[2, :2])
    im1 = ax_sim.imshow(np.log1p(mean_2d.T), origin='lower', aspect='auto', cmap='viridis')
    ax_sim.set_title('2D SFS — mean simulated (log1p)', fontsize=11)
    ax_sim.set_xlabel(f'{p1} derived allele count', fontsize=10)
    ax_sim.set_ylabel(f'{p2} derived allele count', fontsize=10)
    plt.colorbar(im1, ax=ax_sim)

    ax_obs = fig.add_subplot(gs[2, 2:4])
    im2 = ax_obs.imshow(np.log1p(obs_sfs_2d.T), origin='lower', aspect='auto', cmap='viridis')
    ax_obs.set_title('2D SFS — observed (log1p)', fontsize=11)
    ax_obs.set_xlabel(f'{p1} derived allele count', fontsize=10)
    ax_obs.set_ylabel(f'{p2} derived allele count', fontsize=10)
    plt.colorbar(im2, ax=ax_obs)

    ax_diff = fig.add_subplot(gs[2, 4])
    obs_norm = obs_sfs_2d / obs_sfs_2d.sum() if obs_sfs_2d.sum() > 0 else obs_sfs_2d
    sim_norm = mean_2d / mean_2d.sum() if mean_2d.sum() > 0 else mean_2d
    diff = obs_norm - sim_norm
    vmax = np.abs(diff).max()
    im3 = ax_diff.imshow(diff.T, origin='lower', aspect='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    ax_diff.set_title('2D SFS diff\n(obs − sim)', fontsize=10)
    ax_diff.set_xlabel(f'{p1} count', fontsize=9)
    ax_diff.set_ylabel(f'{p2} count', fontsize=9)
    plt.colorbar(im3, ax=ax_diff)

    fig.suptitle(f'Neutral PPC — {model} (Chr2L)', fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="drosophila_three_epoch",
        help="Which demographic model to run the PPC for.",
    )
    args = parser.parse_args()

    mc         = MODEL_CONFIGS[args.model]
    sim_dir    = mc["sim_dir"]
    pop_labels = mc["pop_labels"]

    print("Computing observed stats from polarized VCF...")
    obs_stats, obs_sfs_1, obs_sfs_2, obs_sfs_2d, n1, n2 = compute_obs_stats(REAL_VCF, POPFILE)

    print("\nLoading and projecting simulation SFS...")
    all_sfs = load_sim_sfs(sim_dir, [n1, n2])

    print("\nComputing stats from simulations...")
    sim_stats = [stats_from_sfs(s) for s in all_sfs]

    sim_sfs_1  = np.array([_fill(s.marginalize([1])) for s in all_sfs])
    sim_sfs_2  = np.array([_fill(s.marginalize([0])) for s in all_sfs])
    sim_sfs_2d = np.array([_fill(s) for s in all_sfs])

    print("\nPlotting...")
    plot_results(
        sim_stats, obs_stats,
        sim_sfs_1, sim_sfs_2, sim_sfs_2d,
        obs_sfs_1, obs_sfs_2, obs_sfs_2d,
        pop_labels=pop_labels,
        model=args.model,
        out_path=OUT_DIR / mc["out_name"],
    )

    print("\nSummary (simulated null vs observed):")
    p1, p2 = pop_labels
    for key, label in [
        ('pi_pop1',       f'pi_{p1}'),
        ('pi_pop2',       f'pi_{p2}'),
        ('tajima_d_pop1', f'tajima_d_{p1}'),
        ('tajima_d_pop2', f'tajima_d_{p2}'),
        ('fst',           'fst'),
    ]:
        vals = [s[key] for s in sim_stats if np.isfinite(s.get(key, np.nan))]
        lo, med, hi = np.percentile(vals, [5, 50, 95])
        obs = obs_stats[key]
        pct = float(np.mean(np.array(vals) <= obs)) * 100
        print(f"  {label:20s}: obs={obs:.4g}  sim=[{lo:.4g}, {med:.4g}, {hi:.4g}]  "
              f"percentile={pct:.1f}%")


if __name__ == '__main__':
    main()

"""
Neutral PPC for Drosophila.

Loads pre-computed SFS from neutral simulations, projects each one down to
the observed sample sizes, computes Pi, Tajima's D, FST, and SFS shapes,
then compares to observed Chr2L data via violin plots.
"""

from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import numpy as np
import numpy.ma as ma
import allel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIM_DIR  = Path('/sietch_colab/akapoor/Infer_Demography/experiments/drosophila_three_epoch/simulations')
REAL_VCF = '/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila/Chr2L.diploidGT.vcf.gz'
OUT_DIR  = Path('/sietch_colab/akapoor/Infer_Demography/model_calibration_drosophila_stdpopsim')
OUT_DIR.mkdir(exist_ok=True)

GENOME_LENGTH = 24_000_000  # bp, from bgs.meta.json

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
    c = _fill(sfs_1d) # Convert to a numpy array and replace the masked corners with 0
    n = len(c) - 1 # Number of haplotypes
    # Safety check
    if n < 2:
        return np.nan
    i = np.arange(n + 1, dtype=np.float64) # Allele count index for each bin
    w = 2.0 * i * (n - i) / (n * (n - 1))
    return float(np.dot(w, c) / L)


def tajima_d_from_1d_sfs(sfs_1d) -> float:
    """Tajima's D from unfolded 1D SFS counts (Tajima 1989)."""
    c = _fill(sfs_1d)
    n = len(c) - 1
    if n < 4:
        return np.nan
    seg = c[1:n] # Look at bins 1 ... 39
    S = seg.sum() # Total number of segregating sites
    if S == 0:
        return np.nan

    idx = np.arange(1, n, dtype=np.float64)
    theta_pi = float(np.dot(2.0 * idx * (n - idx) / (n * (n - 1)), seg)) # Numerator of Tajima's pi

    # Watterson's Estimator
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
    return float((theta_pi - theta_W) / np.sqrt(var)) # Tajima's D statistic


def fst_from_2d_sfs(sfs_2d) -> float:
    """Hudson's FST from unfolded 2D SFS counts."""
    counts = _fill(sfs_2d)
    n1 = counts.shape[0] - 1
    n2 = counts.shape[1] - 1
    I, J = np.meshgrid(np.arange(n1 + 1), np.arange(n2 + 1), indexing='ij')

    with np.errstate(divide='ignore', invalid='ignore'):
        p1 = I / n1
        p2 = J / n2
        num = (p1 - p2)**2 - p1*(1-p1)/(n1-1) - p2*(1-p2)/(n2-1) # Between population variance minus sampling noise
        den = p1*(1-p2) + p2*(1-p1) # Probability that two randomly drawn alleles (one from each pop) are different. 

    # Monomorphic
    counts[0, 0] = 0.0
    counts[n1, n2] = 0.0

    total_num = float(np.nansum(num * counts))
    total_den = float(np.nansum(den * counts))
    return total_num / total_den if total_den > 0 else np.nan # FST


def stats_from_sfs(sfs) -> Dict[str, float]:
    m_co = sfs.marginalize([1])
    m_fr = sfs.marginalize([0])
    return {
        'pi_CO':       pi_from_1d_sfs(m_co),
        'pi_FR':       pi_from_1d_sfs(m_fr),
        'tajima_d_CO': tajima_d_from_1d_sfs(m_co),
        'tajima_d_FR': tajima_d_from_1d_sfs(m_fr),
        'fst':         fst_from_2d_sfs(sfs),
    }


# ---------------------------------------------------------------------------
# Load and project simulations
# ---------------------------------------------------------------------------

def load_sim_sfs(sim_dir: Path, target_sizes: List[int]) -> List:
    """Load, project to target_sizes, then fold each simulated SFS."""
    sfs_list = []
    dirs = sorted([d for d in sim_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name))
    for d in dirs:
        p = d / 'SFS.pkl'
        if p.exists():
            with open(p, 'rb') as f:
                sfs = pickle.load(f)
            sfs_list.append(sfs.project(target_sizes).fold())
    print(f"Loaded, projected to {target_sizes}, and folded {len(sfs_list)} simulation SFS files")
    return sfs_list


# ---------------------------------------------------------------------------
# Observed stats from VCF
# ---------------------------------------------------------------------------

def compute_obs_stats(vcf_path: str, L: float = GENOME_LENGTH):
    """Returns scalar_stats, sfs_co, sfs_fr, sfs_2d, n_co, n_fr."""
    callset = allel.read_vcf(vcf_path, fields=['calldata/GT', 'variants/POS', 'samples'])
    samples  = list(callset['samples'])
    co_idx   = [i for i, s in enumerate(samples) if s.startswith('CO')]
    fr_idx   = [i for i, s in enumerate(samples) if s.startswith('FR')]

    gt  = allel.GenotypeArray(callset['calldata/GT'])
    pos = callset['variants/POS']

    ac_co = gt.take(co_idx, axis=1).count_alleles()
    ac_fr = gt.take(fr_idx, axis=1).count_alleles()

    # Inbred lines: each individual is homozygous (0/0 or 1/1), so allele
    # counts are always even.  Treat each individual as one haploid sample by
    # dividing allele counts by 2.
    n_co = len(co_idx)
    n_fr = len(fr_idx)
    print(f"  VCF sample sizes: CO={n_co} individuals, FR={n_fr} individuals (inbred → haploid)")

    dac_co = ac_co[:, 1] // 2
    dac_fr = ac_fr[:, 1] // 2

    # Haploid allele-count matrices for allel.tajima_d / allel.hudson_fst
    ac_hap_co = np.stack([n_co - dac_co, dac_co], axis=1).astype(np.int32)
    ac_hap_fr = np.stack([n_fr - dac_fr, dac_fr], axis=1).astype(np.int32)

    # --- diploid (old) stats for comparison ---
    n_co_dip = len(co_idx) * 2
    n_fr_dip = len(fr_idx) * 2
    dac_co_dip = ac_co[:, 1]
    dac_fr_dip = ac_fr[:, 1]
    pi_co_dip = float(np.sum(2.0 * dac_co_dip * (n_co_dip - dac_co_dip) / (n_co_dip * (n_co_dip - 1)))) / L
    pi_fr_dip = float(np.sum(2.0 * dac_fr_dip * (n_fr_dip - dac_fr_dip) / (n_fr_dip * (n_fr_dip - 1)))) / L
    taj_co_dip = float(allel.tajima_d(ac_co, pos))
    taj_fr_dip = float(allel.tajima_d(ac_fr, pos))
    num_dip, den_dip = allel.hudson_fst(ac_co, ac_fr)
    fst_dip = float(np.nansum(num_dip) / np.nansum(den_dip))

    # Pi normalised by same L as SFS-based pi
    theta_pi_co = float(np.sum(2.0 * dac_co * (n_co - dac_co) / (n_co * (n_co - 1))))
    theta_pi_fr = float(np.sum(2.0 * dac_fr * (n_fr - dac_fr) / (n_fr * (n_fr - 1))))

    # Tajima's D via scikit-allel (on haploid counts)
    taj_co = float(allel.tajima_d(ac_hap_co, pos))
    taj_fr = float(allel.tajima_d(ac_hap_fr, pos))

    # FST (Hudson, on haploid counts)
    num, den = allel.hudson_fst(ac_hap_co, ac_hap_fr)
    fst = float(np.nansum(num) / np.nansum(den))

    print("\n  Diploid (old) vs Individual/haploid (new) observed stats:")
    print(f"  {'stat':<16}  {'diploid':>12}  {'individual':>12}  {'ratio new/old':>14}")
    print(f"  {'-'*58}")
    for name, old, new in [
        ('pi_CO',        pi_co_dip,  theta_pi_co / L),
        ('pi_FR',        pi_fr_dip,  theta_pi_fr / L),
        ('tajima_d_CO',  taj_co_dip, taj_co),
        ('tajima_d_FR',  taj_fr_dip, taj_fr),
        ('fst',          fst_dip,    fst),
    ]:
        ratio = new / old if old != 0 else float('nan')
        print(f"  {name:<16}  {old:>12.5g}  {new:>12.5g}  {ratio:>14.4f}")

    scalar_stats = {
        'pi_CO':       theta_pi_co / L,
        'pi_FR':       theta_pi_fr / L,
        'tajima_d_CO': taj_co,
        'tajima_d_FR': taj_fr,
        'fst':         fst,
    }

    # Folded 1D SFS
    sfs_co = np.zeros(n_co + 1)
    sfs_fr = np.zeros(n_fr + 1)
    for val in dac_co:
        mac = int(min(val, n_co - val))
        if mac > 0:
            sfs_co[mac] += 1
    for val in dac_fr:
        mac = int(min(val, n_fr - val))
        if mac > 0:
            sfs_fr[mac] += 1

    # Folded 2D SFS
    sfs_2d = np.zeros((n_co + 1, n_fr + 1))
    for ci, fj in zip(dac_co, dac_fr):
        mac_co = int(min(ci, n_co - ci))
        mac_fr = int(min(fj, n_fr - fj))
        if mac_co > 0 or mac_fr > 0:
            sfs_2d[mac_co, mac_fr] += 1

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
    sim_sfs_co: np.ndarray,   # (n_sim, n_co+1)
    sim_sfs_fr: np.ndarray,   # (n_sim, n_fr+1)
    sim_sfs_2d: np.ndarray,   # (n_sim, n_co+1, n_fr+1)
    obs_sfs_co: np.ndarray,   # (n_co+1,)
    obs_sfs_fr: np.ndarray,   # (n_fr+1,)
    obs_sfs_2d: np.ndarray,   # (n_co+1, n_fr+1)
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 15))
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.50, wspace=0.38)

    # ---- Row 0: scalar violin plots ----------------------------------------
    scalar_keys   = ['pi_CO', 'pi_FR', 'tajima_d_CO', 'tajima_d_FR', 'fst']
    scalar_labels = ['π (CO)', 'π (FR)', "Tajima's D (CO)", "Tajima's D (FR)", r'$F_{ST}$']

    for col, (key, label) in enumerate(zip(scalar_keys, scalar_labels)):
        ax = fig.add_subplot(gs[0, col])
        vals = [s[key] for s in sim_stats if np.isfinite(s.get(key, np.nan))]
        ax.violinplot(vals, positions=[0], showmedians=True)
        ax.scatter([0], [obs_stats[key]], color='red', zorder=5, s=45, label='observed')
        ax.set_xticks([])
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)

    # ---- Row 1: 1D SFS comparison ------------------------------------------
    # Simulated and observed now have the same n after projection, so allele
    # count is a valid shared x-axis.
    for col, (sim_mat, obs_sfs, pop) in enumerate([
        (sim_sfs_co, obs_sfs_co, 'CO'),
        (sim_sfs_fr, obs_sfs_fr, 'FR'),
    ]):
        ax = fig.add_subplot(gs[1, col*2 : col*2+2])
        norm_sim = np.array([_norm_sfs(s) for s in sim_mat])
        lo = np.percentile(norm_sim, 5,  axis=0)
        hi = np.percentile(norm_sim, 95, axis=0)
        mn = np.mean(norm_sim, axis=0)

        bins = np.arange(len(mn))   # allele count 0..n (same for sim and obs after projection)
        ax.fill_between(bins[1:-1], lo[1:-1], hi[1:-1], alpha=0.3, color='steelblue', label='5–95% CI (sim)')
        ax.plot(bins[1:-1], mn[1:-1], color='steelblue', linewidth=1.5, label='mean sim')
        ax.plot(bins[1:-1], _norm_sfs(obs_sfs)[1:-1], color='red', linewidth=1.5, label='observed')

        ax.set_xlabel('Minor allele count', fontsize=10)
        ax.set_ylabel('Proportion of sites', fontsize=10)
        ax.set_title(f'1D SFS — {pop}', fontsize=11)
        ax.legend(fontsize=8)

    # ---- Row 2: 2D SFS heatmaps --------------------------------------------
    mean_2d = np.mean(sim_sfs_2d, axis=0)

    ax_sim = fig.add_subplot(gs[2, :2])
    im1 = ax_sim.imshow(np.log1p(mean_2d.T), origin='lower', aspect='auto', cmap='viridis')
    ax_sim.set_title('2D SFS — mean simulated (log1p)', fontsize=11)
    ax_sim.set_xlabel('CO minor allele count', fontsize=10)
    ax_sim.set_ylabel('FR minor allele count', fontsize=10)
    plt.colorbar(im1, ax=ax_sim)

    ax_obs = fig.add_subplot(gs[2, 2:4])
    im2 = ax_obs.imshow(np.log1p(obs_sfs_2d.T), origin='lower', aspect='auto', cmap='viridis')
    ax_obs.set_title('2D SFS — observed (log1p)', fontsize=11)
    ax_obs.set_xlabel('CO minor allele count', fontsize=10)
    ax_obs.set_ylabel('FR minor allele count', fontsize=10)
    plt.colorbar(im2, ax=ax_obs)

    # difference panel — shapes now match after projection
    ax_diff = fig.add_subplot(gs[2, 4])
    obs_norm = obs_sfs_2d / obs_sfs_2d.sum() if obs_sfs_2d.sum() > 0 else obs_sfs_2d
    sim_norm = mean_2d / mean_2d.sum() if mean_2d.sum() > 0 else mean_2d
    diff = obs_norm - sim_norm
    vmax = np.abs(diff).max()
    im3 = ax_diff.imshow(diff.T, origin='lower', aspect='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    ax_diff.set_title('2D SFS diff\n(obs − sim)', fontsize=10)
    ax_diff.set_xlabel('CO count', fontsize=9)
    ax_diff.set_ylabel('FR count', fontsize=9)
    plt.colorbar(im3, ax=ax_diff)

    fig.suptitle('Neutral PPC — split_migration_growth (Drosophila Chr2L)', fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Computing observed stats from VCF...")
    obs_stats, obs_sfs_co, obs_sfs_fr, obs_sfs_2d, n_co, n_fr = compute_obs_stats(REAL_VCF)

    print("Loading and projecting simulation SFS...")
    all_sfs = load_sim_sfs(SIM_DIR, [n_co, n_fr])

    print("Computing stats from simulations...")
    sim_stats = [stats_from_sfs(s) for s in all_sfs]

    sim_sfs_co = np.array([_fill(s.marginalize([1])) for s in all_sfs])
    sim_sfs_fr = np.array([_fill(s.marginalize([0])) for s in all_sfs])
    sim_sfs_2d = np.array([_fill(s) for s in all_sfs])

    print("Plotting...")
    plot_results(
        sim_stats, obs_stats,
        sim_sfs_co, sim_sfs_fr, sim_sfs_2d,
        obs_sfs_co, obs_sfs_fr, obs_sfs_2d,
        OUT_DIR / 'neutral_ppc.png',
    )

    print("\nSummary (simulated null vs observed):")
    for key in ['pi_CO', 'pi_FR', 'tajima_d_CO', 'tajima_d_FR', 'fst']:
        vals = [s[key] for s in sim_stats if np.isfinite(s.get(key, np.nan))]
        lo, med, hi = np.percentile(vals, [5, 50, 95])
        obs = obs_stats[key]
        pct = float(np.mean(np.array(vals) <= obs)) * 100
        print(f"  {key:16s}: obs={obs:.4g}  sim=[{lo:.4g}, {med:.4g}, {hi:.4g}]  "
              f"percentile={pct:.1f}%")


if __name__ == '__main__':
    main()

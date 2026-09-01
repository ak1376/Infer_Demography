#!/usr/bin/env python3
# snakemake_scripts/calibration_ppc.py
#
# Model calibration check (posterior-predictive check): compare SFS-derived
# summary stats (pi, Tajima's D, FST) and SFS shape between the real observed
# data and the calibration_simulate replicates (tree-sequence simulations at
# the real-data fitted params). Produces violin/SFS-comparison plots plus a
# JSON percentile summary.
#
# Repurposed from the standalone neutral_ppc_drosophila.py, with two changes:
#  - observed stats come from the pipeline's already-built COMBINED_SFS
#    (summed over whatever AUTOSOMES the active config uses) instead of
#    re-parsing a hardcoded Chr2L VCF -- this also means observed and
#    simulated stats go through the exact same stats_from_sfs() formulas
#    instead of two independently-implemented estimators (scikit-allel for
#    observed, hand-rolled for simulated).
#  - sim_dir is the calibration_simulate replicate directory, not the raw
#    training sim ensemble, so this is a genuine posterior-predictive check
#    (simulate at the fitted params) rather than a prior-predictive one.

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Stats from SFS (shared by observed and simulated -- same formulas both sides)
# ---------------------------------------------------------------------------

def _fill(arr) -> np.ndarray:
    """Return plain float64 array with masked entries set to 0."""
    if isinstance(arr, ma.MaskedArray):
        return ma.filled(arr, 0).astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def pi_from_1d_sfs(sfs_1d, L: float) -> float:
    """Per-site nucleotide diversity from unfolded 1D SFS counts."""
    c = _fill(sfs_1d)
    n = len(c) - 1
    if n < 2:
        return float("nan")
    i = np.arange(n + 1, dtype=np.float64)
    w = 2.0 * i * (n - i) / (n * (n - 1))
    return float(np.dot(w, c) / L)


def tajima_d_from_1d_sfs(sfs_1d) -> float:
    """Tajima's D from unfolded 1D SFS counts (Tajima 1989)."""
    c = _fill(sfs_1d)
    n = len(c) - 1
    if n < 4:
        return float("nan")
    seg = c[1:n]
    S = seg.sum()
    if S == 0:
        return float("nan")

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
        return float("nan")
    return float((theta_pi - theta_W) / np.sqrt(var))


def fst_from_2d_sfs(sfs_2d) -> float:
    """Hudson's FST from unfolded 2D SFS counts."""
    counts = _fill(sfs_2d)
    n1 = counts.shape[0] - 1
    n2 = counts.shape[1] - 1
    I, J = np.meshgrid(np.arange(n1 + 1), np.arange(n2 + 1), indexing="ij")

    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = I / n1
        p2 = J / n2
        num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
        den = p1 * (1 - p2) + p2 * (1 - p1)

    counts[0, 0] = 0.0
    counts[n1, n2] = 0.0
    counts[n1, 0] = 0.0
    counts[0, n2] = 0.0

    total_num = float(np.nansum(num * counts))
    total_den = float(np.nansum(den * counts))
    return total_num / total_den if total_den > 0 else float("nan")


def stats_from_sfs(sfs, L: float) -> Dict[str, float]:
    m1 = sfs.marginalize([1])  # pop dim-0 marginal
    m2 = sfs.marginalize([0])  # pop dim-1 marginal
    return {
        "pi_pop1": pi_from_1d_sfs(m1, L),
        "pi_pop2": pi_from_1d_sfs(m2, L),
        "tajima_d_pop1": tajima_d_from_1d_sfs(m1),
        "tajima_d_pop2": tajima_d_from_1d_sfs(m2),
        "fst": fst_from_2d_sfs(sfs),
    }


def _strip_corners(sfs):
    """Zero out fixed/absent corners so simulated SFS matches the observed SFS."""
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


def _norm_sfs(s: np.ndarray) -> np.ndarray:
    total = s[1:-1].sum()
    return s / total if total > 0 else s


# ---------------------------------------------------------------------------
# Load calibration_simulate replicate SFS
# ---------------------------------------------------------------------------

_REP_RE = re.compile(r"^replicate_(\d+)$")


def load_calibration_sfs(calibration_dir: Path, target_sizes: List[int]) -> List:
    """Load replicate_*/SFS.pkl under calibration_dir, projected to target_sizes."""
    reps = []
    for d in calibration_dir.iterdir():
        m = _REP_RE.match(d.name)
        if d.is_dir() and m:
            reps.append((int(m.group(1)), d))
    reps.sort(key=lambda t: t[0])

    sfs_list = []
    for _, d in reps:
        p = d / "SFS.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                sfs = pickle.load(fh)
            sfs_list.append(_strip_corners(sfs.project(target_sizes)))
    print(f"Loaded and projected to {target_sizes}: {len(sfs_list)}/{len(reps)} calibration replicate SFS files")
    if not sfs_list:
        raise SystemExit(f"No replicate_*/SFS.pkl found under {calibration_dir}")
    return sfs_list


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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
    title: str,
    out_path: Path,
) -> None:
    p1, p2 = pop_labels

    fig = plt.figure(figsize=(22, 15))
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.50, wspace=0.38)

    # ---- Row 0: scalar violin plots ----------------------------------------
    scalar_keys = ["pi_pop1", "pi_pop2", "tajima_d_pop1", "tajima_d_pop2", "fst"]
    scalar_labels = [f"π ({p1})", f"π ({p2})", f"Tajima's D ({p1})", f"Tajima's D ({p2})", r"$F_{ST}$"]

    for col, (key, label) in enumerate(zip(scalar_keys, scalar_labels)):
        ax = fig.add_subplot(gs[0, col])
        vals = [s[key] for s in sim_stats if np.isfinite(s.get(key, np.nan))]
        ax.violinplot(vals, positions=[0], showmedians=True)
        ax.scatter([0], [obs_stats[key]], color="red", zorder=5, s=45, label="observed")
        ax.set_xticks([])
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)

    # ---- Row 1: 1D SFS comparison ------------------------------------------
    for col, (sim_mat, obs_sfs, pop) in enumerate([
        (sim_sfs_1, obs_sfs_1, p1),
        (sim_sfs_2, obs_sfs_2, p2),
    ]):
        ax = fig.add_subplot(gs[1, col * 2: col * 2 + 2])
        norm_sim = np.array([_norm_sfs(s) for s in sim_mat])
        lo = np.percentile(norm_sim, 5, axis=0)
        hi = np.percentile(norm_sim, 95, axis=0)
        mn = np.mean(norm_sim, axis=0)

        bins = np.arange(len(mn))
        ax.fill_between(bins[1:-1], lo[1:-1], hi[1:-1], alpha=0.3, color="steelblue", label="5–95% CI (sim)")
        ax.plot(bins[1:-1], mn[1:-1], color="steelblue", linewidth=1.5, label="mean sim")
        ax.plot(bins[1:-1], _norm_sfs(obs_sfs)[1:-1], color="red", linewidth=1.5, label="observed")

        ax.set_xlabel("Derived allele count", fontsize=10)
        ax.set_ylabel("Proportion of sites", fontsize=10)
        ax.set_title(f"1D SFS (unfolded) — {pop}", fontsize=11)
        ax.legend(fontsize=8)

    # ---- Row 2: 2D SFS heatmaps --------------------------------------------
    mean_2d = np.mean(sim_sfs_2d, axis=0)

    ax_sim = fig.add_subplot(gs[2, :2])
    im1 = ax_sim.imshow(np.log1p(mean_2d.T), origin="lower", aspect="auto", cmap="viridis")
    ax_sim.set_title("2D SFS — mean simulated (log1p)", fontsize=11)
    ax_sim.set_xlabel(f"{p1} derived allele count", fontsize=10)
    ax_sim.set_ylabel(f"{p2} derived allele count", fontsize=10)
    plt.colorbar(im1, ax=ax_sim)

    ax_obs = fig.add_subplot(gs[2, 2:4])
    im2 = ax_obs.imshow(np.log1p(obs_sfs_2d.T), origin="lower", aspect="auto", cmap="viridis")
    ax_obs.set_title("2D SFS — observed (log1p)", fontsize=11)
    ax_obs.set_xlabel(f"{p1} derived allele count", fontsize=10)
    ax_obs.set_ylabel(f"{p2} derived allele count", fontsize=10)
    plt.colorbar(im2, ax=ax_obs)

    ax_diff = fig.add_subplot(gs[2, 4])
    obs_norm = obs_sfs_2d / obs_sfs_2d.sum() if obs_sfs_2d.sum() > 0 else obs_sfs_2d
    sim_norm = mean_2d / mean_2d.sum() if mean_2d.sum() > 0 else mean_2d
    diff = obs_norm - sim_norm
    vmax = np.abs(diff).max()
    im3 = ax_diff.imshow(diff.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_diff.set_title("2D SFS diff\n(obs − sim)", fontsize=10)
    ax_diff.set_xlabel(f"{p1} count", fontsize=9)
    ax_diff.set_ylabel(f"{p2} count", fontsize=9)
    plt.colorbar(im3, ax=ax_diff)

    fig.suptitle(f"Model calibration PPC — {title}", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                     help="Experiment config JSON (demographic_model, num_samples, sequence_length).")
    ap.add_argument("--calibration-dir", required=True, type=Path,
                     help="Directory containing replicate_*/SFS.pkl (calibration_simulate output).")
    ap.add_argument("--combined-sfs", required=True, type=Path,
                     help="Real observed COMBINED_SFS pickle (moments.Spectrum, summed over AUTOSOMES).")
    ap.add_argument("--combined-sfs-meta", type=Path, default=None,
                     help="COMBINED_SFS meta JSON (has 'sequence_length'); falls back to "
                          "config['sequence_length'] if omitted.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--title", default=None,
                     help="Plot title / label; defaults to '<demographic_model> (<variant>/<model_key>)'.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = json.loads(args.config.read_text())
    pop_labels = list(cfg["num_samples"].keys())

    if args.combined_sfs_meta is not None and args.combined_sfs_meta.exists():
        L = float(json.loads(args.combined_sfs_meta.read_text())["sequence_length"])
    else:
        L = float(cfg["sequence_length"])

    print(f"Loading observed SFS <- {args.combined_sfs}  (L={L:,.0f})")
    with open(args.combined_sfs, "rb") as fh:
        obs_sfs = pickle.load(fh)
    obs_sfs = _strip_corners(obs_sfs)
    n1, n2 = obs_sfs.shape[0] - 1, obs_sfs.shape[1] - 1
    print(f"Observed SFS shape: {obs_sfs.shape}  (target_sizes=[{n1}, {n2}])")

    obs_stats = stats_from_sfs(obs_sfs, L)
    obs_sfs_1 = _fill(obs_sfs.marginalize([1]))
    obs_sfs_2 = _fill(obs_sfs.marginalize([0]))
    obs_sfs_2d = _fill(obs_sfs)
    print(f"  pi_{pop_labels[0]}={obs_stats['pi_pop1']:.4g}  pi_{pop_labels[1]}={obs_stats['pi_pop2']:.4g}  "
          f"Taj_D_{pop_labels[0]}={obs_stats['tajima_d_pop1']:.4g}  "
          f"Taj_D_{pop_labels[1]}={obs_stats['tajima_d_pop2']:.4g}  FST={obs_stats['fst']:.4g}")

    print(f"\nLoading calibration replicates <- {args.calibration_dir}")
    all_sfs = load_calibration_sfs(args.calibration_dir, [n1, n2])

    print("\nComputing stats from calibration replicates...")
    sim_stats = [stats_from_sfs(s, L) for s in all_sfs]

    sim_sfs_1 = np.array([_fill(s.marginalize([1])) for s in all_sfs])
    sim_sfs_2 = np.array([_fill(s.marginalize([0])) for s in all_sfs])
    sim_sfs_2d = np.array([_fill(s) for s in all_sfs])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or f"{cfg['demographic_model']} ({args.calibration_dir.parent.name}/{args.calibration_dir.name})"

    print("\nPlotting...")
    plot_results(
        sim_stats, obs_stats,
        sim_sfs_1, sim_sfs_2, sim_sfs_2d,
        obs_sfs_1, obs_sfs_2, obs_sfs_2d,
        pop_labels=pop_labels,
        title=title,
        out_path=args.out_dir / "calibration_ppc.png",
    )

    print("\nSummary (calibration-simulated vs observed):")
    p1, p2 = pop_labels
    summary = {}
    for key, label in [
        ("pi_pop1", f"pi_{p1}"),
        ("pi_pop2", f"pi_{p2}"),
        ("tajima_d_pop1", f"tajima_d_{p1}"),
        ("tajima_d_pop2", f"tajima_d_{p2}"),
        ("fst", "fst"),
    ]:
        vals = [s[key] for s in sim_stats if np.isfinite(s.get(key, np.nan))]
        lo, med, hi = np.percentile(vals, [5, 50, 95])
        obs = obs_stats[key]
        pct = float(np.mean(np.array(vals) <= obs)) * 100
        print(f"  {label:20s}: obs={obs:.4g}  sim=[{lo:.4g}, {med:.4g}, {hi:.4g}]  percentile={pct:.1f}%")
        summary[label] = {
            "observed": obs, "sim_p5": lo, "sim_median": med, "sim_p95": hi,
            "observed_percentile_in_sim": pct,
        }

    (args.out_dir / "calibration_ppc_summary.json").write_text(json.dumps({
        "title": title,
        "n_replicates": len(all_sfs),
        "target_sizes": [n1, n2],
        "sequence_length": L,
        "stats": summary,
    }, indent=2))
    print(f"Saved {args.out_dir / 'calibration_ppc_summary.json'}")


if __name__ == "__main__":
    main()

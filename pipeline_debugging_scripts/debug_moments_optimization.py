#!/usr/bin/env python3
"""
debug_moments_no_anc_only.py

Single-formulation (NO_ANC) end-to-end moments debug.

What it does (per sim):
1) Sample "true" demographic params (log-uniform) within bounds.
2) Build a NO_ANC demes graph where popA carries the ancestral epoch and popB branches at T.
3) Simulate a tree sequence (msprime OR stdpopsim msprime engine).
4) Build observed 2D SFS from the simulated ts (popA,popB).
5) Fit params with nlopt(L-BFGS) in log10 space (Poisson composite LL).
6) Compute expected SFS under TRUE params and under FITTED params.
7) Save diagnostics: expected(true) vs expected(fit) fit stats + scatter plot + graphs.
8) Aggregate across sims:
   - parameter scatterplots (true vs fitted) for each param
   - expected-SFS true-vs-fit error across sims

Notes:
- This script is intentionally "standalone-ish" and does NOT require your pipeline SFS module.
- For stdpopsim backend, the *population names must match* the model populations.
  Use --popA/--popB to match what your define_sps_model(...) produces (e.g. YRI/CEU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import msprime
import demes
import demesdraw
import moments
import nlopt
import numdifftools as nd
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Infer_Demography/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



# Canonical order for THIS script (6 params)
PARAM_NAMES = ["N_ANC", "N_A", "N_B", "T", "mAB", "mBA"]


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def pop_id(ts, name: str) -> int:
    """
    Find a population ID by its demes-population name in ts metadata.
    Works for msprime and stdpopsim(msprime engine) outputs.
    """
    for pid in range(ts.num_populations):
        md = ts.population(pid).metadata
        if isinstance(md, (bytes, bytearray)):
            md = json.loads(md.decode())
        if isinstance(md, dict) and md.get("name") == name:
            return pid
    raise KeyError(f"Population {name!r} not found in ts metadata")


def sample_params_loguniform(rng: np.random.Generator, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Sample each parameter log-uniformly between lb and ub."""
    return 10 ** rng.uniform(np.log10(lb), np.log10(ub))


def save_demes_graph_and_params(sim_dir: Path, g: demes.Graph, pars: np.ndarray, filename_prefix: str) -> None:
    """
    Save demes graph YAML + PNG + params JSON into sim_dir.
    filename_prefix is "true" or "fit".
    """
    sim_dir.mkdir(parents=True, exist_ok=True)

    (sim_dir / f"demes_graph_{filename_prefix}.yaml").write_text(demes.dumps(g), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    demesdraw.tubes(g, ax=ax)
    fig.tight_layout()
    fig.savefig(sim_dir / f"demes_graph_{filename_prefix}.png", dpi=200)
    plt.close(fig)

    (sim_dir / f"{filename_prefix}_params.json").write_text(
        json.dumps({k: float(v) for k, v in zip(PARAM_NAMES, pars)}, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# NO_ANC formulation: return (demes_graph, msprime_demography)
# =============================================================================

def build_no_anc(
    N_ANC: float, N_A: float, N_B: float, T: float, mAB: float, mBA: float,
    *, popA: str, popB: str
) -> Tuple[demes.Graph, msprime.Demography]:
    """
    NO_ANC demes graph:
      popA carries ancestral epoch pre-split:
        - older than T: N_ANC
        - T->0: N_A
      popB splits from popA at T: N_B from T->0
      asymmetric migration after split
    """
    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme(
        popA,
        epochs=[
            dict(start_size=float(N_ANC), end_time=float(T)),
            dict(start_size=float(N_A), end_time=0),
        ],
    )
    b.add_deme(
        popB,
        ancestors=[popA],
        start_time=float(T),
        epochs=[dict(start_size=float(N_B), end_time=0)],
    )

    if float(mAB) > 0:
        b.add_migration(source=popA, dest=popB, rate=float(mAB), start_time=float(T), end_time=0)
    if float(mBA) > 0:
        b.add_migration(source=popB, dest=popA, rate=float(mBA), start_time=float(T), end_time=0)

    g = b.resolve()
    demogr = msprime.Demography.from_demes(g)
    return g, demogr


# =============================================================================
# Simulation backend switch
# =============================================================================

def simulate_ts(
    *,
    sim_backend: str,
    g: demes.Graph,
    demogr: msprime.Demography,
    seqlen: float,
    mu: float,
    recomb: float,
    nA: int,
    nB: int,
    seed: int,
    popA: str,
    popB: str,
    experiment_config: Optional[Dict[str, Any]] = None,
):
    """
    Return a mutated tree sequence.

    - msprime backend: uses demogr + seqlen/recomb + then adds mutations at rate mu.
    - stdpopsim backend: uses define_sps_model(g) + a contig (fallback if cfg not provided).

    IMPORTANT:
      For stdpopsim, the samples dict keys MUST match the populations in the
      stdpopsim demographic model. We use popA/popB for that.
    """
    if sim_backend == "msprime":
        ts = msprime.sim_ancestry(
            samples={popA: int(nA), popB: int(nB)},
            sequence_length=float(seqlen),
            recombination_rate=float(recomb),
            demography=demogr,
            random_seed=int(seed),
        )
        ts = msprime.sim_mutations(ts, rate=float(mu), random_seed=int(seed) + 1)
        return ts

    if sim_backend == "stdpopsim":
        import stdpopsim as sps

        try:
            from src.stdpopsim_wrappers import define_sps_model  # type: ignore
        except Exception as e:
            raise ImportError(
                "stdpopsim backend requires src.stdpopsim_wrappers.define_sps_model(g). "
                "Either provide that module or switch --backend to 'msprime'."
            ) from e

        model = define_sps_model(g)
        samples = {popA: int(nA), popB: int(nB)}  # <-- CRITICAL FIX vs your failing run

        contig = None
        if experiment_config is not None:
            try:
                from src.bgs_intervals import _contig_from_cfg  # type: ignore
                sel = experiment_config.get("selection") or {}
                contig = _contig_from_cfg(experiment_config, sel)
            except Exception:
                contig = None

        if contig is None:
            sp = sps.get_species("HomSap")
            try:
                contig = sp.get_contig(
                    chromosome=None,
                    length=float(seqlen),
                    mutation_rate=float(mu),
                    recombination_rate=float(recomb),
                )
            except TypeError:
                contig = sp.get_contig(
                    chromosome=None,
                    length=float(seqlen),
                    mutation_rate=float(mu),
                )

        eng = sps.get_engine("msprime")
        ts = eng.simulate(model, contig, samples, seed=int(seed))
        return ts

    raise ValueError(f"Unknown sim_backend={sim_backend!r}. Use 'msprime' or 'stdpopsim'.")


# =============================================================================
# Expected SFS helpers
# =============================================================================

def expected_sfs_from_graph(
    g: demes.Graph,
    *,
    sample_size,
    theta: float,
    popA: str,
    popB: str,
) -> moments.Spectrum:
    """Expected SFS from demes graph."""
    return moments.Spectrum.from_demes(
        g,
        sampled_demes=[popA, popB],
        sample_sizes=sample_size,
        theta=float(theta),
    )


def sfs_fit_stats(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> Dict[str, float]:
    """
    Compare two SFS arrays a vs b. Returns scalar diagnostics.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    diff = a - b
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    relerr = np.abs(diff) / denom

    ax = a.ravel()
    bx = b.ravel()
    m = (ax > 0) & (bx > 0)
    if np.any(m):
        corr_log = float(np.corrcoef(np.log10(ax[m]), np.log10(bx[m]))[0, 1])
    else:
        corr_log = float("nan")

    return {
        "sum_a": float(a.sum()),
        "sum_b": float(b.sum()),
        "max_relerr": float(relerr.max()),
        "mean_relerr": float(relerr.mean()),
        "median_relerr": float(np.median(relerr)),
        "p95_relerr": float(np.quantile(relerr, 0.95)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr_log10": corr_log,
    }


def plot_sfs_scatter(x: np.ndarray, y: np.ndarray, out_png: Path, title: str, xlabel: str, ylabel: str):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, "o", markersize=3, alpha=0.5)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =============================================================================
# Moments likelihood + optimization
# =============================================================================

def expected_sfs_for_opt(log10_params, sample_size, mutation_rate, popA: str, popB: str):
    """
    Expected SFS used inside optimization.
    """
    N_ANC, N_A, N_B, T, mAB, mBA = 10 ** log10_params
    g, _ = build_no_anc(N_ANC, N_A, N_B, T, mAB, mBA, popA=popA, popB=popB)
    return moments.Spectrum.from_demes(
        g,
        sampled_demes=[popA, popB],
        sample_sizes=sample_size,
        theta=4 * float(N_ANC) * float(mutation_rate),
    )


def optimize_lbfgs(
    start_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    observed_sfs: moments.Spectrum,
    mutation_rate: float,
    popA: str,
    popB: str,
    verbose: bool = False,
    rtol: float = 1e-8,
) -> np.ndarray:
    assert isinstance(observed_sfs, moments.Spectrum)
    sample_size = [n - 1 for n in observed_sfs.shape]

    def loglikelihood(log10_params):
        exp_sfs = expected_sfs_for_opt(log10_params, sample_size, mutation_rate, popA=popA, popB=popB)
        # Poisson composite LL (no masking here; keep consistent with your working debug script)
        return float(np.sum(np.log(exp_sfs) * observed_sfs - exp_sfs))

    grad_fun = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params, grad):
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fun(log10_params)
        if verbose:
            print(f"loglik: {ll}, params: {10 ** log10_params}")
        return ll

    opt = nlopt.opt(nlopt.LD_LBFGS, start_values.size)
    opt.set_lower_bounds(np.log10(lower_bounds))
    opt.set_upper_bounds(np.log10(upper_bounds))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    fitted_log10 = opt.optimize(np.log10(start_values))
    return 10 ** fitted_log10


# =============================================================================
# One sim
# =============================================================================

def simulate_and_fit_one(
    *,
    sim_backend: str,
    true_pars: np.ndarray,
    seqlen: float,
    mu: float,
    recomb: float,
    nA: int,
    nB: int,
    lb: np.ndarray,
    ub: np.ndarray,
    st: np.ndarray,
    seed: int,
    sim_dir: Path,
    popA: str,
    popB: str,
    experiment_config: Optional[Dict[str, Any]] = None,
    verbose_opt: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
      fitted_params, expected_true_array, expected_fit_array, fit_summary_dict

    FIX: expected SFS grids are defined from the OBSERVED SFS shape (haploid counts),
         so exp_true/exp_fit always match obs_sfs for both msprime and stdpopsim backends.
    """
    # TRUE graph + save
    g_true, demogr_true = build_no_anc(*true_pars, popA=popA, popB=popB)
    save_demes_graph_and_params(sim_dir, g_true, true_pars, filename_prefix="true")

    # Simulate TS
    ts = simulate_ts(
        sim_backend=sim_backend,
        g=g_true,
        demogr=demogr_true,
        seqlen=seqlen,
        mu=mu,
        recomb=recomb,
        nA=nA,
        nB=nB,
        seed=seed,
        popA=popA,
        popB=popB,
        experiment_config=experiment_config,
    )

    # Observed SFS from TS
    A_id = pop_id(ts, popA)
    B_id = pop_id(ts, popB)
    obs_sfs = moments.Spectrum(
        ts.allele_frequency_spectrum(
            sample_sets=[list(ts.samples(A_id)), list(ts.samples(B_id))],
            mode="site",
            polarised=True,
            span_normalise=False,
        )
    )
    np.save(sim_dir / "observed_sfs.npy", np.asarray(obs_sfs, float))

    # -------------------------------------------------------------------------
    # FIX: define sample_size from observed SFS grid (haploid chromosome counts)
    # -------------------------------------------------------------------------
    sample_size = [n - 1 for n in obs_sfs.shape]

    # expected SFS under TRUE params (on matching grid)
    theta_true = 4 * float(true_pars[0]) * (float(mu) * float(seqlen))
    exp_true = np.asarray(
        expected_sfs_from_graph(g_true, sample_size=sample_size, theta=theta_true, popA=popA, popB=popB),
        dtype=float,
    )
    np.save(sim_dir / "expected_sfs_true.npy", exp_true)

    # Sanity: shapes must match
    assert exp_true.shape == obs_sfs.shape, (
        f"Shape mismatch: exp_true {exp_true.shape} vs obs_sfs {obs_sfs.shape}. "
        f"sample_size={sample_size}, nA={nA}, nB={nB}, "
        f"len(ts.samples(A))={len(ts.samples(A_id))}, len(ts.samples(B))={len(ts.samples(B_id))}"
    )

    # Fit
    fitted = optimize_lbfgs(
        st, lb, ub, obs_sfs, float(mu) * float(seqlen),
        popA=popA, popB=popB,
        verbose=verbose_opt
    )
    (sim_dir / "fit_params.json").write_text(
        json.dumps({k: float(v) for k, v in zip(PARAM_NAMES, fitted)}, indent=2),
        encoding="utf-8",
    )

    # FIT graph + save
    g_fit, _ = build_no_anc(*fitted, popA=popA, popB=popB)
    save_demes_graph_and_params(sim_dir, g_fit, fitted, filename_prefix="fit")

    # expected SFS under FITTED params (same matching grid)
    theta_fit = 4 * float(fitted[0]) * (float(mu) * float(seqlen))
    exp_fit = np.asarray(
        expected_sfs_from_graph(g_fit, sample_size=sample_size, theta=theta_fit, popA=popA, popB=popB),
        dtype=float,
    )
    np.save(sim_dir / "expected_sfs_fit.npy", exp_fit)

    # Sanity: shapes must match
    assert exp_fit.shape == obs_sfs.shape, (
        f"Shape mismatch: exp_fit {exp_fit.shape} vs obs_sfs {obs_sfs.shape}. "
        f"sample_size={sample_size}"
    )

    # TRUE vs FIT expected-SFS comparison
    fit_summary = sfs_fit_stats(exp_true, exp_fit)
    (sim_dir / "expected_sfs_true_vs_fit_summary.json").write_text(
        json.dumps(fit_summary, indent=2),
        encoding="utf-8",
    )
    plot_sfs_scatter(
        exp_true, exp_fit,
        sim_dir / "expected_sfs_true_vs_fit_scatter.png",
        title="Expected SFS: TRUE vs FIT",
        xlabel="Expected (TRUE params)",
        ylabel="Expected (FIT params)",
    )

    # Observed vs expected (true/fit) scatters (now safe: same shape)
    plot_sfs_scatter(
        np.asarray(exp_true, float), np.asarray(obs_sfs, float),
        sim_dir / "obs_vs_expected_true_scatter.png",
        title="Observed vs Expected (TRUE)",
        xlabel="Expected (TRUE params)",
        ylabel="Observed",
    )
    plot_sfs_scatter(
        np.asarray(exp_fit, float), np.asarray(obs_sfs, float),
        sim_dir / "obs_vs_expected_fit_scatter.png",
        title="Observed vs Expected (FIT)",
        xlabel="Expected (FIT params)",
        ylabel="Observed",
    )

    return fitted, exp_true, exp_fit, fit_summary


# =============================================================================
# Plotting
# =============================================================================

def scatter_true_vs_fit(true_mat: np.ndarray, fit_mat: np.ndarray, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(PARAM_NAMES):
        x = true_mat[:, j]
        y = fit_mat[:, j]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(x, y, "o", markersize=3)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Moments inferred {name}")
        ax.set_title(f"{name}: inferred vs true ({prefix})")

        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_scatter_true_vs_fit_{name}.png", dpi=200)
        plt.close(fig)


def plot_sfs_error_across_sims(max_relerrs: np.ndarray, mean_relerrs: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(max_relerrs))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, max_relerrs, "o-", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Simulation index")
    ax.set_ylabel("max relative error")
    ax.set_title("Expected SFS TRUE vs FIT: max relerr per sim")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_sfs_true_vs_fit_max_relerr_across_sims.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean_relerrs, "o-", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Simulation index")
    ax.set_ylabel("mean relative error")
    ax.set_title("Expected SFS TRUE vs FIT: mean relerr per sim")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_sfs_true_vs_fit_mean_relerr_across_sims.png", dpi=200)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Moments debug for NO_ANC graph only.")
    p.add_argument("--backend", choices=["msprime", "stdpopsim"], default="msprime")
    p.add_argument("--num-sims", type=int, default=25)
    p.add_argument("--out-root", type=str, default="debug_outputs_no_anc")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--seqlen", type=float, default=1e6)
    p.add_argument("--mu", type=float, default=1e-8)
    p.add_argument("--recomb", type=float, default=1e-8)
    p.add_argument("--nA", type=int, default=5)
    p.add_argument("--nB", type=int, default=5)

    # IMPORTANT: for stdpopsim backend, these should match model population names
    p.add_argument("--popA", type=str, default="A")
    p.add_argument("--popB", type=str, default="B")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    SIM_BACKEND = args.backend
    num_sims = int(args.num_sims)

    out_root = Path(args.out_root)
    plots = ensure_dir(out_root / "plots")
    sims_dir = ensure_dir(out_root / "sims")

    popA = args.popA
    popB = args.popB

    # bounds + start (same style as your working debug script)
    lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8, 1e-8], dtype=float)
    ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3, 1e-3], dtype=float)

    st = (lb + ub) / 2

    seqlen = float(args.seqlen)
    mu = float(args.mu)
    recomb = float(args.recomb)
    nA = int(args.nA)
    nB = int(args.nB)

    rng = np.random.default_rng(int(args.seed))

    # Optional: pass a config dict if you want stdpopsim contig to be built via your pipeline.
    experiment_config: Optional[Dict[str, Any]] = None

    true_mat = np.zeros((num_sims, 6), dtype=float)
    fit_mat = np.zeros((num_sims, 6), dtype=float)

    max_relerr = np.zeros(num_sims, dtype=float)
    mean_relerr = np.zeros(num_sims, dtype=float)
    corr_log10 = np.zeros(num_sims, dtype=float)

    for i in range(num_sims):
        theta = sample_params_loguniform(rng, lb, ub)  # "true" params
        true_mat[i, :] = theta

        seed_base = int(rng.integers(1, 2**31 - 1000))
        seed = seed_base + 1

        sim_dir = ensure_dir(sims_dir / f"sim_{i:04d}")

        print(
            f"[{i+1:>3}/{num_sims}] backend={SIM_BACKEND} popA={popA} popB={popB} "
            f"seqlen={seqlen} mu={mu} recomb={recomb} nA={nA} nB={nB} seed={seed}"
        )

        fitted, exp_true, exp_fit, s = simulate_and_fit_one(
            sim_backend=SIM_BACKEND,
            true_pars=theta,
            seqlen=seqlen,
            mu=mu,
            recomb=recomb,
            nA=nA,
            nB=nB,
            lb=lb,
            ub=ub,
            st=st,
            seed=seed,
            sim_dir=sim_dir,
            popA=popA,
            popB=popB,
            experiment_config=experiment_config,
            verbose_opt=False,
        )

        fit_mat[i, :] = fitted
        max_relerr[i] = s["max_relerr"]
        mean_relerr[i] = s["mean_relerr"]
        corr_log10[i] = s["corr_log10"]

        print(f"    true: {theta}")
        print(f"    fit : {fitted}")
        print(f"    expected-SFS TRUE vs FIT: max_relerr={max_relerr[i]:.3e} mean_relerr={mean_relerr[i]:.3e} corr_log10={corr_log10[i]:.3f}")

    np.save(out_root / "true_params.npy", true_mat)
    np.save(out_root / "fit_params.npy", fit_mat)

    scatter_true_vs_fit(true_mat, fit_mat, plots, prefix=f"NO_ANC_{SIM_BACKEND}_{popA}_{popB}")
    plot_sfs_error_across_sims(max_relerr, mean_relerr, plots)

    (out_root / "expected_sfs_true_vs_fit_across_sims.json").write_text(
        json.dumps(
            {
                "popA": popA,
                "popB": popB,
                "backend": SIM_BACKEND,
                "max_relerr_per_sim": [float(x) for x in max_relerr],
                "mean_relerr_per_sim": [float(x) for x in mean_relerr],
                "corr_log10_per_sim": [float(x) for x in corr_log10],
                "max_of_max_relerr": float(np.max(max_relerr)),
                "mean_of_mean_relerr": float(np.mean(mean_relerr)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Backend:   {SIM_BACKEND}")
    print(f"popA/popB: {popA}/{popB}")
    print(f"Out root:  {out_root}")
    print(f"Plots:     {plots}")


if __name__ == "__main__":
    main()

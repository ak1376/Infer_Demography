#!/usr/bin/env python3
"""
debug_moments_compare_two_formulations_backend.py

Paired sims comparing two demes formulations:

(1) WITH explicit ancestral deme "ANC"
(2) NO ancestral deme (A carries ancestral epoch)

Simulation backends:
- SIM_BACKEND="msprime"   : msprime.sim_ancestry + msprime.sim_mutations
- SIM_BACKEND="stdpopsim" : stdpopsim.get_engine("msprime").simulate(model, contig, samples)

Adds expected-SFS equivalence checks:
- For each sim (same theta for both formulations), compute
    moments expected SFS from demes graph for each formulation
  and save:
    - expected_sfs_with_anc.npy / expected_sfs_no_anc.npy
    - expected_sfs_diff.npy, expected_sfs_relerr.npy
    - expected_sfs_compare.png (scatter, log-log)
    - expected_sfs_relerr.png (heatmap of log10 rel err)
    - expected_sfs_summary.json

Also aggregates across sims:
- max rel err per sim plot
- mean rel err per sim plot
- pooled scatter across all sims and all entries

NOTE: SLiM is intentionally ignored.
"""

from __future__ import annotations

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


PARAM_NAMES = ["N_ANC", "N_A", "N_B", "T", "mAB", "mBA"]


# =============================================================================
# Utilities
# =============================================================================

def pop_id(ts, name: str) -> int:
    for pid in range(ts.num_populations):
        md = ts.population(pid).metadata
        if isinstance(md, (bytes, bytearray)):
            md = json.loads(md.decode())
        if isinstance(md, dict) and md.get("name") == name:
            return pid
    raise KeyError(f"Population {name!r} not found in ts metadata")


def sample_params_loguniform(rng: np.random.Generator, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return 10 ** rng.uniform(np.log10(lb), np.log10(ub))


def save_demes_graph_and_params(sim_dir: Path, g: demes.Graph, true_pars: np.ndarray) -> None:
    sim_dir.mkdir(parents=True, exist_ok=True)

    (sim_dir / "demes_graph.yaml").write_text(demes.dumps(g), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    demesdraw.tubes(g, ax=ax)
    fig.tight_layout()
    fig.savefig(sim_dir / "demes_graph.png", dpi=200)
    plt.close(fig)

    (sim_dir / "true_params.json").write_text(
        json.dumps({k: float(v) for k, v in zip(PARAM_NAMES, true_pars)}, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Two formulations: return (demes_graph, msprime_demography)
# =============================================================================

def build_with_anc(N_ANC, N_A, N_B, T, mAB, mBA) -> Tuple[demes.Graph, msprime.Demography]:
    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme("ANC", epochs=[dict(start_size=float(N_ANC), end_time=float(T))])
    b.add_deme("A", ancestors=["ANC"], epochs=[dict(start_size=float(N_A), end_time=0)])
    b.add_deme("B", ancestors=["ANC"], epochs=[dict(start_size=float(N_B), end_time=0)])

    if mAB > 0:
        b.add_migration(source="A", dest="B", rate=float(mAB), start_time=T, end_time=0)
    if mBA > 0:
        b.add_migration(source="B", dest="A", rate=float(mBA), start_time=T, end_time=0)

    g = b.resolve()
    demogr = msprime.Demography.from_demes(g)
    return g, demogr


def build_no_anc(N_ANC, N_A, N_B, T, mAB, mBA) -> Tuple[demes.Graph, msprime.Demography]:
    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme(
        "A",
        epochs=[
            dict(start_size=float(N_ANC), end_time=float(T)),
            dict(start_size=float(N_A),   end_time=0),
        ],
    )
    b.add_deme(
        "B",
        ancestors=["A"],
        start_time=float(T),
        epochs=[dict(start_size=float(N_B), end_time=0)],
    )

    if mAB > 0:
        b.add_migration(source="A", dest="B", rate=float(mAB), start_time=T, end_time=0)
    if mBA > 0:
        b.add_migration(source="B", dest="A", rate=float(mBA), start_time=T, end_time=0)

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
    experiment_config: Optional[Dict[str, Any]] = None,
):
    """
    Return a mutated tree sequence.

    - msprime backend: uses demogr + seqlen/recomb + then adds mutations at rate mu.
    - stdpopsim backend: uses define_sps_model(g) + a contig.
      If experiment_config and _contig_from_cfg exist, uses them.
      Else uses sp.get_contig(...) fallback.
    """
    if sim_backend == "msprime":
        ts = msprime.sim_ancestry(
            samples={"A": nA, "B": nB},
            sequence_length=seqlen,
            recombination_rate=recomb,
            demography=demogr,
            random_seed=seed,
        )
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1)
        return ts

    if sim_backend == "stdpopsim":
        import stdpopsim as sps

        try:
            from src.stdpopsim_wrappers import define_sps_model  # type: ignore
        except Exception as e:
            raise ImportError(
                "stdpopsim backend requires src.stdpopsim_wrappers.define_sps_model(g). "
                "Either provide that module or switch SIM_BACKEND to 'msprime'."
            ) from e

        model = define_sps_model(g)
        samples = {"A": int(nA), "B": int(nB)}

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

def expected_sfs_from_graph(g: demes.Graph, sample_size, theta: float) -> moments.Spectrum:
    """
    Expected SFS from demes graph, directly.
    """
    return moments.Spectrum.from_demes(
        g,
        sampled_demes=["A", "B"],
        sample_sizes=sample_size,
        theta=theta,
    )


def compare_expected_sfs_and_save(
    *,
    exp_with: np.ndarray,
    exp_no: np.ndarray,
    out_dir: Path,
    eps: float = 1e-30,
) -> Dict[str, float]:
    """
    Save per-sim expected-SFS comparison artifacts into out_dir.
    Returns summary stats.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_with = np.asarray(exp_with, dtype=float)
    exp_no = np.asarray(exp_no, dtype=float)

    diff = exp_with - exp_no
    denom = np.maximum(np.maximum(np.abs(exp_with), np.abs(exp_no)), eps)
    relerr = np.abs(diff) / denom  # elementwise relative error

    np.save(out_dir / "expected_sfs_with_anc.npy", exp_with)
    np.save(out_dir / "expected_sfs_no_anc.npy", exp_no)
    np.save(out_dir / "expected_sfs_diff.npy", diff)
    np.save(out_dir / "expected_sfs_relerr.npy", relerr)

    # Scatter plot (log-log)
    x = exp_with.flatten()
    y = exp_no.flatten()
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, "o", markersize=3)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expected SFS (WITH_ANC)")
    ax.set_ylabel("Expected SFS (NO_ANC)")
    ax.set_title("Expected SFS: WITH_ANC vs NO_ANC")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_sfs_compare.png", dpi=200)
    plt.close(fig)

    # Heatmap of log10 relerr
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(np.log10(relerr + eps), aspect="auto")
    ax.set_title("log10(relative error) of expected SFS")
    ax.set_xlabel("B freq bin")
    ax.set_ylabel("A freq bin")
    fig.colorbar(img, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "expected_sfs_relerr.png", dpi=200)
    plt.close(fig)

    summary = {
        "max_relerr": float(np.max(relerr)),
        "mean_relerr": float(np.mean(relerr)),
        "median_relerr": float(np.median(relerr)),
        "p95_relerr": float(np.quantile(relerr, 0.95)),
    }
    (out_dir / "expected_sfs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# =============================================================================
# Moments likelihood + optimization
# =============================================================================

def expected_sfs(log10_params, sample_size, mutation_rate, builder_fn):
    """
    builder_fn returns (demes_graph, msprime_demography)
    We compute expectation directly from demes graph.
    """
    N_ANC, N_A, N_B, T, mAB, mBA = 10 ** log10_params
    g, _ = builder_fn(N_ANC, N_A, N_B, T, mAB, mBA)
    return moments.Spectrum.from_demes(
        g,
        sampled_demes=["A", "B"],
        sample_sizes=sample_size,
        theta=4 * N_ANC * mutation_rate,
    )


def optimize_lbfgs(
    start_values,
    lower_bounds,
    upper_bounds,
    observed_sfs,
    mutation_rate,
    builder_fn,
    verbose=False,
    rtol=1e-8,
):
    assert isinstance(observed_sfs, moments.Spectrum)
    sample_size = [n - 1 for n in observed_sfs.shape]

    def loglikelihood(log10_params):
        exp_sfs = expected_sfs(log10_params, sample_size, mutation_rate, builder_fn)
        return np.sum(np.log(exp_sfs) * observed_sfs - exp_sfs)

    def gradient(log10_params):
        return nd.Gradient(loglikelihood, n=1, step=1e-4)(log10_params)

    def objective(log10_params, grad):
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = gradient(log10_params)
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
# One sim for one formulation
# =============================================================================

def simulate_and_fit_one(
    *,
    sim_backend: str,
    builder_fn,
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
    experiment_config: Optional[Dict[str, Any]] = None,
    verbose_opt: bool = False,
) -> Tuple[np.ndarray, demes.Graph, np.ndarray]:
    """
    Returns:
      fitted_params, demes_graph, expected_sfs_array
    """
    g, demogr = builder_fn(*true_pars)

    # save graph + params
    save_demes_graph_and_params(sim_dir, g, true_pars)

    # expected SFS for this formulation (for model equivalence checks)
    sample_size = [nA - 1, nB - 1]
    theta = 4 * float(true_pars[0]) * (mu * seqlen)
    exp = np.asarray(expected_sfs_from_graph(g, sample_size=sample_size, theta=theta), dtype=float)
    np.save(sim_dir / "expected_sfs.npy", exp)

    ts = simulate_ts(
        sim_backend=sim_backend,
        g=g,
        demogr=demogr,
        seqlen=seqlen,
        mu=mu,
        recomb=recomb,
        nA=nA,
        nB=nB,
        seed=seed,
        experiment_config=experiment_config,
    )

    A = pop_id(ts, "A")
    B = pop_id(ts, "B")
    obs_sfs = moments.Spectrum(
        ts.allele_frequency_spectrum(
            sample_sets=[list(ts.samples(A)), list(ts.samples(B))],
            mode="site",
            polarised=True,
            span_normalise=False,
        )
    )

    fitted = optimize_lbfgs(st, lb, ub, obs_sfs, mu * seqlen, builder_fn, verbose=verbose_opt)

    (sim_dir / "fitted_params.json").write_text(
        json.dumps({k: float(v) for k, v in zip(PARAM_NAMES, fitted)}, indent=2),
        encoding="utf-8",
    )

    return fitted, g, exp


# =============================================================================
# Plotting
# =============================================================================

def scatter_true_vs_fit(true_mat, fit_mat, out_dir: Path, prefix: str):
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


def scatter_fit_vs_fit(fit_with_anc, fit_no_anc, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(PARAM_NAMES):
        x = fit_with_anc[:, j]
        y = fit_no_anc[:, j]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(x, y, "o", markersize=3)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Moments inferred {name} (WITH_ANC)")
        ax.set_ylabel(f"Moments inferred {name} (NO_ANC)")
        ax.set_title(f"{name}: WITH_ANC vs NO_ANC")

        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_fit_vs_fit_{name}.png", dpi=200)
        plt.close(fig)


def plot_expected_sfs_error_across_sims(max_relerrs, mean_relerrs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(max_relerrs))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, max_relerrs, "o-", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Simulation index")
    ax.set_ylabel("max relative error")
    ax.set_title("Expected SFS mismatch: max relerr per sim")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_sfs_max_relerr_across_sims.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean_relerrs, "o-", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Simulation index")
    ax.set_ylabel("mean relative error")
    ax.set_title("Expected SFS mismatch: mean relerr per sim")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_sfs_mean_relerr_across_sims.png", dpi=200)
    plt.close(fig)


def plot_expected_sfs_pooled_scatter(pooled_with, pooled_no, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_with = np.asarray(pooled_with, dtype=float)
    pooled_no = np.asarray(pooled_no, dtype=float)
    mask = (pooled_with > 0) & (pooled_no > 0)
    x = pooled_with[mask]
    y = pooled_no[mask]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, "o", markersize=2, alpha=0.35)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expected SFS (WITH_ANC)")
    ax.set_ylabel("Expected SFS (NO_ANC)")
    ax.set_title("Expected SFS pooled across sims")
    fig.tight_layout()
    fig.savefig(out_dir / "expected_sfs_expected_compare_across_sims.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    # ---------------- user controls ----------------
    SIM_BACKEND = "msprime"     # <-- set to "stdpopsim" to use stdpopsim engine
    num_sims = 25

    out_root = Path("debug_outputs_compare_formulations")
    out_with = out_root / "with_anc"
    out_no   = out_root / "no_anc"
    plots    = out_root / "plots"

    seqlen = 1e6
    mu = 1e-8
    recomb = 1e-8
    nA = 5
    nB = 5

    lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8, 1e-8], dtype=float)
    ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3, 1e-3], dtype=float)
    st = (lb + ub) / 2

    rng = np.random.default_rng(123)

    # Optional: if you want stdpopsim to use your pipeline contig builder,
    # pass a config dict (otherwise fallback contig is used).
    experiment_config: Optional[Dict[str, Any]] = None
    # ------------------------------------------------

    (out_with / "sims").mkdir(parents=True, exist_ok=True)
    (out_no / "sims").mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    true_mat = np.zeros((num_sims, 6), dtype=float)
    fit_with = np.zeros((num_sims, 6), dtype=float)
    fit_no   = np.zeros((num_sims, 6), dtype=float)

    # expected-SFS comparison aggregates
    max_relerrs = np.zeros(num_sims, dtype=float)
    mean_relerrs = np.zeros(num_sims, dtype=float)
    pooled_with = []
    pooled_no = []

    for i in range(num_sims):
        theta = sample_params_loguniform(rng, lb, ub)
        true_mat[i, :] = theta

        seed_base = int(rng.integers(1, 2**31 - 100))
        seed_with = seed_base + 1
        seed_no   = seed_base + 10

        sim_dir_with = out_with / "sims" / f"sim_{i:04d}"
        sim_dir_no   = out_no   / "sims" / f"sim_{i:04d}"

        fit_with[i, :], g_with, exp_with = simulate_and_fit_one(
            sim_backend=SIM_BACKEND,
            builder_fn=build_with_anc,
            true_pars=theta,
            seqlen=seqlen,
            mu=mu,
            recomb=recomb,
            nA=nA,
            nB=nB,
            lb=lb,
            ub=ub,
            st=st,
            seed=seed_with,
            sim_dir=sim_dir_with,
            experiment_config=experiment_config,
            verbose_opt=False,
        )

        fit_no[i, :], g_no, exp_no = simulate_and_fit_one(
            sim_backend=SIM_BACKEND,
            builder_fn=build_no_anc,
            true_pars=theta,
            seqlen=seqlen,
            mu=mu,
            recomb=recomb,
            nA=nA,
            nB=nB,
            lb=lb,
            ub=ub,
            st=st,
            seed=seed_no,
            sim_dir=sim_dir_no,
            experiment_config=experiment_config,
            verbose_opt=False,
        )

        # ---- expected SFS comparison (same theta) ----
        # Save comparison artifacts into BOTH dirs for convenience
        summary = compare_expected_sfs_and_save(
            exp_with=exp_with,
            exp_no=exp_no,
            out_dir=sim_dir_with,
        )
        compare_expected_sfs_and_save(
            exp_with=exp_with,
            exp_no=exp_no,
            out_dir=sim_dir_no,
        )

        max_relerrs[i] = summary["max_relerr"]
        mean_relerrs[i] = summary["mean_relerr"]
        pooled_with.append(np.asarray(exp_with).ravel())
        pooled_no.append(np.asarray(exp_no).ravel())

        print(f"[{i+1:>3}/{num_sims}] theta={theta}")
        print(f"    WITH_ANC fit: {fit_with[i,:]}")
        print(f"    NO_ANC   fit: {fit_no[i,:]}")
        print(f"    expected-SFS max_relerr={max_relerrs[i]:.3e}, mean_relerr={mean_relerrs[i]:.3e}")

    np.save(out_root / "true_params.npy", true_mat)
    np.save(out_root / "fit_with_anc.npy", fit_with)
    np.save(out_root / "fit_no_anc.npy", fit_no)

    scatter_true_vs_fit(true_mat, fit_with, plots, prefix=f"WITH_ANC_{SIM_BACKEND}")
    scatter_true_vs_fit(true_mat, fit_no,   plots, prefix=f"NO_ANC_{SIM_BACKEND}")
    scatter_fit_vs_fit(fit_with, fit_no, plots)

    # Aggregate expected-SFS mismatch plots
    plot_expected_sfs_error_across_sims(max_relerrs, mean_relerrs, plots)

    pooled_with = np.concatenate(pooled_with) if len(pooled_with) else np.array([])
    pooled_no = np.concatenate(pooled_no) if len(pooled_no) else np.array([])
    plot_expected_sfs_pooled_scatter(pooled_with, pooled_no, plots)

    (out_root / "expected_sfs_mismatch_across_sims.json").write_text(
        json.dumps(
            {
                "max_relerr_per_sim": [float(x) for x in max_relerrs],
                "mean_relerr_per_sim": [float(x) for x in mean_relerrs],
                "max_of_max_relerr": float(np.max(max_relerrs)),
                "mean_of_mean_relerr": float(np.mean(mean_relerrs)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Backend:       {SIM_BACKEND}")
    print(f"WITH_ANC sims: {out_with / 'sims'}")
    print(f"NO_ANC sims:   {out_no / 'sims'}")
    print(f"Plots:         {plots}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
debug_dadi.py

Demes-built IM-symmetric (A carries ancestral epoch; B branches at T) with
msprime simulation + moments expected SFS + nlopt(L-BFGS) optimization +
1D likelihood surface plots.

Population naming:
  A = YRI
  B = CEU
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import msprime
import demes
import demesdraw
import moments
import nlopt
import numdifftools as nd
import matplotlib.pyplot as plt


# =============================================================================
# Demography (demes -> msprime)
# =============================================================================

def build_demes_graph(N_anc: float, N_A: float, N_B: float, T: float, m: float) -> demes.Graph:
    """
    Demes graph equivalent to your IM_symmetric_model structure, but with names A/B:

      - A exists from present back to infinity
      - A has size N_A for time in [0, T)
      - A has size N_anc for time in [T, inf)
      - B splits from A at time T and exists for [0, T)
      - symmetric migration between A and B during [0, T)

    Times in generations ago.
    """
    assert T > 0, "T must be > 0"
    assert N_anc > 0 and N_A > 0 and N_B > 0, "population sizes must be > 0"
    assert m >= 0, "migration rate must be >= 0"

    b = demes.Builder(time_units="generations", generation_time=1)

    # Root extant deme A
    b.add_deme(
        "A",
        epochs=[
            dict(start_size=float(N_anc), end_time=float(T)),  # older epoch
            dict(start_size=float(N_A), end_time=0),           # recent epoch
        ],
    )

    # B branches at T
    b.add_deme(
        "B",
        ancestors=["A"],
        start_time=float(T),
        epochs=[dict(start_size=float(N_B), end_time=0)],
    )

    # Symmetric migration only when both exist: [0, T]
    if m > 0:
        b.add_migration(source="A", dest="B", rate=float(m), start_time=float(T), end_time=0)
        b.add_migration(source="B", dest="A", rate=float(m), start_time=float(T), end_time=0)

    return b.resolve()


def save_demes_graph(g: demes.Graph, prefix: str, out_dir: str | Path = ".") -> Tuple[Path, Path]:
    """
    Save the demes graph as YAML + PNG.
    Returns (yaml_path, png_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / prefix

    yaml_path = prefix_path.with_suffix(".demes.yaml")
    png_path = prefix_path.with_suffix(".png")

    # YAML
    # g.dump(yaml_path)

    # PNG
    fig, ax = plt.subplots(figsize=(7, 4.5))
    demesdraw.tubes(g, ax=ax)
    ax.set_xlabel("Time (generations ago)")
    ax.set_ylabel("Effective population size")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return yaml_path, png_path


def msprime_model(N_anc: float, N_A: float, N_B: float, T: float, m: float,
                 *, save_graph: bool = False, graph_prefix: str = "IM_symmetric_AB",
                 graph_out_dir: str | Path = ".") -> msprime.Demography:
    """
    Build demography via demes, then convert to msprime.

    Uses population names "A" and "B" so the rest of your code stays unchanged:
      - samples={"A":5,"B":5}
      - sampled_demes=["A","B"]
    """
    g = build_demes_graph(N_anc, N_A, N_B, T, m)

    if save_graph:
        yaml_path, png_path = save_demes_graph(g, prefix=graph_prefix, out_dir=graph_out_dir)
        print(f"[saved] {yaml_path}")
        print(f"[saved] {png_path}")

    return msprime.Demography.from_demes(g)


# =============================================================================
# moments expected SFS + likelihood
# =============================================================================

def expected_sfs(log10_params: np.ndarray, sample_size: Sequence[int], mutation_rate: float) -> moments.Spectrum:
    """
    log10_params corresponds to:
      [N_anc, N_A, N_B, T, m] in log10 space.
    mutation_rate should be per-base mu * sequence_length (theta uses 4*N_anc*muL).
    """
    N_anc, N_A, N_B, T, m = 10 ** log10_params
    demogr = msprime_model(N_anc, N_A, N_B, T, m, save_graph=False)
    return moments.Spectrum.from_demes(
        demogr.to_demes(),
        sampled_demes=["A", "B"],
        sample_sizes=list(sample_size),
        theta=4 * N_anc * mutation_rate,
    )


def optimize_lbfgs(
    start_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    observed_sfs: moments.Spectrum,
    mutation_rate: float,  # mu * sequence_length
    verbose: bool = False,
    rtol: float = 1e-8,
):
    """
    Maximize Poisson composite log-likelihood:
      sum_{i} [ obs_i * log(exp_i) - exp_i ]
    in log10 parameter space.
    """
    assert isinstance(observed_sfs, moments.Spectrum)
    sample_size = [n - 1 for n in observed_sfs.shape]

    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = expected_sfs(log10_params, sample_size, mutation_rate)
        # moments Spectrum supports elementwise ops; cast to np arrays for safety
        exp = np.asarray(exp_sfs, dtype=float)
        obs = np.asarray(observed_sfs, dtype=float)
        # avoid log(0)
        exp = np.maximum(exp, 1e-300)
        return float(np.sum(np.log(exp) * obs - exp))

    grad_fn = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"loglik: {ll:.6f}  log10_params: {log10_params}")
        return ll

    opt = nlopt.opt(nlopt.LD_LBFGS, start_values.size)
    opt.set_lower_bounds(np.log10(lower_bounds))
    opt.set_upper_bounds(np.log10(upper_bounds))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    fitted_log10 = opt.optimize(np.log10(start_values))
    fitted = 10 ** fitted_log10
    fitted_sfs = expected_sfs(fitted_log10, sample_size, mutation_rate)
    code = opt.last_optimize_result()

    return fitted, fitted_sfs, code


def parameter_grid_loglik(
    param_values: np.ndarray,
    what_to_vary: int,
    grid_of_values: np.ndarray,
    observed_sfs: moments.Spectrum,
    mutation_rate: float,
):
    assert isinstance(observed_sfs, moments.Spectrum)
    sample_size = [n - 1 for n in observed_sfs.shape]

    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = expected_sfs(log10_params, sample_size, mutation_rate)
        exp = np.asarray(exp_sfs, dtype=float)
        obs = np.asarray(observed_sfs, dtype=float)
        exp = np.maximum(exp, 1e-300)
        return float(np.sum(np.log(exp) * obs - exp))

    ll_surface = []
    base = np.log10(param_values).astype(float)

    for p in grid_of_values:
        x = base.copy()
        x[what_to_vary] = np.log10(p)
        ll_surface.append(loglikelihood(x))

    return ll_surface


# =============================================================================
# Diagnostics / plots
# =============================================================================

def sanity_check(num_reps: int = 100, out_dir: str | Path = ".") -> None:
    """
    Compare:
      - msprime Monte Carlo expected SFS (branch mode, averaged)
      - moments expected SFS from demes graph

    Also saves the demes graph used in this sanity check.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqlen = 1e6
    mu = 1e-8
    params = np.array([1e4, 1e3, 2e4, 1e4, 1e-6], dtype=float)  # [N_anc, N_A, N_B, T, m]

    demogr = msprime_model(*params, save_graph=True, graph_prefix="sanity_IM_symmetric_AB", graph_out_dir=out_dir)

    tsg = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        sequence_length=seqlen,
        recombination_rate=1e-8,
        demography=demogr,
        random_seed=1,
        num_replicates=num_reps,
    )

    mc_sfs = None
    for ts in tsg:
        tmp = ts.allele_frequency_spectrum(
            sample_sets=[list(ts.samples(population=p)) for p in [0, 1]],
            mode="branch",
            span_normalise=False,
            polarised=True,
        ) * mu
        mc_sfs = tmp if mc_sfs is None else (mc_sfs + tmp)

    mc_sfs /= num_reps

    exp_sfs = expected_sfs(np.log10(params), [n - 1 for n in mc_sfs.shape], mu * seqlen)

    fig_path = out_dir / "sanity-check.png"
    plt.figure()
    plt.plot(exp_sfs, mc_sfs, "o", color="black", markersize=4)
    mx = float(np.asarray(exp_sfs).mean())
    plt.axline((mx, mx), (mx + 1, mx + 1), linestyle="dashed", color="red")
    plt.xlabel("moments expected SFS")
    plt.ylabel("msprime Monte Carlo SFS (branch)")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fig_path}")


def plot_1d(fitted_pars: np.ndarray, obs_sfs: moments.Spectrum, muL: float,
            param_idx: int, grid: np.ndarray, xlabel: str, true_val: float,
            out_path: str | Path) -> None:
    ll = parameter_grid_loglik(fitted_pars, param_idx, grid, obs_sfs, muL)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(grid, ll, "-o", markersize=3)
    plt.axvline(true_val)
    plt.xscale("log")
    plt.ylabel("Loglikelihood")
    plt.xlabel(xlabel)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    OUTDIR = Path("debug_outputs")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) Sanity check + save demes graph
    sanity_check(num_reps=100, out_dir=OUTDIR)

    # 2) Simulate data
    true_pars = np.array([1e4, 1e3, 2e4, 2e4, 1e-6], dtype=float)  # [N_anc, N_A, N_B, T, m]
    seqlen = 5e7
    mu = 1e-8
    muL = mu * seqlen

    # Save the "true" graph too
    _ = msprime_model(*true_pars, save_graph=True, graph_prefix="true_IM_symmetric_AB", graph_out_dir=OUTDIR)

    ts = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        sequence_length=seqlen,
        recombination_rate=1e-8,
        demography=msprime_model(*true_pars, save_graph=False),
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=2)

    obs_sfs = moments.Spectrum(
        ts.allele_frequency_spectrum(
            sample_sets=[list(ts.samples(population=p)) for p in [0, 1]],
            mode="site",
            span_normalise=False,
            polarised=True,
        )
    )

    # 3) Optimize
    lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8], dtype=float)
    ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3], dtype=float)

    # Start at the geometric mean of bounds
    st = np.sqrt(lb * ub)

    fitted_pars, _, code = optimize_lbfgs(st, lb, ub, obs_sfs, muL, verbose=True)
    print(f"\nopt_code: {code}")
    print(f"fitted_pars: {fitted_pars}")
    print(f"true_pars:   {true_pars}\n")

    # Save fitted graph
    _ = msprime_model(*fitted_pars, save_graph=True, graph_prefix="fitted_IM_symmetric_AB", graph_out_dir=OUTDIR)

    # 4) 1D likelihood surfaces
    grid_N = np.logspace(2, 5, 51)
    grid_T = np.logspace(2, 5, 51)
    grid_m = np.logspace(-8, -3, 51)

    plot_1d(fitted_pars, obs_sfs, muL, 0, grid_N, "Ancestral size N_anc", true_pars[0], OUTDIR / "loglik_surface_Nanc.png")
    plot_1d(fitted_pars, obs_sfs, muL, 1, grid_N, "Size N_A",           true_pars[1], OUTDIR / "loglik_surface_NA.png")
    plot_1d(fitted_pars, obs_sfs, muL, 2, grid_N, "Size N_B",           true_pars[2], OUTDIR / "loglik_surface_NB.png")
    plot_1d(fitted_pars, obs_sfs, muL, 3, grid_T, "Split time T",       true_pars[3], OUTDIR / "loglik_surface_T.png")
    plot_1d(fitted_pars, obs_sfs, muL, 4, grid_m, "Migration rate m",   true_pars[4], OUTDIR / "loglik_surface_m.png")

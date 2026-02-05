#!/usr/bin/env python3
"""
debug_moments_multiple_sims.py

- Demes model with NO separate ancestral-only deme (A carries ancestral epoch)
- Multiple simulations with randomly sampled parameters
- Moments optimization for each simulation
- Save true demes graph + params per simulation
- Scatterplots of moments-inferred vs true parameters across simulations
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import msprime
import demes
import demesdraw
import moments
import nlopt
import numdifftools as nd
import matplotlib.pyplot as plt


# =============================================================================
# Utilities
# =============================================================================

PARAM_NAMES = ["N_ANC", "N_A", "N_B", "T", "mAB", "mBA"]


def pop_id(ts, name: str) -> int:
    """Return population ID by name (robust to ordering)."""
    for pid in range(ts.num_populations):
        md = ts.population(pid).metadata
        if isinstance(md, (bytes, bytearray)):
            md = json.loads(md.decode())
        if isinstance(md, dict) and md.get("name") == name:
            return pid
    raise KeyError(f"Population {name!r} not found")


def sample_params_loguniform(rng, lb, ub):
    """Log-uniform sampling between bounds."""
    return 10 ** rng.uniform(np.log10(lb), np.log10(ub))


# =============================================================================
# Demography: no separate ancestral deme
# =============================================================================

def msprime_model(N_ANC, N_A, N_B, T, mAB, mBA):
    """
    Demes graph (no separate ANC deme) -> msprime demography

    A:
      - ancestral epoch (older than T): size N_ANC
      - recent epoch (T -> 0): size N_A
    B:
      - splits from A at time T
      - size N_B from T -> 0
    Asymmetric migration after split.
    """
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

    dg = b.resolve()
    return msprime.Demography.from_demes(dg)


# =============================================================================
# Moments likelihood + optimization
# =============================================================================

def expected_sfs(log10_params, sample_size, mutation_rate):
    N_ANC, N_A, N_B, T, mAB, mBA = 10 ** log10_params
    demogr = msprime_model(N_ANC, N_A, N_B, T, mAB, mBA)
    return moments.Spectrum.from_demes(
        demogr.to_demes(),
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
    verbose=False,
    rtol=1e-8,
):
    sample_size = [n - 1 for n in observed_sfs.shape]

    def loglikelihood(log10_params):
        exp_sfs = expected_sfs(log10_params, sample_size, mutation_rate)
        return np.sum(np.log(exp_sfs) * observed_sfs - exp_sfs)

    def gradient(log10_params):
        return nd.Gradient(loglikelihood, step=1e-4)(log10_params)

    def objective(log10_params, grad):
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = gradient(log10_params)
        if verbose:
            print("loglik:", ll, "params:", 10 ** log10_params)
        return ll

    opt = nlopt.opt(nlopt.LD_LBFGS, len(start_values))
    opt.set_lower_bounds(np.log10(lower_bounds))
    opt.set_upper_bounds(np.log10(upper_bounds))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    fitted_log10 = opt.optimize(np.log10(start_values))
    return 10 ** fitted_log10


# =============================================================================
# Per-simulation outputs
# =============================================================================

def save_true_demes(sim_dir: Path, true_pars: np.ndarray):
    sim_dir.mkdir(parents=True, exist_ok=True)

    demogr = msprime_model(*true_pars)
    dg = demogr.to_demes()

    (sim_dir / "true_params.json").write_text(
        json.dumps(dict(zip(PARAM_NAMES, map(float, true_pars))), indent=2)
    )
    (sim_dir / "demes_graph.yaml").write_text(demes.dumps(dg))

    fig, ax = plt.subplots(figsize=(7, 4))
    demesdraw.tubes(dg, ax=ax)
    fig.tight_layout()
    fig.savefig(sim_dir / "demes_graph.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Main multi-sim driver
# =============================================================================

def main():
    # ---------------- user controls ----------------
    num_sims = 25
    out_root = Path("debug_outputs_moments")
    seqlen = 1e7
    mu = 1e-8
    recomb = 1e-8
    nA = 5
    nB = 5
    rng = np.random.default_rng(123)

    lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8, 1e-8])
    ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3, 1e-3])
    st = (lb + ub) / 2
    # ------------------------------------------------

    sims_dir = out_root / "sims"
    plots_dir = out_root / "plots"
    sims_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    true_mat = np.zeros((num_sims, 6))
    fit_mat  = np.zeros((num_sims, 6))

    for i in range(num_sims):
        true_pars = sample_params_loguniform(rng, lb, ub)
        true_mat[i] = true_pars

        sim_dir = sims_dir / f"sim_{i:04d}"
        save_true_demes(sim_dir, true_pars)

        ts = msprime.sim_ancestry(
            samples={"A": nA, "B": nB},
            sequence_length=seqlen,
            recombination_rate=recomb,
            demography=msprime_model(*true_pars),
            random_seed=int(rng.integers(1, 2**31)),
        )
        ts = msprime.sim_mutations(ts, rate=mu)

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

        fitted = optimize_lbfgs(st, lb, ub, obs_sfs, mu * seqlen)
        fit_mat[i] = fitted

        (sim_dir / "fitted_params.json").write_text(
            json.dumps(dict(zip(PARAM_NAMES, map(float, fitted))), indent=2)
        )

        print(f"[{i+1:>3}/{num_sims}] true={true_pars}  fitted={fitted}")

    # =============================================================================
    # Scatterplots across simulations
    # =============================================================================

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
        ax.set_title(f"{name}: inferred vs true")

        fig.tight_layout()
        fig.savefig(plots_dir / f"scatter_{name}.png", dpi=200)
        plt.close(fig)

    np.save(out_root / "true_params.npy", true_mat)
    np.save(out_root / "fitted_params.npy", fit_mat)

    print("\nDone.")
    print(f"Per-sim outputs: {sims_dir}")
    print(f"Scatterplots:    {plots_dir}")


if __name__ == "__main__":
    main()

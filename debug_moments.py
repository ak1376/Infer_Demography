#!/usr/bin/env python3
"""
debug_moments_compare_two_formulations.py

Paired sims comparing two demes formulations:

(1) WITH explicit ancestral deme "ANC":
    ANC -> split at T -> A, B ; migration A<->B after split

(2) NO ancestral deme (A carries ancestral epoch):
    A has two epochs: N_ANC (older than T), N_A (T->0)
    B splits from A at T ; migration A<->B after split

For each simulation i:
  - sample parameters theta_i (differs across sims)
  - run both formulations with SAME theta_i
  - simulate, infer with moments, save per-sim outputs in separate dirs

Also makes scatterplots:
  - inferred vs true (separately for each formulation)
  - inferred(with_ANC) vs inferred(no_ANC) across sims for each parameter
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
# Globals
# =============================================================================

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


def save_demes_graph_and_params(sim_dir: Path, demogr: msprime.Demography, true_pars: np.ndarray) -> None:
    sim_dir.mkdir(parents=True, exist_ok=True)

    dg = demogr.to_demes()
    (sim_dir / "demes_graph.yaml").write_text(demes.dumps(dg), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    demesdraw.tubes(dg, ax=ax)
    fig.tight_layout()
    fig.savefig(sim_dir / "demes_graph.png", dpi=200)
    plt.close(fig)

    (sim_dir / "true_params.json").write_text(
        json.dumps({k: float(v) for k, v in zip(PARAM_NAMES, true_pars)}, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Two formulations
# =============================================================================

def msprime_model_with_anc(N_ANC, N_A, N_B, T, mAB, mBA) -> msprime.Demography:
    """
    Explicit ancestral deme:
      ANC (size N_ANC) until time T
      splits into A (N_A) and B (N_B) at T
      asymmetric migration between A and B after split
    """
    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme("ANC", epochs=[dict(start_size=float(N_ANC), end_time=float(T))])
    b.add_deme("A", ancestors=["ANC"], epochs=[dict(start_size=float(N_A), end_time=0)])
    b.add_deme("B", ancestors=["ANC"], epochs=[dict(start_size=float(N_B), end_time=0)])

    if mAB > 0:
        b.add_migration(source="A", dest="B", rate=float(mAB), start_time=T, end_time=0)
    if mBA > 0:
        b.add_migration(source="B", dest="A", rate=float(mBA), start_time=T, end_time=0)

    dg = b.resolve()
    return msprime.Demography.from_demes(dg)


def msprime_model_no_anc(N_ANC, N_A, N_B, T, mAB, mBA) -> msprime.Demography:
    """
    No ancestral deme:
      A carries ancestral epoch pre-split:
        - older than T: N_ANC
        - T->0: N_A
      B splits from A at T: N_B from T->0
      asymmetric migration after split
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
# Moments likelihood + optimization (unchanged logic)
# =============================================================================

def expected_sfs(log10_params, sample_size, mutation_rate, model_fn):
    N_ANC, N_A, N_B, T, mAB, mBA = 10 ** log10_params
    demogr = model_fn(N_ANC, N_A, N_B, T, mAB, mBA)
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
    model_fn,
    verbose=False,
    rtol=1e-8,
):
    assert isinstance(observed_sfs, moments.Spectrum)
    sample_size = [n - 1 for n in observed_sfs.shape]

    def loglikelihood(log10_params):
        exp_sfs = expected_sfs(log10_params, sample_size, mutation_rate, model_fn)
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
    demogr: msprime.Demography,
    model_fn,
    true_pars: np.ndarray,
    seqlen: float,
    mu: float,
    recomb: float,
    nA: int,
    nB: int,
    lb: np.ndarray,
    ub: np.ndarray,
    st: np.ndarray,
    seed_ancestry: int,
    seed_mut: int,
    sim_dir: Path,
    verbose_opt: bool = False,
) -> np.ndarray:
    # save graph + params
    save_demes_graph_and_params(sim_dir, demogr, true_pars)

    ts = msprime.sim_ancestry(
        samples={"A": nA, "B": nB},
        sequence_length=seqlen,
        recombination_rate=recomb,
        demography=demogr,
        random_seed=seed_ancestry,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed_mut)

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

    fitted = optimize_lbfgs(st, lb, ub, obs_sfs, mu * seqlen, model_fn, verbose=verbose_opt)

    (sim_dir / "fitted_params.json").write_text(
        json.dumps({k: float(v) for k, v in zip(PARAM_NAMES, fitted)}, indent=2),
        encoding="utf-8",
    )

    return fitted


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


# =============================================================================
# Main
# =============================================================================

def main():
    # ---------------- user controls ----------------
    num_sims = 25

    out_root = Path("debug_outputs_compare_formulations")
    out_with = out_root / "with_anc"
    out_no   = out_root / "no_anc"
    plots    = out_root / "plots"

    seqlen = 5e7
    mu = 1e-8
    recomb = 1e-8
    nA = 5
    nB = 5

    lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8, 1e-8], dtype=float)
    ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3, 1e-3], dtype=float)
    st = (lb + ub) / 2

    rng = np.random.default_rng(123)
    # ------------------------------------------------

    (out_with / "sims").mkdir(parents=True, exist_ok=True)
    (out_no / "sims").mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    true_mat = np.zeros((num_sims, 6), dtype=float)
    fit_with = np.zeros((num_sims, 6), dtype=float)
    fit_no   = np.zeros((num_sims, 6), dtype=float)

    for i in range(num_sims):
        # paired parameters
        theta = sample_params_loguniform(rng, lb, ub)
        true_mat[i, :] = theta

        # deterministic paired seeds (so comparisons are fair-ish)
        # (still different between sims)
        seed_base = int(rng.integers(1, 2**31 - 10))
        seed_with_anc_anc = seed_base + 1
        seed_with_anc_mut = seed_base + 2
        seed_no_anc_anc   = seed_base + 3
        seed_no_anc_mut   = seed_base + 4

        # build demographies
        dem_with = msprime_model_with_anc(*theta)
        dem_no   = msprime_model_no_anc(*theta)

        sim_dir_with = out_with / "sims" / f"sim_{i:04d}"
        sim_dir_no   = out_no   / "sims" / f"sim_{i:04d}"

        fit_with[i, :] = simulate_and_fit_one(
            demogr=dem_with,
            model_fn=msprime_model_with_anc,
            true_pars=theta,
            seqlen=seqlen,
            mu=mu,
            recomb=recomb,
            nA=nA,
            nB=nB,
            lb=lb,
            ub=ub,
            st=st,
            seed_ancestry=seed_with_anc_anc,
            seed_mut=seed_with_anc_mut,
            sim_dir=sim_dir_with,
            verbose_opt=False,
        )

        fit_no[i, :] = simulate_and_fit_one(
            demogr=dem_no,
            model_fn=msprime_model_no_anc,
            true_pars=theta,
            seqlen=seqlen,
            mu=mu,
            recomb=recomb,
            nA=nA,
            nB=nB,
            lb=lb,
            ub=ub,
            st=st,
            seed_ancestry=seed_no_anc_anc,
            seed_mut=seed_no_anc_mut,
            sim_dir=sim_dir_no,
            verbose_opt=False,
        )

        print(f"[{i+1:>3}/{num_sims}] theta={theta}")
        print(f"    WITH_ANC fit: {fit_with[i,:]}")
        print(f"    NO_ANC   fit: {fit_no[i,:]}")

    # save matrices
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "true_params.npy", true_mat)
    np.save(out_root / "fit_with_anc.npy", fit_with)
    np.save(out_root / "fit_no_anc.npy", fit_no)

    # scatterplots: true vs fit for each formulation
    scatter_true_vs_fit(true_mat, fit_with, plots, prefix="WITH_ANC")
    scatter_true_vs_fit(true_mat, fit_no,   plots, prefix="NO_ANC")

    # scatterplots: fit vs fit across formulations
    scatter_fit_vs_fit(fit_with, fit_no, plots)

    # small summary json
    rows = []
    for i in range(num_sims):
        row = {"sim": i}
        row.update({f"true_{k}": float(true_mat[i, j]) for j, k in enumerate(PARAM_NAMES)})
        row.update({f"with_anc_{k}": float(fit_with[i, j]) for j, k in enumerate(PARAM_NAMES)})
        row.update({f"no_anc_{k}": float(fit_no[i, j]) for j, k in enumerate(PARAM_NAMES)})
        rows.append(row)
    (out_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"WITH_ANC sims: {out_with / 'sims'}")
    print(f"NO_ANC sims:   {out_no / 'sims'}")
    print(f"Plots:         {plots}")


if __name__ == "__main__":
    main()

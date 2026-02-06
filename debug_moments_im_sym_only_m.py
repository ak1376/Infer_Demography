#!/usr/bin/env python3
"""
debug_moments_im_sym_only_m.py

Sanity check: sample ONLY migration rate m, keep all other parameters fixed,
and fit ONLY m using moments (IM symmetric).

NEW: Optional 1D likelihood scan over m (log-spaced grid) for a subset of sims:
- saves m_grid.npy, ll_grid.npy, ll_vs_m.png in each sim directory

Formulations:
(1) WITH explicit ancestral deme "ANC"
(2) NO ancestral deme (YRI carries ancestral epoch)

Backends:
- msprime
- stdpopsim (optional; requires your define_sps_model)

Outputs:
- per-sim directories with demes graph + true/fitted m
- plots: true_m vs fitted_m (and formulation vs formulation if both)
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


POP1 = "YRI"
POP2 = "CEU"


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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sample_loguniform(rng: np.random.Generator, lb: float, ub: float) -> float:
    return float(10 ** rng.uniform(np.log10(lb), np.log10(ub)))


def save_demes_graph(sim_dir: Path, g: demes.Graph) -> None:
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / "demes_graph.yaml").write_text(demes.dumps(g), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    demesdraw.tubes(g, ax=ax)
    fig.tight_layout()
    fig.savefig(sim_dir / "demes_graph.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Two formulations (same as your IM symmetric)
# =============================================================================

def build_with_anc(N_anc, N_YRI, N_CEU, T_split, m) -> Tuple[demes.Graph, msprime.Demography]:
    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme("ANC", epochs=[dict(start_size=float(N_anc), end_time=float(T_split))])
    b.add_deme(POP1, ancestors=["ANC"], epochs=[dict(start_size=float(N_YRI), end_time=0)])
    b.add_deme(POP2, ancestors=["ANC"], epochs=[dict(start_size=float(N_CEU), end_time=0)])

    if float(m) > 0:
        b.add_migration(source=POP1, dest=POP2, rate=float(m), start_time=float(T_split), end_time=0)
        b.add_migration(source=POP2, dest=POP1, rate=float(m), start_time=float(T_split), end_time=0)

    g = b.resolve()
    demogr = msprime.Demography.from_demes(g)
    return g, demogr


def build_no_anc(N_anc, N_YRI, N_CEU, T_split, m) -> Tuple[demes.Graph, msprime.Demography]:
    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme(
        POP1,
        epochs=[
            dict(start_size=float(N_anc), end_time=float(T_split)),
            dict(start_size=float(N_YRI), end_time=0),
        ],
    )
    b.add_deme(
        POP2,
        ancestors=[POP1],
        start_time=float(T_split),
        epochs=[dict(start_size=float(N_CEU), end_time=0)],
    )

    if float(m) > 0:
        b.add_migration(source=POP1, dest=POP2, rate=float(m), start_time=float(T_split), end_time=0)
        b.add_migration(source=POP2, dest=POP1, rate=float(m), start_time=float(T_split), end_time=0)

    g = b.resolve()
    demogr = msprime.Demography.from_demes(g)
    return g, demogr


FORMULATION_BUILDERS = {"with_anc": build_with_anc, "no_anc": build_no_anc}


# =============================================================================
# Simulation
# =============================================================================

def simulate_ts(
    *,
    sim_backend: str,
    g: demes.Graph,
    demogr: msprime.Demography,
    seqlen: float,
    mu: float,
    recomb: float,
    nYRI: int,
    nCEU: int,
    seed: int,
    experiment_config: Optional[Dict[str, Any]] = None,
):
    if sim_backend == "msprime":
        ts = msprime.sim_ancestry(
            samples={POP1: nYRI, POP2: nCEU},
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
        samples = {POP1: int(nYRI), POP2: int(nCEU)}

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

    raise ValueError(f"Unknown sim_backend={sim_backend!r}.")


# =============================================================================
# Moments likelihood for ONLY m
# =============================================================================

def expected_sfs_only_m(
    log10_m: np.ndarray,
    *,
    sample_size,
    mut_rate_times_L: float,
    builder_fn,
    N_anc: float,
    N_YRI: float,
    N_CEU: float,
    T_split: float,
) -> moments.Spectrum:
    m = float(10 ** float(log10_m[0]))
    g, _ = builder_fn(N_anc, N_YRI, N_CEU, T_split, m)
    return moments.Spectrum.from_demes(
        g,
        sampled_demes=[POP1, POP2],
        sample_sizes=sample_size,
        theta=4 * float(N_anc) * float(mut_rate_times_L),
    )


def loglik_m_only(
    *,
    m: float,
    observed_sfs: moments.Spectrum,
    mut_rate_times_L: float,
    builder_fn,
    N_anc: float,
    N_YRI: float,
    N_CEU: float,
    T_split: float,
    eps: float = 1e-300,
) -> float:
    """
    The same Poisson composite log-likelihood used by optimize_m_only:
      sum_{bins} [ obs * log(exp) - exp ]
    """
    sample_size = [n - 1 for n in observed_sfs.shape]
    exp_sfs = expected_sfs_only_m(
        np.array([np.log10(m)], dtype=float),
        sample_size=sample_size,
        mut_rate_times_L=mut_rate_times_L,
        builder_fn=builder_fn,
        N_anc=N_anc,
        N_YRI=N_YRI,
        N_CEU=N_CEU,
        T_split=T_split,
    )
    exp = np.asarray(exp_sfs, dtype=float)
    obs = np.asarray(observed_sfs, dtype=float)
    exp = np.maximum(exp, eps)  # avoid log(0)
    return float(np.sum(np.log(exp) * obs - exp))


def save_ll_curve_over_m(
    *,
    sim_dir: Path,
    observed_sfs: moments.Spectrum,
    builder_fn,
    N_anc: float,
    N_YRI: float,
    N_CEU: float,
    T_split: float,
    mut_rate_times_L: float,
    m_true: float,
    m_hat: float,
    m_lb: float,
    m_ub: float,
    n_grid: int = 200,
):
    """
    Save a 1D likelihood curve over a log-spaced grid of m values.
    """
    sim_dir = ensure_dir(sim_dir)

    m_grid = np.logspace(np.log10(m_lb), np.log10(m_ub), int(n_grid))
    ll_grid = np.array(
        [
            loglik_m_only(
                m=float(m),
                observed_sfs=observed_sfs,
                mut_rate_times_L=mut_rate_times_L,
                builder_fn=builder_fn,
                N_anc=N_anc,
                N_YRI=N_YRI,
                N_CEU=N_CEU,
                T_split=T_split,
            )
            for m in m_grid
        ],
        dtype=float,
    )

    np.save(sim_dir / "m_grid.npy", m_grid)
    np.save(sim_dir / "ll_grid.npy", ll_grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(m_grid, ll_grid, "-")
    ax.set_xscale("log")
    ax.set_xlabel("m")
    ax.set_ylabel("Composite log-likelihood")
    ax.set_title("1D likelihood scan over m")

    # vertical lines for truth and fitted
    ax.axvline(float(m_true), linestyle="--")
    ax.axvline(float(m_hat), linestyle=":")

    # annotate a bit
    ax.text(
        0.02,
        0.02,
        f"true={m_true:.3e}\nfit={m_hat:.3e}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )

    fig.tight_layout()
    fig.savefig(sim_dir / "ll_vs_m.png", dpi=200)
    plt.close(fig)


def optimize_m_only(
    *,
    m_start: float,
    m_lb: float,
    m_ub: float,
    observed_sfs: moments.Spectrum,
    mut_rate_times_L: float,
    builder_fn,
    N_anc: float,
    N_YRI: float,
    N_CEU: float,
    T_split: float,
    verbose: bool = False,
    rtol: float = 1e-8,
) -> float:
    sample_size = [n - 1 for n in observed_sfs.shape]

    def loglikelihood(log10_m_vec):
        exp_sfs = expected_sfs_only_m(
            log10_m_vec,
            sample_size=sample_size,
            mut_rate_times_L=mut_rate_times_L,
            builder_fn=builder_fn,
            N_anc=N_anc,
            N_YRI=N_YRI,
            N_CEU=N_CEU,
            T_split=T_split,
        )
        return np.sum(np.log(exp_sfs) * observed_sfs - exp_sfs)

    def grad(log10_m_vec):
        return nd.Gradient(loglikelihood, n=1, step=1e-4)(log10_m_vec)

    def objective(log10_m_vec, g_out):
        ll = loglikelihood(log10_m_vec)
        if g_out.size > 0:
            g_out[:] = grad(log10_m_vec)
        if verbose:
            print(f"loglik={ll:.6f}, m={10**float(log10_m_vec[0]):.3e}")
        return ll

    opt = nlopt.opt(nlopt.LD_LBFGS, 1)
    opt.set_lower_bounds([np.log10(m_lb)])
    opt.set_upper_bounds([np.log10(m_ub)])
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    fitted_log10 = opt.optimize(np.array([np.log10(m_start)], dtype=float))
    return float(10 ** float(fitted_log10[0]))


# =============================================================================
# Plotting
# =============================================================================

def plot_true_vs_fit_m(true_m: np.ndarray, fit_m: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(true_m, fit_m, "o", markersize=3)
    lo = min(true_m.min(), fit_m.min())
    hi = max(true_m.max(), fit_m.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True m")
    ax.set_ylabel("Inferred m")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_fit_vs_fit_m(fit_with: np.ndarray, fit_no: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fit_with, fit_no, "o", markersize=3)
    lo = min(fit_with.min(), fit_no.min())
    hi = max(fit_with.max(), fit_no.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inferred m (WITH_ANC)")
    ax.set_ylabel("Inferred m (NO_ANC)")
    ax.set_title("m: WITH_ANC vs NO_ANC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Sample ONLY m; keep other IM-symmetric params fixed; fit ONLY m.")
    p.add_argument("--backend", choices=["msprime", "stdpopsim"], default="msprime")
    p.add_argument("--formulations", choices=["with_anc", "no_anc", "both"], default="both")
    p.add_argument("--num-sims", type=int, default=50)
    p.add_argument("--out-root", type=str, default="debug_outputs_im_sym_only_m")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--seqlen", type=float, default=1e6)
    p.add_argument("--mu", type=float, default=1e-8)
    p.add_argument("--recomb", type=float, default=1e-8)
    p.add_argument("--nYRI", type=int, default=10)
    p.add_argument("--nCEU", type=int, default=10)

    # Fixed parameters
    p.add_argument("--N-anc", dest="N_anc", type=float, default=2e4)
    p.add_argument("--N-YRI", dest="N_YRI", type=float, default=2e4)
    p.add_argument("--N-CEU", dest="N_CEU", type=float, default=2e4)
    p.add_argument("--T-split", dest="T_split", type=float, default=2e3)

    # m sampling + m optimization bounds
    p.add_argument("--m-lb", type=float, default=1e-8)
    p.add_argument("--m-ub", type=float, default=1e-3)
    p.add_argument("--m-start", type=float, default=1e-5)

    # NEW: likelihood scan controls
    p.add_argument("--ll-scan", action="store_true", help="Compute and save 1D log-likelihood curve vs m.")
    p.add_argument("--ll-scan-first-k", type=int, default=5, help="Only do ll-scan for the first K sims.")
    p.add_argument("--ll-scan-points", type=int, default=200, help="Number of grid points in ll scan.")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    out_root = Path(args.out_root)
    plots = ensure_dir(out_root / "plots")

    rng = np.random.default_rng(int(args.seed))

    # fixed params
    N_anc = float(args.N_anc)
    N_YRI = float(args.N_YRI)
    N_CEU = float(args.N_CEU)
    T_split = float(args.T_split)

    # sample/opt bounds for m
    m_lb = float(args.m_lb)
    m_ub = float(args.m_ub)
    m_start = float(args.m_start)

    # simulation args
    seqlen = float(args.seqlen)
    mu = float(args.mu)
    recomb = float(args.recomb)
    nYRI = int(args.nYRI)
    nCEU = int(args.nCEU)
    SIM_BACKEND = args.backend

    # choose formulations
    if args.formulations == "both":
        run_formulations = ["with_anc", "no_anc"]
        do_pairwise = True
    else:
        run_formulations = [args.formulations]
        do_pairwise = False

    # storage
    true_m = np.zeros(args.num_sims, dtype=float)
    fit_m: Dict[str, np.ndarray] = {f: np.zeros(args.num_sims, dtype=float) for f in run_formulations}

    experiment_config: Optional[Dict[str, Any]] = None
    mut_rate_times_L = float(mu) * float(seqlen)

    for i in range(args.num_sims):
        m_true = sample_loguniform(rng, m_lb, m_ub)
        true_m[i] = m_true

        seed_base = int(rng.integers(1, 2**31 - 1000))

        for f in run_formulations:
            builder_fn = FORMULATION_BUILDERS[f]
            sim_dir = ensure_dir(out_root / f / "sims" / f"sim_{i:04d}")

            # build + save graph
            g, demogr = builder_fn(N_anc, N_YRI, N_CEU, T_split, m_true)
            save_demes_graph(sim_dir, g)

            (sim_dir / "true_m.json").write_text(json.dumps({"m": float(m_true)}, indent=2), encoding="utf-8")

            # simulate
            ts = simulate_ts(
                sim_backend=SIM_BACKEND,
                g=g,
                demogr=demogr,
                seqlen=seqlen,
                mu=mu,
                recomb=recomb,
                nYRI=nYRI,
                nCEU=nCEU,
                seed=seed_base + (1 if f == "with_anc" else 10),
                experiment_config=experiment_config,
            )

            YRI = pop_id(ts, POP1)
            CEU = pop_id(ts, POP2)

            obs_sfs = moments.Spectrum(
                ts.allele_frequency_spectrum(
                    sample_sets=[list(ts.samples(YRI)), list(ts.samples(CEU))],
                    mode="site",
                    polarised=True,
                    span_normalise=False,
                )
            )

            # fit ONLY m
            m_hat = optimize_m_only(
                m_start=m_start,
                m_lb=m_lb,
                m_ub=m_ub,
                observed_sfs=obs_sfs,
                mut_rate_times_L=mut_rate_times_L,
                builder_fn=builder_fn,
                N_anc=N_anc,
                N_YRI=N_YRI,
                N_CEU=N_CEU,
                T_split=T_split,
                verbose=False,
            )

            fit_m[f][i] = m_hat
            (sim_dir / "fitted_m.json").write_text(json.dumps({"m": float(m_hat)}, indent=2), encoding="utf-8")

            # NEW: likelihood scan (optionally only for first K sims)
            if args.ll_scan and (i < int(args.ll_scan_first_k)):
                save_ll_curve_over_m(
                    sim_dir=sim_dir,
                    observed_sfs=obs_sfs,
                    builder_fn=builder_fn,
                    N_anc=N_anc,
                    N_YRI=N_YRI,
                    N_CEU=N_CEU,
                    T_split=T_split,
                    mut_rate_times_L=mut_rate_times_L,
                    m_true=m_true,
                    m_hat=m_hat,
                    m_lb=m_lb,
                    m_ub=m_ub,
                    n_grid=int(args.ll_scan_points),
                )

        # logging
        msg = f"[{i+1:>3}/{args.num_sims}] true m={m_true:.3e} | "
        msg += " | ".join([f"{f} mhat={fit_m[f][i]:.3e}" for f in run_formulations])
        print(msg)

    # plots
    for f in run_formulations:
        plot_true_vs_fit_m(
            true_m,
            fit_m[f],
            plots / f"{f.upper()}_{SIM_BACKEND}_m_scatter_true_vs_fit.png",
            title=f"m only: true vs inferred ({f}, {SIM_BACKEND})",
        )

    if do_pairwise:
        plot_fit_vs_fit_m(
            fit_m["with_anc"],
            fit_m["no_anc"],
            plots / f"{SIM_BACKEND}_m_scatter_fit_with_vs_no.png",
        )

    np.save(out_root / "true_m.npy", true_m)
    for f in run_formulations:
        np.save(out_root / f"fit_m_{f}.npy", fit_m[f])

    print("\nDone.")
    print(f"Backend:      {SIM_BACKEND}")
    print(f"Formulations: {run_formulations}")
    print(f"Out root:     {out_root}")
    print(f"Plots:        {plots}")


if __name__ == "__main__":
    main()

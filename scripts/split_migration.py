#!/usr/bin/env python3
"""
split_migration.py – windows ⇒ LD stats ⇒ comparison PDF ⇒ optimisation
==================================================================

Modes:
  - momentsld : your original LD pipeline + optimisation (default)
  - moments   : Poisson SFS optimisation
  - dadi      : Poisson SFS optimisation

SFS modes save, for each replicate:
  <exp_root>/inferences/sim_<rep>/(Moments|Dadi)/profiles_grid.png

Best fits are cached per replicate & mode: best_fit.pkl
"""
from __future__ import annotations
import argparse, importlib, json, logging, pickle, subprocess, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# NLOpt + finite-diff gradients are used for SFS modes
import nlopt
import numdifftools as nd

# moments / momentsLD are always required
import moments
import ray

# ── project paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR  = PROJECT_ROOT / "snakemake_scripts"
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Moments_LD_theoretical import split_asym_mig_MomentsLD

SIM_SCRIPT = SCRIPTS_DIR / "simulation.py"
WIN_SCRIPT = SCRIPTS_DIR / "simulate_window.py"
LD_SCRIPT  = SCRIPTS_DIR / "compute_ld_window.py"

# =============== user toggles for SFS optimisation =======================
USE_BOBYQA = False     # False → LD_LBFGS (finite-diff grad). True → LN_BOBYQA (derivative-free)
VERBOSE_SFS = True
RTOL_SFS = 1e-8
MAXEVAL_SFS = 800
EPS_SFS = 1e-300        # guard for log(0) in Poisson LL
# ========================================================================


# ─── helpers (shared) ────────────────────────────────────────────────────
def _ensure_sampled_params(cfg_file: Path, exp_root: Path, rep: int) -> None:
    sim_root = exp_root / "simulations"
    pkl      = sim_root / str(rep) / "sampled_params.pkl"
    if pkl.exists():
        return
    sim_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(SIM_SCRIPT),
            "--simulation-dir",    str(sim_root),
            "--experiment-config", str(cfg_file),
            "--model-type",        json.loads(cfg_file.read_text())["demographic_model"],
            "--simulation-number", str(rep),
        ],
        check=True,
    )

# ─── LD (momentsLD) path ─────────────────────────────────────────────────
def _run_one_window_and_ld(cfg: Path, exp_root: Path, rep: int,
                           win: int, r_bins: str):
    sim_dir = exp_root / "simulations" / str(rep)
    win_dir = exp_root / "inferences" / f"sim_{rep}" / "MomentsLD" / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    if not (win_dir / f"window_{win}.vcf.gz").exists():
        subprocess.run(
            [
                sys.executable, str(WIN_SCRIPT),
                "--sim-dir", str(sim_dir),
                "--rep-index", str(win),
                "--config-file", str(cfg),
                "--out-dir", str(win_dir),
            ],
            check=True,
        )

    ld_root = exp_root / "inferences" / f"sim_{rep}" / "MomentsLD"
    ld_pkl  = ld_root / "LD_stats" / f"LD_stats_window_{win}.pkl"
    if not ld_pkl.exists():
        subprocess.run(
            [
                sys.executable, str(LD_SCRIPT),
                "--sim-dir", str(ld_root),
                "--window-index", str(win),
                "--config-file", str(cfg),
                "--r-bins", r_bins,
            ],
            check=True,
        )

def _plot_comparison(cfg_json: dict, sampled_params: dict,
                     mv: dict, r_vec: np.ndarray, out_dir: Path):
    pdf = out_dir / f"{cfg_json['demographic_model']}_comparison.pdf"
    if pdf.exists():
        return
    demo_mod  = importlib.import_module("simulation")
    demo_func = getattr(demo_mod, f"{cfg_json['demographic_model']}_model")
    graph     = demo_func(sampled_params)

    demes = list(cfg_json["num_samples"].keys())
    y     = moments.Demes.LD(graph, sampled_demes=demes,
                             rho=4*sampled_params["N0"]*r_vec)
    y     = moments.LD.LDstats(
        [(yl+yr)/2 for yl,yr in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops, pop_ids=y.pop_ids)
    y     = moments.LD.Inference.sigmaD2(y)

    stats = [
        ["DD_0_0"], ["DD_0_1"], ["DD_1_1"],
        ["Dz_0_0_0"], ["Dz_0_1_1"], ["Dz_1_1_1"],
        ["pi2_0_0_1_1"], ["pi2_0_1_0_1"], ["pi2_1_1_1_1"],
    ]
    labs  = [
        [r"$D_0^2$"], [r"$D_0D_1$"], [r"$D_1^2$"],
        [r"$Dz_{0,0,0}$"], [r"$Dz_{0,1,1}$"], [r"$Dz_{1,1,1}$"],
        [r"$\pi_{2;0,0,1,1}$"], [r"$\pi_{2;0,1,0,1}$"],
        [r"$\pi_{2;1,1,1,1}$"],
    ]
    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y, mv["means"][:-1], mv["varcovs"][:-1],
        rs=r_vec, stats_to_plot=stats, labels=labs,
        rows=3, plot_vcs=True, show=False, fig_size=(6, 4))
    fig.savefig(pdf, dpi=300); plt.close(fig)
    logging.info("PDF written → %s", pdf.name)

def _aggregate_and_optimise_momentsld(cfg: Path, exp_root: Path, rep: int, r_bins: str):
    root   = exp_root / "inferences" / f"sim_{rep}" / "MomentsLD"
    statsD = root / "LD_stats"
    mean   = root / "means.varcovs.pkl"
    boot   = root / "bootstrap_sets.pkl"
    best   = root / "best_fit.pkl"

    ld_stats = {int(p.stem.split("_")[-1]): pickle.loads(p.read_bytes())
                for p in statsD.glob("LD_stats_window_*.pkl")}
    if not ld_stats:
        logging.warning("rep %s: no LD pickles – skip aggregation", rep)
        return

    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    pickle.dump(mv, mean.open("wb"))
    pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_stats), boot.open("wb"))

    cfg_json = json.loads(cfg.read_text())
    r_vec    = np.array([float(x) for x in r_bins.split(',')])
    params   = pickle.loads(
        (exp_root / "simulations" / str(rep) / "sampled_params.pkl").read_bytes())
    _plot_comparison(cfg_json, params, mv, r_vec, root)

    if cfg_json["demographic_model"] != "split_migration" or best.exists():
        return

    priors = cfg_json["priors"]; pm = {k:(lo+hi)/2 for k,(lo,hi) in priors.items()}
    p0 = [pm["N1"]/pm["N0"], pm["N2"]/pm["N0"],
          2*pm["m12"]*pm["N0"], 2*pm["m21"]*pm["N0"], pm["t_split"]/(2*pm["N0"]), pm["N0"]]

    opt, ll = moments.LD.Inference.optimize_log_lbfgsb(
        p0, [mv["means"], mv["varcovs"]],
        [split_asym_mig_MomentsLD],
        rs=r_vec, verbose=1)
    best_phys = dict(zip(["N1","N2","m12","m21", "t_split", "N0"],
                         moments.LD.Util.rescale_params(opt, ["nu","nu","m","m","T","Ne"])))
    pickle.dump({"best_params": best_phys, "best_lls": ll}, best.open("wb"))
    logging.info("rep %s: optimisation finished (LL=%.2f)", rep, ll)

# ─── SFS (moments/dadi) path ─────────────────────────────────────────────

def _save_profiles_grid(
    root_dir: Path,
    names: list[str],
    grids: list[np.ndarray],
    observed_sfs,
    muL: float,
    backend: str,
    fitted_pars: np.ndarray,
):
    """Make a 2x3 grid of scatterplots: LL vs parameter grid, save once."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=150, constrained_layout=True)
    axes = axes.ravel()
    for i, (ax, name, grid) in enumerate(zip(axes, names, grids)):
        y = []
        for v in grid:
            test = fitted_pars.copy()
            test[i] = float(v)
            y.append(_poisson_ll(np.log10(test), observed_sfs, muL, backend))
        ax.scatter(grid, y, s=10)
        ax.axvline(fitted_pars[i], linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel(name)
        if i % 3 == 0:
            ax.set_ylabel("Poisson log-likelihood")
        ax.grid(alpha=0.25, which="both")
    outpng = root_dir / "profiles_grid.png"
    fig.savefig(outpng)
    plt.close(fig)
    logging.info("  saved %s", outpng.name)

def _default_pts(sample_sizes: List[int]) -> List[int]:
    nmax = int(max(sample_sizes))
    return [nmax + 10, nmax + 20, nmax + 30]

def _msprime_model(N_ANC, N1, N2, T, m12, m21):
    import msprime
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=N1)
    dem.add_population(name="B", initial_size=N2)
    dem.add_population(name="ANC", initial_size=N_ANC)
    dem.add_population_split(time=T, ancestral="ANC", derived=["A","B"])
    dem.migration_matrix = np.array([[0, m12, 0],[m21, 0, 0],[0,0,0]])
    return dem

def _expected_sfs(log10_params: np.ndarray,
                  sample_sizes: List[int],
                  mu_times_L: float,
                  backend: str,
                  pts: List[int] | None = None):
    N_ANC, N1, N2, T, m12, m21 = 10 ** np.asarray(log10_params, float)
    demog = _msprime_model(N_ANC, N1, N2, T, m12, m21)

    if backend == "moments":
        import moments as sfs_backend
        fs = sfs_backend.Spectrum.from_demes(
            demog.to_demes(), sampled_demes=["A","B"], sample_sizes=sample_sizes
        )
    elif backend == "dadi":
        import dadi as sfs_backend
        if pts is None:
            pts = _default_pts(sample_sizes)
        fs = sfs_backend.Spectrum.from_demes(
            demog.to_demes(), sampled_demes=["A","B"], sample_sizes=sample_sizes, pts=pts
        )
    else:
        raise ValueError("backend must be 'moments' or 'dadi'")

    # θ scaling for Poisson counts
    fs = fs * (4.0 * N_ANC * mu_times_L)
    return fs

def _poisson_ll(log10_params, observed_sfs, mu_times_L, backend):
    ns = getattr(observed_sfs, "sample_sizes", None)
    if ns is None:
        ns = [n - 1 for n in observed_sfs.shape]
    pts = _default_pts(ns) if backend == "dadi" else None
    model = _expected_sfs(log10_params, ns, mu_times_L, backend, pts=pts)
    if getattr(observed_sfs, "folded", False):
        model = model.fold()
    model = np.maximum(model, EPS_SFS)
    return float(np.sum(np.log(model) * observed_sfs - model))

def _optimize_sfs_poisson(
    start_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    observed_sfs,
    mu_times_L: float,
    backend: str,
) -> Tuple[np.ndarray, object, float, int]:
    """
    Maximize Poisson composite log-likelihood in log10-parameter space.
    Uses NLopt with either LD_LBFGS (default) or LN_BOBYQA (if USE_BOBYQA=True).
    Robustly recovers best-so-far on optimizer errors.
    """
    # Convert bounds/start to log10 space
    start_values = np.asarray(start_values, float)
    lb = np.log10(np.asarray(lower_bounds, float))
    ub = np.log10(np.asarray(upper_bounds, float))
    x0 = np.log10(start_values)

    # Objective (returns scalar LL to maximize)
    def obj(xlog10):
        return _poisson_ll(xlog10, observed_sfs, mu_times_L, backend)

    # Finite-diff gradient for L-BFGS (ignored by BOBYQA)
    grad_fn = nd.Gradient(obj, step=1e-4)

    def objective(xlog10, grad):
        ll = obj(xlog10)
        if grad.size > 0 and not USE_BOBYQA:
            grad[:] = grad_fn(xlog10)
        if VERBOSE_SFS:
            print(f"[LL={ll:.6g}] log10_params={np.array2string(xlog10, precision=4)}")
        return ll

    # Choose optimizer
    opt = nlopt.opt(nlopt.LN_BOBYQA if USE_BOBYQA else nlopt.LD_LBFGS, x0.size)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_max_objective(objective)
    opt.set_ftol_rel(RTOL_SFS)
    opt.set_xtol_rel(RTOL_SFS)
    opt.set_maxeval(MAXEVAL_SFS)

    # Optimize with robust recovery
    try:
        x_hat = opt.optimize(x0)
        status = opt.last_optimize_result()
        best_ll = opt.last_optimum_value()
    except Exception as e:
        if VERBOSE_SFS:
            print(f"[WARN] nlopt exception: {e}. Trying to recover best-so-far.")
        try:
            x_hat = opt.last_optimum_x()
            best_ll = opt.last_optimum_value()
            status = opt.last_optimize_result()
        except Exception:
            # Fall back to start if even recovery fails
            x_hat = x0
            best_ll = obj(x0)
            status = -1  # custom "failed" code

    # Build outputs at the final point
    fitted = 10 ** x_hat
    ns = getattr(observed_sfs, "sample_sizes", None)
    if ns is None:
        ns = [n - 1 for n in observed_sfs.shape]
    pts = _default_pts(ns) if backend == "dadi" else None
    fitted_sfs = _expected_sfs(x_hat, ns, mu_times_L, backend, pts=pts)
    if getattr(observed_sfs, "folded", False):
        fitted_sfs = fitted_sfs.fold()

    return fitted, fitted_sfs, float(best_ll), int(status)

def _simulate_observed_sfs(cfg_json: dict, params: dict):
    """
    Create a single big dataset to act as observed SFS (consistent across backends).
    Pulls sensible defaults if keys are absent.
    """
    import msprime
    seqlen = int(cfg_json.get("sequence_length", cfg_json.get("L", 5e7)))
    mu = float(cfg_json.get("mutation_rate", cfg_json.get("mu", 1e-8)))
    nA = int(cfg_json.get("num_samples", {}).get("A", 5))
    nB = int(cfg_json.get("num_samples", {}).get("B", 5))

    dem = _msprime_model(params["N0"], params["N1"], params["N2"],
                         params["t_split"], params["m12"], params["m21"])

    ts = msprime.sim_ancestry(
        samples={"A": nA, "B": nB},
        sequence_length=seqlen,
        recombination_rate=float(cfg_json.get("recombination_rate", 1e-8)),
        demography=dem,
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=2)
    obs = ts.allele_frequency_spectrum(
        sample_sets=[list(ts.samples(population=p)) for p in [0, 1]],
        mode="site", span_normalise=False, polarised=True,
    )
    return obs, mu * seqlen  # (array, μ·L)

def _aggregate_and_optimise_sfs(cfg: Path, exp_root: Path, rep: int, backend: str):
    """
    SFS optimisation (moments or dadi) with Poisson composite likelihood.
    """
    # paths
    root = exp_root / "inferences" / f"sim_{rep}" / ( "Moments" if backend=="moments" else "Dadi" )
    root.mkdir(parents=True, exist_ok=True)
    best = root / "best_fit.pkl"

    if best.exists():
        logging.info("rep %d: %s best_fit.pkl exists – skipping", rep, backend)
        return

    # inputs
    cfg_json = json.loads(cfg.read_text())
    params   = pickle.loads((exp_root / "simulations" / str(rep) / "sampled_params.pkl").read_bytes())

    # observed SFS & μ·L
    obs_array, muL = _simulate_observed_sfs(cfg_json, params)

    # wrap observed counts
    if backend == "moments":
        observed_sfs = moments.Spectrum(obs_array)
    else:
        import dadi
        observed_sfs = dadi.Spectrum(obs_array)

    # bounds from priors; start = geometric mid
    pri = cfg_json["priors"]
    lb = np.array([pri["N0"][0], pri["N1"][0], pri["N2"][0], pri["t_split"][0], pri["m12"][0], pri["m21"][0]], float)
    ub = np.array([pri["N0"][1], pri["N1"][1], pri["N2"][1], pri["t_split"][1], pri["m12"][1], pri["m21"][1]], float)
    # parameter order for optimisation [N_ANC, N1, N2, T, m12, m21]
    start = np.sqrt(lb * ub)

    fitted_pars, fitted_sfs, best_ll, status = _optimize_sfs_poisson(
        start_values=start,
        lower_bounds=lb,
        upper_bounds=ub,
        observed_sfs=observed_sfs,
        mu_times_L=muL,
        backend=backend,
    )

    # save fit
    out = {"best_params": dict(zip(["N_ANC","N1","N2","T","m12","m21"], map(float, fitted_pars))),
           "best_ll": best_ll, "status": status}
    pickle.dump(out, best.open("wb"))
    logging.info("rep %d: %s SFS optimisation done (LL=%.3f, status=%d)",
                 rep, backend, best_ll, status)

    # build grids and save a single 2×3 sheet of scatterplots
    names = ["N_ANC","N1","N2","T","m12","m21"]
    grids = []
    for i, name in enumerate(names):
        center = fitted_pars[i]
        lo, hi = lb[i], ub[i]
        if i <= 3:
            gmin = max(lo, center/50.0); gmax = min(hi, center*50.0)
        else:
            gmin = max(lo, center/100.0); gmax = min(hi, center*100.0)
        gmin = max(gmin, 1e-12)
        if gmax <= gmin*1.0001: gmax = min(hi, gmin*10.0)
        grids.append(np.logspace(np.log10(gmin), np.log10(gmax), 41))

    _save_profiles_grid(root, names, grids, observed_sfs, muL, backend, fitted_pars)

# ─── CLI / main ───────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser("windows ⇒ LD ⇒ PDF ⇒ optimisation; Ray-parallel")
    p.add_argument("-c","--config", required=True, type=Path)
    p.add_argument("-e","--exp-root", required=True, type=Path)
    p.add_argument("-r","--rep-index", type=int, nargs="+", default=[0],
                   help="Replicate indices, e.g. -r 0 1 2")
    p.add_argument("-n","--window-index", type=int, nargs="+", default=[0],
                   help="Window indices, e.g. -n 0 1 2 … (momentsld only)")
    p.add_argument("--r-bins", help="Comma-separated r-bin edges (momentsld only)")
    p.add_argument("--mode", choices=["momentsld","moments","dadi"],
                   default="momentsld", help="Which optimisation path to run")
    p.add_argument("--ray-address", default=None)
    p.add_argument("--no-ray", action="store_true")
    p.add_argument("-v","--verbose", action="count", default=0)
    return p.parse_args()

def main():
    a = _parse_args()
    logging.basicConfig(
        level=logging.WARNING - 10*min(a.verbose,2),
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # ensure sampled_params for every replicate
    for rep in a.rep_index:
        _ensure_sampled_params(a.config, a.exp_root, rep)

    if a.mode == "momentsld":
        if not a.r_bins:
            raise SystemExit("--r-bins is required in momentsld mode (comma-separated)")
        # ---------- window & LD tasks -------------------------------------
        if a.no_ray:
            for rep in a.rep_index:
                for win in a.window_index:
                    _run_one_window_and_ld(a.config, a.exp_root, rep, win, a.r_bins)
        else:
            ray.init(address=a.ray_address, ignore_reinit_error=True)
            tasks = [ray.remote(_run_one_window_and_ld).remote(
                        a.config, a.exp_root, rep, win, a.r_bins)
                     for rep in a.rep_index for win in a.window_index]
            ray.get(tasks)
            ray.shutdown()

        # ---------- aggregation & optimisation per replicate --------------
        for rep in a.rep_index:
            _aggregate_and_optimise_momentsld(a.config, a.exp_root, rep, a.r_bins)

    else:
        # SFS modes: skip window/LD work and do SFS Poisson optimisation
        backend = a.mode  # "moments" or "dadi"
        for rep in a.rep_index:
            _aggregate_and_optimise_sfs(a.config, a.exp_root, rep, backend)

# ─── entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

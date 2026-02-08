#!/usr/bin/env python3
"""
debug_moments_fit_model.py

Debug script to test src/moments_inference.py::fit_model end-to-end and
compare against ground truth.

What it does
------------
Given a simulation directory (or explicit file paths), it:
1) Loads observed SFS (moments.Spectrum) and sampled true params (dict)
2) Loads experiment_config (JSON) to get priors, mu, L, etc.
3) Runs fit_model(...) using your new moments_inference.py (nlopt LBFGS, log10 space)
4) Generates expected SFS from demes graph for:
      - true params
      - fitted params
5) Computes and saves:
      - parameter table + ratios/log-diffs
      - SFS comparison plots (heatmaps, log-log scatters)
      - goodness-of-fit stats (LLs, correlations, MARE)
      - expected SFS arrays (*.npy)

Key correctness checks
----------------------
- Uses *exact* moments.from_demes conventions matching your debug script:
    haploid sample sizes = [axis_len - 1 for axis_len in sfs.shape]
    sampled_demes order  = sfs.pop_ids
    theta                = 4 * N_anc * mu * L  (N_anc must be first prior key)

Usage
-----
Example:
  python debug_moments_fit_model.py \
    --sim-dir experiments/IM_symmetric/simulations/0 \
    --config experiments/IM_symmetric/experiment_config.json \
    --out debug_moments_fit_model_out \
    --start geomean

or explicitly:
  python debug_moments_fit_model.py \
    --sfs experiments/IM_symmetric/simulations/0/SFS.pkl \
    --true experiments/IM_symmetric/simulations/0/sampled_params.pkl \
    --config experiments/IM_symmetric/experiment_config.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import moments

# Make src importable
import sys
ROOT = Path(__file__).resolve().parents[1]  # adjust if script lives elsewhere
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demes_models import IM_symmetric_model  # your model callable(dict)->demes.Graph
from src.moments_inference import fit_model      # your rewritten fit_model


# --------------------------- IO helpers ---------------------------

def load_pickle(p: Path) -> Any:
    with p.open("rb") as f:
        return pickle.load(f)

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------- expected SFS helpers ----------------------

def expected_sfs_from_params(
    params: Dict[str, float],
    sfs: moments.Spectrum,
    experiment_config: Dict[str, Any],
    demo_model=IM_symmetric_model,
) -> moments.Spectrum:
    """
    Expected SFS using the same conventions as moments_inference.fit_model:
      - sampled_demes: sfs.pop_ids
      - haploid sample sizes: [n-1 for n in sfs.shape]
      - theta: 4*N0*mu*L where N0 is params[first prior key], typically N_anc
    """
    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("Observed sfs is missing pop_ids; cannot determine deme order.")

    haploid_sizes = [int(n) - 1 for n in sfs.shape]

    priors = experiment_config["priors"]
    first_key = list(priors.keys())[0]
    if first_key not in params:
        raise KeyError(f"Expected first prior key {first_key!r} to be present in params dict.")

    mu = float(experiment_config["mutation_rate"])
    L = float(experiment_config["genome_length"])
    N0 = float(params[first_key])
    theta = 4.0 * N0 * mu * L

    graph = demo_model({k: float(v) for k, v in params.items()})

    return moments.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=theta,
    )


# ---------------------- comparison helpers ----------------------

def param_table(true_params: Dict[str, float], fit_params: Dict[str, float], param_order: List[str]) -> str:
    lines = []
    lines.append("=== Parameter Comparison ===")
    lines.append(f"{'Parameter':<12} {'True':>15} {'Fitted':>15} {'Ratio(F/T)':>12} {'|log10(F/T)|':>14}")
    lines.append("-" * 72)
    for p in param_order:
        t = float(true_params[p])
        f = float(fit_params[p])
        ratio = (f / t) if t != 0 else np.inf
        logdiff = abs(np.log10(ratio)) if ratio > 0 else np.inf
        lines.append(f"{p:<12} {t:>15.6e} {f:>15.6e} {ratio:>12.4f} {logdiff:>14.3f}")
    return "\n".join(lines)

def goodness_of_fit(obs: moments.Spectrum, exp_true: moments.Spectrum, exp_fit: moments.Spectrum) -> Dict[str, float]:
    obs_arr = np.asarray(obs, float)
    t_arr = np.asarray(exp_true, float)
    f_arr = np.asarray(exp_fit, float)

    eps = 1e-300
    ll_true = float(np.sum(obs_arr * np.log(t_arr + eps) - t_arr))
    ll_fit  = float(np.sum(obs_arr * np.log(f_arr + eps) - f_arr))

    o = obs_arr.ravel()
    t = t_arr.ravel()
    f = f_arr.ravel()
    mask = (o > 0) & (t > 0) & (f > 0)

    if mask.sum() > 5:
        r2_true = float(np.corrcoef(np.log(o[mask]), np.log(t[mask]))[0, 1] ** 2)
        r2_fit  = float(np.corrcoef(np.log(o[mask]), np.log(f[mask]))[0, 1] ** 2)
        mare_true = float(np.mean(np.abs((o[mask] - t[mask]) / (t[mask] + 1e-12))))
        mare_fit  = float(np.mean(np.abs((o[mask] - f[mask]) / (f[mask] + 1e-12))))
    else:
        r2_true = r2_fit = np.nan
        mare_true = mare_fit = np.nan

    return {
        "log_likelihood_true": ll_true,
        "log_likelihood_fit": ll_fit,
        "ll_improvement_fit_minus_true": ll_fit - ll_true,
        "r2_log_true": r2_true,
        "r2_log_fit": r2_fit,
        "mare_true": mare_true,
        "mare_fit": mare_fit,
    }

def plot_sfs_panels(obs: moments.Spectrum, exp_true: moments.Spectrum, exp_fit: moments.Spectrum, out_dir: Path) -> None:
    obs_arr = np.asarray(obs, float)
    t_arr = np.asarray(exp_true, float)
    f_arr = np.asarray(exp_fit, float)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(obs_arr, origin="lower", aspect="auto")
    axes[0, 0].set_title("Observed SFS")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(t_arr, origin="lower", aspect="auto")
    axes[0, 1].set_title("Expected SFS (True params)")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(f_arr, origin="lower", aspect="auto")
    axes[0, 2].set_title("Expected SFS (Fitted params)")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # log-log scatters (mask <=0)
    def _scatter(ax, x, y, title, xlabel):
        x = x.ravel()
        y = y.ravel()
        m = (x > 0) & (y > 0)
        if m.sum() == 0:
            ax.set_title(title + " (no positive entries)")
            return
        ax.scatter(x[m], y[m], s=8, alpha=0.4)
        lo = min(x[m].min(), y[m].min())
        hi = max(x[m].max(), y[m].max())
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Observed")
        ax.set_title(title)

    _scatter(axes[1, 0], t_arr, obs_arr, "Observed vs Expected (True)", "Expected (True)")
    _scatter(axes[1, 1], f_arr, obs_arr, "Observed vs Expected (Fitted)", "Expected (Fitted)")

    # standardized residuals
    resid = (obs_arr - f_arr) / np.sqrt(f_arr + 1e-12)
    im5 = axes[1, 2].imshow(resid, origin="lower", aspect="auto")
    axes[1, 2].set_title("(Obs - Fit) / sqrt(Fit)")
    plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)

    for ax in axes.flat:
        ax.set_xlabel(ax.get_xlabel() or "")
        ax.set_ylabel(ax.get_ylabel() or "")

    fig.tight_layout()
    fig.savefig(out_dir / "sfs_comparison.png", dpi=250)
    plt.close(fig)


# ---------------------- start vector helpers ----------------------

def start_vector_from_priors(priors: Dict[str, List[float]], mode: str) -> np.ndarray:
    """
    mode:
      - "geomean": geometric mean of bounds (best default for log-space)
      - "mid": arithmetic midpoint
      - "lb": lower bounds
      - "ub": upper bounds
    """
    lb = np.array([priors[k][0] for k in priors.keys()], float)
    ub = np.array([priors[k][1] for k in priors.keys()], float)
    if mode == "geomean":
        return np.sqrt(lb * ub)
    if mode == "mid":
        return (lb + ub) / 2.0
    if mode == "lb":
        return lb
    if mode == "ub":
        return ub
    raise ValueError(f"Unknown start mode: {mode!r}")


# ------------------------------ CLI ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug moments_inference.fit_model against truth and expected SFS.")
    p.add_argument("--sim-dir", type=str, default=None,
                   help="Simulation directory that contains SFS.pkl and sampled_params.pkl")
    p.add_argument("--sfs", type=str, default=None, help="Path to observed SFS.pkl")
    p.add_argument("--true", type=str, default=None, help="Path to sampled_params.pkl (truth dict)")
    p.add_argument("--config", type=str, required=True, help="Path to experiment_config.json (must include priors, mutation_rate, genome_length)")
    p.add_argument("--out", type=str, default="debug_moments_fit_model_out", help="Output directory for plots/stats")
    p.add_argument("--start", choices=["geomean", "mid", "lb", "ub"], default="geomean",
                   help="How to choose start_vec from priors")
    p.add_argument("--save-expected", action="store_true", help="Save expected SFS arrays for obs/true/fit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = ensure_dir(Path(args.out))

    # resolve paths
    if args.sim_dir is not None:
        sim_dir = Path(args.sim_dir)
        sfs_path = sim_dir / "SFS.pkl"
        true_path = sim_dir / "sampled_params.pkl"
    else:
        if args.sfs is None or args.true is None:
            raise SystemExit("Provide either --sim-dir or both --sfs and --true.")
        sfs_path = Path(args.sfs)
        true_path = Path(args.true)

    cfg_path = Path(args.config)

    # load
    print(f"Loading observed SFS:  {sfs_path}")
    obs_sfs = load_pickle(sfs_path)
    if not isinstance(obs_sfs, moments.Spectrum):
        # If your pipeline pickles raw arrays, wrap them:
        obs_sfs = moments.Spectrum(obs_sfs)

    print(f"Loading true params:   {true_path}")
    true_params = load_pickle(true_path)
    true_params = {k: float(v) for k, v in true_params.items()}

    print(f"Loading config:        {cfg_path}")
    experiment_config = load_json(cfg_path)

    # sanity: pop_ids
    if not getattr(obs_sfs, "pop_ids", None):
        # try to recover from config, else fail loudly
        # (you can hardcode if you want)
        obs_sfs.pop_ids = ["YRI", "CEU"]

    # build start_vec from priors
    priors = experiment_config["priors"]
    start_vec = start_vector_from_priors(priors, args.start)

    # make fit_model save fitted expected SFS into this debug directory (optional)
    experiment_config = dict(experiment_config)
    experiment_config["out_dir"] = str(out_dir)

    print("\n=== Running fit_model (moments) ===")
    best_params_list, best_lls = fit_model(
        sfs=obs_sfs,
        start_vec=start_vec,
        demo_model=IM_symmetric_model,
        experiment_config=experiment_config,
    )
    opt_vec = np.asarray(best_params_list[0], float)
    ll_fit = float(best_lls[0])

    # convert opt vec -> dict using prior order (must match your moments_inference)
    param_names = list(priors.keys())
    fit_params = {k: float(v) for k, v in zip(param_names, opt_vec)}

    # expected SFS under true and fitted
    print("Generating expected SFS under true and fitted params...")
    exp_true = expected_sfs_from_params(true_params, obs_sfs, experiment_config, demo_model=IM_symmetric_model)
    exp_fit  = expected_sfs_from_params(fit_params,  obs_sfs, experiment_config, demo_model=IM_symmetric_model)

    # compute true LL too (composite Poisson)
    stats = goodness_of_fit(obs_sfs, exp_true, exp_fit)

    # report
    print("\n" + param_table(true_params, fit_params, param_order=param_names))
    print("\n=== Likelihood / fit stats ===")
    for k, v in stats.items():
        if isinstance(v, float) and np.isfinite(v):
            print(f"{k:>28}: {v:.6g}")
        else:
            print(f"{k:>28}: {v}")

    print(f"\nfit_model returned ll: {ll_fit:.6g}  (should match stats['log_likelihood_fit'] closely)")

    # save artifacts
    save_json(out_dir / "fit_params.json", fit_params)
    save_json(out_dir / "true_params.json", true_params)
    save_json(out_dir / "fit_stats.json", stats)

    (out_dir / "report.txt").write_text(
        param_table(true_params, fit_params, param_order=param_names)
        + "\n\n=== Likelihood / fit stats ===\n"
        + "\n".join([f"{k}: {v}" for k, v in stats.items()])
        + f"\n\nfit_model_returned_ll: {ll_fit}\n",
        encoding="utf-8",
    )

    plot_sfs_panels(obs_sfs, exp_true, exp_fit, out_dir)

    if args.save_expected:
        np.save(out_dir / "obs_sfs.npy", np.asarray(obs_sfs, float))
        np.save(out_dir / "expected_true.npy", np.asarray(exp_true, float))
        np.save(out_dir / "expected_fit.npy", np.asarray(exp_fit, float))

    print(f"\nWrote outputs to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

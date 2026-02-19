#!/usr/bin/env python3
"""
moments_inference.py

Moments SFS inference for demes-based models with NLopt LD_LBFGS.

What this version does (per your requests):
- Uses a GEOMETRIC-midpoint start based on bounds (overrides incoming start_vec),
  with optional mild jitter controlled by experiment_config:
    * experiment_config["start_jitter_sigma"] = 0.2   # log10-space
    * experiment_config["opt_seed"] = OPT             # reproducible per optimization
- Computes mask-safe Poisson composite log-likelihood:
    LL = sum( sfs * log(model + eps) - model )
- Optional 1D profile likelihood curves (hold others fixed at xhat):
    * experiment_config["generate_profiles"] = True
    * experiment_config["profile_points"] = 41
    * experiment_config["profile_widen"] = 0.5
    * experiment_config["profile_make_plots"] = True
  Writes <save_dir>/likelihood_plots/profile_<PARAM>.npz and optionally .png

IMPORTANT: No cfg["out_dir"] is used. No cfg output directory keys are required.
If you want to write likelihood plots, pass save_dir=... when calling fit_model.
Otherwise nothing is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import moments
import nlopt
import numdifftools as nd


# ───────────────────────── diffusion helper ────────────────────────────
def _diffusion_sfs(
    log_space_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    param_names: List[str],
    sampled_demes: List[str],
    haploid_sizes: List[int],
    experiment_config: Dict[str, Any],
    *,
    return_graph: bool = False,
    folded: bool = False,
):
    real_space_vec = 10 ** log_space_vec
    p_dict = {k: float(v) for k, v in zip(param_names, real_space_vec)}

    graph = demo_model(p_dict)

    muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
    # Convention: first parameter is N_ANC/N0 for theta scaling
    N0 = float(p_dict[param_names[0]])
    theta = 4.0 * N0 * muL

    fs = moments.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=theta,
    )

    if folded:
        fs = fs.fold()

    if return_graph:
        return fs, graph, p_dict, theta, muL
    return fs


# ───────────────────────── profile likelihood helpers ─────────────────────────
def _profile_1d(
    *,
    xhat_log10: np.ndarray,
    param_names: List[str],
    lb_full: np.ndarray,
    ub_full: np.ndarray,
    loglikelihood_fn: Callable[[np.ndarray], float],
    n_points: int = 41,
    widen: float = 0.5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    1D profile likelihood for each parameter:
      sweep x_i over a grid in log10-space, keeping others fixed at xhat.

    widen:
      - 0.0 => full bound range [lb, ub]
      - >0  => centered around xhat using +/- widen * (prior_span), clipped to bounds
    """
    lb_log10 = np.log10(lb_full)
    ub_log10 = np.log10(ub_full)

    out: Dict[str, Dict[str, np.ndarray]] = {}

    for i, p in enumerate(param_names):
        lo = lb_log10[i]
        hi = ub_log10[i]

        if widen > 0:
            span = hi - lo
            lo_g = max(lo, xhat_log10[i] - widen * span)
            hi_g = min(hi, xhat_log10[i] + widen * span)
        else:
            lo_g, hi_g = lo, hi

        grid = np.linspace(lo_g, hi_g, int(n_points))
        ll = np.empty_like(grid)

        x = xhat_log10.copy()
        for k, g in enumerate(grid):
            x[i] = g
            ll[k] = loglikelihood_fn(x)

        out[p] = {"grid_log10": grid, "ll": ll}

    return out


def _save_profiles(
    profiles: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    *,
    make_plots: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for p, d in profiles.items():
        np.savez(out_dir / f"profile_{p}.npz", grid_log10=d["grid_log10"], ll=d["ll"])

    if not make_plots:
        return

    import matplotlib.pyplot as plt

    for p, d in profiles.items():
        grid_log10 = d["grid_log10"]
        ll = d["ll"]
        ll_max = float(np.max(ll))
        dLL = ll_max - ll

        plt.figure()
        plt.plot(10 ** grid_log10, dLL)
        plt.xscale("log")
        plt.xlabel(p)
        plt.ylabel("Δ log-likelihood (max - ll)")
        plt.title(f"Profile likelihood: {p}")
        plt.tight_layout()
        plt.savefig(out_dir / f"profile_{p}.png", dpi=150)
        plt.close()


# ───────────────────────── optimisation wrapper ────────────────────────
def fit_model(
    sfs: moments.Spectrum,
    start_vec: np.ndarray,  # kept for backward compatibility; overridden below
    demo_model: Callable[[Dict[str, float]], Any],
    experiment_config: Dict[str, Any],
    param_order: Optional[List[str]] = None,
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-12,
    *,
    save_dir: Optional[str | Path] = None,  # <-- NEW: where to put likelihood_plots/
) -> Tuple[np.ndarray, float]:
    """
    Returns (fitted_real_params, ll_hat).

    Start behavior:
      - overrides incoming start_vec to geometric midpoint of bounds
      - optional mild jitter around that midpoint via config keys:
          start_jitter_sigma (float, log10 space; e.g. 0.2)
          opt_seed (int; e.g. OPT index for reproducible multi-start)

    Likelihood plots:
      - If save_dir is provided AND experiment_config["generate_profiles"] is True,
        1D profiles are written to:
            <save_dir>/likelihood_plots/
    """

    assert isinstance(sfs, moments.Spectrum), "sfs must be a moments.Spectrum"

    # ---- parameter order ----
    if param_order is None:
        param_order = list(experiment_config["parameter_order"])
    param_names = list(param_order)

    # ---- bounds ----
    priors = experiment_config.get("priors", experiment_config.get("parameters", {}))
    if not priors:
        raise ValueError("experiment_config must include 'priors' (or 'parameters').")

    lb_full = np.array([float(priors[p][0]) for p in param_names], dtype=float)
    ub_full = np.array([float(priors[p][1]) for p in param_names], dtype=float)

    if np.any(lb_full <= 0) or np.any(ub_full <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb_full, ub_full) if lo <= 0 or hi <= 0]
        raise ValueError(f"All bounds must be positive for log10 optimization. Bad: {bad}")

    # ---- build geometric-midpoint start (+ optional jitter) ----
    start_vec = 10 ** ((np.log10(lb_full) + np.log10(ub_full)) / 2.0)

    sigma = float(experiment_config.get("start_jitter_sigma", 0.0))
    if sigma > 0:
        seed = experiment_config.get("opt_seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        start_vec = start_vec * (10 ** rng.normal(0.0, sigma, size=start_vec.shape))

    start_vec = np.clip(start_vec, lb_full, ub_full)

    # ---- SFS demes order ----
    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("Observed SFS has no pop_ids; cannot infer sampled_demes order.")

    haploid_sizes = [n - 1 for n in sfs.shape]
    obs_folded = bool(getattr(sfs, "folded", False))

    # ---- Poisson composite log-likelihood (MASK SAFE) ----
    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = _diffusion_sfs(
            log_space_vec=log10_params,
            demo_model=demo_model,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            experiment_config=experiment_config,
            folded=obs_folded,
        )
        return float((np.log(exp_sfs + eps) * sfs - exp_sfs).sum())

    import time

    # --- quick timing diagnostic (one eval) ---
    t0 = time.time()
    ll0 = loglikelihood(np.log10(start_vec))
    dt = time.time() - t0
    print(f"[diagnostic] one loglik eval took {dt:.3f}s  (ll={ll0:.6g})")

    grad_fn = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {log10_params}")
        return ll

    # ---- optimizer setup ----
    start_vec = np.asarray(start_vec, dtype=float)
    if start_vec.shape != (len(param_names),):
        raise ValueError(f"start_vec shape {start_vec.shape} != ({len(param_names)},)")

    opt = nlopt.opt(nlopt.LD_LBFGS, start_vec.size)
    opt.set_lower_bounds(np.log10(lb_full))
    opt.set_upper_bounds(np.log10(ub_full))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    # ---- run optimization ----
    x0 = np.log10(start_vec)
    xhat = opt.optimize(x0)

    ll_hat = loglikelihood(xhat)
    fitted_real = 10 ** xhat

    # ---- optional profile likelihood curves -> <save_dir>/likelihood_plots/ ----
    if save_dir is not None and bool(experiment_config.get("generate_profiles", False)):
        base = Path(save_dir)
        like_dir = base / "likelihood_plots"
        like_dir.mkdir(parents=True, exist_ok=True)

        profiles = _profile_1d(
            xhat_log10=xhat,
            param_names=param_names,
            lb_full=lb_full,
            ub_full=ub_full,
            loglikelihood_fn=loglikelihood,
            n_points=int(experiment_config.get("profile_points", 41)),
            widen=float(experiment_config.get("profile_widen", 0.5)),
        )
        _save_profiles(
            profiles,
            like_dir,
            make_plots=bool(experiment_config.get("profile_make_plots", True)),
        )

    return fitted_real, float(ll_hat)

#!/usr/bin/env python3
"""
src/moments_inference_real.py

Real-data moments SFS inference where we:
- optimize demography "shape" in DIMENSIONLESS units:
    sizes: ratios r = N_X / N_ANC
    time:  tau = T / (2*N_ANC)
    migration: M = 2*N_ANC*m
- profile out theta analytically under Poisson composite likelihood
- convert theta_hat -> N_ANC_implied = theta_hat / (4*mu*L)
- convert all inferred params back to ABSOLUTE units using N_ANC_implied

Assumptions:
- param_order includes the same names as your model expects:
    ['N_ANC', 'N_CO', 'N_FR0', 'N_FR1', 'T', 'm_CO_FR', 'm_FR_CO', ...]
- In real-data mode, these names are interpreted as:
    N_ANC  : placeholder (typically fixed to 1.0 via bounds [1,1])
    N_*    : ratios r = N_*/N_ANC
    T      : tau = T/(2*N_ANC)
    m_*    : M = 2*N_ANC*m

Priors:
- uses experiment_config["_active_priors"] if present
- else uses experiment_config["priors_real_data_analysis"]

Return:
    (best_params_abs: Dict[str,float], ll_hat: float, theta_hat: float, N_ANC_implied: float)

Also supports saving 1D profile curves (in SCALED space) if:
  experiment_config["generate_profiles"] == True
  and save_dir is provided
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import moments
import nlopt
import numdifftools as nd

from src.inference_utils import build_scaled_param_dict, scaled_to_absolute_params, lhs_start_log10, profile_1d, save_profiles


# ------------------------------ mask-safe helpers ------------------------------
def _mask_safe_sum(sfs: moments.Spectrum, arr: np.ndarray) -> float:
    mask = getattr(sfs, "mask", None)
    if mask is None:
        return float(np.asarray(arr).sum())
    mask = np.asarray(mask, dtype=bool)
    return float(np.asarray(arr)[~mask].sum())


def _theta_hat_poisson_mle(sfs: moments.Spectrum, base_sfs: np.ndarray) -> float:
    s_obs = _mask_safe_sum(sfs, np.asarray(sfs))
    s_base = _mask_safe_sum(sfs, np.asarray(base_sfs))
    if s_base <= 0:
        return 0.0
    return s_obs / s_base


def _mask_safe_poisson_ll(sfs: moments.Spectrum, exp_sfs: np.ndarray, eps: float) -> float:
    obs = np.asarray(sfs)
    exp = np.asarray(exp_sfs)
    ll = obs * np.log(exp + eps) - exp
    mask = getattr(sfs, "mask", None)
    if mask is None:
        return float(ll.sum())
    mask = np.asarray(mask, dtype=bool)
    return float(ll[~mask].sum())


# ------------------------------ diffusion call --------------------------------
def _base_sfs_theta1_from_scaled(
    log10_params: np.ndarray,
    *,
    demo_model_abs: Callable[[Dict[str, float]], Any],
    param_names: List[str],
    sampled_demes: List[str],
    haploid_sizes: List[int],
    folded: bool,
) -> moments.Spectrum:
    """
    Compute base SFS with theta=1 using a "shape-only" demography.

    Trick:
      convert scaled -> absolute with N_ANC=1.0 (so it’s pure shape),
      then compute SFS with theta=1.0.
    """
    vec_real = 10 ** log10_params
    p_scaled = build_scaled_param_dict(param_names, vec_real)

    # shape-only absolute params with N_ANC=1
    p_abs_shape = scaled_to_absolute_params(p_scaled, N_anc_abs=1.0, time_scale="2N")

    graph = demo_model_abs(p_abs_shape)

    fs = moments.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=1.0,
    )
    if folded:
        fs = fs.fold()
    return fs


# ------------------------------ main optimizer --------------------------------
def fit_model_realdata_scaled(
    *,
    sfs: moments.Spectrum,
    demo_model_abs: Callable[[Dict[str, float]], Any],
    experiment_config: Dict[str, Any],
    param_order: Optional[List[str]] = None,
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-12,
    save_dir: Optional[str | Path] = None,
    fixed_params: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float, float, float]:
    """
    Returns:
      (best_params_abs, ll_hat, theta_hat, N_ANC_implied)

    Optimizer runs in scaled space (ratios/tau/M).
    Returned best_params_abs are ABSOLUTE with N_ANC = implied N_ANC.
    fixed_params: scaled-space values to hold constant (lb=ub=value).
    """
    assert isinstance(sfs, moments.Spectrum)

    if param_order is None:
        param_order = list(experiment_config["parameter_order"])
    param_names = list(param_order)

    priors = experiment_config.get("_active_priors", None)
    if priors is None:
        priors = experiment_config.get("priors_real_data_analysis", None)
    if priors is None:
        priors = experiment_config.get("priors", experiment_config.get("parameters", {}))
    if not priors:
        raise ValueError("Real-data scaled inference needs scaled priors (prefer _active_priors or priors_real_data_analysis).")

    lb = np.array([float(priors[p][0]) for p in param_names], dtype=float)
    ub = np.array([float(priors[p][1]) for p in param_names], dtype=float)
    if fixed_params:
        for p, v in fixed_params.items():
            if p in param_names:
                i = param_names.index(p)
                lb[i] = ub[i] = float(v)
    if np.any(lb <= 0) or np.any(ub <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb, ub) if lo <= 0 or hi <= 0]
        raise ValueError(f"All scaled bounds must be > 0 for log10 optimization. Bad: {bad}")

    # start = LHS point for this run (diverse, well-spread across prior)
    x0 = lhs_start_log10(lb, ub, experiment_config)
    x0_real = 10 ** x0

    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("Observed SFS has no pop_ids; cannot infer sampled_demes order.")
    haploid_sizes = [n - 1 for n in sfs.shape]
    obs_folded = bool(getattr(sfs, "folded", False))

    # loglik in scaled space with theta profiled
    def loglikelihood(log10_params: np.ndarray) -> float:
        base = _base_sfs_theta1_from_scaled(
            log10_params,
            demo_model_abs=demo_model_abs,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            folded=obs_folded,
        )
        base_arr = np.asarray(base)
        theta_hat = _theta_hat_poisson_mle(sfs, base_arr)
        exp_arr = theta_hat * base_arr
        return _mask_safe_poisson_ll(sfs, exp_arr, eps=eps)

    # --- quick timing diagnostic (one eval) ---
    import time
    t0 = time.time()
    ll0 = loglikelihood(x0)
    dt = time.time() - t0
    print(f"[diagnostic real] one loglik eval took {dt:.3f}s  (ll={ll0:.6g})")

    grad_fn = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {log10_params}")
        return ll

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(param_names))
    opt.set_lower_bounds(np.log10(lb))
    opt.set_upper_bounds(np.log10(ub))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    xhat = opt.optimize(x0)
    ll_hat = float(loglikelihood(xhat))

    # compute theta_hat at optimum
    base_hat = _base_sfs_theta1_from_scaled(
        xhat,
        demo_model_abs=demo_model_abs,
        param_names=param_names,
        sampled_demes=sampled_demes,
        haploid_sizes=haploid_sizes,
        folded=obs_folded,
    )
    theta_hat = float(_theta_hat_poisson_mle(sfs, np.asarray(base_hat)))

    mu = float(experiment_config["mutation_rate"])
    L = float(experiment_config["genome_length"])
    muL = mu * L
    if muL <= 0:
        raise ValueError("mutation_rate * genome_length must be > 0 for implied N_ANC.")
    N_anc_implied = float(theta_hat) / float(4.0 * muL)

    # convert scaled params at optimum to ABSOLUTE using implied N_ANC
    p_scaled_hat = build_scaled_param_dict(param_names, 10 ** xhat)
    best_params_abs = scaled_to_absolute_params(
        p_scaled_hat,
        N_anc_abs=N_anc_implied,
        time_scale="2N",
    )

    # optional scaled profile likelihood curves
    if save_dir is not None and bool(experiment_config.get("generate_profiles", False)):
        like_dir = Path(save_dir) / "likelihood_plots_scaled"
        profiles = profile_1d(
            xhat_log10=xhat,
            param_names=param_names,
            lb_full=lb,
            ub_full=ub,
            loglikelihood_fn=loglikelihood,
            n_points=int(experiment_config.get("profile_points", 41)),
            widen=float(experiment_config.get("profile_widen", 0.5)),
        )
        save_profiles(
            profiles,
            like_dir,
            make_plots=bool(experiment_config.get("profile_make_plots", True)),
        )

    return best_params_abs, ll_hat, theta_hat, N_anc_implied

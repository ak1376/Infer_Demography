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


# ------------------------------ profiles (scaled space) ------------------------
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
        plt.title(f"Scaled profile likelihood: {p}")
        plt.tight_layout()
        plt.savefig(out_dir / f"profile_{p}.png", dpi=150)
        plt.close()


# ------------------------------ scaling logic --------------------------------
def _build_scaled_param_dict(param_names: List[str], vec_real: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(param_names, vec_real)}


def _scaled_to_absolute_params(
    p_scaled: Dict[str, float],
    *,
    N_anc_abs: float,
    time_scale: str = "2N",   # "2N" means T_abs = 2*N_ANC*tau
) -> Dict[str, float]:
    """
    Convert scaled ("shape") parameters to absolute params.

    Scaled interpretation:
      - N_ANC is a placeholder (ignored); absolute comes from theta
      - sizes (N_*) are ratios: N_*_abs = r_* * N_ANC_abs
      - time T is tau: T_abs = (2*N_ANC_abs)*tau
      - migrations m_* are M: m_abs = M / (2*N_ANC_abs)
    """
    if N_anc_abs <= 0:
        raise ValueError(f"N_anc_abs must be > 0; got {N_anc_abs}")

    out = dict(p_scaled)
    out["N_ANC"] = float(N_anc_abs)

    # sizes: treat any key starting with "N_" (except N_ANC) as ratio
    for k, v in list(out.items()):
        if k.startswith("N_") and k != "N_ANC":
            out[k] = float(v) * float(N_anc_abs)

    # time: T is tau
    if "T" in out:
        tau = float(out["T"])
        if time_scale == "2N":
            out["T"] = float(2.0 * N_anc_abs * tau)
        else:
            raise ValueError(f"Unknown time_scale={time_scale}")

    # migration: keys starting with "m_" are treated as M = 2Nanc*m
    for k, v in list(out.items()):
        if k.startswith("m_"):
            M = float(v)
            out[k] = float(M) / float(2.0 * N_anc_abs)

    return out


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
    p_scaled = _build_scaled_param_dict(param_names, vec_real)

    # shape-only absolute params with N_ANC=1
    p_abs_shape = _scaled_to_absolute_params(p_scaled, N_anc_abs=1.0, time_scale="2N")

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
) -> Tuple[Dict[str, float], float, float, float]:
    """
    Returns:
      (best_params_abs, ll_hat, theta_hat, N_ANC_implied)

    Optimizer runs in scaled space (ratios/tau/M).
    Returned best_params_abs are ABSOLUTE with N_ANC = implied N_ANC.
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
    if np.any(lb <= 0) or np.any(ub <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb, ub) if lo <= 0 or hi <= 0]
        raise ValueError(f"All scaled bounds must be > 0 for log10 optimization. Bad: {bad}")

    # start = geometric midpoint (+ optional jitter)
    x0_real = 10 ** ((np.log10(lb) + np.log10(ub)) / 2.0)
    sigma = float(experiment_config.get("start_jitter_sigma", 0.0))
    if sigma > 0:
        seed = experiment_config.get("opt_seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        x0_real = x0_real * (10 ** rng.normal(0.0, sigma, size=x0_real.shape))
    x0_real = np.clip(x0_real, lb, ub)
    x0 = np.log10(x0_real)

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

    opt = nlopt.opt(nlopt.LD_LBFGS, len(param_names))
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
    p_scaled_hat = _build_scaled_param_dict(param_names, 10 ** xhat)
    best_params_abs = _scaled_to_absolute_params(
        p_scaled_hat,
        N_anc_abs=N_anc_implied,
        time_scale="2N",
    )

    # optional scaled profile likelihood curves
    if save_dir is not None and bool(experiment_config.get("generate_profiles", False)):
        like_dir = Path(save_dir) / "likelihood_plots_scaled"
        profiles = _profile_1d(
            xhat_log10=xhat,
            param_names=param_names,
            lb_full=lb,
            ub_full=ub,
            loglikelihood_fn=loglikelihood,
            n_points=int(experiment_config.get("profile_points", 41)),
            widen=float(experiment_config.get("profile_widen", 0.5)),
        )
        _save_profiles(
            profiles,
            like_dir,
            make_plots=bool(experiment_config.get("profile_make_plots", True)),
        )

    return best_params_abs, ll_hat, theta_hat, N_anc_implied

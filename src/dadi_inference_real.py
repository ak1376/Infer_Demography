#!/usr/bin/env python3
"""
src/dadi_inference_real.py

Real-data dadi SFS inference where we:
- optimize demography "shape" in DIMENSIONLESS units:
    sizes: ratios r = N_X / N_ANC
    time:  tau = T / (2*N_ANC)
    migration: M = 2*N_ANC*m
- profile out theta analytically under Poisson composite likelihood
- convert theta_hat -> N_ANC_implied = theta_hat / (4*mu*L)
- convert all inferred params back to ABSOLUTE units using N_ANC_implied

Assumptions (same as moments_inference_real.py):
- param_order includes names your model expects, e.g.
    ['N_ANC','N_CO','N_FR0','N_FR1','T','m_CO_FR','m_FR_CO',...]
- In real-data mode, these names are interpreted as:
    N_ANC  : placeholder (usually bounds [1,1] in scaled priors)
    N_*    : ratios r = N_*/N_ANC
    T      : tau = T/(2*N_ANC)
    m_*    : M = 2*N_ANC*m

Priors:
- uses experiment_config["_active_priors"] if present
- else uses experiment_config["priors_real_data_analysis"]
- else falls back to experiment_config["priors"] / ["parameters"] (NOT recommended for real data)

Return:
    (best_params_abs: Dict[str,float], ll_hat: float, theta_hat: float, N_ANC_implied: float)

Notes:
- We compute a BASE dadi SFS with theta=1 by constructing a demography with N_ANC=1
  (shape-only), then profile theta using theta_hat = sum(obs)/sum(base) (mask-safe).
- Likelihood is Poisson composite: sum(obs*log(exp+eps) - exp) over unmasked entries.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import dadi
import nlopt
import numdifftools as nd

from src.inference_utils import (
    build_scaled_param_dict,
    scaled_to_absolute_params,
    lhs_start_log10,
    profile_1d,
    save_profiles,
)


# ------------------------------ mask-safe helpers ------------------------------
def _mask_safe_sum(sfs: dadi.Spectrum, arr: np.ndarray) -> float:
    mask = getattr(sfs, "mask", None)
    if mask is None:
        return float(np.asarray(arr).sum())
    mask = np.asarray(mask, dtype=bool)
    return float(np.asarray(arr)[~mask].sum())


def _theta_hat_poisson_mle(sfs: dadi.Spectrum, base_sfs: np.ndarray) -> float:
    s_obs = _mask_safe_sum(sfs, np.asarray(sfs))
    s_base = _mask_safe_sum(sfs, np.asarray(base_sfs))
    if s_base <= 0:
        return 0.0
    return s_obs / s_base


def _mask_safe_poisson_ll(sfs: dadi.Spectrum, exp_sfs: np.ndarray, eps: float) -> float:
    obs = np.asarray(sfs)
    exp = np.asarray(exp_sfs)
    ll = obs * np.log(exp + eps) - exp
    mask = getattr(sfs, "mask", None)
    if mask is None:
        return float(ll.sum())
    mask = np.asarray(mask, dtype=bool)
    return float(ll[~mask].sum())


# ------------------------------ dadi expected SFS ------------------------------
def _choose_pts_l(sfs: dadi.Spectrum, experiment_config: Dict[str, Any]) -> List[int]:
    # Allow explicit override
    pts_l_cfg = experiment_config.get("pts_l", None)
    if pts_l_cfg is not None:
        pts_l = [int(x) for x in pts_l_cfg]
        if len(pts_l) < 1:
            raise ValueError("experiment_config['pts_l'] must have >= 1 entries.")
        return pts_l

    # Default heuristic (same spirit as your simulation dadi_inference.py)
    pts_base = experiment_config.get("pts_base", [40, 50, 60])
    pts_base = [int(x) for x in pts_base]

    diploid_ns = [(dim - 1) // 2 for dim in sfs.shape]
    n_max_hap = max(2 * n for n in diploid_ns)
    return [int(n_max_hap + p) for p in pts_base]


def _base_sfs_theta1_from_scaled(
    log10_params: np.ndarray,
    *,
    demo_model_abs: Callable[[Dict[str, float]], Any],
    param_names: List[str],
    sampled_demes: List[str],
    ns_haploid: Tuple[int, ...],
    pts_l: List[int],
    folded: bool,
) -> dadi.Spectrum:
    """
    Compute base SFS with theta=1 using a "shape-only" demography.

    Trick:
      convert scaled -> absolute with N_ANC=1.0 (shape-only),
      then compute dadi SFS with theta=1.0 (i.e., DO NOT multiply by theta).
    """
    vec_real = 10**log10_params
    p_scaled = build_scaled_param_dict(param_names, vec_real)

    # shape-only absolute params with N_ANC=1
    p_abs_shape = scaled_to_absolute_params(p_scaled, N_anc_abs=1.0, time_scale="2N")
    graph = demo_model_abs(p_abs_shape)

    fs = dadi.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=list(ns_haploid),
        pts=pts_l,
    )

    if folded:
        fs = fs.fold()
    return fs


# ------------------------------ main optimizer --------------------------------
def fit_model_realdata_scaled(
    *,
    sfs: dadi.Spectrum,
    demo_model_abs: Callable[[Dict[str, float]], Any],
    experiment_config: Dict[str, Any],
    param_order: Optional[List[str]] = None,
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-12,
    save_dir: Optional[Path] = None,
    fixed_params: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float, float, float]:
    """
    Returns:
      (best_params_abs, ll_hat, theta_hat, N_ANC_implied)

    Optimizer runs in scaled space (ratios/tau/M).
    Returned best_params_abs are ABSOLUTE with N_ANC = implied N_ANC.
    """
    assert isinstance(sfs, dadi.Spectrum)

    # ---- optional GPU toggle ----
    use_gpu = bool(experiment_config.get("use_gpu_dadi", False))
    dadi.cuda_enabled(use_gpu)

    if param_order is None:
        param_order = list(experiment_config["parameter_order"])
    param_names = list(param_order)

    priors = experiment_config.get("_active_priors", None)
    if priors is None:
        priors = experiment_config.get("priors_real_data_analysis", None)
    if priors is None:
        priors = experiment_config.get(
            "priors", experiment_config.get("parameters", {})
        )
    if not priors:
        raise ValueError(
            "Real-data scaled inference needs scaled priors (prefer _active_priors or priors_real_data_analysis)."
        )

    lb = np.array([float(priors[p][0]) for p in param_names], dtype=float)
    ub = np.array([float(priors[p][1]) for p in param_names], dtype=float)
    if fixed_params:
        for p, v in fixed_params.items():
            if p in param_names:
                i = param_names.index(p)
                lb[i] = ub[i] = float(v)
    if np.any(lb <= 0) or np.any(ub <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb, ub) if lo <= 0 or hi <= 0]
        raise ValueError(
            f"All scaled bounds must be > 0 for log10 optimization. Bad: {bad}"
        )

    # start = LHS point for this run (diverse, well-spread across prior)
    x0 = lhs_start_log10(lb, ub, experiment_config)
    x0_real = 10**x0

    sampled_demes = list(getattr(sfs, "pop_ids", []) or [])
    if not sampled_demes:
        raise ValueError(
            "Observed dadi SFS has no pop_ids; cannot infer sampled_demes order."
        )

    # dadi uses haploid ns = dim - 1
    ns_haploid = tuple(int(dim - 1) for dim in sfs.shape)
    obs_folded = bool(getattr(sfs, "folded", False))

    # pts grid
    pts_l = _choose_pts_l(sfs, experiment_config)

    # Extrapolation wrapper (like your sim code)
    def _raw_wrapper(
        params_vec: np.ndarray, ns_local: Tuple[int, ...], pts: List[int]
    ) -> dadi.Spectrum:
        # params_vec is in REAL (scaled) space already here for base; inside wrapper we are called by extrap func
        p_scaled = build_scaled_param_dict(param_names, np.asarray(params_vec, float))

        # shape-only absolute params with N_ANC=1
        p_abs_shape = scaled_to_absolute_params(
            p_scaled, N_anc_abs=1.0, time_scale="2N"
        )
        graph = demo_model_abs(p_abs_shape)

        fs = dadi.Spectrum.from_demes(
            graph,
            sampled_demes=sampled_demes,
            sample_sizes=list(ns_local),
            pts=pts,
        )
        if obs_folded:
            fs = fs.fold()
        return fs

    func_ex = dadi.Numerics.make_extrap_func(_raw_wrapper)

    BAD_LL = -1e300

    # loglik in scaled space with theta profiled
    last_eval: Dict[str, Any] = {"log10_params": None, "ll": None}

    def loglikelihood(log10_params: np.ndarray) -> float:
        last_eval["log10_params"] = np.array(log10_params, copy=True)

        # enforce bounds
        log10_params = np.clip(log10_params, np.log10(lb), np.log10(ub))
        params_real = 10 ** np.asarray(log10_params, float)

        try:
            base = func_ex(params_real, ns_haploid, pts_l)
            base_arr = np.asarray(base)

            theta_hat = _theta_hat_poisson_mle(sfs, base_arr)
            exp_arr = theta_hat * base_arr
            ll = _mask_safe_poisson_ll(sfs, exp_arr, eps=eps)

            if not np.isfinite(ll):
                last_eval["ll"] = BAD_LL
                return BAD_LL

            last_eval["ll"] = float(ll)
            return float(ll)
        except Exception as e:
            if verbose:
                print("DADI(real) FAIL at", log10_params, "err:", repr(e))
            last_eval["ll"] = BAD_LL
            return BAD_LL

    # --- quick timing diagnostic (one eval) ---
    import time

    t0 = time.time()
    ll0 = loglikelihood(x0)
    dt = time.time() - t0
    print(
        f"[diagnostic dadi real] one loglik eval took {dt:.3f}s  (ll={ll0:.6g})  pts_l={pts_l}  ns={ns_haploid}"
    )

    # gradient
    step = float(experiment_config.get("dadi_grad_step", 3e-4))
    grad_fn = nd.Gradient(loglikelihood, step=step)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            if ll <= BAD_LL / 10:
                grad[:] = 0.0
            else:
                g = grad_fn(log10_params)
                grad[:] = g if np.all(np.isfinite(g)) else 0.0
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {np.asarray(log10_params)}")
        return float(ll)

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(param_names))
    opt.set_lower_bounds(np.log10(lb))
    opt.set_upper_bounds(np.log10(ub))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)
    opt.set_maxeval(int(experiment_config.get("dadi_maxeval", 500)))

    # Optional fallback (mirrors your sim code style)
    debug_txt: Optional[str] = None

    def _build_debug_txt(tag: str) -> str:
        lines: List[str] = []
        lines.append(f"tag = {tag}")
        lines.append(f"loglik = {last_eval.get('ll', None)}")
        lines.append(f"pts_l = {pts_l}")
        lines.append(f"ns_haploid = {ns_haploid}")
        lines.append(f"folded_obs = {obs_folded}")

        log10_p = last_eval.get("log10_params", None)
        if log10_p is None:
            lines.append("log10_params = None")
            return "\n".join(lines) + "\n"

        real_p = 10 ** np.asarray(log10_p, float)
        lines.append(f"log10_params = {np.asarray(log10_p, float).tolist()}")
        lines.append(f"real_params = {np.asarray(real_p, float).tolist()}")

        try:
            _ = func_ex(real_p, ns_haploid, pts_l)
            lines.append("base_sfs_recomputed = True")
        except Exception as e:
            lines.append("base_sfs_recomputed = False")
            lines.append(f"base_sfs_error = {repr(e)}")

        return "\n".join(lines) + "\n"

    try:
        xhat = opt.optimize(x0)
    except nlopt.runtime_error:
        print(
            "[DADI REAL DEBUG] LD_LBFGS runtime_error; capturing debug state and falling back to LN_COBYLA"
        )
        debug_txt = _build_debug_txt("lbfgs_runtime_error")

        x_start = last_eval.get("log10_params", None)
        if x_start is None:
            x_start = x0

        opt_fb = nlopt.opt(nlopt.LN_COBYLA, len(param_names))
        opt_fb.set_lower_bounds(np.log10(lb))
        opt_fb.set_upper_bounds(np.log10(ub))
        opt_fb.set_ftol_rel(rtol)
        opt_fb.set_maxeval(int(experiment_config.get("dadi_maxeval_cobyla", 2000)))
        opt_fb.set_max_objective(objective)

        try:
            xhat = opt_fb.optimize(np.asarray(x_start, float))
        except nlopt.runtime_error:
            print(
                "[DADI REAL DEBUG] LN_COBYLA runtime_error too; capturing debug state and re-raising"
            )
            debug_txt = _build_debug_txt("cobyla_runtime_error")
            raise

    ll_hat = float(loglikelihood(xhat))

    # theta_hat at optimum
    params_hat_real = 10 ** np.asarray(xhat, float)
    base_hat = func_ex(params_hat_real, ns_haploid, pts_l)
    theta_hat = float(_theta_hat_poisson_mle(sfs, np.asarray(base_hat)))

    mu = float(experiment_config["mutation_rate"])
    L = float(experiment_config["genome_length"])
    muL = mu * L
    if muL <= 0:
        raise ValueError("mutation_rate * genome_length must be > 0 for implied N_ANC.")
    N_anc_implied = float(theta_hat) / float(4.0 * muL)

    # convert scaled params at optimum to ABSOLUTE using implied N_ANC
    p_scaled_hat = build_scaled_param_dict(param_names, params_hat_real)
    best_params_abs = scaled_to_absolute_params(
        p_scaled_hat,
        N_anc_abs=N_anc_implied,
        time_scale="2N",
    )

    if use_gpu:
        dadi.cuda_enabled(False)

    # You can optionally return debug_txt by stashing into best_params_abs or separately;
    # runner below saves it as "debug_txt" in best_fit.pkl.
    if debug_txt is not None:
        best_params_abs["_debug_txt_present"] = 1.0  # harmless marker if you like

    # optional profile likelihood curves
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

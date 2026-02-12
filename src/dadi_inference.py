#!/usr/bin/env python3
"""
dadi_inference.py – single-run dadi optimisation (ALL params free)

Matches the structure + optimization logic of moments_inference.py:
- log10-space optimization with NLopt LD_LBFGS
- numdifftools gradient
- mask-safe Poisson composite LL: sum( sfs*log(model+eps) - model )
- uses experiment_config["parameter_order"] and priors just like moments_inference
- preserves deme labels/order via sfs.pop_ids

Dadi-specific:
- expected SFS computed with dadi.Spectrum.from_demes(graph, ..., pts=pts_l)
- extrapolation via dadi.Numerics.make_extrap_func
- dynamic pts_l grid based on sample sizes + config pts offsets
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import dadi
import nlopt
import numdifftools as nd


# ───────────────────────── dadi expected SFS helper ────────────────────────────
def _dadi_expected_sfs(
    log_space_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    param_names: List[str],
    sampled_demes: List[str],
    ns: Tuple[int, ...],                 # haploid sample sizes for dadi
    pts_l: List[int],                    # dadi integration grid (3 values)
    experiment_config: Dict[str, Any],
):
    """
    Build expected SFS using dadi from a demes graph.

    Returns expected *counts* (theta-scaled): dadi_model * theta
    where theta = 4 * N0 * mu * L and N0 is param_names[0].
    """
    real_space_vec = 10 ** log_space_vec
    p_dict = {k: float(v) for k, v in zip(param_names, real_space_vec)}

    graph = demo_model(p_dict)

    # Unscaled dadi Spectrum
    model_unscaled = dadi.Spectrum.from_demes(
        graph,
        sample_sizes=list(ns),          # haploid ns
        sampled_demes=sampled_demes,    # preserve labels/order
        pts=pts_l,
    )

    muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
    N0 = float(p_dict[param_names[0]])
    theta = 4.0 * N0 * muL

    return model_unscaled * theta


# ───────────────────────── optimisation wrapper ────────────────────────
def fit_model(
    sfs: dadi.Spectrum,
    start_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    experiment_config: Dict[str, Any],
    param_order: Optional[List[str]] = None,
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """
    Mirrors moments_inference.fit_model signature and optimization behavior.

    Returns:
      fitted_real (np.ndarray), ll_hat (float)
    """
    # assert isinstance(sfs, dadi.Spectrum), "sfs must be a dadi.Spectrum"

    # ---- optional GPU toggle (kept minimal) ----
    use_gpu = bool(experiment_config.get("use_gpu_dadi", False))
    dadi.cuda_enabled(use_gpu)

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

    # ---- deme labels/order from SFS ----
    sampled_demes = list(getattr(sfs, "pop_ids", []) or [])
    if not sampled_demes:
        raise ValueError("Observed dadi SFS has no pop_ids; cannot infer sampled_demes order.")

    # dadi uses haploid ns = dim-1
    ns = tuple(int(dim - 1) for dim in sfs.shape)

    # ---- dadi pts grid (dynamic like your older script) ----
    # Your config had experiment_config["optimization"]["dadi"]["pts"] = [a,b,c] offsets
    # pts_base = experiment_config["optimization"]["dadi"]["pts"]
    pts_base = [10,20,30]

    # replicate your prior dynamic rule:
    # compute diploid n_i = (dim-1)//2, then n_max_hap = max(2*n_i) which equals max(dim-1) (for typical even dims)
    diploid_ns = [(dim - 1) // 2 for dim in sfs.shape]
    n_max_hap = max(2 * n for n in diploid_ns)
    pts_l = [int(n_max_hap + pts_base[0]), int(n_max_hap + pts_base[1]), int(n_max_hap + pts_base[2])]

    # ---- extrapolation wrapper ----
    def raw_wrapper(params_vec, ns_local, pts):
        # params_vec is real-space (not log space)
        # we want the same expected SFS as _dadi_expected_sfs does
        # but dadi.make_extrap_func expects this signature
        # NOTE: eps handled in likelihood, not here
        p_dict = {k: float(v) for k, v in zip(param_names, params_vec)}
        graph = demo_model(p_dict)

        model_unscaled = dadi.Spectrum.from_demes(
            graph,
            sample_sizes=list(ns_local),
            sampled_demes=sampled_demes,
            pts=pts,
        )

        muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
        N0 = float(p_dict[param_names[0]])
        theta = 4.0 * N0 * muL
        return model_unscaled * theta

    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    # ---- optimizer setup (IDENTICAL pattern to moments_inference) ----
    start_vec = np.asarray(start_vec, dtype=float)
    if start_vec.shape != (len(param_names),):
        raise ValueError(f"start_vec shape {start_vec.shape} != ({len(param_names)},)")

    # log10 bounds + start
    x0 = np.log10(start_vec)

    opt = nlopt.opt(nlopt.LD_LBFGS, start_vec.size)
    opt.set_lower_bounds(np.log10(lb_full))
    opt.set_upper_bounds(np.log10(ub_full))
    opt.set_ftol_rel(rtol)

    # ---- Poisson composite log-likelihood (MASK SAFE) ----
    def loglikelihood(log10_params: np.ndarray) -> float:
        params = 10.0 ** np.asarray(log10_params, float)
        try:
            model = func_ex(params, ns, pts_l)  # dadi.Spectrum expected counts
            if not isinstance(model, dadi.Spectrum):
                model = dadi.Spectrum(model)

            # mask-safe Spectrum arithmetic; avoid log(0) via +eps
            return float((sfs * np.log(model + eps) - model).sum())
        except Exception:
            return -np.inf

    grad_fn = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {log10_params}")
        return ll

    opt.set_max_objective(objective)

    # ---- run optimization ----
    xhat = opt.optimize(x0)
    ll_hat = loglikelihood(xhat)
    fitted_real = 10 ** xhat

    # ---- optional save (same as moments_inference) ----
    out_dir = experiment_config.get("out_dir", None)
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fitted_sfs = func_ex(fitted_real, ns, pts_l)
        if not isinstance(fitted_sfs, dadi.Spectrum):
            fitted_sfs = dadi.Spectrum(fitted_sfs)

        np.save(out_path / "expected_sfs_fitted.npy", np.asarray(fitted_sfs))

        fitted_dict = {p: float(v) for p, v in zip(param_names, fitted_real)}
        (out_path / "fitted_params.json").write_text(
            __import__("json").dumps(fitted_dict, indent=2, sort_keys=False)
        )

    # ---- GPU cleanup ----
    if use_gpu:
        dadi.cuda_enabled(False)

    return fitted_real, ll_hat

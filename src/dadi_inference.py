#!/usr/bin/env python3
"""
dadi_inference.py – single-run dadi optimisation (ALL params free)

Robust + pipeline-friendly:
- Never returns ±inf/NaN to NLopt (finite BAD_LL penalty)
- Guards numdifftools gradients so LD_LBFGS can't crash from NaN grads
- Tracks last evaluated parameters + LL
- On nlopt.runtime_error: dumps observed/expected SFS + params
- Fallback: if LD_LBFGS runtime_error → run LN_COBYLA starting from last eval (or x0)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import dadi
import nlopt
import numdifftools as nd


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
    Mirrors moments_inference.fit_model signature/behavior.

    Returns:
      fitted_real (np.ndarray), ll_hat (float)
    """

    # ---- optional GPU toggle ----
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

    # ---- deme labels ----
    sampled_demes = list(getattr(sfs, "pop_ids", []) or [])
    if not sampled_demes:
        raise ValueError("Observed dadi SFS has no pop_ids; cannot infer sampled_demes order.")

    # dadi uses haploid ns = dim - 1
    ns = tuple(int(dim - 1) for dim in sfs.shape)

    # ---- dadi pts grid ----
    # (kept as your current heuristic)
    pts_base = [10, 20, 30]
    diploid_ns = [(dim - 1) // 2 for dim in sfs.shape]
    n_max_hap = max(2 * n for n in diploid_ns)
    pts_l = [int(n_max_hap + p) for p in pts_base]

    # ---- extrapolation wrapper ----
    def raw_wrapper(params_vec, ns_local, pts):
        p_dict = dict(zip(param_names, params_vec))
        graph = demo_model(p_dict)

        fs = dadi.Spectrum.from_demes(
            graph,
            sample_sizes=list(ns_local),
            sampled_demes=sampled_demes,
            pts=pts,
        )

        muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
        theta = 4.0 * float(p_dict[param_names[0]]) * muL
        return fs * theta

    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    # ---- optimizer setup ----
    start_vec = np.asarray(start_vec, dtype=float)
    if start_vec.shape != (len(param_names),):
        raise ValueError(f"start_vec shape {start_vec.shape} != ({len(param_names)},)")
    start_vec = np.minimum(np.maximum(start_vec, lb_full), ub_full)
    x0 = np.log10(start_vec)

    # ---- constants / debug tracker ----
    BAD_LL = -1e300
    last_eval: Dict[str, Any] = {"log10_params": None, "ll": None}

    # ---- likelihood (keeps Spectrum/masked ops exactly) ----
    def loglikelihood(log10_params):
        last_eval["log10_params"] = np.array(log10_params, copy=True)

        log10_params = np.clip(log10_params, np.log10(lb_full), np.log10(ub_full))
        params = 10.0 ** np.asarray(log10_params, float)

        try:
            model = func_ex(params, ns, pts_l)
            ll = float((sfs * np.log(model + eps) - model).sum())
            if not np.isfinite(ll):
                last_eval["ll"] = BAD_LL
                return BAD_LL
            last_eval["ll"] = ll
            return ll
        except Exception as e:
            if verbose:
                print("DADI FAIL at", log10_params, "err:", repr(e))
            last_eval["ll"] = BAD_LL
            return BAD_LL

    # keep your step (you changed to 3e-4)
    grad_fn = nd.Gradient(loglikelihood, step=3e-4)

    def objective(log10_params, grad):
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            if ll <= BAD_LL / 10:
                grad[:] = 0.0
            else:
                g = grad_fn(log10_params)
                grad[:] = g if np.all(np.isfinite(g)) else 0.0
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {np.asarray(log10_params)}")
        return ll

    # ---- debug dump helper ----
    def _dump_debug(tag: str) -> None:
        dbg_root = Path(experiment_config.get("out_dir", "dadi_debugging"))
        dbg_dir = dbg_root / "dadi_runtime_debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        log10_p = last_eval.get("log10_params", None)
        if log10_p is None:
            (dbg_dir / f"{tag}_info.txt").write_text("No last_eval log10_params captured.\n")
            return

        real_p = 10 ** np.asarray(log10_p, float)

        # save observed SFS
        np.save(dbg_dir / f"{tag}_observed_sfs.npy", np.asarray(sfs))
        np.save(dbg_dir / f"{tag}_log10_params.npy", np.asarray(log10_p, float))
        np.save(dbg_dir / f"{tag}_real_params.npy", np.asarray(real_p, float))

        # recompute + save expected SFS at last params
        expected_ok = True
        try:
            expected = func_ex(real_p, ns, pts_l)
            np.save(dbg_dir / f"{tag}_expected_sfs.npy", np.asarray(expected))
        except Exception as e:
            expected_ok = False
            if verbose:
                print(f"Failed to recompute expected SFS for debug dump ({tag}):", repr(e))

        (dbg_dir / f"{tag}_info.txt").write_text(
            f"tag = {tag}\n"
            f"loglik = {last_eval.get('ll', None)}\n"
            f"log10_params = {np.asarray(log10_p, float).tolist()}\n"
            f"real_params = {np.asarray(real_p, float).tolist()}\n"
            f"pts_l = {pts_l}\n"
            f"ns = {ns}\n"
            f"expected_saved = {expected_ok}\n"
        )

    # ---- optimizer factory ----
    def _make_opt(algorithm: int) -> nlopt.opt:
        o = nlopt.opt(algorithm, x0.size)
        o.set_lower_bounds(np.log10(lb_full))
        o.set_upper_bounds(np.log10(ub_full))
        o.set_ftol_rel(rtol)

        # keep bounded runtime in pipelines (optional, but recommended)
        maxeval = int(experiment_config.get("dadi_maxeval", 500))
        o.set_maxeval(maxeval)

        o.set_max_objective(objective)
        return o

    # ---- run optimization: LBFGS then fallback COBYLA ----
    try:
        opt = _make_opt(nlopt.LD_LBFGS)
        xhat = opt.optimize(x0)

    except nlopt.runtime_error:
        print("[DADI DEBUG] LD_LBFGS runtime_error; dumping state and falling back to LN_COBYLA")
        _dump_debug("lbfgs_runtime_error")

        x_start = last_eval.get("log10_params", None)
        if x_start is None:
            x_start = x0

        # COBYLA ignores grad and is much harder to crash
        opt_fb = _make_opt(nlopt.LN_COBYLA)
        try:
            xhat = opt_fb.optimize(np.asarray(x_start, float))
        except nlopt.runtime_error:
            print("[DADI DEBUG] LN_COBYLA runtime_error too; dumping state and re-raising")
            _dump_debug("cobyla_runtime_error")
            raise

    # ---- finalize ----
    ll_hat = loglikelihood(xhat)
    fitted_real = 10 ** np.asarray(xhat, float)

    # ---- cleanup ----
    if use_gpu:
        dadi.cuda_enabled(False)

    return fitted_real, ll_hat

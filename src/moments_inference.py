#!/usr/bin/env python3
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
):
    real_space_vec = 10 ** log_space_vec
    p_dict = {k: float(v) for k, v in zip(param_names, real_space_vec)}

    graph = demo_model(p_dict)

    muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
    N0 = float(p_dict[param_names[0]])
    theta = 4.0 * N0 * muL

    fs = moments.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=theta,
    )

    if return_graph:
        return fs, graph, p_dict, theta, muL
    return fs


# ───────────────────────── optimisation wrapper ────────────────────────
def fit_model(
    sfs: moments.Spectrum,
    start_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    experiment_config: Dict[str, Any],
    param_order: Optional[List[str]] = None,
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], List[float]]:

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

    # ---- SFS demes order ----
    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("Observed SFS has no pop_ids; cannot infer sampled_demes order.")

    haploid_sizes = [n - 1 for n in sfs.shape]

    # ---- Poisson composite log-likelihood (MASK SAFE) ----
    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = _diffusion_sfs(
            log_space_vec=log10_params,
            demo_model=demo_model,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            experiment_config=experiment_config,
        )
        # IMPORTANT: keep Spectrum arithmetic (no np.asarray)
        return float((np.log(exp_sfs + eps) * sfs - exp_sfs).sum())

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

    # ---- optional save ----
    out_dir = experiment_config.get("out_dir", None)
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fitted_sfs = _diffusion_sfs(
            log_space_vec=xhat,
            demo_model=demo_model,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            experiment_config=experiment_config,
        )

        np.save(out_path / "expected_sfs_fitted.npy", np.asarray(fitted_sfs))

        fitted_dict = {p: float(v) for p, v in zip(param_names, fitted_real)}
        (out_path / "fitted_params.json").write_text(
            __import__("json").dumps(fitted_dict, indent=2, sort_keys=False)
        )

    return fitted_real, ll_hat

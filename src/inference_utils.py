#!/usr/bin/env python3
"""
inference_utils.py - Shared utilities for parameter fixing and optimization
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np


def lhs_start_log10(
    lb: np.ndarray,
    ub: np.ndarray,
    experiment_config: Dict[str, Any],
) -> np.ndarray:
    """
    Return a starting point in log10 space using Latin Hypercube Sampling.

    Generates a full LHS grid of shape (n_runs, n_params) once, keyed by the
    global config seed, so every run sees the same grid and each picks a
    different, well-spread row.  Fixed parameters (lb == ub) are left at their
    fixed log10 value regardless of the LHS draw.

    Parameters
    ----------
    lb, ub : bounds arrays (real, positive)
    experiment_config : must contain
        "seed"             – global RNG seed for reproducible grid
        "opt_seed"         – run index (0-based), set by Snakemake wildcard
        "num_optimizations"– total number of parallel runs (rows in LHS grid)
    """
    from scipy.stats.qmc import LatinHypercube

    lb_log10 = np.log10(lb)
    ub_log10 = np.log10(ub)

    n_runs = int(experiment_config.get("num_optimizations", 5))
    global_seed = int(experiment_config.get("seed", 42))
    run_idx = int(experiment_config.get("opt_seed", 0))
    n_params = len(lb)

    sampler = LatinHypercube(d=n_params, seed=global_seed)
    unit_samples = sampler.random(n=n_runs)  # (n_runs, n_params) in [0, 1]

    # Map [0, 1] -> [lb_log10, ub_log10]
    log10_starts = lb_log10 + unit_samples * (ub_log10 - lb_log10)

    # Fixed params (lb == ub): clamp to the fixed value so LHS doesn't perturb them
    fixed_mask = lb == ub
    log10_starts[:, fixed_mask] = lb_log10[fixed_mask]

    x0 = log10_starts[run_idx % n_runs]
    return np.clip(x0, lb_log10, ub_log10)


def build_scaled_param_dict(
    param_names: List[str], vec_real: np.ndarray
) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(param_names, vec_real)}


def scaled_to_absolute_params(
    p_scaled: Dict[str, float],
    *,
    N_anc_abs: float,
    time_scale: str = "2N",
) -> Dict[str, float]:
    """
    Convert scaled ("shape") parameters to absolute params.

    Scaled interpretation:
      - N_ANC is a placeholder (ignored); absolute comes from theta or is passed in
      - sizes (N_*) are ratios: N_*_abs = r_* * N_ANC_abs
      - time T / T_* are tau: T_abs = 2 * N_ANC_abs * tau
      - migrations m_* are M = 2*N*m: m_abs = M / (2 * N_ANC_abs)
    """
    if N_anc_abs <= 0:
        raise ValueError(f"N_anc_abs must be > 0; got {N_anc_abs}")

    out = dict(p_scaled)
    out["N_ANC"] = float(N_anc_abs)

    for k, v in list(out.items()):
        if k.startswith("N_") and k != "N_ANC":
            out[k] = float(v) * float(N_anc_abs)

    for k, v in list(out.items()):
        if k == "T" or k.startswith("T_"):
            if time_scale == "2N":
                out[k] = float(2.0 * N_anc_abs * float(v))
            else:
                raise ValueError(f"Unknown time_scale={time_scale}")

    for k, v in list(out.items()):
        if k.startswith("m_"):
            out[k] = float(v) / float(2.0 * N_anc_abs)

    return out


def absolute_to_scaled_params(
    p_abs: Dict[str, float],
    *,
    N_anc_abs: float,
    time_scale: str = "2N",
) -> Dict[str, float]:
    """
    Convert absolute demographic params to scaled ("shape") params.

    Inverse of scaled_to_absolute_params:
      N_ANC  → kept as absolute (anchor for rho in MomentsLD)
      N_*    → ratio r = N_*/N_ANC_abs
      T / T_*→ dimensionless tau = T / (2*N_ANC_abs)
      m_*    → dimensionless M = 2*N_ANC_abs*m
    """
    if N_anc_abs <= 0:
        raise ValueError(f"N_anc_abs must be > 0; got {N_anc_abs}")

    out = dict(p_abs)
    out["N_ANC"] = float(N_anc_abs)

    for k, v in list(out.items()):
        if k.startswith("N_") and k != "N_ANC":
            out[k] = float(v) / float(N_anc_abs)

    for k, v in list(out.items()):
        if k == "T" or k.startswith("T_"):
            if time_scale == "2N":
                out[k] = float(v) / (2.0 * float(N_anc_abs))
            else:
                raise ValueError(f"Unknown time_scale={time_scale}")

    for k, v in list(out.items()):
        if k.startswith("m_"):
            out[k] = float(v) * 2.0 * float(N_anc_abs)

    return out


def build_fixed_param_mapper(
    param_names: List[str], fixed_params: Dict[str, float]
) -> Tuple[List[int], List[int], Dict[str, float]]:
    """
    Create mappings for handling fixed parameters in optimization.

    Returns:
        free_indices: Indices of parameters that will be optimized
        fixed_indices: Indices of parameters that are fixed
        fixed_values_dict: Dictionary of fixed parameter values
    """
    fixed_indices = [i for i, name in enumerate(param_names) if name in fixed_params]
    free_indices = [i for i, name in enumerate(param_names) if name not in fixed_params]

    # Validate fixed parameters
    for param_name, value in fixed_params.items():
        if param_name not in param_names:
            raise ValueError(
                f"Fixed parameter '{param_name}' not found in model parameters: {param_names}"
            )

    return free_indices, fixed_indices, fixed_params


def create_param_vector_with_fixed(
    free_params: np.ndarray,
    param_names: List[str],
    fixed_params: Dict[str, float],
    free_indices: List[int],
) -> np.ndarray:
    """
    Reconstruct full parameter vector from free parameters and fixed values.
    """
    full_params = np.zeros(len(param_names))

    # Fill in free parameters
    for i, free_idx in enumerate(free_indices):
        full_params[free_idx] = free_params[i]

    # Fill in fixed parameters
    for i, param_name in enumerate(param_names):
        if param_name in fixed_params:
            full_params[i] = fixed_params[param_name]

    return full_params


def validate_fixed_params_bounds(
    fixed_params: Dict[str, float],
    param_names: List[str],
    lower_bounds: List[float],
    upper_bounds: List[float],
) -> None:
    """
    Validate that fixed parameter values are within specified bounds.
    """
    for param_name, value in fixed_params.items():
        if param_name in param_names:
            idx = param_names.index(param_name)
            if not (lower_bounds[idx] <= value <= upper_bounds[idx]):
                raise ValueError(
                    f"Fixed parameter {param_name}={value} is outside bounds "
                    f"[{lower_bounds[idx]}, {upper_bounds[idx]}]"
                )


def extract_free_bounds(
    param_names: List[str],
    lower_bounds: List[float],
    upper_bounds: List[float],
    fixed_params: Dict[str, float],
) -> Tuple[List[float], List[float]]:
    """
    Extract bounds for only the free (non-fixed) parameters.
    """
    free_lower = []
    free_upper = []

    for i, param_name in enumerate(param_names):
        if param_name not in fixed_params:
            free_lower.append(lower_bounds[i])
            free_upper.append(upper_bounds[i])

    return free_lower, free_upper


def create_fixed_params_list_for_moments(
    param_names: List[str], fixed_params: Dict[str, float]
) -> List[float | None]:
    """
    Create fixed_params list in the format expected by moments.Inference.optimize_*

    Returns a list where:
    - Fixed parameters have their values
    - Free parameters have None
    """
    fixed_list = []
    for param_name in param_names:
        if param_name in fixed_params:
            fixed_list.append(fixed_params[param_name])
        else:
            fixed_list.append(None)

    return fixed_list


def create_fixed_params_list_for_dadi(
    param_names: List[str], fixed_params: Dict[str, float]
) -> List[int | None]:
    """
    Create fixed_params list in the format expected by dadi.Inference.opt

    Returns a list where:
    - Fixed parameters have their indices
    - Free parameters have None
    """
    fixed_list = []
    for i, param_name in enumerate(param_names):
        if param_name in fixed_params:
            fixed_list.append(i)
        else:
            fixed_list.append(None)

    return fixed_list


def profile_1d(
    *,
    xhat_log10: np.ndarray,
    param_names: List[str],
    lb_full: np.ndarray,
    ub_full: np.ndarray,
    loglikelihood_fn,
    n_points: int = 41,
    widen: float = 0.5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute 1-D profile likelihood slices around the MLE in log10 space."""
    lb_log10 = np.log10(lb_full)
    ub_log10 = np.log10(ub_full)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i, p in enumerate(param_names):
        lo, hi = lb_log10[i], ub_log10[i]
        if widen > 0:
            span = hi - lo
            lo = max(lo, xhat_log10[i] - widen * span)
            hi = min(hi, xhat_log10[i] + widen * span)
        grid = np.linspace(lo, hi, int(n_points))
        ll = np.empty_like(grid)
        x = xhat_log10.copy()
        for k, g in enumerate(grid):
            x[i] = g
            ll[k] = loglikelihood_fn(x)
        out[p] = {"grid_log10": grid, "ll": ll, "xhat_log10": xhat_log10[i]}
    return out


def save_profiles(
    profiles: Dict[str, Dict[str, np.ndarray]],
    out_dir,
    *,
    make_plots: bool = True,
    title_prefix: str = "Scaled profile likelihood",
) -> None:
    """Save profile likelihood npz files and optional PNG plots."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, d in profiles.items():
        np.savez(out_dir / f"profile_{p}.npz", grid_log10=d["grid_log10"], ll=d["ll"])
    if not make_plots:
        return
    import matplotlib.pyplot as plt

    for p, d in profiles.items():
        ll_max = float(np.max(d["ll"]))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(10 ** d["grid_log10"], ll_max - d["ll"])
        ax.axvline(
            10 ** d["xhat_log10"], color="red", linestyle="--", lw=1, label="MLE"
        )
        ax.set_xscale("log")
        ax.set_xlabel(p)
        ax.set_ylabel("Δ log-likelihood (max − ll)")
        ax.set_title(f"{title_prefix}: {p}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / f"profile_{p}.png", dpi=150)
        plt.close(fig)

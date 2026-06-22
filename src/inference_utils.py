#!/usr/bin/env python3
"""
inference_utils.py - Shared utilities for parameter fixing and optimization
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def build_scaled_param_dict(param_names: List[str], vec_real: np.ndarray) -> Dict[str, float]:
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

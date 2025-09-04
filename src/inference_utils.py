#!/usr/bin/env python3
"""
inference_utils.py - Shared utilities for parameter fixing and optimization
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def build_fixed_param_mapper(
    param_names: List[str],
    fixed_params: Dict[str, float]
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
            raise ValueError(f"Fixed parameter '{param_name}' not found in model parameters: {param_names}")
    
    return free_indices, fixed_indices, fixed_params


def create_param_vector_with_fixed(
    free_params: np.ndarray,
    param_names: List[str],
    fixed_params: Dict[str, float],
    free_indices: List[int]
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
    upper_bounds: List[float]
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
    fixed_params: Dict[str, float]
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
    param_names: List[str],
    fixed_params: Dict[str, float]
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
    param_names: List[str],
    fixed_params: Dict[str, float]
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

#!/usr/bin/env python3
"""
src/sfs_inference_runner.py

Shared inference “glue” for dadi + moments. This is the module that lets your
Snakemake wrapper stay lightweight.

Responsibilities:
- Load SFS + experiment config
- Load demography model function (--model-py "module:function")
- Parse/validate fixed parameters (same semantics as MomentsLD)
- Build start_dict (geometric mean for free params, fixed value for fixed params)
- Run dadi and/or moments inference using src/dadi_inference.py and src/moments_inference.py
- (Optional) generate 1D likelihood profiles + plots per parameter

Your thin CLI wrapper should just parse args and call run_cli(...).

Notes / deliberate choices:
- We avoid any Snakemake assumptions about directory layout.
- We keep conversion between dadi/moments Spectrum types here.
- We DO NOT reach into moments_inference internals except _diffusion_sfs.
  If you want cleaner separation later, expose a public expected_sfs() in
  both dadi_inference and moments_inference.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib
import json
import pickle
from collections import Counter

import numpy as np
import matplotlib
import moments
import dadi 
# Ensure headless behavior for cluster jobs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Imports for inference backends
# ---------------------------------------------------------------------
# IMPORTANT: This module is in src/, so "import dadi_inference" resolves if src/
# is on sys.path (your wrapper does that). If you import from project root, you
# may prefer "from src import dadi_inference" etc.
import dadi_inference
import moments_inference


# =============================================================================
# Parameter ordering / validation helpers (general across demographic models)
# =============================================================================

def _validate_parameter_order(config: Dict[str, Any]) -> List[str]:
    """
    Returns the canonical parameter order (from config) and validates:
      - parameter_order exists and is non-empty
      - no duplicate names
    """
    if "parameter_order" not in config or not config["parameter_order"]:
        raise ValueError("Config must include a non-empty 'parameter_order' list.")
    order = list(config["parameter_order"])

    counts = Counter(order)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        raise ValueError(f"'parameter_order' contains duplicates: {dups}")

    return order


def _validate_parameterization(
    param_order: List[str],
    priors: Dict[str, Any],
    fixed_params: Dict[str, float],
) -> None:
    """
    Enforces that:
      - every param in param_order is either fixed or has a prior
      - priors do not contain params not present in param_order
      - fixed_params do not contain params not present in param_order
    """
    order_set = set(param_order)
    prior_set = set(priors.keys())
    fixed_set = set(fixed_params.keys())

    # Each param in order must be covered either by fixed or priors
    missing_coverage = [p for p in param_order if (p not in fixed_set and p not in prior_set)]
    if missing_coverage:
        raise ValueError(
            "Parameters in 'parameter_order' missing from both priors and fixed params: "
            f"{missing_coverage}"
        )

    # Priors shouldn't include extra params
    extra_priors = sorted(list(prior_set - order_set))
    if extra_priors:
        raise ValueError(
            "Config priors include parameters not in 'parameter_order': "
            f"{extra_priors}"
        )

    # Fixed params shouldn't include extra params
    extra_fixed = sorted(list(fixed_set - order_set))
    if extra_fixed:
        raise ValueError(
            "Fixed parameters include parameters not in 'parameter_order': "
            f"{extra_fixed}"
        )


def _build_start_dict_from_config(
    param_order: List[str],
    priors: Dict[str, Any],
    fixed_params: Dict[str, float],
) -> Dict[str, float]:
    """
    Returns a dict with exactly the keys in param_order:
      - fixed param value if fixed
      - geometric mean of (lower, upper) if free
    """
    start_dict: Dict[str, float] = {}

    for p in param_order:
        if p in fixed_params:
            start_dict[p] = float(fixed_params[p])
        else:
            lower, upper = priors[p]
            lower = float(lower)
            upper = float(upper)
            if lower <= 0 or upper <= 0:
                raise ValueError(
                    f"Bounds for '{p}' must be positive for geometric mean; got {lower}, {upper}"
                )
            start_dict[p] = float(lower / 2 + upper / 2)  # Midpoint of the prior distribution

    # Sanity: exact key match
    missing = [p for p in param_order if p not in start_dict]
    extra = [p for p in start_dict.keys() if p not in set(param_order)]
    if missing or extra:
        raise ValueError(
            f"Internal error: start_dict key mismatch. missing={missing}, extra={extra}"
        )

    return start_dict

def _save_results(
    mode: str,
    best_params: Optional[Dict[str, float]],
    best_ll: Optional[float],
    param_order: List[str],
    fixed_params: Dict[str, float],
    outdir: Path,
) -> Path:
    """
    Save inference results to best_fit.pkl file (compatible with Snakemake expectations).
    
    Returns:
        Path to the saved file
    """
    mode_outdir = outdir / mode
    mode_outdir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "mode": mode,
        "best_params": best_params,
        "best_ll": float(best_ll) if best_ll is not None else None,
        "param_order": param_order,
        "fixed_params": fixed_params,
    }
    
    out_pkl = mode_outdir / "best_fit.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(result, f)
    
    ll_str = f"{best_ll:.6g}" if best_ll is not None else "None"
    print(f"[{mode}] Results saved: LL={ll_str} → {out_pkl}")
    
    return out_pkl



# =============================================================================
# Main CLI entry
# =============================================================================

def run_cli(
    mode: str,
    sfs_file: Path,
    config_file: Path,
    model_py: str,
    outdir: Path,
    ground_truth: Optional[Path] = None,
    generate_profiles: bool = False,
    profile_grid_points: int = 41,
    verbose: bool = False,
):
    """
    mode: "dadi", "moments", or "both"
    sfs_file: filepath of the pickle file that contains the SFS (dadi.Spectrum or moments.Spectrum)
    config_file: filepath of the JSON experiment config (see example in configs/)
    model_py: string of the form "module:function" that can be imported to get a function that returns a demes.Graph when called with a param dict
    outdir: parent output directory. For --mode both writes into outdir/{dadi,moments}
    ground_truth: Optional filepath of a pickle or JSON file containing the ground truth simulation parameters. If I specify fixed parameters then it will fix the parameter to the ground truth. But otherwise this is only used for ad hoc visualization of results.
    generate_profiles: Whether to generate 1D likelihood profiles for each parameter. This can be slow, so it's optional.\
    profile_grid_points: Number of grid points for likelihood profiles. Only relevant if generate_profiles is True.
    verbose: Whether to print verbose output during inference.
    """

    # Load the SFS
    with open(sfs_file, "rb") as f:
        sfs = pickle.load(f)

    # Load the experiment config
    with open(config_file, "r") as f:
        config = json.load(f)

    # Load the model function
    module_name, func_name = model_py.split(":")
    module = importlib.import_module(module_name)
    model_func = getattr(module, func_name)

    mode_outdir = outdir / mode
    mode_outdir.mkdir(parents=True, exist_ok=True)

    # Get model function signature for parameter validation (optional / informational)
    import inspect
    sig = inspect.signature(model_func)

    # ------------------------------------------------------------------
    # Step 1: Build fixed_params (if any)
    # ------------------------------------------------------------------
    fixed_params: Dict[str, float] = {}
    fixed_param_names = list(config.get("fixed_parameters", {}).keys())

    if fixed_param_names:
        if ground_truth is None:
            raise ValueError(
                f"fixed_parameters specified ({fixed_param_names}) but no ground_truth provided"
            )

        # (Your current behavior assumes pickle; keep minimal change)
        with open(ground_truth, "rb") as f:
            gt_params = pickle.load(f)

        for p in fixed_param_names:
            if p not in gt_params:
                raise ValueError(
                    f"Parameter '{p}' is specified as fixed but not found in ground truth parameters"
                )
            fixed_params[p] = float(gt_params[p])

    # ------------------------------------------------------------------
    # Step 2: Parameter ordering + priors validation (GENERAL)
    # ------------------------------------------------------------------
    param_order = _validate_parameter_order(config)

    # Use "parameters" if present, else "priors" (your original logic)
    priors = config.get("parameters", config.get("priors", {}))
    if not priors:
        raise ValueError("Config must include 'priors' (or 'parameters') with bounds.")

    _validate_parameterization(param_order, priors, fixed_params)

    # ------------------------------------------------------------------
    # Step 3: Build start_dict and ordered vectors
    # ------------------------------------------------------------------
    start_dict = _build_start_dict_from_config(param_order, priors, fixed_params)

    print(f"Model function: {module_name}:{func_name}  signature={sig}")
    print(f"Parameter order: {param_order}")
    print(f"Start dict (ordered): {[ (p, start_dict[p]) for p in param_order ]}")

    start_arr = np.array([start_dict[p] for p in param_order], dtype=float)

    # Now perturb the starting value slightly (order preserved)
    start_perturbed = moments.Misc.perturb_params(start_arr, fold=0.1)

    print(f"Starting values for optimization (ordered): {start_perturbed}")

    # ------------------------------------------------------------------
    # Step 4: Run inference (unchanged; keep your current structure)
    # ------------------------------------------------------------------

    # Example: moments inference
    if mode == "moments":

        sfs = moments.Spectrum(sfs)
        sfs.pop_ids = list(config['num_samples'].keys())

        print(sfs)
        print(sfs.pop_ids)

        assert isinstance(
            sfs, moments.Spectrum
        ), "SFS must be a moments.Spectrum when mode is moments"
        fitted_real, ll_value = moments_inference.fit_model(
            sfs=sfs,
            demo_model=model_func,
            experiment_config=config,
            start_vec=start_perturbed,
            param_order=param_order,   # <-- add this
            verbose=verbose,
        )

        print(f'Fitted parameters (ordered): {fitted_real}')
        print(f'Max log-likelihood: {ll_value}')

        # Convert fitted array back to dict for saving
        best_params = {param_order[i]: float(fitted_real[i]) for i in range(len(param_order))}
        
        # Save results
        _save_results(
            mode=mode,
            best_params=best_params,
            best_ll=ll_value,
            param_order=param_order,
            fixed_params=fixed_params,
            outdir=outdir,
        )

    # Example: dadi inference

    if mode == "dadi":

        sfs = dadi.Spectrum(sfs)
        sfs.pop_ids = list(config['num_samples'].keys())
        print(sfs)
        print(sfs.pop_ids)

        # assert isinstance(
        #     sfs, dadi.Spectrum
        # ), "SFS must be a dadi.Spectrum when mode is dadi"
        fitted_params, ll_value = dadi_inference.fit_model(
            sfs=sfs,
            demo_model=model_func,
            experiment_config=config,
            start_vec=start_perturbed,
            param_order=param_order,   # <-- add this
            verbose=verbose,
        )

        print(f'Fitted parameters (ordered): {fitted_params}')
        print(f'Max log-likelihood: {ll_value}')

        # Save results
        _save_results(
            mode=mode,
            best_params=fitted_params,
            best_ll=ll_value,
            param_order=param_order,
            fixed_params=fixed_params,
            outdir=outdir,
        )



    # if mode == "dadi":
    #     ... call dadi_inference.fit_model(...)

    # if mode == "both":
    #     ... run both into separate subdirs ...

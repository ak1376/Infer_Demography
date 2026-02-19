#!/usr/bin/env python3
"""
src/sfs_inference_runner.py

Shared inference “glue” for dadi + moments.

Key behaviors:
- fixed parameters come from config["fixed_parameters"] and are filled from ground_truth
- start vector uses fixed values for fixed params
- only FREE params are perturbed (fixed params are NOT perturbed)
- FIXED params are ALSO CONSTRAINED during optimization by setting their bounds to [v, v]
  in the config passed to the backend (moments/dadi), so they truly cannot move.
- dadi runtime debug txt is saved next to best_fit.pkl if present
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import importlib
import json
import pickle
from collections import Counter
import copy

import numpy as np
import matplotlib
import moments
import dadi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401

import dadi_inference
import moments_inference


# =============================================================================
# Parameter ordering / validation helpers
# =============================================================================

def _validate_parameter_order(config: Dict[str, Any]) -> List[str]:
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
    order_set = set(param_order)
    prior_set = set(priors.keys())
    fixed_set = set(fixed_params.keys())

    missing_coverage = [p for p in param_order if (p not in fixed_set and p not in prior_set)]
    if missing_coverage:
        raise ValueError(
            "Parameters in 'parameter_order' missing from both priors and fixed params: "
            f"{missing_coverage}"
        )

    extra_priors = sorted(list(prior_set - order_set))
    if extra_priors:
        raise ValueError(
            "Config priors include parameters not in 'parameter_order': "
            f"{extra_priors}"
        )

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
                    f"Bounds for '{p}' must be positive for midpoint; got {lower}, {upper}"
                )
            start_dict[p] = float(lower / 2 + upper / 2)

    missing = [p for p in param_order if p not in start_dict]
    extra = [p for p in start_dict.keys() if p not in set(param_order)]
    if missing or extra:
        raise ValueError(
            f"Internal error: start_dict key mismatch. missing={missing}, extra={extra}"
        )

    return start_dict


def _perturb_only_free_params(
    start_arr: np.ndarray,
    param_order: List[str],
    fixed_params: Dict[str, float],
    *,
    fold: float = 0.1,
) -> np.ndarray:
    """
    Perturb only parameters that are NOT fixed.
    Fixed params are re-imposed exactly.
    """
    start_arr = np.asarray(start_arr, dtype=float)
    out = start_arr.copy()

    free_idx = [i for i, p in enumerate(param_order) if p not in fixed_params]
    if free_idx:
        free_vals = out[free_idx]
        free_perturbed = moments.Misc.perturb_params(free_vals, fold=fold)
        out[free_idx] = free_perturbed

    for i, p in enumerate(param_order):
        if p in fixed_params:
            out[i] = float(fixed_params[p])

    return out


def _save_results(
    mode: str,
    best_params: Optional[Dict[str, float]],
    best_ll: Optional[float],
    param_order: List[str],
    fixed_params: Dict[str, float],
    outdir: Path,
) -> Path:
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


def _apply_fixed_bounds_to_config(
    config: Dict[str, Any],
    priors: Dict[str, Any],
    fixed_params: Dict[str, float],
) -> Dict[str, Any]:
    """
    Return a config copy where fixed params have bounds exactly [v, v],
    so the optimizer cannot move them.
    Also writes the merged priors dict back to BOTH keys ("priors" and "parameters")
    because your backends may read either.
    """
    cfg = copy.deepcopy(config)
    pri = copy.deepcopy(priors)

    for p, v in fixed_params.items():
        v = float(v)
        if v <= 0:
            raise ValueError(f"Fixed parameter '{p}' must be > 0 for log10 optimization; got {v}")
        pri[p] = [v, v]

    cfg["priors"] = pri
    cfg["parameters"] = pri
    return cfg


# =============================================================================
# Main entry
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
) -> None:
    # Load SFS
    with open(sfs_file, "rb") as f:
        sfs = pickle.load(f)

    # Load config
    with open(config_file, "r") as f:
        config = json.load(f)

    # Load model function
    module_name, func_name = model_py.split(":")
    module = importlib.import_module(module_name)
    model_func = getattr(module, func_name)

    # Signature (informational)
    import inspect
    sig = inspect.signature(model_func)

    # ------------------------------------------------------------------
    # Step 1: fixed params
    # ------------------------------------------------------------------
    fixed_params: Dict[str, float] = {}
    fixed_param_names = list(config.get("fixed_parameters", {}).keys())

    if fixed_param_names:
        if ground_truth is None:
            raise ValueError(
                f"fixed_parameters specified ({fixed_param_names}) but no ground_truth provided"
            )
        with open(ground_truth, "rb") as f:
            gt_params = pickle.load(f)
        for p in fixed_param_names:
            if p not in gt_params:
                raise ValueError(
                    f"Parameter '{p}' is specified as fixed but not found in ground truth parameters"
                )
            fixed_params[p] = float(gt_params[p])

    # ------------------------------------------------------------------
    # Step 2: ordering + priors
    # ------------------------------------------------------------------
    param_order = _validate_parameter_order(config)

    # Build priors dict (support either key)
    priors = config.get("priors", config.get("parameters", {}))
    if not priors:
        raise ValueError("Config must include 'priors' (or 'parameters') with bounds.")

    _validate_parameterization(param_order, priors, fixed_params)

    # IMPORTANT: create a backend config where fixed params have bounds [v, v]
    config_fit = _apply_fixed_bounds_to_config(config, priors, fixed_params)
    priors_fit = config_fit["priors"]

    # ------------------------------------------------------------------
    # Step 3: start vector (fixed values) + perturb only free
    # ------------------------------------------------------------------
    start_dict = _build_start_dict_from_config(param_order, priors_fit, fixed_params)

    print(f"Model function: {module_name}:{func_name}  signature={sig}")
    print(f"Parameter order: {param_order}")
    print(f"Fixed params: {fixed_params}")
    print(f"Start dict (ordered): {[ (p, start_dict[p]) for p in param_order ]}")

    start_arr = np.array([start_dict[p] for p in param_order], dtype=float)
    start_perturbed = _perturb_only_free_params(
        start_arr=start_arr,
        param_order=param_order,
        fixed_params=fixed_params,
        fold=0.1,
    )
    print(f"Starting values for optimization (ordered): {start_perturbed}")

    # ------------------------------------------------------------------
    # Step 4: inference
    # ------------------------------------------------------------------
    if mode == "moments":
        sfs_m = moments.Spectrum(sfs)
        sfs_m.pop_ids = list(config_fit["num_samples"].keys())

        # Save profiles under the SAME directory where best_fit.pkl will live:
        # outdir/moments/likelihood_plots/
        moments_dir = outdir / "moments"
        moments_dir.mkdir(parents=True, exist_ok=True)

        # Do NOT add any output dir to cfg. Just toggle profile behavior in-memory.
        config_fit_local = config_fit
        if generate_profiles:
            config_fit_local = copy.deepcopy(config_fit)
            config_fit_local["generate_profiles"] = True
            config_fit_local["profile_points"] = int(profile_grid_points)
            # optional defaults if not set in JSON
            config_fit_local.setdefault("profile_widen", 0.5)
            config_fit_local.setdefault("profile_make_plots", True)

        fitted_real, ll_value = moments_inference.fit_model(
            sfs=sfs_m,
            demo_model=model_func,
            experiment_config=config_fit_local,   # <-- uses fixed bounds
            start_vec=start_perturbed,
            param_order=param_order,
            verbose=verbose,
            save_dir=moments_dir,                # <-- KEY: enables likelihood_plots location
        )

        best_params = {param_order[i]: float(fitted_real[i]) for i in range(len(param_order))}
        # enforce fixed exactly (belt + suspenders)
        for p, v in fixed_params.items():
            best_params[p] = float(v)

        _save_results(
            mode=mode,
            best_params=best_params,
            best_ll=ll_value,
            param_order=param_order,
            fixed_params=fixed_params,
            outdir=outdir,
        )
        return

    if mode == "dadi":
        sfs_d = dadi.Spectrum(sfs)
        sfs_d.pop_ids = list(config_fit["num_samples"].keys())

        result = dadi_inference.fit_model(
            sfs=sfs_d,
            demo_model=model_func,
            experiment_config=config_fit,   # <-- uses fixed bounds
            start_vec=start_perturbed,
            param_order=param_order,
            verbose=verbose,
        )

        fitted_params = result.params
        ll_value = result.loglik

        best_params = {param_order[i]: float(fitted_params[i]) for i in range(len(param_order))}
        for p, v in fixed_params.items():
            best_params[p] = float(v)

        _save_results(
            mode=mode,
            best_params=best_params,
            best_ll=ll_value,
            param_order=param_order,
            fixed_params=fixed_params,
            outdir=outdir,
        )

        if result.debug_txt is not None:
            sim = config_fit.get("_simulation_number", config_fit.get("simulation_number", "NA"))
            run = config_fit.get("_run_number", config_fit.get("run_number", "NA"))
            dbg_path = (outdir / mode) / f"dadi_runtime_debug_sim{sim}_run{run}.txt"
            dbg_path.write_text(result.debug_txt)
            print(f"[dadi] Runtime debug saved → {dbg_path}")

        return

    if mode == "both":
        run_cli(
            mode="moments",
            sfs_file=sfs_file,
            config_file=config_file,
            model_py=model_py,
            outdir=outdir,
            ground_truth=ground_truth,
            generate_profiles=generate_profiles,
            profile_grid_points=profile_grid_points,
            verbose=verbose,
        )
        run_cli(
            mode="dadi",
            sfs_file=sfs_file,
            config_file=config_file,
            model_py=model_py,
            outdir=outdir,
            ground_truth=ground_truth,
            generate_profiles=generate_profiles,
            profile_grid_points=profile_grid_points,
            verbose=verbose,
        )
        return

    raise ValueError("mode must be one of: 'dadi', 'moments', 'both'")

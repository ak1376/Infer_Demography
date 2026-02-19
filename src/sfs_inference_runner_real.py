#!/usr/bin/env python3
"""
src/sfs_inference_runner_real.py

Real-data runner:
- moments OR dadi
- uses scaled optimization (ratios/tau/M), profiles theta, returns ABS params
- writes best_fit.pkl with:
    best_params: ABSOLUTE params (including N_ANC = implied)
    theta_hat: float
    N_ANC_implied_from_theta: float
    theta_mode: "profiled_unit_scaled"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import importlib
import json
import pickle
from collections import Counter

import moments
import dadi

from src.moments_inference_real import fit_model_realdata_scaled as fit_moments_real_scaled
from src.dadi_inference_real import fit_model_realdata_scaled as fit_dadi_real_scaled


def _validate_parameter_order(config: Dict[str, Any]) -> List[str]:
    if "parameter_order" not in config or not config["parameter_order"]:
        raise ValueError("Config must include a non-empty 'parameter_order' list.")
    order = list(config["parameter_order"])
    counts = Counter(order)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        raise ValueError(f"'parameter_order' contains duplicates: {dups}")
    return order


def _save_results_real(
    *,
    outdir: Path,
    mode: str,
    best_params: Dict[str, float],
    best_ll: float,
    param_order: List[str],
    fixed_params: Dict[str, float],
    theta_hat: float,
    N_ANC_implied_from_theta: float,
    theta_mode: str,
    debug_txt: Optional[str] = None,
) -> Path:
    mode_outdir = outdir / mode
    mode_outdir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "mode": mode,
        "best_params": best_params,
        "best_ll": float(best_ll),
        "param_order": param_order,
        "fixed_params": fixed_params,
        "theta_hat": float(theta_hat),
        "N_ANC_implied_from_theta": float(N_ANC_implied_from_theta),
        "theta_mode": str(theta_mode),
    }
    if debug_txt is not None:
        result["debug_txt"] = str(debug_txt)

    out_pkl = mode_outdir / "best_fit.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(result, f)

    print(f"[{mode}-real] Results saved: LL={best_ll:.6g} → {out_pkl}")
    return out_pkl


def run_cli_real(
    *,
    sfs_file: Path,
    config_file: Path,
    model_py: str,
    outdir: Path,
    mode: str = "moments",            # "moments" or "dadi"
    verbose: bool = False,
) -> None:
    with open(sfs_file, "rb") as f:
        sfs = pickle.load(f)

    with open(config_file, "r") as f:
        config = json.load(f)

    module_name, func_name = model_py.split(":")
    module = importlib.import_module(module_name)
    model_func = getattr(module, func_name)

    param_order = _validate_parameter_order(config)

    # Fixed params on real data usually empty
    fixed_params: Dict[str, float] = {}

    mode = str(mode).lower().strip()
    if mode not in {"moments", "dadi"}:
        raise ValueError(f"mode must be 'moments' or 'dadi'; got {mode}")

    debug_txt: Optional[str] = None

    if mode == "moments":
        sfs_m = moments.Spectrum(sfs)
        sfs_m.pop_ids = list(config["num_samples"].keys())

        best_params_abs, ll_hat, theta_hat, Nanc_implied = fit_moments_real_scaled(
            sfs=sfs_m,
            demo_model_abs=model_func,
            experiment_config=config,
            param_order=param_order,
            verbose=verbose,
        )

    else:
        sfs_d = dadi.Spectrum(sfs)
        sfs_d.pop_ids = list(config["num_samples"].keys())

        # dadi real scaled
        best_params_abs, ll_hat, theta_hat, Nanc_implied = fit_dadi_real_scaled(
            sfs=sfs_d,
            demo_model_abs=model_func,
            experiment_config=config,
            param_order=param_order,
            verbose=verbose,
        )
        # if you later want to persist debug_txt, you can return it from fit_dadi_real_scaled;
        # for now, we don't have it unless you extend the function signature.

    # enforce any fixed params if you ever add them
    for p, v in fixed_params.items():
        best_params_abs[p] = float(v)

    _save_results_real(
        outdir=outdir,
        mode=mode,
        best_params=best_params_abs,
        best_ll=ll_hat,
        param_order=param_order,
        fixed_params=fixed_params,
        theta_hat=theta_hat,
        N_ANC_implied_from_theta=Nanc_implied,
        theta_mode="profiled_unit_scaled",
        debug_txt=debug_txt,
    )

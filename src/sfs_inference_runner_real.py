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

from src.moments_inference_real import (
    fit_model_realdata_scaled as fit_moments_real_scaled,
)
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


def _x0_log10_from_fit(path: Path, param_order: List[str]) -> List[float]:
    """log10 SCALED start vector (param_order) from an earlier real fit's
    ABSOLUTE best_params (best_fit.pkl, single or top-K, or sfs_fit.json)."""
    import math
    from src.inference_utils import absolute_to_scaled_params

    if path.suffix == ".json":
        d = json.loads(path.read_text())
    else:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
    bp, ll = d["best_params"], d["best_ll"]
    if isinstance(bp, list):
        bp = bp[max(range(len(ll)), key=lambda i: ll[i])]
    p_abs = {k: float(v) for k, v in bp.items()}
    p_scaled = absolute_to_scaled_params(p_abs, N_anc_abs=p_abs["N_ANC"])
    p_scaled["N_ANC"] = 1.0   # scaled N_ANC is the unit placeholder
    return [math.log10(p_scaled[p]) for p in param_order]


def run_cli_real(
    *,
    sfs_file: Path,
    config_file: Path,
    model_py: str,
    outdir: Path,
    mode: str = "moments",  # "moments" or "dadi"
    opt_seed: Optional[int] = None,
    real_sequence_length: Optional[float] = None,
    verbose: bool = False,
    fix: Optional[Dict[str, float]] = None,
    x0_from: Optional[Path] = None,
) -> None:
    """fix: extra SCALED params to hold constant, on top of the config's
    fixed_parameters (profile likelihoods pin one at each grid value).
    x0_from: (moments only) warm-start from an earlier fit -- a best_fit.pkl
    (single or top-K, the best entry is used) or sfs_fit.json with ABSOLUTE
    best_params -- converted to scaled units; fixed params override it."""
    with open(sfs_file, "rb") as f:
        sfs = pickle.load(f)

    with open(config_file, "r") as f:
        config = json.load(f)

    if opt_seed is not None:
        config["opt_seed"] = int(opt_seed)
    if real_sequence_length is not None:
        config["real_sequence_length"] = float(real_sequence_length)

    module_name, func_name = model_py.split(":")
    module = importlib.import_module(module_name)
    model_func = getattr(module, func_name)

    param_order = _validate_parameter_order(config)

    # Read numeric fixed params from config (skip special string values like "sampled")
    fixed_params: Dict[str, float] = {
        k: float(v)
        for k, v in config.get("fixed_parameters", {}).items()
        if isinstance(v, (int, float))
    }
    fixed_params.update({k: float(v) for k, v in (fix or {}).items()})
    unknown = sorted(set(fixed_params) - set(param_order))
    if unknown:
        raise ValueError(f"fixed params not in parameter_order: {unknown}")

    x0_log10 = None
    if x0_from is not None:
        if str(mode).lower().strip() != "moments":
            raise ValueError("x0_from is only supported for mode='moments'")
        x0_log10 = _x0_log10_from_fit(Path(x0_from), param_order)

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
            fixed_params=fixed_params,
            verbose=verbose,
            save_dir=outdir / "moments",
            x0_log10=x0_log10,
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
            fixed_params=fixed_params,
            verbose=verbose,
            save_dir=outdir / "dadi",
        )
        # if you later want to persist debug_txt, you can return it from fit_dadi_real_scaled;
        # for now, we don't have it unless you extend the function signature.

    # (fixed params need no special handling here: fixed_params are SCALED
    # values held at lb == ub during the fit, and the fit already converts
    # every param -- fixed ones included -- to ABSOLUTE units in
    # best_params_abs. Overwriting them with the scaled value would be wrong.)

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

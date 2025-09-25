#!/usr/bin/env python3
# snakemake_scripts/moments_dadi_inference.py
# Lightweight CLI wrapper for dadi and moments inference

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Callable
import argparse
import importlib
import importlib.util
import json
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make src importable (for simulation.py, dadi_inference, moments_inference)
# ---------------------------------------------------------------------
SYS_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SYS_SRC))

import dadi_inference
import moments_inference


# =====================================================================
# CLI
# =====================================================================
def _parse_args():
    p = argparse.ArgumentParser("CLI wrapper for dadi/moments inference")
    p.add_argument("--mode", choices=["dadi", "moments", "both"], required=True)
    p.add_argument("--sfs-file", type=Path, required=True,
                   help="Pickle of dadi.Spectrum | moments.Spectrum")
    p.add_argument("--config", type=Path, required=True,
                   help="JSON experiment configuration file")
    p.add_argument("--model-py", type=str, required=True,
                   help="module:function returning demes.Graph when called with a param dict")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Parent output directory. For --mode both, writes into outdir/{dadi,moments}")
    p.add_argument("--ground-truth", type=Path,
                   help="Pickle or JSON file with ground truth simulation parameters")
    p.add_argument("--generate-profiles", action="store_true",
                   help="Generate 1D likelihood profiles for each parameter")
    p.add_argument("--profile-grid-points", type=int, default=41,
                   help="Number of grid points for likelihood profiles")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


# =====================================================================
# IO helpers
# =====================================================================
def load_sfs(path: Path):
    """Load SFS (moments.Spectrum or dadi.Spectrum) from pickle."""
    return pickle.loads(path.read_bytes())


def _load_json_or_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    # Try pickle first, then JSON
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return json.loads(path.read_text())


def load_ground_truth(gt_path: Path) -> Dict[str, float]:
    """Best-effort loader for ground-truth param dicts."""
    gt = _load_json_or_pickle(gt_path)
    if isinstance(gt, dict):
        for key in ("parameters", "true_params", "ground_truth"):
            if key in gt and isinstance(gt[key], dict):
                return {k: float(v) for k, v in gt[key].items()}
        # fall back: top-level flat dict
        return {k: float(v) for k, v in gt.items() if isinstance(v, (int, float, np.floating))}
    raise ValueError(f"Unsupported ground truth format: {type(gt)}")


# =====================================================================
# Fixed-parameter & starts
# =====================================================================
def handle_fixed_parameters(config: Dict, sampled_params: Dict | None, param_names: List[str]) -> Dict[str, float]:
    """
    Parse config['fixed_parameters'] where values can be:
      number → fix to that value
      'sampled' or 'true' → fix to sampled_params[param]
    """
    out = {}
    fixed_cfg = config.get("fixed_parameters", {}) or {}
    for p in param_names:
        if p not in fixed_cfg:
            continue
        spec = fixed_cfg[p]
        if isinstance(spec, (int, float, np.floating)):
            out[p] = float(spec)
        elif isinstance(spec, str) and spec.lower() in {"sampled", "true"}:
            if sampled_params is None or p not in sampled_params:
                raise ValueError(f"Cannot fix {p} to sampled value: not available")
            out[p] = float(sampled_params[p])
        else:
            raise ValueError(f"Invalid fixed spec for {p}: {spec}")
    return out


def create_start_dict_with_fixed(config: Dict, fixed_params: Dict[str, float]) -> Dict[str, float]:
    """Start at geometric mean of prior unless fixed."""
    priors = config["priors"]
    start = {}
    for p, (lo, hi) in priors.items():
        if p in fixed_params:
            start[p] = float(fixed_params[p])
        else:
            start[p] = float(np.sqrt(float(lo) * float(hi)))
    return start


# =====================================================================
# Likelihood profiles (optional)
# =====================================================================
def classify_parameter_type(param_name: str) -> str:
    p = param_name.lower()
    if any(x in p for x in ["n0", "n1", "n2", "size", "anc", "afr", "eur", "bottleneck", "recover"]):
        return "population_size"
    if any(x in p for x in ["t", "time", "split", "expansion", "divergence"]):
        return "time"
    if any(x in p for x in ["m", "mig", "flow"]):
        return "migration"
    return "other"


def create_parameter_grid(param_name: str, fitted_value: float, lower_bound: float,
                          upper_bound: float, grid_points: int = 41) -> np.ndarray:
    kind = classify_parameter_type(param_name)
    fold = 50.0 if kind in {"population_size", "time"} else (100.0 if kind == "migration" else 20.0)
    gmin = max(lower_bound, fitted_value / fold)
    gmax = min(upper_bound, fitted_value * fold)
    gmin = max(gmin, 1e-12)
    if gmax <= gmin * 1.0001:
        gmax = min(upper_bound, gmin * 10.0)
    return np.logspace(np.log10(gmin), np.log10(gmax), grid_points)


def compute_likelihood_profile(mode: str, param_values: np.ndarray, param_idx: int,
                               grid_values: np.ndarray, sfs, demo_func: Callable,
                               config: Dict, fixed_params: Dict[str, float]) -> np.ndarray:
    """Compute Poisson log-likelihood profile for one parameter."""
    from collections import OrderedDict
    like = []

    # resolve sample sizes from SFS (preserve pop_ids if present)
    if hasattr(sfs, "pop_ids") and sfs.pop_ids:
        ns = OrderedDict((pid, (sfs.shape[i] - 1) // 2) for i, pid in enumerate(sfs.pop_ids))
    else:
        ns = OrderedDict((f"pop{i}", (n - 1) // 2) for i, n in enumerate(sfs.shape))

    param_names = list(config["priors"].keys())

    for val in grid_values:
        vec = param_values.copy()
        vec[param_idx] = float(val)
        try:
            if mode == "dadi":
                expected = dadi_inference.diffusion_sfs_dadi(
                    vec, ns, lambda pd: _call_demo(demo_func, pd, config),
                    config["mutation_rate"], config["genome_length"], pts=[50, 60, 70]
                )
            else:
                expected = moments_inference._diffusion_sfs(
                    vec, lambda pd: _call_demo(demo_func, pd, config),
                    param_names, ns, config
                )
            if getattr(sfs, "folded", False):
                expected = expected.fold()
            expected = np.maximum(expected, 1e-300)
            ll = float(np.sum(sfs * np.log(expected) - expected))
        except Exception as e:
            print(f"[profile] {param_names[param_idx]}={val} error: {e}")
            ll = -np.inf
        like.append(ll)
    return np.array(like)


def create_likelihood_plot(param_name: str, grid_values: np.ndarray,
                           likelihood_values: np.ndarray, fitted_value: float,
                           true_value: float | None, mode: str, outdir: Path) -> Path:
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(grid_values, likelihood_values, "-o", linewidth=2.0, markersize=4, alpha=0.85)
    plt.axvline(fitted_value, ls="-", lw=2, color="tab:red", label=f"MLE {fitted_value:.4g}")
    if true_value is not None and not np.isnan(true_value):
        plt.axvline(true_value, ls="--", lw=2, color="tab:green", label=f"True {true_value:.4g}")
    plt.xscale("log")
    plt.ylabel("Log-likelihood")
    plt.xlabel(param_name)
    plt.title(f"{mode.title()} profile: {param_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    outfile = outdir / f"likelihood_profile_{mode}_{param_name}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    return outfile


def generate_likelihood_profiles(mode: str, sfs, fitted_params: Dict[str, float],
                                 config: Dict, demo_func: Callable,
                                 ground_truth_params: Dict[str, float],
                                 outdir: Path, fixed_params: Dict[str, float],
                                 grid_points: int = 21):
    prof_dir = outdir / "likelihood_profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    names = list(config["priors"].keys())
    vals  = np.array([fitted_params[n] for n in names], float)
    lows  = np.array([config["priors"][n][0] for n in names], float)
    highs = np.array([config["priors"][n][1] for n in names], float)

    saved = []
    for i, n in enumerate(names):
        if n in fixed_params:
            print(f"[profiles] skip fixed {n}")
            continue
        grid = create_parameter_grid(n, vals[i], lows[i], highs[i], grid_points=grid_points)
        like = compute_likelihood_profile(mode, vals, i, grid, sfs, demo_func, config, fixed_params)
        tval = ground_truth_params.get(n) if ground_truth_params else None
        saved.append(create_likelihood_plot(n, grid, like, vals[i], tval, mode, prof_dir))
        print(f"[profiles] saved {saved[-1].name}")
    return saved


# =====================================================================
# Adapter: call demography builders with (params, config) if possible
# =====================================================================
def _call_demo(demo_func: Callable, params_vec_or_dict, config: Dict):
    """
    Accept either:
      * list/array ordered as config['priors'] keys, or
      * dict of parameter -> value.
    Call builder as demo_func(param_dict, config) if signature supports it,
    else fall back to demo_func(param_dict).
    """
    # Convert to dict using the config priors order if we were given a vector
    if not isinstance(params_vec_or_dict, dict):
        keys = list(config["priors"].keys())
        param_dict = {k: float(v) for k, v in zip(keys, params_vec_or_dict)}
    else:
        param_dict = {k: float(v) for k, v in params_vec_or_dict.items()}

    # Try (params, config), else (params)
    try:
        return demo_func(param_dict, config)
    except TypeError:
        return demo_func(param_dict)


# =====================================================================
# Inference driver
# =====================================================================
def run_inference_mode(mode: str, sfs, config: Dict, demo_func: Callable,
                       start_dict: Dict[str, float], outdir: Path,
                       fixed_params: Dict[str, float], do_profiles: bool,
                       profile_grid_points: int):
    """Run a single mode and save result pickle."""
    mode_out = outdir / mode
    mode_out.mkdir(parents=True, exist_ok=True)

    if mode == "dadi":
        import dadi
        if not isinstance(sfs, dadi.Spectrum):
            pop_ids = getattr(sfs, "pop_ids", None)
            sfs = dadi.Spectrum(np.array(sfs))
            if pop_ids is not None:
                sfs.pop_ids = pop_ids

        param_lists, ll_list = dadi_inference.fit_model(
            sfs=sfs,
            start_dict=start_dict,
            demo_model=lambda d: _call_demo(demo_func, d, config),
            experiment_config=config,
            fixed_params=fixed_params
        )

    else:  # moments
        import moments
        if not isinstance(sfs, moments.Spectrum):
            sfs = moments.Spectrum(np.array(sfs))

        param_lists, ll_list = moments_inference.fit_model(
            sfs=sfs,
            start_dict=start_dict,
            demo_model=lambda d: _call_demo(demo_func, d, config),
            experiment_config=config,
            fixed_params=fixed_params
        )

    best_params = param_lists[0] if param_lists else None
    best_ll     = ll_list[0]     if ll_list     else None

    result = {
        "mode": mode,
        "best_params": ({k: float(v) for k, v in zip(start_dict.keys(), best_params)}
                        if best_params is not None else None),
        "best_ll": (float(best_ll) if best_ll is not None else None),
        "param_order": list(start_dict.keys()),
        "fixed_params": fixed_params,
    }
    out_pkl = mode_out / "best_fit.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(result, f)
    print(f"[{mode}] finished  LL={result['best_ll']}  → {out_pkl}")

    # Likelihood profiles (optional)
    if best_params is not None and do_profiles:
        try:
            fitted = {k: float(v) for k, v in zip(start_dict.keys(), best_params)}
            # Try to find a ground-truth nearby (optional)
            gt = {}
            gt_path = outdir.parent / "sampled_params.pkl"
            if gt_path.exists():
                try:
                    with gt_path.open("rb") as f:
                        gt = pickle.load(f)
                except Exception:
                    gt = {}
            generate_likelihood_profiles(
                mode, sfs, fitted, config, lambda d: _call_demo(demo_func, d, config),
                gt, mode_out, fixed_params, grid_points=profile_grid_points
            )
        except Exception as e:
            print(f"[{mode}] profiles failed: {e}")

    return result


# =====================================================================
# Main
# =====================================================================
def main():
    args = _parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load config + SFS
    config = json.loads(args.config.read_text())
    sfs    = load_sfs(args.sfs_file)

    # Ground truth (optional)
    sampled_params = None
    if args.ground_truth and args.ground_truth.exists():
        try:
            sampled_params = load_ground_truth(args.ground_truth)
        except Exception as e:
            print(f"[WARN] Could not load ground truth: {e}")

    # Import the builder function
    if ":" not in args.model_py:
        raise ValueError("--model-py must be 'module:function'")
    mod_name, func_name = args.model_py.split(":", 1)

    if mod_name.startswith("src."):
        # Resolve path to ./src/<module>.py
        module_file = mod_name.replace("src.", "").replace(".", "/") + ".py"
        module_path = SYS_SRC / module_file
        if not module_path.exists():
            # fallback: just last segment, e.g. src.split_isolation → split_isolation.py
            module_path = SYS_SRC / (mod_name.split(".")[-1] + ".py")
        spec = importlib.util.spec_from_file_location(mod_name.split(".")[-1], module_path)
        assert spec and spec.loader, f"Cannot load module: {module_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw_demo_func = getattr(module, func_name)
    else:
        raw_demo_func = getattr(importlib.import_module(mod_name), func_name)

    # Wrap so we can always call with config if supported
    def demo_func(param_dict):
        try:
            return raw_demo_func(param_dict, config)  # prefers (params, config)
        except TypeError:
            return raw_demo_func(param_dict)          # fallback for legacy signature


    # Fixed parameters & starts
    param_names  = list(config["priors"].keys())
    fixed_params = handle_fixed_parameters(config, sampled_params, param_names)
    start_dict   = create_start_dict_with_fixed(config, fixed_params)

    if args.verbose:
        print(f"Starting params: {start_dict}")
        if fixed_params:
            print(f"Fixed params:  {fixed_params}")

    # Run
    if args.mode == "both":
        for mode in ("dadi", "moments"):
            run_inference_mode(
                mode, sfs, config, demo_func, start_dict, args.outdir,
                fixed_params, do_profiles=args.generate_profiles,
                profile_grid_points=args.profile_grid_points
            )
    else:
        run_inference_mode(
            args.mode, sfs, config, demo_func, start_dict, args.outdir,
            fixed_params, do_profiles=args.generate_profiles,
            profile_grid_points=args.profile_grid_points
        )


if __name__ == "__main__":
    main()

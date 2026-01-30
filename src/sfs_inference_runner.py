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

import numpy as np
import matplotlib

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
# IO helpers
# =============================================================================


def load_sfs(path: Path):
    """Load SFS from pickle file (dadi.Spectrum or moments.Spectrum)."""
    return pickle.loads(path.read_bytes())


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_ground_truth(path: Path) -> Dict[str, float]:
    """
    Load ground-truth params. Supports:
      - pickle (dict)
      - json (dict)
    """
    if path.suffix.lower() in {".json"}:
        d = json.loads(path.read_text())
        return {k: float(v) for k, v in d.items()}
    # default: pickle
    with path.open("rb") as f:
        d = pickle.load(f)
    return {k: float(v) for k, v in d.items()}


# =============================================================================
# Fixed parameter handling (shared with MomentsLD semantics)
# =============================================================================


def handle_fixed_parameters(
    config: Dict[str, Any],
    sampled_params: Optional[Dict[str, float]],
    param_names: List[str],
) -> Dict[str, float]:
    """
    Parse config["fixed_parameters"] and return dict {param_name: fixed_value}.

    fixed spec options:
      - numeric: fix to that number
      - "sampled" or "true": fix to sampled_params[param_name]
    """
    fixed_params: Dict[str, float] = {}
    fixed_config = config.get("fixed_parameters", {}) or {}

    for p in param_names:
        if p not in fixed_config:
            continue
        spec = fixed_config[p]

        if isinstance(spec, (int, float)):
            fixed_params[p] = float(spec)
        elif isinstance(spec, str) and spec.lower() in {"sampled", "true"}:
            if sampled_params is None or p not in sampled_params:
                raise ValueError(f"Cannot fix {p} to sampled value: not available")
            fixed_params[p] = float(sampled_params[p])
        else:
            raise ValueError(f"Invalid fixed parameter spec for {p}: {spec}")

    return fixed_params


def create_start_dict_with_fixed(
    config: Dict[str, Any],
    fixed_params: Dict[str, float],
) -> Dict[str, float]:
    """
    Starting dict aligned with config['priors'] iteration order.
    - fixed params: fixed value
    - free params: geometric mean sqrt(low*high)
    """
    priors: Dict[str, List[float]] = config["priors"]
    start_dict: Dict[str, float] = {}

    for p, (low, high) in priors.items():
        if p in fixed_params:
            start_dict[p] = float(fixed_params[p])
        else:
            start_dict[p] = float(np.sqrt(float(low) * float(high)))

    return start_dict


# =============================================================================
# Model loading: --model-py "module:function"
# =============================================================================


def load_demo_func(model_py: str, *, project_root: Optional[Path] = None):
    """
    Load a function returning demes.Graph given a param dict.

    Accepts:
      - "some_module:func"
      - "src.some_module:func"

    If model_py begins with "src.", we try to load it from a file path under
    <project_root>/src/ to be robust to sys.path differences.

    Args:
        model_py: "module:function"
        project_root: root directory containing src/. If None, inferred as:
                      this file's parent (src/) -> project_root = parent of src/

    Returns:
        callable
    """
    if ":" not in model_py:
        raise ValueError("--model-py must be in format 'module:function'")

    mod_name, func_name = model_py.split(":", 1)

    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    # Special handling for src.* modules: load by file to avoid sys.path weirdness.
    if mod_name.startswith("src."):
        module_file = mod_name.replace("src.", "").replace(".", "/") + ".py"
        module_path = project_root / "src" / module_file

        if not module_path.exists():
            # fallback: assume last segment directly under src/
            module_path = project_root / "src" / (mod_name.split(".")[-1] + ".py")

        if not module_path.exists():
            raise FileNotFoundError(f"Cannot find module file for {mod_name}: {module_path}")

        import importlib.util

        spec = importlib.util.spec_from_file_location(mod_name.split(".")[-1], module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    # Normal import
    module = importlib.import_module(mod_name)
    return getattr(module, func_name)


# =============================================================================
# SFS conversions (keep pop_ids when possible)
# =============================================================================


def ensure_dadi_spectrum(sfs):
    import dadi

    if isinstance(sfs, dadi.Spectrum):
        return sfs

    # convert array-like moments.Spectrum or np.ndarray
    if hasattr(sfs, "shape"):
        pop_ids = getattr(sfs, "pop_ids", None)
        sfs_data = np.array(sfs)
        out = dadi.Spectrum(sfs_data)
        if pop_ids is not None:
            out.pop_ids = pop_ids
        return out

    raise ValueError(f"Cannot convert SFS to dadi.Spectrum: {type(sfs)}")


def ensure_moments_spectrum(sfs):
    import moments

    if isinstance(sfs, moments.Spectrum):
        return sfs

    if hasattr(sfs, "shape"):
        pop_ids = getattr(sfs, "pop_ids", None)
        sfs_data = np.array(sfs)
        out = moments.Spectrum(sfs_data)
        if pop_ids is not None:
            out.pop_ids = pop_ids
        return out

    raise ValueError(f"Cannot convert SFS to moments.Spectrum: {type(sfs)}")


# =============================================================================
# Likelihood profiles
# =============================================================================


def classify_parameter_type(param_name: str) -> str:
    """
    Returns one of: 'population_size', 'time', 'migration', 'other'
    """
    p = param_name.lower()

    # NOTE: fixed a bug you had: missing comma between "n2" and "size"
    if any(x in p for x in ["n0", "n1", "n2", "size", "anc", "afr", "eur", "bottleneck", "recover"]):
        return "population_size"
    if any(x in p for x in ["t", "time", "split", "expansion", "divergence"]):
        return "time"
    if any(x in p for x in ["m", "mig", "flow"]):
        return "migration"
    return "other"


def create_parameter_grid(
    param_name: str,
    fitted_value: float,
    lower_bound: float,
    upper_bound: float,
    grid_points: int = 41,
) -> np.ndarray:
    """
    Create a log-spaced grid around fitted_value, clipped to [lower_bound, upper_bound].
    """
    param_type = classify_parameter_type(param_name)

    # fold ranges: choose wide multiplicative ranges
    if param_type in {"population_size", "time"}:
        fold_range = 50.0
    elif param_type == "migration":
        fold_range = 100.0
    else:
        fold_range = 20.0

    gmin = max(float(lower_bound), float(fitted_value) / fold_range)
    gmax = min(float(upper_bound), float(fitted_value) * fold_range)

    gmin = max(gmin, 1e-12)
    if gmax <= gmin * 1.0001:
        gmax = min(float(upper_bound), gmin * 10.0)

    return np.logspace(np.log10(gmin), np.log10(gmax), int(grid_points))


def _infer_sample_sizes_from_sfs(sfs) -> Dict[str, int]:
    """
    Infer diploid sample sizes per pop from sfs pop_ids + shape.
    Returns dict pop_id -> diploid_n (== (axis_len-1)//2)
    """
    if hasattr(sfs, "pop_ids") and getattr(sfs, "pop_ids") is not None and len(sfs.pop_ids) > 0:
        return {pop_id: (sfs.shape[i] - 1) // 2 for i, pop_id in enumerate(sfs.pop_ids)}

    # try 'sample_sizes' if present
    if hasattr(sfs, "sample_sizes"):
        ss = getattr(sfs, "sample_sizes")
        # could be list or dict; prefer dict
        if isinstance(ss, dict):
            return {k: int(v) for k, v in ss.items()}
        # list: make generic names
        return {f"pop{i}": int(v) for i, v in enumerate(ss)}

    # fallback: generic names
    return {f"pop{i}": (n - 1) // 2 for i, n in enumerate(sfs.shape)}


def compute_likelihood_profile(
    *,
    mode: str,
    param_values: np.ndarray,
    param_idx: int,
    grid_values: np.ndarray,
    sfs,
    demo_func,
    config: Dict[str, Any],
    fixed_params: Dict[str, float],
) -> np.ndarray:
    """
    Compute Poisson log-likelihood across grid_values for one parameter.

    NOTE:
    - We skip fixed param indices in the caller; this still works if called anyway.
    - For moments expected SFS, this uses moments_inference._diffusion_sfs (private).
    """
    param_names = list(config["priors"].keys())
    ll_vals: List[float] = []

    for val in grid_values:
        test = np.asarray(param_values, float).copy()
        test[param_idx] = float(val)

        try:
            sample_sizes = _infer_sample_sizes_from_sfs(sfs)

            if mode == "dadi":
                expected = dadi_inference.diffusion_sfs_dadi(
                    params_vec=test,
                    param_names=param_names,
                    sample_sizes=sample_sizes,
                    demo_model=demo_func,
                    mutation_rate=float(config["mutation_rate"]),
                    sequence_length=float(config["genome_length"]),
                    pts=[50, 60, 70],
                    config=config,
                )
            elif mode == "moments":
                expected = moments_inference._diffusion_sfs(
                    init_vec=test,
                    demo_model=demo_func,
                    param_names=param_names,
                    sample_sizes=sample_sizes,  # OrderedDict not strictly required by moments_inference
                    experiment_config=config,
                )
            else:
                raise ValueError("mode must be 'dadi' or 'moments'")

            if getattr(sfs, "folded", False):
                expected = expected.fold()

            expected = np.maximum(np.array(expected), 1e-300)
            obs = np.array(sfs)
            ll = float(np.sum(obs * np.log(expected) - expected))
            ll_vals.append(ll)

        except Exception as e:
            print(f"[profile:{mode}] error at {val}: {e}")
            ll_vals.append(float("-inf"))

    return np.asarray(ll_vals, float)


def create_likelihood_plot(
    *,
    param_name: str,
    grid_values: np.ndarray,
    likelihood_values: np.ndarray,
    fitted_value: float,
    true_value: Optional[float],
    mode: str,
    outdir: Path,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=150)

    plt.plot(
        grid_values,
        likelihood_values,
        "-o",
        linewidth=2.0,
        markersize=4,
        label="Profile LL",
        alpha=0.85,
    )
    plt.axvline(x=float(fitted_value), linestyle="-", linewidth=2, label=f"MLE: {fitted_value:.4g}")
    if true_value is not None and not np.isnan(true_value):
        plt.axvline(x=float(true_value), linestyle="--", linewidth=2, label=f"True: {true_value:.4g}")

    plt.xscale("log")
    plt.ylabel("Log-likelihood")
    plt.xlabel(param_name)
    plt.title(f"{mode.title()} - Likelihood Profile: {param_name}")
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outfile = outdir / f"likelihood_profile_{mode}_{param_name}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    return outfile


def generate_likelihood_profiles(
    *,
    mode: str,
    sfs,
    fitted_params: Dict[str, float],
    config: Dict[str, Any],
    demo_func,
    ground_truth_params: Optional[Dict[str, float]],
    outdir: Path,
    fixed_params: Dict[str, float],
    grid_points: int = 41,
) -> List[Path]:
    """
    Generate 1D likelihood profiles for each free parameter.
    """
    profiles_dir = outdir / "likelihood_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    param_names = list(config["priors"].keys())
    param_values = np.array([fitted_params[name] for name in param_names], dtype=float)
    lower_bounds = np.array([config["priors"][name][0] for name in param_names], dtype=float)
    upper_bounds = np.array([config["priors"][name][1] for name in param_names], dtype=float)

    print(f"[profiles:{mode}] generating profiles for {len(param_names)} params (skipping fixed)...")

    saved: List[Path] = []
    for i, pname in enumerate(param_names):
        if pname in fixed_params:
            print(f"  - skip fixed: {pname}")
            continue

        grid = create_parameter_grid(
            param_name=pname,
            fitted_value=float(param_values[i]),
            lower_bound=float(lower_bounds[i]),
            upper_bound=float(upper_bounds[i]),
            grid_points=int(grid_points),
        )

        ll = compute_likelihood_profile(
            mode=mode,
            param_values=param_values,
            param_idx=i,
            grid_values=grid,
            sfs=sfs,
            demo_func=demo_func,
            config=config,
            fixed_params=fixed_params,
        )

        true_val = ground_truth_params.get(pname) if ground_truth_params else None
        out = create_likelihood_plot(
            param_name=pname,
            grid_values=grid,
            likelihood_values=ll,
            fitted_value=float(param_values[i]),
            true_value=true_val,
            mode=mode,
            outdir=profiles_dir,
        )
        saved.append(out)
        print(f"    saved {out.name}")

    print(f"[profiles:{mode}] done → {profiles_dir} ({len(saved)} plots)")
    return saved


# =============================================================================
# Inference run (dadi/moments/both)
# =============================================================================


@dataclass
class InferenceResult:
    mode: str
    best_params: Optional[Dict[str, float]]
    best_ll: Optional[float]
    param_order: List[str]
    fixed_params: Dict[str, float]
    out_pkl: Path


def run_inference_mode(
    *,
    mode: str,
    sfs,
    config: Dict[str, Any],
    demo_func,
    start_dict: Dict[str, float],
    outdir: Path,
    fixed_params: Dict[str, float],
    generate_profiles: bool = False,
    profile_grid_points: int = 41,
    ground_truth_params: Optional[Dict[str, float]] = None,
) -> InferenceResult:
    """
    Run inference using either dadi or moments. Writes:
      outdir/<mode>/best_fit.pkl
      outdir/<mode>/likelihood_profiles/*.png   (optional)
    """
    mode_outdir = outdir / mode
    mode_outdir.mkdir(parents=True, exist_ok=True)

    if mode == "dadi":
        sfs_dadi = ensure_dadi_spectrum(sfs)
        param_lists, ll_list = dadi_inference.fit_model(
            sfs=sfs_dadi,
            start_dict=start_dict,
            demo_model=demo_func,
            experiment_config=config,
            fixed_params=fixed_params,
        )
        sfs_used = sfs_dadi

    elif mode == "moments":
        sfs_mom = ensure_moments_spectrum(sfs)
        param_lists, ll_list = moments_inference.fit_model(
            sfs=sfs_mom,
            start_dict=start_dict,
            demo_model=demo_func,
            experiment_config=config,
            fixed_params=fixed_params,
        )
        sfs_used = sfs_mom

    else:
        raise ValueError("mode must be 'dadi' or 'moments'")

    best_params_vec = param_lists[0] if param_lists else None
    best_ll = ll_list[0] if ll_list else None

    if best_params_vec is not None:
        best_params = {k: float(v) for k, v in zip(start_dict.keys(), best_params_vec)}
    else:
        best_params = None

    result_dict = {
        "mode": mode,
        "best_params": best_params,
        "best_ll": (float(best_ll) if best_ll is not None else None),
        "param_order": list(start_dict.keys()),
        "fixed_params": {k: float(v) for k, v in fixed_params.items()},
    }

    out_pkl = mode_outdir / "best_fit.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(result_dict, f)

    ll_str = f"{best_ll:.6g}" if best_ll is not None else "None"
    print(f"[{mode}] finished  LL={ll_str}  → {out_pkl}")

    # Likelihood profiles
    if generate_profiles and best_params is not None:
        try:
            generate_likelihood_profiles(
                mode=mode,
                sfs=sfs_used,
                fitted_params=best_params,
                config=config,
                demo_func=demo_func,
                ground_truth_params=ground_truth_params,
                outdir=mode_outdir,
                fixed_params=fixed_params,
                grid_points=int(profile_grid_points),
            )
        except Exception as e:
            print(f"[profiles:{mode}] WARNING failed: {e}")

    return InferenceResult(
        mode=mode,
        best_params=best_params,
        best_ll=(float(best_ll) if best_ll is not None else None),
        param_order=list(start_dict.keys()),
        fixed_params=fixed_params,
        out_pkl=out_pkl,
    )


def run_cli(
    *,
    mode: str,
    sfs_file: Path,
    config_file: Path,
    model_py: str,
    outdir: Path,
    ground_truth: Optional[Path] = None,
    generate_profiles: bool = False,
    profile_grid_points: int = 41,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Programmatic entrypoint used by your thin Snakemake wrapper.

    Returns a dict with per-mode results.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_file)
    sfs = load_sfs(sfs_file)

    # Load ground truth / sampled params (optional)
    sampled_params = None
    ground_truth_params = None
    if ground_truth is not None:
        if ground_truth.exists():
            try:
                sampled_params = load_ground_truth(ground_truth)
                ground_truth_params = sampled_params
                if verbose:
                    print(f"[INFO] Loaded ground truth from {ground_truth}")
            except Exception as e:
                print(f"[WARN] Could not load ground truth: {e}")
        else:
            print(f"[WARN] Ground truth path provided but not found: {ground_truth}")

    # Model function
    demo_func = load_demo_func(model_py)

    # Fixed params + start dict
    param_names = list(config["priors"].keys())
    fixed_params = handle_fixed_parameters(config, sampled_params, param_names)
    start_dict = create_start_dict_with_fixed(config, fixed_params)

    if verbose:
        print(f"[runner] start_dict: {start_dict}")
        if fixed_params:
            print(f"[runner] fixed_params: {fixed_params}")

    results: Dict[str, Any] = {}

    if mode == "both":
        for m in ("dadi", "moments"):
            res = run_inference_mode(
                mode=m,
                sfs=sfs,
                config=config,
                demo_func=demo_func,
                start_dict=start_dict,
                outdir=outdir,
                fixed_params=fixed_params,
                generate_profiles=generate_profiles,
                profile_grid_points=profile_grid_points,
                ground_truth_params=ground_truth_params,
            )
            results[m] = {
                "best_params": res.best_params,
                "best_ll": res.best_ll,
                "out_pkl": str(res.out_pkl),
            }
    else:
        res = run_inference_mode(
            mode=mode,
            sfs=sfs,
            config=config,
            demo_func=demo_func,
            start_dict=start_dict,
            outdir=outdir,
            fixed_params=fixed_params,
            generate_profiles=generate_profiles,
            profile_grid_points=profile_grid_points,
            ground_truth_params=ground_truth_params,
        )
        results[mode] = {
            "best_params": res.best_params,
            "best_ll": res.best_ll,
            "out_pkl": str(res.out_pkl),
        }

    return results

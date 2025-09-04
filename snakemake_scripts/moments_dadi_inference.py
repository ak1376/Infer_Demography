#!/usr/bin/env python3
# snakemake_scripts/moments_dadi_inference.py
# Lightweight CLI wrapper for dadi and moments inference

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import argparse
import importlib
import json
import pickle
import sys

import numpy as np

# Add src to path so we can import our inference modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dadi_inference
import moments_inference


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
    p.add_argument("--fix-params", action="append", default=[],
                   help="Parameter names to fix at ground truth values, e.g. --fix-params N0 (repeatable)")
    p.add_argument("--fix", action="append", default=[],
                   help="Fix params by name with explicit values, e.g. --fix N0=10000 (repeatable)")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def load_sfs(path: Path):
    """Load SFS from pickle file."""
    return pickle.loads(path.read_bytes())


def load_ground_truth(gt_path: Path) -> Dict[str, float]:
    """Load ground truth parameters from pickle or JSON file."""
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    # Try to load as pickle first, then fall back to JSON
    try:
        with gt_path.open("rb") as f:
            gt_data = pickle.load(f)
    except (pickle.UnpicklingError, UnicodeDecodeError):
        # Fall back to JSON if pickle fails
        try:
            gt_data = json.loads(gt_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not load ground truth file as pickle or JSON: {e}")
    
    # Handle different possible structures in the ground truth data
    if isinstance(gt_data, dict):
        # Check for nested parameter structures
        if "parameters" in gt_data:
            return {k: float(v) for k, v in gt_data["parameters"].items()}
        elif "true_params" in gt_data:
            return {k: float(v) for k, v in gt_data["true_params"].items()}
        elif "ground_truth" in gt_data:
            return {k: float(v) for k, v in gt_data["ground_truth"].items()}
        else:
            # Assume the top level contains the parameters directly
            return {k: float(v) for k, v in gt_data.items() if isinstance(v, (int, float))}
    else:
        raise ValueError(f"Ground truth data must be a dictionary, got {type(gt_data)}")


def parse_explicit_fixed(items: List[str]) -> Dict[str, float]:
    """Parse --fix name=value arguments."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--fix requires name=value (got {it!r})")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def combine_fixed_params(
    explicit_fixed: List[str], 
    fix_param_names: List[str], 
    ground_truth_path: Path | None
) -> Dict[str, float]:
    """Combine explicitly fixed parameters with ground-truth fixed parameters."""
    fixed_by_name = {}
    
    # First, add explicitly fixed parameters (--fix name=value)
    explicit_dict = parse_explicit_fixed(explicit_fixed)
    fixed_by_name.update(explicit_dict)
    
    # Then, add ground truth parameters for specified parameter names
    if ground_truth_path and fix_param_names:
        gt_params = load_ground_truth(ground_truth_path)
        for param_name in fix_param_names:
            param_name = param_name.strip()
            if param_name in gt_params:
                if param_name in fixed_by_name:
                    print(f"[WARN] Parameter {param_name} specified in both --fix and --fix-params. "
                          f"Using explicit value {fixed_by_name[param_name]} instead of ground truth {gt_params[param_name]}")
                else:
                    fixed_by_name[param_name] = float(gt_params[param_name])
                    print(f"[INFO] Fixing {param_name} = {gt_params[param_name]} (from ground truth)")
            else:
                print(f"[WARN] Parameter {param_name} not found in ground truth file")
    
    return fixed_by_name


def create_start_dict_with_fixed(config: Dict, fixed_params: Dict[str, float]) -> Dict[str, float]:
    """Create starting parameter dictionary, incorporating fixed parameters."""
    priors = config["priors"]
    start_dict = {}
    
    for param_name, (low, high) in priors.items():
        if param_name in fixed_params:
            start_dict[param_name] = fixed_params[param_name]
        else:
            # Use geometric mean as starting point
            start_dict[param_name] = float(np.sqrt(low * high))
    
    return start_dict


def run_inference_mode(mode: str, sfs, config: Dict, demo_func, start_dict: Dict[str, float], outdir: Path, fixed_params: Dict[str, float]):
    """Run inference using either dadi or moments."""
    mode_outdir = outdir / mode
    mode_outdir.mkdir(parents=True, exist_ok=True)
    
    if mode == "dadi":
        import dadi
        if not isinstance(sfs, dadi.Spectrum):
            # Convert if needed, preserving pop_ids
            if hasattr(sfs, 'shape'):
                # Preserve pop_ids if they exist
                pop_ids = getattr(sfs, 'pop_ids', None)
                sfs_data = np.array(sfs)
                sfs = dadi.Spectrum(sfs_data)
                if pop_ids is not None:
                    sfs.pop_ids = pop_ids
            else:
                raise ValueError(f"Cannot convert SFS to dadi.Spectrum: {type(sfs)}")
        
        param_lists, ll_list = dadi_inference.fit_model(
            sfs=sfs,
            start_dict=start_dict,
            demo_model=demo_func,
            experiment_config=config,
            fixed_params=fixed_params
        )
        
    elif mode == "moments":
        import moments
        if not isinstance(sfs, moments.Spectrum):
            # Convert if needed  
            if hasattr(sfs, 'shape'):
                sfs = moments.Spectrum(np.array(sfs))
            else:
                raise ValueError(f"Cannot convert SFS to moments.Spectrum: {type(sfs)}")
            
        param_lists, ll_list = moments_inference.fit_model(
            sfs=sfs,
            start_dict=start_dict,
            demo_model=demo_func,
            experiment_config=config,
            fixed_params=fixed_params
        )
    
    # Save results
    best_params = param_lists[0] if param_lists else None
    best_ll = ll_list[0] if ll_list else None
    
    result = {
        "mode": mode,
        "best_params": {k: float(v) for k, v in zip(start_dict.keys(), best_params)} if best_params is not None else None,
        "best_ll": float(best_ll) if best_ll is not None else None,
        "param_order": list(start_dict.keys()),
        "fixed_params": fixed_params,  # Record what was requested to be fixed
    }
    
    out_pkl = mode_outdir / "best_fit.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(result, f)
    
    ll_str = f"{best_ll:.6g}" if best_ll is not None else "None"
    print(f"[{mode}] finished  LL={ll_str}  → {out_pkl}")
    return result


def main():
    args = _parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration and SFS
    config = json.loads(args.config.read_text())
    sfs = load_sfs(args.sfs_file)
    
    # Load demographic model function
    if ":" in args.model_py:
        mod_name, func_name = args.model_py.split(":")
        
        # Handle src module imports specially
        if mod_name.startswith("src."):
            module_file = mod_name.replace("src.", "").replace(".", "/") + ".py"
            module_path = Path(__file__).parent.parent / "src" / module_file
            if not module_path.exists():
                module_path = Path(__file__).parent.parent / "src" / (mod_name.split(".")[-1] + ".py")
            
            # Load module from file
            import importlib.util
            spec = importlib.util.spec_from_file_location(mod_name.split(".")[-1], module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            demo_func = getattr(module, func_name)
        else:
            demo_func = getattr(importlib.import_module(mod_name), func_name)
    else:
        raise ValueError("--model-py must be in format 'module:function'")
    
    # Handle parameter fixing
    fixed_params = combine_fixed_params(args.fix, args.fix_params, args.ground_truth)
    
    # Create starting parameter dictionary
    start_dict = create_start_dict_with_fixed(config, fixed_params)
    
    if args.verbose:
        print(f"Starting parameters: {start_dict}")
        if fixed_params:
            print(f"Fixed parameters: {fixed_params}")
    
    # Run inference
    if args.mode == "both":
        for mode in ["dadi", "moments"]:
            run_inference_mode(mode, sfs, config, demo_func, start_dict, args.outdir, fixed_params)
    else:
        run_inference_mode(args.mode, sfs, config, demo_func, start_dict, args.outdir, fixed_params)


if __name__ == "__main__":
    main()

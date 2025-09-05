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


def handle_fixed_parameters(config, sampled_params, param_names):
    """
    Parse configuration to determine which parameters should be fixed.
    Same logic as MomentsLD.
    
    Args:
        config: Configuration dictionary
        sampled_params: Dictionary of sampled parameter values (optional)
        param_names: List of all parameter names
        
    Returns:
        Dictionary mapping parameter names to fixed values (or None if not fixed)
    """
    fixed_params = {}
    fixed_config = config.get("fixed_parameters", {})
    
    for param_name in param_names:
        if param_name not in fixed_config:
            continue
            
        fixed_spec = fixed_config[param_name]
        
        if isinstance(fixed_spec, (int, float)):
            fixed_params[param_name] = float(fixed_spec)
        elif isinstance(fixed_spec, str) and fixed_spec.lower() in ["sampled", "true"]:
            if sampled_params is None or param_name not in sampled_params:
                raise ValueError(f"Cannot fix {param_name} to sampled value: not available")
            fixed_params[param_name] = float(sampled_params[param_name])
        else:
            raise ValueError(f"Invalid fixed parameter specification for {param_name}: {fixed_spec}")
    
    return fixed_params


def create_start_dict_with_fixed(config, fixed_params):
    """Create starting parameters dictionary with fixed params"""
    start_dict = {}
    for param, prior in config["priors"].items():
        if param in fixed_params:
            start_dict[param] = fixed_params[param]
        else:
            # Random start from uniform prior
            start_dict[param] = np.random.uniform(prior[0], prior[1])
    return start_dict


def load_ground_truth(ground_truth_path):
    """Load ground truth parameters"""
    with open(ground_truth_path, 'rb') as f:
        return pickle.load(f)


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
    
    # Load ground truth parameters if available
    sampled_params = None
    if args.ground_truth:
        try:
            sampled_params = load_ground_truth(args.ground_truth)
        except Exception as e:
            print(f"[WARN] Could not load ground truth: {e}")
    
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
    
    # Handle parameter fixing (same as MomentsLD)
    param_names = list(config["priors"].keys())
    fixed_params = handle_fixed_parameters(config, sampled_params, param_names)
    
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

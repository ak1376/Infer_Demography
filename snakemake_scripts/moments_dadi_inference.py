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
import matplotlib.pyplot as plt

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
    p.add_argument("--generate-profiles", action="store_true",
                   help="Generate 1D likelihood profiles for each parameter")
    p.add_argument("--profile-grid-points", type=int, default=41,
                   help="Number of grid points for likelihood profiles")
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


def classify_parameter_type(param_name: str) -> str:
    """
    Classify parameter type to determine appropriate grid range.
    
    Returns:
        'population_size' for population size parameters
        'time' for time parameters  
        'migration' for migration rate parameters
        'other' for other parameters
    """
    param_lower = param_name.lower()
    
    if any(x in param_lower for x in ['n0', 'n1', 'n2' 'size', 'anc', 'afr', 'eur', 'bottleneck', 'recover']):
        return 'population_size'
    elif any(x in param_lower for x in ['t', 'time', 'split', 'expansion', 'divergence']):
        return 'time'
    elif any(x in param_lower for x in ['m', 'mig', 'flow']):
        return 'migration'
    else:
        return 'other'


def create_parameter_grid(param_name: str, fitted_value: float, lower_bound: float, 
                         upper_bound: float, grid_points: int = 41) -> np.ndarray:
    """
    Create a logarithmic grid around the fitted parameter value.
    """
    param_type = classify_parameter_type(param_name)
    
    # Set grid range based on parameter type
    if param_type in ['population_size', 'time']:
        # Wider range for population sizes and times
        fold_range = 50.0
    elif param_type == 'migration':
        # Narrower range for migration rates (often very small)
        fold_range = 100.0
    else:
        # Default range for other parameters
        fold_range = 20.0
    
    # Calculate grid bounds
    gmin = max(lower_bound, fitted_value / fold_range)
    gmax = min(upper_bound, fitted_value * fold_range)
    
    # Ensure minimum separation
    gmin = max(gmin, 1e-12)
    if gmax <= gmin * 1.0001:
        gmax = min(upper_bound, gmin * 10.0)
    
    return np.logspace(np.log10(gmin), np.log10(gmax), grid_points)


def compute_likelihood_profile(mode: str, param_values: np.ndarray, param_idx: int,
                              grid_values: np.ndarray, sfs, demo_func, config: Dict,
                              fixed_params: Dict[str, float]) -> np.ndarray:
    """Compute likelihood profile for a single parameter."""
    likelihood_values = []
    param_names = list(config["priors"].keys())
    
    for param_val in grid_values:
        # Create test parameter vector
        test_params = param_values.copy()
        test_params[param_idx] = param_val
        
        try:
            if mode == "dadi":
                # Use dadi to compute expected SFS
                if hasattr(sfs, 'pop_ids') and sfs.pop_ids:
                    # Use the actual population IDs from the SFS
                    sample_sizes = {pop_id: (sfs.shape[i] - 1) // 2 for i, pop_id in enumerate(sfs.pop_ids)}
                elif hasattr(sfs, 'sample_sizes'):
                    # If sample_sizes attribute exists, use it directly
                    sample_sizes = sfs.sample_sizes
                else:
                    # Fall back to inferring from shape - use generic names but this might fail
                    from collections import OrderedDict
                    sample_sizes = OrderedDict()
                    for i, n in enumerate(sfs.shape):
                        sample_sizes[f"pop{i}"] = (n - 1) // 2
                
                expected = dadi_inference.diffusion_sfs_dadi(
                    test_params, 
                    sample_sizes, 
                    demo_func, 
                    config["mutation_rate"], 
                    config["genome_length"],
                    pts=[50, 60, 70]  # Standard grid
                )
                
            elif mode == "moments":
                # Use moments to compute expected SFS
                if hasattr(sfs, 'pop_ids') and sfs.pop_ids:
                    # Use the actual population IDs from the SFS
                    sample_sizes = {pop_id: (sfs.shape[i] - 1) // 2 for i, pop_id in enumerate(sfs.pop_ids)}
                elif hasattr(sfs, 'sample_sizes'):
                    # If sample_sizes attribute exists, use it directly
                    sample_sizes = sfs.sample_sizes
                else:
                    # Fall back to inferring from shape
                    from collections import OrderedDict
                    sample_sizes = OrderedDict()
                    for i, n in enumerate(sfs.shape):
                        sample_sizes[f"pop{i}"] = (n - 1) // 2
                
                expected = moments_inference._diffusion_sfs(
                    test_params,
                    demo_func,
                    param_names,
                    sample_sizes,
                    config
                )
            
            if hasattr(sfs, 'folded') and sfs.folded:
                expected = expected.fold()
            
            # Compute Poisson log-likelihood
            expected = np.maximum(expected, 1e-300)  # Avoid log(0)
            ll = np.sum(sfs * np.log(expected) - expected)
            
            likelihood_values.append(ll)
            
        except Exception as e:
            print(f"Error computing likelihood for {param_val}: {e}")
            likelihood_values.append(-np.inf)
    
    return np.array(likelihood_values)


def create_likelihood_plot(param_name: str, grid_values: np.ndarray, 
                          likelihood_values: np.ndarray, fitted_value: float,
                          true_value: float, mode: str, outdir: Path) -> Path:
    """Create and save a likelihood profile plot."""
    plt.figure(figsize=(8, 6), dpi=150)
    
    # Plot likelihood profile
    plt.plot(grid_values, likelihood_values, "-o", linewidth=2.0, markersize=4, 
             label="Profile LL", color='blue', alpha=0.8)
    
    # Add vertical line for MLE
    plt.axvline(x=fitted_value, linestyle="-", linewidth=2, 
               label=f"MLE: {fitted_value:.4g}", color='red')
    
    # Add vertical line for true value
    if true_value is not None and not np.isnan(true_value):
        plt.axvline(x=true_value, linestyle="--", linewidth=2,
                   label=f"True: {true_value:.4g}", color='green')
    
    # Formatting
    plt.xscale("log")
    plt.ylabel("Log-likelihood", fontsize=12)
    plt.xlabel(param_name, fontsize=12)
    plt.title(f"{mode.title()} - Likelihood Profile: {param_name}", fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    outfile = outdir / f"likelihood_profile_{mode}_{param_name}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    
    return outfile


def generate_likelihood_profiles(mode: str, sfs, fitted_params: Dict[str, float],
                                config: Dict, demo_func, ground_truth_params: Dict[str, float],
                                outdir: Path, fixed_params: Dict[str, float]):
    """Generate likelihood profiles for all parameters."""
    profiles_dir = outdir / "likelihood_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parameter information
    param_names = list(config["priors"].keys())
    param_values = np.array([fitted_params[name] for name in param_names])
    lower_bounds = np.array([config["priors"][name][0] for name in param_names])
    upper_bounds = np.array([config["priors"][name][1] for name in param_names])
    
    print(f"Generating {mode} likelihood profiles for {len(param_names)} parameters...")
    
    saved_plots = []
    
    for i, param_name in enumerate(param_names):
        # Skip fixed parameters
        if param_name in fixed_params:
            print(f"  Skipping fixed parameter: {param_name}")
            continue
            
        print(f"  Processing parameter {i+1}/{len(param_names)}: {param_name}")
        
        # Create parameter grid
        grid_values = create_parameter_grid(
            param_name, param_values[i], lower_bounds[i], upper_bounds[i], grid_points=21
        )
        
        # Compute likelihood profile
        likelihood_values = compute_likelihood_profile(
            mode, param_values, i, grid_values, sfs, demo_func, config, fixed_params
        )
        
        # Get true value if available
        true_value = ground_truth_params.get(param_name) if ground_truth_params else None
        
        # Create and save plot
        plot_file = create_likelihood_plot(
            param_name, grid_values, likelihood_values, param_values[i],
            true_value, mode, profiles_dir
        )
        saved_plots.append(plot_file)
        print(f"    Saved: {plot_file.name}")
    
    print(f"Generated {len(saved_plots)} likelihood profile plots in {profiles_dir}")
    return saved_plots


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
    
    # Generate likelihood profiles if optimization was successful
    if best_params is not None:
        try:
            print(f"Generating likelihood profiles for {mode}...")
            fitted_params_dict = {k: float(v) for k, v in zip(start_dict.keys(), best_params)}
            
            # Load ground truth parameters if available - try to find them in the workspace
            ground_truth_params = {}
            # For simulated data, we often have a sim_params.pkl file
            sim_params_file = outdir.parent / "sim_params.pkl" if outdir.parent.exists() else None
            if sim_params_file and sim_params_file.exists():
                try:
                    with open(sim_params_file, 'rb') as f:
                        ground_truth_params = pickle.load(f)
                except:
                    pass
            
            generate_likelihood_profiles(
                mode=mode,
                sfs=sfs,
                fitted_params=fitted_params_dict,
                config=config,
                demo_func=demo_func,
                ground_truth_params=ground_truth_params,
                outdir=mode_outdir,
                fixed_params=fixed_params
            )
        except Exception as e:
            print(f"Warning: Failed to generate likelihood profiles for {mode}: {e}")
    
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
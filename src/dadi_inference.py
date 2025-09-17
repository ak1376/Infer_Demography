#!/usr/bin/env python3
"""
dadi_inference.py – single‑run dadi optimisation
------------------------------------------------
* Dynamic pts grid (based on observed SFS).
* Bounds taken straight from JSON priors.
* Uses raw‑wrapper  ➜  make_extrap_func  ➜  dadi.Inference.opt.
* No files written: everything prints to stdout/stderr.
"""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import datetime
import json
import numpy as np
import dadi
import nlopt
import numdifftools as nd


# ── helper: expected SFS via dadi‑Demes ───────────────────────────────────
def diffusion_sfs_dadi(
    params: list[float],
    sample_sizes: OrderedDict[str, int],
    demo_model,                      # callable(dict) → demes.Graph
    mutation_rate: float,
    sequence_length: int,
    pts: list[int],
) -> dadi.Spectrum:
    # p_dict = {
    #     "N0": params[0],
    #     "N1": params[1],
    #     "N2": params[2],
    #     "m": params[3],
    #     "t_split": params[4]
    # }

    # p_dict = {
    #     "N0": params[0],
    #     "N_bottleneck": params[1],
    #     "N_recover": params[2],
    #     "t_bottleneck_start": params[3],
    #     "t_bottleneck_end": params[4]
    # }

    p_dict = {
        "N0": params[0], 
        "N1": params[1],
        "N2": params[2],
        "m12": params[3],
        "m21": params[4],
        "t_split": params[5]
    }

    # Drosophila three epoch model parameters
    # p_dict = {
    #     "N0": params[0],                    # Ancestral population size
    #     "AFR": params[1],                   # Post expansion African population size  
    #     "EUR_bottleneck": params[2],        # European bottleneck pop size
    #     "EUR_recover": params[3],           # Modern European population size after recovery
    #     "T_AFR_expansion": params[4],       # Expansion of population in Africa
    #     "T_AFR_EUR_split": params[5],       # African-European Divergence
    #     "T_EUR_expansion": params[6]        # European population expansion
    # }
    
    graph  = demo_model(p_dict)

    # print(f' Sample Sizes: {sample_sizes}')
    haploid_sizes = [2 * n for n in sample_sizes.values()]
    # print(f' Haploid Sizes: {haploid_sizes}')
    sampled_demes = list(sample_sizes.keys())

    fs = dadi.Spectrum.from_demes(
        graph,
        sample_sizes = haploid_sizes,
        sampled_demes= sampled_demes,
        pts          = pts,
    )
    fs *= 4 * params[0] * mutation_rate * sequence_length
    return fs


# ── main fitting function (called by your pipeline) ──────────────────────
def fit_model(
    sfs: dadi.Spectrum,
    start_dict: dict[str, float],
    demo_model,
    experiment_config: dict,
    sampled_params: dict | None = None,
    fixed_params: dict[str, float] | None = None,
):
    """
    Run one dadi optimisation; return ([best_params], [best_ll]).
    
    Args:
        sfs: Observed site frequency spectrum
        start_dict: Starting parameter values
        demo_model: Demographic model function
        experiment_config: Configuration dictionary
        sampled_params: Legacy parameter (kept for compatibility)
        fixed_params: Dictionary of parameter names and values to fix during optimization
    """
    priors = experiment_config["priors"]

    # order / start vector / bounds ---------------------------------------
    param_names = list(start_dict.keys())
    p0          = np.array([start_dict[p] for p in param_names])
    lower_b     = [priors[p][0] for p in param_names]
    upper_b     = [priors[p][1] for p in param_names]

    # dynamic pts grid -----------------------------------------------------
    # Handle cases where pop_ids might be None after conversion
    if hasattr(sfs, 'pop_ids') and sfs.pop_ids is not None:
        sample_sizes = OrderedDict(
            (pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape)
        )
    else:
        # Fallback: use generic population names based on SFS dimensions
        pop_names = [f"pop{i}" for i in range(len(sfs.shape))]
        sample_sizes = OrderedDict(
            (pop, (n - 1) // 2) for pop, n in zip(pop_names, sfs.shape)
        )
    n_max_hap = max(2 * n for n in sample_sizes.values())
    pts_l     = [n_max_hap + 20, n_max_hap + 40, n_max_hap + 60]

    # wrap model -----------------------------------------------------------
    mut_rate = experiment_config["mutation_rate"]
    L        = experiment_config["genome_length"]

    def raw_wrapper(params, ns, pts):
        return diffusion_sfs_dadi(
            params, sample_sizes, demo_model, mut_rate, L, pts
        )

    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    # Handle fixed parameters ----------------------------------------------
    import inference_utils
    
    fixed = None
    if fixed_params:
        # Use flexible parameter fixing
        free_indices, fixed_indices, _ = inference_utils.build_fixed_param_mapper(
            param_names, fixed_params
        )
        
        # Validate bounds for fixed parameters
        inference_utils.validate_fixed_params_bounds(
            fixed_params, param_names, lower_b, upper_b
        )
        
        # Create fixed params list for dadi (uses parameter values, not indices)
        fixed = [fixed_params.get(name) for name in param_names]
        
        print(f"  fixing parameters: {fixed_params}")
        
    elif experiment_config["demographic_model"] == "bottleneck" and sampled_params:
        # Legacy bottleneck-specific fixing (kept for backwards compatibility)
        fixed = [
            sampled_params.get("N0"),
            sampled_params.get("N_bottleneck"),
            None, None, None,
        ]

    # optimisation --------------------------------------------------------
    print("▶ dadi custom NLopt optimisation started –", datetime.datetime.now().isoformat(timespec='seconds'))
    print("  lower bounds:", lower_b)
    print("  upper bounds:", upper_b)

    # Use custom NLopt optimization instead of built-in dadi optimization
    seed = dadi.Misc.perturb_params(p0, fold=0.1)
    
    # Convert to log10 space for optimization
    start_log10 = np.log10(np.maximum(seed, 1e-300))  
    lower_log10 = np.log10(np.maximum(lower_b, 1e-300))
    upper_log10 = np.log10(upper_b)
    
    # Handle fixed parameters for NLopt
    if fixed_params:
        # Create mapping for free parameters
        free_indices = [i for i, name in enumerate(param_names) if name not in fixed_params]
        fixed_indices = [i for i, name in enumerate(param_names) if name in fixed_params]
        fixed_values_log10 = [np.log10(max(fixed_params[param_names[i]], 1e-300)) for i in fixed_indices]
        
        # Optimization will be over free parameters only
        free_start = start_log10[free_indices]
        free_lower = lower_log10[free_indices] 
        free_upper = upper_log10[free_indices]
        
        def expand_params(free_params_log10):
            """Expand free parameters to full parameter vector"""
            full_params_log10 = np.zeros(len(param_names))
            full_params_log10[free_indices] = free_params_log10
            full_params_log10[fixed_indices] = fixed_values_log10
            return full_params_log10
    else:
        free_start = start_log10
        free_lower = lower_log10
        free_upper = upper_log10
        expand_params = lambda x: x
    
    def objective_function(free_params_log10, gradient):
        """NLopt objective function (maximize likelihood)"""
        full_params_log10 = expand_params(free_params_log10)
        full_params = 10 ** full_params_log10
        
        try:
            # Compute expected SFS
            expected = func_ex(full_params, sample_sizes, pts_l)
            if sfs.folded:
                expected = expected.fold()
            
            # Poisson log-likelihood
            ll = np.sum(sfs * np.log(np.maximum(expected, 1e-300)) - expected)
            
            # Compute gradient if requested
            if gradient.size > 0:
                def obj_for_grad(x_log10):
                    x_full = 10 ** expand_params(x_log10)
                    exp_sfs = func_ex(x_full, sample_sizes, pts_l)
                    if sfs.folded:
                        exp_sfs = exp_sfs.fold()
                    return np.sum(sfs * np.log(np.maximum(exp_sfs, 1e-300)) - exp_sfs)
                
                grad_fn = nd.Gradient(obj_for_grad, step=1e-4)
                gradient[:] = grad_fn(free_params_log10)
            
            # Print real-time progress (same as moments)
            print(f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(free_params_log10), precision=4)}")
            
            return ll
        except Exception as e:
            print(f"Error in objective: {e}")
            return -np.inf
    
    # Set up and run NLopt optimization
    # opt = nlopt.opt(nlopt.LN_COBYLA, len(free_start))
    opt = nlopt.opt(nlopt.LD_LBFGS, len(free_start))
    opt.set_lower_bounds(free_lower)
    opt.set_upper_bounds(free_upper) 
    opt.set_max_objective(objective_function)
    opt.set_ftol_rel(1e-8)
    opt.set_maxeval(10000)
    
    try:
        best_free_log10 = opt.optimize(free_start)
        best_ll = opt.last_optimum_value()
        status = opt.last_optimize_result()
    except Exception as e:
        print(f"Optimization failed: {e}")
        best_free_log10 = free_start
        best_ll = objective_function(free_start, np.array([]))
        status = nlopt.FAILURE
    
    # Convert back to original scale
    best_full_log10 = expand_params(best_free_log10)
    best_p = 10 ** best_full_log10

    print("✔ finished –", datetime.datetime.now().isoformat(timespec='seconds'))
    print("  LL  :", best_ll)
    print("  params:", best_p)

    return [best_p], [best_ll]   # keep list‑of‑one format


# ── optional CLI for quick testing ---------------------------------------
if __name__ == "__main__":
    import argparse, importlib, pickle

    cli = argparse.ArgumentParser("Standalone dadi single‑fit (no files written)")
    cli.add_argument("--sfs-file", required=True, type=Path)
    cli.add_argument("--config",   required=True, type=Path)
    cli.add_argument("--model-py", required=True, type=str,
                     help="python:module.function returning demes.Graph")
    args = cli.parse_args()

    sfs = pickle.loads(args.sfs_file.read_bytes())
    cfg = json.loads(args.config.read_text())

    mod_path, func_name = args.model_py.split(":")
    demo_func = getattr(importlib.import_module(mod_path), func_name)

    start = {k: (lo+hi)/2 for k,(lo,hi) in cfg["priors"].items()}

    fit_model(sfs, start, demo_func, cfg)
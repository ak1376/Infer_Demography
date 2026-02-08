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
import moments

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

def run_cli(mode: str, sfs_file: Path, config_file: Path, model_py: str, outdir: Path,
            ground_truth: Optional[Path] = None, generate_profiles: bool = False,
            profile_grid_points: int = 41, verbose: bool = False):
    '''
    mode: "dadi", "moments", or "both"
    sfs_file: filepath of the pickle file that contains the SFS (dadi.Spectrum or moments.Spectrum)
    config_file: filepath of the JSON experiment config (see example in configs/)
    model_py: string of the form "module:function" that can be imported to get a function that returns a demes.Graph when called with a param dict
    outdir: parent output directory. For --mode both writes into outdir/{dadi,moments}
    ground_truth: Optional filepath of a pickle or JSON file containing the ground truth simulation parameters. If I specify fixed parameters then it will fix the parameter to the ground truth. But otherwise this is only used for ad hoc visualization of results.
    generate_profiles: Whether to generate 1D likelihood profiles for each parameter. This can be slow, so it's optional.\
    profile_grid_points: Number of grid points for likelihood profiles. Only relevant if generate_profiles is True.
    verbose: Whether to print verbose output during inference.
    '''

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

    # The arguments that should be passed to inference methods should be as follows: 
    # 1. The observed SFS 
    # 2. The demes model function 
    # 3. The starting guess for the parameters. 
    # 4. The fixed parameters (if any)
    # 5. The output directory to write results to

    # Step 1: Create the starting guess for the parameters. I can either calculate the midpoint of the prior distribution on each parameter as the starting point. Or I could take the geometric mean. 
    start_dict = {}
    fixed_params = {}

    # Find the fixed parameters 
    fixed_param_names = config.get('fixed_parameters', {})

    if ground_truth is not None:
        with open(ground_truth, "rb") as f:
            gt_params = pickle.load(f)
        for param_name in fixed_param_names:
            if param_name not in gt_params:
                raise ValueError(f"Parameter {param_name} is specified as fixed but not found in ground truth parameters")
            fixed_params[param_name] = gt_params[param_name]
    else:
        for param_name in fixed_param_names:
            raise ValueError(f"Parameter {param_name} is specified as fixed but no ground truth parameters provided to fix it to")
        
    # For the fixed parameters, the upper and lower bound should be the same and equal to the fixed value. For the free parameters, the upper and lower bound should be taken from the config file. The starting guess should be the geometric mean of the upper and lower bound.
    for param_name, bounds in config['parameters'].items():
        if param_name in fixed_params:
            start_dict[param_name] = fixed_params[param_name]
        else:
            lower, upper = bounds
            if lower <= 0 or upper <= 0:
                raise ValueError(f"Bounds for parameter {param_name} must be positive to calculate geometric mean")
            start_dict[param_name] = np.sqrt(lower * upper)

    
    # Now I need to create a numpy array of the starting values in the same order as the parameters expected by the model function. I can get the parameter names from the model function's signature.
    import inspect
    sig = inspect.signature(model_func)
    param_names = list(sig.parameters.keys())
    start_arr = np.array([start_dict[param_name] for param_name in param_names])

    # Extract the sample sizes 
    
    
    # Now perturb the starting value slightly 
    start_perturbed = moments.Misc.perturb_params(start_arr, fold=0.1)

    # Now I can run moments inference

    # Run inference
    if mode == "moments":
        # Assert that the SFS is a moments.Spectrum
        assert isinstance(sfs, moments.Spectrum), "SFS must be a moments.Spectrum when mode is moments"
        moments_inference.fit_model(
            sfs = sfs,
            demo_model = model_func,
            experiment_config=config,
            start_vec = start_perturbed,
        )




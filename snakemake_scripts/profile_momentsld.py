#!/usr/bin/env python3
"""
Standalone script to compute 1D profile likelihoods for an existing MomentsLD result.

Usage:
    python snakemake_scripts/profile_momentsld.py \
        --ld-root experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD \
        --config  config_files/experiment_config_split_migration_growth.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.MomentsLD_inference import (
    _profile_1d_ld,
    _load_demographic_function,
    DEFAULT_R_BINS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ld-root", required=True, type=Path)
    p.add_argument("--config",  required=True, type=Path)
    p.add_argument("--n-points", type=int, default=None,
                   help="Override profile_points from config")
    args = p.parse_args()

    config = json.loads(args.config.read_text())

    best_fit = pickle.loads((args.ld_root / "best_fit.pkl").read_bytes())
    mv       = pickle.loads((args.ld_root / "means.varcovs.pkl").read_bytes())

    optimal_params = np.array(list(best_fit["best_params"].values()))
    param_names    = list(best_fit["best_params"].keys())

    priors       = config["priors"]
    lower_bounds = np.array([priors[p][0] for p in param_names])
    upper_bounds = np.array([priors[p][1] for p in param_names])

    # means.varcovs.pkl is the empirical_data dict used by objective_function
    empirical_data = mv
    populations    = list(config.get("num_samples", {}).keys())
    normalization  = config.get("ld_normalization", 0)
    r_bins         = np.array(config.get("r_bins", DEFAULT_R_BINS))
    demo_function  = _load_demographic_function(config)

    n_points = args.n_points or int(config.get("profile_points", 21))
    out_dir  = args.ld_root / "likelihood_plots_scaled"

    print(f"Computing profiles for {param_names} → {out_dir}")
    _profile_1d_ld(
        xhat_log10    = np.log10(np.maximum(optimal_params, 1e-300)),
        param_names   = param_names,
        lb_log10      = np.log10(np.maximum(lower_bounds, 1e-300)),
        ub_log10      = np.log10(upper_bounds),
        demographic_model = demo_function,
        r_bins        = r_bins,
        empirical_data= empirical_data,
        populations   = populations,
        normalization = normalization,
        n_points      = n_points,
        widen         = float(config.get("profile_widen", 0.5)),
        out_dir       = out_dir,
        make_plots    = bool(config.get("profile_make_plots", True)),
    )
    print("Done.")


if __name__ == "__main__":
    main()

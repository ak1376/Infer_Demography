#!/usr/bin/env python3
# godambe_correction_LRT/src/sfs_fit_one_start.py
"""
Logic for running ONE LHS-seeded moments SFS fit (one optimizer start) for one
arm + model. Each opt_index is its own Snakemake job, mirroring the LD side's
fit_simple_ld.py / collect_best_fit.py split. Collected + turned into the
FIM/railing diagnostic by src/sfs_collect_and_fim.py.

CLI wrapper: godambe_correction_LRT/snakemake_scripts/sfs_fit_one_start.py
"""

from __future__ import annotations

import importlib

from src.moments_inference_real import fit_model_realdata_scaled


def run_one_sfs_start(arm, model, sfs, cfg, opt_index, n_starts, param_order):
    """Run one LHS-seeded optimizer start of `model` on `sfs`, using `cfg`
    (the loaded experiment_config dict) with the LHS grid sized to `n_starts`
    and seeded at `opt_index`.

    Returns a dict: seed, best_abs, ll_hat, theta_hat, N_anc_implied.
    """
    model_func = getattr(
        importlib.import_module("src.demes_models"), f"{model}_model"
    )

    cfg_run = dict(cfg)
    cfg_run["num_optimizations"] = n_starts   # size the LHS grid to match
    cfg_run["opt_seed"] = opt_index

    best_abs, ll_hat, theta_hat, N_anc_implied = fit_model_realdata_scaled(
        sfs=sfs,
        demo_model_abs=model_func,
        experiment_config=cfg_run,
        param_order=param_order,
        verbose=False,
    )

    return dict(
        seed=opt_index,
        best_abs=best_abs,
        ll_hat=float(ll_hat),
        theta_hat=float(theta_hat),
        N_anc_implied=float(N_anc_implied),
    )

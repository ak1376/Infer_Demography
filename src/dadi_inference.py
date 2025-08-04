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


# ── helper: expected SFS via dadi‑Demes ───────────────────────────────────
def diffusion_sfs_dadi(
    params: list[float],
    sample_sizes: OrderedDict[str, int],
    demo_model,                      # callable(dict) → demes.Graph
    mutation_rate: float,
    sequence_length: int,
    pts: list[int],
) -> dadi.Spectrum:
    p_dict = {
        "N0": params[0],
        "N1": params[1],
        "N2": params[2],
        "m": params[3],
        "t_split": params[4]
    }
    graph  = demo_model(p_dict)

    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())

    fs = dadi.Spectrum.from_demes(
        graph,
        sample_sizes = haploid_sizes,
        sampled_demes= sampled_demes,
        pts          = pts,
    )
    fs *= mutation_rate * sequence_length
    return fs


# ── main fitting function (called by your pipeline) ──────────────────────
def fit_model(
    sfs: dadi.Spectrum,
    start_dict: dict[str, float],
    demo_model,
    experiment_config: dict,
    sampled_params: dict | None = None,
):
    """
    Run one dadi optimisation; return ([best_params], [best_ll]).
    """
    priors = experiment_config["priors"]

    # order / start vector / bounds ---------------------------------------
    param_names = list(start_dict.keys())
    p0          = np.array([start_dict[p] for p in param_names])
    lower_b     = [priors[p][0] for p in param_names]
    upper_b     = [priors[p][1] for p in param_names]

    # dynamic pts grid -----------------------------------------------------
    sample_sizes = OrderedDict(
        (pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape)
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

    # optional fixed params (bottleneck example) --------------------------
    fixed = None
    if experiment_config["demographic_model"] == "bottleneck":
        fixed = [
            sampled_params.get("N0"),
            sampled_params.get("N_bottleneck"),
            None, None, None,
        ]

    # optimisation --------------------------------------------------------
    print("▶ dadi optimisation started –", datetime.datetime.now().isoformat(timespec='seconds'))
    print("  lower bounds:", lower_b)
    print("  upper bounds:", upper_b)

    seed = dadi.Misc.perturb_params(p0, fold=0.1)
    best_p, best_ll = dadi.Inference.opt(
        seed, sfs, func_ex, pts=pts_l,
        lower_bound=lower_b, upper_bound=upper_b,
        algorithm=nlopt.LN_BOBYQA,
        maxeval=10_000,
        verbose=1,
        fixed_params=fixed,
    )

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

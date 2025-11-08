#!/usr/bin/env python3
"""
dadi_inference.py – single-run dadi optimisation

• Uses param names from start_dict order (no hard-coded N0/N1/etc).
• Sample sizes taken from sfs.pop_ids to preserve your config labels.
• Wraps a demes-based model builder; config can be threaded by your wrapper.
• Optional fixed parameters.
• NLopt (COBYLA) in log10 space; Poisson LL objective.
"""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import datetime
import numpy as np
import dadi
import nlopt
import numdifftools as nd


# ── helper: expected SFS via dadi-Demes ───────────────────────────────────
def diffusion_sfs_dadi(
    params_vec,
    param_names,  # list[str]: order matches params_vec
    sample_sizes,  # OrderedDict[str,int] diploid counts per pop
    demo_model,  # callable(param_dict) → demes.Graph  (your wrapper may accept config)
    mutation_rate: float,
    sequence_length: float,
    pts,  # [int, int, int]
    config=None,  # passed through to demo_model by your wrapper (if supported)
):
    """
    Build an expected SFS for dadi given a parameter vector and demes model.
    Theta is scaled using the first parameter (assumed ancestral-size-like).
    """
    p_dict = {k: float(v) for k, v in zip(param_names, params_vec)}
    # Your moments_dadi_inference wrapper should have created demo_model so that:
    #   demo_model(p_dict)  works, and internally forwards `config` if supported.
    graph = demo_model(p_dict)

    fs = dadi.Spectrum.from_demes(
        graph,
        sample_sizes=[2 * n for n in sample_sizes.values()],  # haploid counts
        sampled_demes=list(sample_sizes.keys()),
        pts=pts,
    )
    theta = (
        4.0
        * float(p_dict[param_names[0]])
        * float(mutation_rate)
        * float(sequence_length)
    )
    fs *= theta
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
        sfs: dadi.Spectrum (folded or unfolded)
        start_dict: {param_name: start_value} — order defines param order
        demo_model: callable(param_dict) → demes.Graph  (wrapper may accept config)
        experiment_config: JSON dict with 'priors', 'mutation_rate', 'genome_length'
        sampled_params: (kept for backwards compat; unused unless you add logic)
        fixed_params: {param_name: fixed_value}
    """
    priors = experiment_config["priors"]

    # ── parameter order / vectors / bounds ───────────────────────────────
    param_names = list(start_dict.keys())
    p0 = np.array([start_dict[p] for p in param_names], dtype=float)
    lower_b = np.array([priors[p][0] for p in param_names], dtype=float)
    upper_b = np.array([priors[p][1] for p in param_names], dtype=float)

    # ── sample sizes from SFS (preserve your config pop labels) ──────────
    if hasattr(sfs, "pop_ids") and sfs.pop_ids is not None:
        sample_sizes = OrderedDict(
            (pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape)
        )
    else:
        # fallback generic names
        pop_names = [f"pop{i}" for i in range(len(sfs.shape))]
        sample_sizes = OrderedDict(
            (pop, (n - 1) // 2) for pop, n in zip(pop_names, sfs.shape)
        )

    # dynamic integration grid (safe, slightly generous)
    n_max_hap = max(2 * n for n in sample_sizes.values())
    pts_l = [n_max_hap + 20, n_max_hap + 40, n_max_hap + 60]

    mut_rate = float(experiment_config["mutation_rate"])
    L = float(experiment_config["genome_length"])

    # Closure that matches dadi's expected signature f(params, ns, pts)
    def raw_wrapper(params_vec, ns, pts):
        return diffusion_sfs_dadi(
            params_vec,
            param_names=param_names,
            sample_sizes=sample_sizes,
            demo_model=demo_model,
            mutation_rate=mut_rate,
            sequence_length=L,
            pts=pts,
            config=experiment_config,  # forwarded if your wrapper accepts it
        )

    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    # ── fixed parameter handling (free vs fixed indices) ─────────────────
    fixed_params = fixed_params or {}
    free_idx = [i for i, n in enumerate(param_names) if n not in fixed_params]
    fix_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
    fix_vals = np.array(
        [max(float(fixed_params[param_names[i]]), 1e-300) for i in fix_idx], dtype=float
    )

    # Bound check for fixeds
    for i, v in zip(fix_idx, fix_vals):
        if not (lower_b[i] <= v <= upper_b[i]):
            raise ValueError(
                f"Fixed value {param_names[i]}={v} outside bounds [{lower_b[i]}, {upper_b[i]}]."
            )

    # ── log10 space setup ────────────────────────────────────────────────
    seed = dadi.Misc.perturb_params(p0, fold=0.1)
    start_log10 = np.log10(np.maximum(seed, 1e-300))
    lower_log10 = np.log10(np.maximum(lower_b, 1e-300))
    upper_log10 = np.log10(np.maximum(upper_b, 1e-300))

    if free_idx:
        free_start = start_log10[free_idx]
        free_lower = lower_log10[free_idx]
        free_upper = upper_log10[free_idx]
    else:
        # all fixed
        free_start = np.array([], dtype=float)
        free_lower = np.array([], dtype=float)
        free_upper = np.array([], dtype=float)

    def expand_free_to_full(log10_free_vec: np.ndarray) -> np.ndarray:
        full = np.zeros_like(start_log10)
        if free_idx:
            full[free_idx] = log10_free_vec
        if fix_idx:
            full[fix_idx] = np.log10(fix_vals)
        return full

    # ── objective (maximize Poisson LL) ──────────────────────────────────
    def objective_log10(log10_free_vec, grad):
        full_log10 = expand_free_to_full(np.asarray(log10_free_vec, dtype=float))
        full_params = 10.0**full_log10
        try:
            expected = func_ex(full_params, sample_sizes, pts_l)
            if getattr(sfs, "folded", False):
                expected = expected.fold()
            expected = np.maximum(expected, 1e-300)
            ll = float(np.sum(sfs * np.log(expected) - expected))

            if grad.size > 0:

                def ll_only(x_log10):
                    x_full = 10.0 ** expand_free_to_full(x_log10)
                    e = func_ex(x_full, sample_sizes, pts_l)
                    if getattr(sfs, "folded", False):
                        e = e.fold()
                    e = np.maximum(e, 1e-300)
                    return float(np.sum(sfs * np.log(e) - e))

                grad_fn = nd.Gradient(ll_only, step=1e-4)
                grad[:] = grad_fn(log10_free_vec)

            print(
                f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(log10_free_vec), precision=4)}"
            )
            return ll
        except Exception as e:
            print(f"Error in objective: {e}")
            return -np.inf

    # ── optimise ────────────────────────────────────────────────────────
    print(
        "▶ dadi NLopt optimisation –",
        datetime.datetime.now().isoformat(timespec="seconds"),
    )
    print("  lower bounds:", lower_b)
    print("  upper bounds:", upper_b)
    if free_idx:
        opt = nlopt.opt(nlopt.LN_COBYLA, len(free_start))
        opt.set_lower_bounds(free_lower)
        opt.set_upper_bounds(free_upper)
        opt.set_max_objective(objective_log10)
        opt.set_ftol_rel(1e-8)
        opt.set_maxeval(10000)

        try:
            best_free_log10 = opt.optimize(free_start)
            best_ll = opt.last_optimum_value()
            status = opt.last_optimize_result()
        except Exception as e:
            print(f"Optimization failed: {e}")
            best_free_log10 = free_start
            best_ll = objective_log10(free_start, np.array([]))
            status = nlopt.FAILURE
        best_full_log10 = expand_free_to_full(best_free_log10)
    else:
        # everything fixed: just evaluate once
        best_full_log10 = expand_free_to_full(np.array([], dtype=float))
        best_ll = objective_log10(np.array([], dtype=float), np.array([]))
        status = nlopt.SUCCESS

    best_params = 10.0**best_full_log10

    print("✔ finished –", datetime.datetime.now().isoformat(timespec="seconds"))
    print("  status :", status)
    print("  LL     :", best_ll)
    print("  params :", best_params)

    return [best_params], [best_ll]


# ── quick CLI for ad-hoc testing ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse, importlib, pickle, json

    cli = argparse.ArgumentParser("Standalone dadi single-fit (no files written)")
    cli.add_argument("--sfs-file", required=True, type=Path)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument(
        "--model-py",
        required=True,
        type=str,
        help="python:module.function returning demes.Graph",
    )
    args = cli.parse_args()

    sfs = pickle.loads(args.sfs_file.read_bytes())
    cfg = json.loads(args.config.read_text())

    mod_path, func_name = args.model_py.split(":")
    demo_func = getattr(importlib.import_module(mod_path), func_name)

    start = {k: (lo + hi) / 2 for k, (lo, hi) in cfg["priors"].items()}

    fit_model(sfs, start, demo_func, cfg)

#!/usr/bin/env python3
"""
dadi_inference.py – single-run dadi optimisation

• Uses param names from start_dict order (no hard-coded N0/N1/etc).
• Sample sizes taken from sfs.pop_ids to preserve your config labels.
• Wraps a demes-based model builder; config can be threaded by your wrapper.
• Optional fixed parameters.
• NLopt (COBYLA) in log10 space; Poisson LL objective.

CHANGES (per your request):
  1) REMOVE ALL GRADIENT COMPUTATION (numdifftools) — COBYLA is derivative-free.
  2) PRINT ONLY EVERY 25 EVALUATIONS, but with the SAME EXACT INFO STRING as before.
  3) SPEED: adjust NLopt hyperparams to reduce wall time (early stopping + fewer max evals).
     (Does NOT change model / pts_l / objective formula.)
"""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import datetime
import numpy as np
import dadi
import nlopt

# REMOVED: import numdifftools as nd


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

    # (kept) GPU flag exists but we don't change behavior here
    _use_gpu = config.get("use_gpu_dadi", False) if config else False

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
    """
    # ── GPU configuration ───────────────────────────────────────────────
    use_gpu = experiment_config.get("use_gpu_dadi", False)

    if use_gpu:
        print(f"[DADI GPU] Enabling GPU acceleration...")
        dadi.cuda_enabled(True)

        # Keep your GPU debug block (unchanged)
        try:
            import pycuda  # noqa: F401
            import pycuda.driver as cuda

            cuda.init()

            device_count = cuda.Device.count()
            if device_count > 0:
                device = cuda.Device(0)
                device_name = device.name()
                memory_info = cuda.mem_get_info()
                free_memory = memory_info[0] / 1024**3  # GB
                total_memory = memory_info[1] / 1024**3  # GB

                print(f"[DADI GPU] Successfully initialized GPU: {device_name}")
                print(
                    f"[DADI GPU] GPU memory: {free_memory:.1f} GB free / {total_memory:.1f} GB total"
                )
                print(f"[DADI GPU] dadi.cuda_enabled() = {dadi.cuda_enabled()}")
            else:
                print(f"[DADI GPU] Warning: No CUDA devices found")
        except Exception as e:
            print(f"[DADI GPU] Warning: GPU initialization failed: {e}")
    else:
        print(f"[DADI GPU] GPU acceleration disabled (use_gpu_dadi=False)")
        dadi.cuda_enabled(False)

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
        pop_names = [f"pop{i}" for i in range(len(sfs.shape))]
        sample_sizes = OrderedDict(
            (pop, (n - 1) // 2) for pop, n in zip(pop_names, sfs.shape)
        )

    # dynamic integration grid (UNCHANGED from your script)
    n_max_hap = max(2 * n for n in sample_sizes.values())
    pts_l = [n_max_hap + 10, n_max_hap + 20, n_max_hap + 30]

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
            config=experiment_config,
        )

    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    # ── fixed parameter handling (free vs fixed indices) ─────────────────
    fixed_params = fixed_params or {}
    free_idx = [i for i, n in enumerate(param_names) if n not in fixed_params]
    fix_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
    fix_vals = np.array(
        [max(float(fixed_params[param_names[i]]), 1e-300) for i in fix_idx], dtype=float
    )

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
    # CHANGE: ignore gradients completely (COBYLA is derivative-free)
    eval_counter = {"n": 0}
    PRINT_EVERY = 1

    def objective_log10(log10_free_vec, grad):
        _ = grad  # unused on purpose
        full_log10 = expand_free_to_full(np.asarray(log10_free_vec, dtype=float))
        full_params = 10.0**full_log10
        try:
            expected = func_ex(full_params, sample_sizes, pts_l)
            if getattr(sfs, "folded", False):
                expected = expected.fold()
            expected = np.maximum(expected, 1e-300)
            ll = float(np.sum(sfs * np.log(expected) - expected))

            # SAME EXACT INFO STRING, just printed every 25 evals (and eval #1)
            eval_counter["n"] += 1
            if eval_counter["n"] == 1 or (eval_counter["n"] % PRINT_EVERY) == 0:
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

        # SPEED CHANGES (NLopt hyperparams only):
        # - looser ftol for faster exit
        # - add xtol (stop when params stop moving)
        # - fewer max evals
        opt.set_ftol_rel(1e-5)  # was 1e-8
        opt.set_xtol_rel(1e-4)  # new
        opt.set_maxeval(2000)  # was 10000
        opt.set_initial_step(0.1)  # new, good default in log10 space

        try:
            import time

            if use_gpu:
                print(f"[DADI GPU] Starting GPU-accelerated optimization...")
            else:
                print(f"[DADI CPU] Starting CPU optimization...")

            start_time = time.time()

            best_free_log10 = opt.optimize(free_start)
            best_ll = opt.last_optimum_value()
            status = opt.last_optimize_result()

            end_time = time.time()
            optimization_time = end_time - start_time
            if use_gpu:
                print(
                    f"[DADI GPU] Optimization completed in {optimization_time:.2f} seconds"
                )
            else:
                print(
                    f"[DADI CPU] Optimization completed in {optimization_time:.2f} seconds"
                )

        except Exception as e:
            print(f"Optimization failed: {e}")
            best_free_log10 = free_start
            best_ll = objective_log10(free_start, np.array([]))
            status = nlopt.FAILURE
        best_full_log10 = expand_free_to_full(best_free_log10)
    else:
        best_full_log10 = expand_free_to_full(np.array([], dtype=float))
        best_ll = objective_log10(np.array([], dtype=float), np.array([]))
        status = nlopt.SUCCESS

    best_params = 10.0**best_full_log10

    print("✔ finished –", datetime.datetime.now().isoformat(timespec="seconds"))
    print("  status :", status)
    print("  LL     :", best_ll)
    print("  params :", best_params)

    # ── GPU cleanup ──────────────────────────────────────────────────────
    if use_gpu:
        print(f"[DADI GPU] Disabling GPU acceleration...")
        dadi.cuda_enabled(False)
        print(f"[DADI GPU] GPU disabled. dadi.cuda_enabled() = {dadi.cuda_enabled()}")

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

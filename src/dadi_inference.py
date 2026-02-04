#!/usr/bin/env python3
"""
dadi_inference.py – single-run dadi optimisation (NO fixed/free params)

• Uses param names from start_dict order (no hard-coded N0/N1/etc).
• Sample sizes taken from sfs.pop_ids to preserve your config labels.
• Wraps a demes-based model builder.
• NLopt (LD_LBFGS) in log10 space; Poisson LL objective (manual form).
• Theta is explicit: model is multiplied by (4 * params[0] * mu * L).
"""

from __future__ import annotations
from collections import OrderedDict
import numpy as np
import dadi
import nlopt
import numdifftools as nd


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
    # ── GPU configuration ───────────────────────────────────────────────
    use_gpu = experiment_config.get("use_gpu_dadi", False)

    if use_gpu:
        print("[DADI GPU] Enabling GPU acceleration...")
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
                free_b, total_b = cuda.mem_get_info()
                free_gb = free_b / 1024**3
                total_gb = total_b / 1024**3

                print(f"[DADI GPU] Successfully initialized GPU: {device_name}")
                print(f"[DADI GPU] GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
                print(f"[DADI GPU] dadi.cuda_enabled() = {dadi.cuda_enabled()}")
            else:
                print("[DADI GPU] Warning: No CUDA devices found")
        except Exception as e:
            print(f"[DADI GPU] Warning: GPU initialization failed: {e}")
    else:
        print("[DADI GPU] GPU acceleration disabled (use_gpu_dadi=False)")
        dadi.cuda_enabled(False)

    priors = experiment_config["priors"]

    # ── parameter order / vectors / bounds ───────────────────────────────
    param_names = list(start_dict.keys())
    print(f"Parameter names (in order): {param_names}")

    p0 = np.array([start_dict[p] for p in param_names], dtype=float)
    lower_b = np.array([priors[p][0] for p in param_names], dtype=float)
    upper_b = np.array([priors[p][1] for p in param_names], dtype=float)

    # ── sample sizes from SFS (preserve your config pop labels) ──────────
    if hasattr(sfs, "pop_ids") and sfs.pop_ids is not None:
        sampled_demes = list(sfs.pop_ids)
        sample_sizes = OrderedDict((pop, (n - 1) // 2) for pop, n in zip(sampled_demes, sfs.shape))
    # else:
    #     sampled_demes = [f"pop{i}" for i in range(len(sfs.shape))]
    #     sample_sizes = OrderedDict((pop, (n - 1) // 2) for pop, n in zip(sampled_demes, sfs.shape))

    print(f"Sampled demes: {sampled_demes}")
    print(f"Sample sizes: {sample_sizes}")

    # dynamic integration grid (unchanged from your script)
    n_max_hap = max(2 * n for n in sample_sizes.values())
    pts_base = experiment_config["optimization"]["dadi"]["pts"]
    pts_l = [n_max_hap + pts_base[0], n_max_hap + pts_base[1], n_max_hap + pts_base[2]]

    print(pts_l)

    mut_rate = float(experiment_config["mutation_rate"])
    L = float(experiment_config["genome_length"])

    # haploid ns for dadi
    ns = tuple(int(dim - 1) for dim in sfs.shape)

    # Closure that matches dadi's expected signature f(params, ns, pts)
    # Returns expected *counts* (theta is explicit via params[0])
    def raw_wrapper(params_vec, ns_local, pts):
        p_dict = {k: float(v) for k, v in zip(param_names, params_vec)}
        graph = demo_model(p_dict)

        model = dadi.Spectrum.from_demes(
            graph,
            sample_sizes=list(ns_local),   # haploid sample sizes
            sampled_demes=sampled_demes,   # preserve labels/order
            pts=pts,
        )

        print("=" * 40)
        print(f"Model (dadi.Spectrum) created with params: {p_dict}")
        print(f"Model (dadi.Spectrum) created with shape: {model.shape}")
        print(f'Model SFS (dadi spectrum): {model}')
        print(f'Graph: {graph}')
        print("=" * 40)

        theta = 4.0 * float(params_vec[0]) * mut_rate * L

        print(f'Theta: {theta}')
        print(f'Scaled SFS: {model * theta}')
        return model * theta

    func_ex = dadi.Numerics.make_extrap_func(raw_wrapper)

    # --- log10 space (optimize ALL params) ---
    seed = dadi.Misc.perturb_params(p0, fold=experiment_config["optimization"]["perturb_fold"])
    x0 = np.log10(np.maximum(seed, 1e-300))
    lb = np.log10(np.maximum(lower_b, 1e-300))
    ub = np.log10(np.maximum(upper_b, 1e-300))

    eval_counter = {"n": 0}
    print_every = int(experiment_config["optimization"]["dadi"].get("print_every", 5000)) #TODO: Change this later

    def obj_log10(xlog10):
        params = 10.0 ** np.asarray(xlog10, float)
        try:
            model = func_ex(params, ns, pts_l)
            model = dadi.Spectrum(np.maximum(model, 1e-300))

            # Poisson LL up to an additive constant
            ll = float(np.sum(sfs * np.log(model) - model))
        except Exception as e:
            print(f"Error in objective: {e}")
            return -np.inf

        eval_counter["n"] += 1
        if eval_counter["n"] % print_every == 0:
            print(f"[LL={ll:.6g}] log10={np.array2string(np.asarray(xlog10), precision=4)}")
        return ll

    grad_fn = nd.Gradient(obj_log10, step=experiment_config["optimization"]["dadi"]["grad_step"])

    def nlopt_objective(x, grad):
        ll = obj_log10(x)
        if grad.size > 0:
            grad[:] = grad_fn(x)
        return ll

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
    opt.set_lower_bounds(lb)   # log10
    opt.set_upper_bounds(ub)   # log10
    opt.set_max_objective(nlopt_objective)
    opt.set_initial_step(experiment_config["optimization"]["dadi"]["initial_step"])
    opt.set_ftol_rel(experiment_config["optimization"]["dadi"]["ftol_rel"])
    opt.set_maxeval(experiment_config["optimization"]["dadi"]["maxeval"])

    xhat_log10 = opt.optimize(x0)
    ll_val = opt.last_optimum_value()
    opt_params = 10.0 ** xhat_log10

    # ── GPU cleanup ──────────────────────────────────────────────────────
    if use_gpu:
        print("[DADI GPU] Disabling GPU acceleration...")
        dadi.cuda_enabled(False)
        print(f"[DADI GPU] GPU disabled. dadi.cuda_enabled() = {dadi.cuda_enabled()}")

    return [opt_params], [ll_val]

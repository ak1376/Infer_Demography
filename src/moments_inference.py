#!/usr/bin/env python3
# moments_inference.py – single-run moments optimisation
# Matches dadi_inference.py conventions (param order, pop labels, fixing, log10).

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import moments
import nlopt
import numdifftools as nd
import datetime


# ── expected SFS via moments-Demes ─────────────────────────────────────
def _diffusion_sfs(
    params_vec:     np.ndarray,
    demo_model,                       # callable(param_dict) → demes.Graph
    param_names:    List[str],        # order matches params_vec
    sample_sizes:   OrderedDict[str, int],  # diploid counts per pop
    experiment_config: Dict,
) -> moments.Spectrum:
    """
    Build an expected SFS for a given parameter vector.
    Moments picks integration internally; we supply theta explicitly.
    """
    p_dict = {k: float(v) for k, v in zip(param_names, params_vec)}

    graph = demo_model(p_dict)  # your wrapper may internally use experiment_config if needed

    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())

    theta = (
        float(p_dict[param_names[0]])     # first param treated as ancestral size
        * 4.0
        * float(experiment_config["mutation_rate"])
        * float(experiment_config["genome_length"])
    )

    return moments.Spectrum.from_demes(
        graph,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
        theta=theta,
    )


# ── optional scatter plot helper (unchanged) ───────────────────────────
def save_scatterplots(
    true_vecs:   List[Dict[str, float]],
    est_vecs:    List[Dict[str, float]],
    ll_vec:      List[float],
    param_names: List[str],
    outfile:     Path,
    *,
    label: str = "moments",
) -> None:
    norm   = colors.Normalize(vmin=min(ll_vec), vmax=max(ll_vec))
    cmap   = cm.get_cmap("viridis")
    colour = cmap(norm(ll_vec))

    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for i, p in enumerate(param_names):
        ax = axes[0, i]
        x  = [d[p] for d in true_vecs]
        y  = [d[p] for d in est_vecs]
        ax.scatter(x, y, s=20, c=colour)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.7, color="grey")
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"{label} {p}")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="log-likelihood")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ── main optimiser wrapper ─────────────────────────────────────────────
def fit_model(
    sfs,
    start_dict: Dict[str, float],         # initial parameter dict
    demo_model,                           # callable → demes.Graph
    experiment_config: Dict,
    *,
    sampled_params: Dict | None = None,   # kept for backwards compatibility
    fixed_params: Dict[str, float] | None = None,
) -> Tuple[List[np.ndarray], List[float]]:
    priors = experiment_config["priors"]

    # param order & vectors / bounds
    param_names = list(start_dict.keys())
    start_vec   = np.array([start_dict[p] for p in param_names], dtype=float)
    lb_full     = np.array([priors[p][0] for p in param_names], dtype=float)
    ub_full     = np.array([priors[p][1] for p in param_names], dtype=float)

    # sample sizes (diploid) from SFS; fall back if pop_ids missing
    if hasattr(sfs, "pop_ids") and sfs.pop_ids is not None:
        sample_sizes = OrderedDict((pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape))
    else:
        pop_names = [f"pop{i}" for i in range(len(sfs.shape))]
        sample_sizes = OrderedDict((pop, (n - 1) // 2) for pop, n in zip(pop_names, sfs.shape))

    # small perturbation for robustness
    seed_vec   = moments.Misc.perturb_params(start_vec, fold=0.1)
    start_full = np.clip(seed_vec, lb_full, ub_full)

    # fixed parameter handling
    fixed_by_name = dict(fixed_params or {})
    fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_by_name]
    free_idx  = [i for i, n in enumerate(param_names) if n not in fixed_by_name]

    x_full0 = start_full.copy()
    for i in fixed_idx:
        x_full0[i] = float(fixed_by_name[param_names[i]])

    # validate fixed within bounds
    for i in fixed_idx:
        v = x_full0[i]
        if not (lb_full[i] <= v <= ub_full[i]):
            raise ValueError(f"Fixed value {param_names[i]}={v} outside bounds [{lb_full[i]}, {ub_full[i]}].")

    # objective in log10 space (match dadi path)
    def pack_free_to_full(x_free: np.ndarray) -> np.ndarray:
        x_full = x_full0.copy()
        for j, i in enumerate(free_idx):
            x_full[i] = float(x_free[j])
        return x_full

    def obj_log10(xlog10_free: np.ndarray) -> float:
        x_free = 10.0 ** np.asarray(xlog10_free, dtype=float)
        x_full = pack_free_to_full(x_free)
        try:
            expected = _diffusion_sfs(x_full, demo_model, param_names, sample_sizes, experiment_config)
            expected = np.maximum(expected, 1e-300)
            if getattr(sfs, "folded", False):
                expected = expected.fold()
            ll = float(np.sum(sfs * np.log(expected) - expected))
            return ll
        except Exception as e:
            print(f"Error in objective: {e}")
            return -np.inf

    # all fixed? just evaluate once
    if len(free_idx) == 0:
        opt_params = x_full0
        ll_val = obj_log10(np.array([], dtype=float))
        status = "ALL_FIXED_EVAL"
    else:
        lb_free = np.array([lb_full[i] for i in free_idx], float)
        ub_free = np.array([ub_full[i] for i in free_idx], float)
        x0_free = np.array([x_full0[i] for i in free_idx], float)

        grad_fn = nd.Gradient(obj_log10, step=1e-4)

        def nlopt_objective(xlog10_free, grad):
            ll = obj_log10(xlog10_free)
            if grad.size > 0:
                grad[:] = grad_fn(xlog10_free)
            print(f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(xlog10_free), precision=4)}")
            return ll

        opt = nlopt.opt(nlopt.LD_LBFGS, len(free_idx))
        opt.set_lower_bounds(np.log10(np.maximum(lb_free, 1e-300)))
        opt.set_upper_bounds(np.log10(np.maximum(ub_free, 1e-300)))
        opt.set_max_objective(nlopt_objective)
        opt.set_ftol_rel(1e-8)
        opt.set_maxeval(10000)

        try:
            x_free_hat_log10 = opt.optimize(np.log10(np.maximum(x0_free, 1e-300)))
            ll_val = opt.last_optimum_value()
            status = opt.last_optimize_result()
        except Exception as e:
            print(f"NLopt optimization failed: {e}")
            x_free_hat_log10 = np.log10(np.maximum(x0_free, 1e-300))
            ll_val = obj_log10(x_free_hat_log10)
            status = nlopt.FAILURE

        x_free_hat = 10.0 ** x_free_hat_log10
        opt_params = pack_free_to_full(x_free_hat)

    # short log (optional)
    log_dir  = Path(experiment_config.get("log_dir", ".")) / "moments"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "optim_single.txt").write_text(
        "# moments custom NLopt optimisation\n"
        f"# finished: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"# status: {status}\n"
        f"# best_ll: {ll_val}\n"
        f"# opt_params: {opt_params.tolist()}\n"
    )

    return [opt_params], [ll_val]

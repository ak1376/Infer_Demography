# moments_inference.py ── “split-models” helper mirroring dadi version
# -------------------------------------------------------------------
# Changes vs your current version:
#   1) demes order comes from sfs.pop_ids (NOT from priors or diploid sample_sizes)
#   2) haploid sample sizes passed to moments are [n-1 for n in sfs.shape] (debug-script match)
#   3) composite Poisson log-likelihood uses log(exp + eps) to avoid -inf
#   4) saves fitted expected SFS to out_dir if provided in experiment_config
#   5) keeps nlopt LD_LBFGS + log10 parameterization exactly like your debug script

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import moments
import nlopt
import numdifftools as nd


# ───────────────────────── diffusion helper ────────────────────────────
# NOTE: This helper is now made consistent with the debug-script convention:
# sample_sizes passed to moments.from_demes are HAPLOID sizes = (axis_len - 1).
def _diffusion_sfs(
    init_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],  # callable(param_dict) → demes.Graph
    param_names: List[str],
    sampled_demes: List[str],
    haploid_sizes: List[int],
    experiment_config: Dict,
) -> moments.Spectrum:
    """
    Build a moments expected SFS for a given parameter vector.
    Uses debug-script conventions:
      - sampled_demes ordering is explicit (typically sfs.pop_ids)
      - sample_sizes are haploid sizes (axis_len - 1)
      - theta = 4 * N0 * mu * L, where N0 is param_names[0]
    """
    p_dict = dict(zip(param_names, map(float, init_vec)))
    print("p_dict being passed to demo_model:", p_dict)
    graph = demo_model(p_dict)

    muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
    N0 = float(p_dict[param_names[0]])
    theta = 4.0 * N0 * muL

    return moments.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=theta,
    )


# ───────────────────────── scatter-plot helper ─────────────────────────
def save_scatterplots(
    true_vecs: List[Dict[str, float]],
    est_vecs: List[Dict[str, float]],
    ll_vec: List[float],
    param_names: List[str],
    outfile: Path,
    *,
    label: str = "moments",
) -> None:
    """
    Draw one scatter panel per parameter, coloured by log-likelihood, and
    place a single colour-bar outside the grid.
    """
    norm = colors.Normalize(vmin=min(ll_vec), vmax=max(ll_vec))
    cmap = cm.get_cmap("viridis")
    colour = cmap(norm(ll_vec))

    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)

    for i, p in enumerate(param_names):
        ax = axes[0, i]
        x = [d[p] for d in true_vecs]
        y = [d[p] for d in est_vecs]

        ax.scatter(x, y, s=20, c=colour)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.7, color="grey")

        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"{label} {p}")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="log-likelihood"
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ───────────────────────── optimisation wrapper ────────────────────────
def fit_model(
    sfs,
    start_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],  # callable → demes.Graph
    experiment_config: Dict,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run a **single** moments optimisation (nlopt.LD_LBFGS) in log10 space,
    matching your debugging script.

    Also saves the final fitted expected SFS if experiment_config["out_dir"] is set.
    """
    # ----- settings / priors ------------------------------------------
    priors = experiment_config["priors"]
    param_names = list(priors.keys())  # must match demo_model expected ordering

    lb_full = np.array([priors[p][0] for p in param_names], dtype=float)
    ub_full = np.array([priors[p][1] for p in param_names], dtype=float)

    # ----- match debug script sample-size semantics --------------------
    # moments.from_demes expects HAPLOID sample sizes (n_hap = axis_len - 1)
    haploid_sizes = [int(n) - 1 for n in sfs.shape]

    # very important: keep deme order aligned to the observed SFS
    # (this is the cleanest + most consistent convention)
    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("sfs.pop_ids is missing/empty; cannot determine sampled_demes order.")

    # ----- constants ---------------------------------------------------
    muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
    eps = float(experiment_config.get("moments_ll_eps", 1e-300))
    ftol_rel = float(experiment_config.get("nlopt_ftol_rel", 1e-8))

    # ----- expected SFS under params (log10 space) ---------------------
    def expected_sfs_from_log10(log10_params: np.ndarray) -> moments.Spectrum:
        params = 10.0 ** np.asarray(log10_params, dtype=float)
        init_vec = np.asarray(params, dtype=float)

        # use the unified helper so conventions are identical everywhere
        return _diffusion_sfs(
            init_vec=init_vec,
            demo_model=demo_model,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            experiment_config=experiment_config,
        )

    # ----- composite Poisson log-likelihood (debug-script match) -------
    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = expected_sfs_from_log10(log10_params)
        exp_arr = np.asarray(exp_sfs, dtype=float)
        obs_arr = np.asarray(sfs, dtype=float)
        return float(np.sum(np.log(exp_arr + eps) * obs_arr - exp_arr))

    grad_fun = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(x, grad):
        ll = loglikelihood(x)
        if grad.size > 0:
            grad[:] = grad_fun(x)
        return ll

    # Checking
    print("param_names:", param_names)
    print("start_vec:", start_vec)
    print("lb:", lb_full)
    print("ub:", ub_full)
    print("sfs type:", type(sfs))
    print("sfs.pop_ids:", getattr(sfs, "pop_ids", None))
    print("sfs.shape:", sfs.shape)
    print("haploid_sizes:", haploid_sizes)
    print("sampled_demes used:", sampled_demes)

    # sanity: expected SFS at start
    exp0 = expected_sfs_from_log10(np.log10(start_vec))
    print("exp0 min/max:", np.min(exp0), np.max(exp0), "zeros:", np.sum(np.asarray(exp0)==0))


    # ----- nlopt (LBFGS) in log10 space --------------------------------
    x0 = np.log10(np.asarray(start_vec, dtype=float))
    lb = np.log10(lb_full)
    ub = np.log10(ub_full)

    opt = nlopt.opt(nlopt.LD_LBFGS, len(param_names))
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_max_objective(objective)
    opt.set_ftol_rel(ftol_rel)

    x_hat = opt.optimize(x0)  # log10 params
    ll_val = float(opt.last_optimum_value())
    opt_params = 10.0 ** np.asarray(x_hat, dtype=float)  # natural-space params

    # ----- fitted expected SFS under the optimized params --------------
    fitted_sfs = expected_sfs_from_log10(np.log10(opt_params))

    # Save if out_dir provided
    out_dir = experiment_config.get("out_dir", None)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "moments_fitted_expected_sfs.npy", np.asarray(fitted_sfs, dtype=float))

    # ----- wrap & return ----------------------------------------------
    best_params = [opt_params]  # list length 1
    best_lls = [ll_val]
    return best_params, best_lls

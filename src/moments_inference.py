# moments_inference.py ── “split-models” helper mirroring dadi version
# -------------------------------------------------------------------
# ➊ Uses the same parameter-dict convention as dadi_inference.py.
# ➋ Grid-points (pts) are *not* passed to moments.Spectrum.from_demes()
#    because that function doesn’t accept them.  Moments chooses
#    integration points internally.
# ➌ Lower/upper bounds for the optimiser are taken directly from the
#    JSON priors, so you edit them only in one place.
#
# NOTE:  scatter-plot helper is unchanged and kept verbatim.

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import moments


# ───────────────────────── diffusion helper ────────────────────────────
def _diffusion_sfs(
    init_vec: np.ndarray,
    demo_model,  # callable(param_dict) → demes.Graph
    param_names: List[str],
    sample_sizes: OrderedDict[str, int],
    experiment_config: Dict,
) -> moments.Spectrum:
    """
    Build a frequency spectrum for a given parameter vector (`init_vec`).
    No `pts` argument is supplied – moments picks a sensible grid itself.
    """
    p_dict = dict(zip(param_names, init_vec))
    # print(f"Fitting parameters: {p_dict}")

    graph = demo_model(p_dict)

    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())

    theta = (
        p_dict[param_names[0]]  # N0 as reference
        * 4
        * experiment_config["mutation_rate"]
        * experiment_config["genome_length"]
    )

    return moments.Spectrum.from_demes(
        graph,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
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

    Parameters
    ----------
    true_vecs, est_vecs : list[dict]
        Matching dictionaries of “true” and estimated parameter values.
    ll_vec : list[float]
        Log-likelihoods (one per point) – used only for colour scale.
    param_names : list[str]
        Order of parameters to plot.
    outfile : pathlib.Path
        PNG file to write.
    label : str, default "moments"
        Text prefix on the y–axis (“moments N0”, …).
    """
    # ── colour mapping ────────────────────────────────────────────────
    norm = colors.Normalize(vmin=min(ll_vec), vmax=max(ll_vec))
    cmap = cm.get_cmap("viridis")
    colour = cmap(norm(ll_vec))

    # ── scatter panels ────────────────────────────────────────────────
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

    # ── reserve space & add colour-bar axis ───────────────────────────
    fig.subplots_adjust(right=0.88)  # grid occupies left 88 %
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [l, b, w, h] in figure coords

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="log-likelihood"
    )

    # ── final tweaks & save ───────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 0.88, 1])  # keep panels inside the grid box
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ───────────────────────── optimisation wrapper ────────────────────────
def fit_model(
    sfs,
    start_dict: Dict[str, float],  # initial parameter *dict*
    demo_model,  # callable → demes.Graph
    experiment_config: Dict,
    *,
    sampled_params: Dict | None = None,
    fixed_params: Dict[str, float] | None = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run a **single** moments optimisation and return the best parameter
    vector / log‑likelihood wrapped in 1‑element lists.

    Args:
        sfs: Observed site frequency spectrum
        start_dict: Starting parameter values
        demo_model: Demographic model function
        experiment_config: Configuration dictionary
        sampled_params: Legacy parameter for bottleneck model (kept for compatibility)
        fixed_params: Dictionary of parameter names and values to fix during optimization
    """
    # ----- pull settings ------------------------------------------------
    priors = experiment_config["priors"]
    # # we keep TOP_K in the cfg for symmetry, but enforce it to be 1 here
    # top_k  = experiment_config.get("top_k", 1)
    # assert top_k == 1, "Using one optimiser run ⇒ top_k must be 1"

    # ----- order & initial vector --------------------------------------
    param_names = list(start_dict.keys())
    start_vec = np.array([start_dict[p] for p in param_names])

    # ----- sample sizes (convert SFS axes → diploid counts) -------------
    sample_sizes = OrderedDict(
        (pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape)
    )

    # --- PARAM → GRAPH sanity check -----------------------------------
    p_dict = dict(zip(param_names, start_vec))
    g = demo_model(p_dict)

    def _list_migs(graph):
        rows = []
        for m in getattr(graph, "migrations", []):
            # demes.Graph has either 'demes=[a,b]' (old) or (source,dest) (new)
            src = getattr(m, "source", None)
            dst = getattr(m, "dest", None)
            if src is None or dst is None:
                # older demes uses 'demes' 2-list
                pair = getattr(m, "demes", None)
                if pair:
                    src, dst = pair[0], pair[1]
            rate = getattr(m, "rate", None)
            rows.append((src, dst, rate))
        return rows

    print("[CHECK] param_names:", param_names)
    print("[CHECK] start_vec  :", start_vec)
    print("[CHECK] fixed_params:", fixed_params if fixed_params else {})
    print("[CHECK] migrations in graph (src→dst, rate):", _list_migs(g))

    # ----- bounds from priors ------------------------------------------
    lower_bounds = [priors[p][0] for p in param_names]
    upper_bounds = [priors[p][1] for p in param_names]

    # ----- single optimisation call ------------------------------------
    import datetime
    from pathlib import Path

    log_dir = Path(experiment_config.get("log_dir", ".")) / "moments"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "optim_single.txt"

    # (optional) add a small perturbation; remove if you want deterministic starts
    seed_vec = moments.Misc.perturb_params(start_vec, fold=0.1)

    # Handle fixed parameters ----------------------------------------------
    import inference_utils

    fixed_params_list = None
    if fixed_params:
        # Use flexible parameter fixing
        free_indices, fixed_indices, _ = inference_utils.build_fixed_param_mapper(
            param_names, fixed_params
        )

        # Validate bounds for fixed parameters
        inference_utils.validate_fixed_params_bounds(
            fixed_params, param_names, lower_bounds, upper_bounds
        )

        # Create fixed params list for moments
        fixed_params_list = inference_utils.create_fixed_params_list_for_moments(
            param_names, fixed_params
        )

        print(f"  fixing parameters: {fixed_params}")

    elif experiment_config["demographic_model"] == "bottleneck" and sampled_params:
        # Legacy bottleneck-specific fixing (kept for backwards compatibility)
        fixed_params_list = [
            sampled_params.get("N0"),
            sampled_params.get("N_bottleneck"),
            None,
            None,
            None,
        ]

    # Custom NLopt optimization (based on original sfs_optimize_cli.py patterns)
    import datetime
    import nlopt
    import numdifftools as nd
    from pathlib import Path

    log_dir = Path(experiment_config.get("log_dir", ".")) / "moments"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "optim_single.txt"

    # (optional) add a small perturbation; remove if you want deterministic starts
    seed_vec = moments.Misc.perturb_params(start_vec, fold=0.1)

    # Convert to numpy arrays
    lb_full = np.array(lower_bounds, float)
    ub_full = np.array(upper_bounds, float)
    start_full = np.clip(seed_vec, lb_full, ub_full)

    # Prepare fixed parameters
    fixed_by_name = fixed_params if fixed_params else {}

    # Build parameter packing for fixed parameters
    fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_by_name]
    free_idx = [i for i, n in enumerate(param_names) if n not in fixed_by_name]

    # Initialize full parameter vector with fixed values
    x_full0 = start_full.copy()
    for i in fixed_idx:
        x_full0[i] = float(fixed_by_name[param_names[i]])

    # Validate fixed parameter bounds
    for i in fixed_idx:
        v = x_full0[i]
        if not (lb_full[i] <= v <= ub_full[i]):
            raise ValueError(
                f"Fixed value {param_names[i]}={v} outside bounds [{lb_full[i]}, {ub_full[i]}]."
            )

    print(f"  fixing parameters: {fixed_by_name}")

    # If all parameters are fixed, just evaluate and return
    if len(free_idx) == 0:
        opt_params = x_full0
        expected = _diffusion_sfs(
            opt_params, demo_model, param_names, sample_sizes, experiment_config
        )
        ll_val = float(np.sum(sfs * np.log(np.maximum(expected, 1e-300)) - expected))
    else:
        # Prepare free parameter optimization
        lb_free = np.array([lb_full[i] for i in free_idx], float)
        ub_free = np.array([ub_full[i] for i in free_idx], float)
        x0_free = np.array([x_full0[i] for i in free_idx], float)

        def pack_free_to_full(x_free):
            x_full = x_full0.copy()
            for j, i in enumerate(free_idx):
                x_full[i] = float(x_free[j])
            return x_full

        def obj_free_log10(xlog10_free):
            """Objective function in log10 space for free parameters"""
            x_free = 10.0 ** np.asarray(xlog10_free, float)
            x_full = pack_free_to_full(x_free)
            try:
                expected = _diffusion_sfs(
                    x_full, demo_model, param_names, sample_sizes, experiment_config
                )
                expected = np.maximum(expected, 1e-300)
                ll = float(np.sum(sfs * np.log(expected) - expected))
                return ll
            except Exception as e:
                print(f"Error in objective: {e}")
                return -np.inf

        # Finite difference gradient
        grad_fn = nd.Gradient(obj_free_log10, step=1e-4)

        def nlopt_objective(xlog10_free, grad):
            ll = obj_free_log10(xlog10_free)
            if grad.size > 0:
                grad[:] = grad_fn(xlog10_free)
            print(
                f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(xlog10_free), precision=4)}"
            )
            return ll

        # Set up NLopt optimizer
        opt = nlopt.opt(nlopt.LD_LBFGS, len(free_idx))
        opt.set_lower_bounds(np.log10(lb_free))
        opt.set_upper_bounds(np.log10(ub_free))
        opt.set_max_objective(nlopt_objective)
        opt.set_ftol_rel(1e-8)
        opt.set_maxeval(500)

        try:
            x_free_hat_log10 = opt.optimize(np.log10(x0_free))
            ll_val = opt.last_optimum_value()
            status = opt.last_optimize_result()
        except Exception as e:
            print(f"NLopt optimization failed: {e}")
            x_free_hat_log10 = np.log10(x0_free)
            ll_val = obj_free_log10(x_free_hat_log10)
            status = nlopt.FAILURE

        # Convert back to full parameter space
        x_free_hat = 10.0**x_free_hat_log10
        opt_params = pack_free_to_full(x_free_hat)

    # write short optimiser log
    log_file.write_text(
        "# moments custom NLopt optimisation\n"
        f"# finished: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"# status: {status}\n"
        f"# best_ll: {ll_val}\n"
        f"# opt_params: {opt_params}\n"
    )

    # ----- wrap & return ------------------------------------------------
    best_params = [opt_params]  # lists of length 1
    best_lls = [ll_val]
    return best_params, best_lls

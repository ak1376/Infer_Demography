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
from typing      import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import moments


# ───────────────────────── diffusion helper ────────────────────────────
def _diffusion_sfs(
    init_vec:     np.ndarray,
    demo_model,                       # callable(param_dict) → demes.Graph
    param_names:  List[str],
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
        p_dict[param_names[0]]          # N0 as reference
        * 4
        * experiment_config["mutation_rate"]
        * experiment_config["genome_length"]
    )

    return moments.Spectrum.from_demes(
        graph,
        sample_sizes  = haploid_sizes,
        sampled_demes = sampled_demes,
        theta         = theta,
    )


# ───────────────────────── scatter-plot helper ─────────────────────────
def save_scatterplots(
    true_vecs:   List[Dict[str, float]],
    est_vecs:    List[Dict[str, float]],
    ll_vec:      List[float],
    param_names: List[str],
    outfile:     Path,
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
    norm   = colors.Normalize(vmin=min(ll_vec), vmax=max(ll_vec))
    cmap   = cm.get_cmap("viridis")
    colour = cmap(norm(ll_vec))

    # ── scatter panels ────────────────────────────────────────────────
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)

    for i, p in enumerate(param_names):
        ax = axes[0, i]
        x  = [d[p] for d in true_vecs]
        y  = [d[p] for d in est_vecs]

        ax.scatter(x, y, s=20, c=colour)
        ax.plot(ax.get_xlim(), ax.get_xlim(),
                ls="--", lw=0.7, color="grey")

        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"{label} {p}")

    # ── reserve space & add colour-bar axis ───────────────────────────
    fig.subplots_adjust(right=0.88)       # grid occupies left 88 %
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [l, b, w, h] in figure coords

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax,
                 label="log-likelihood")

    # ── final tweaks & save ───────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 0.88, 1])  # keep panels inside the grid box
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ───────────────────────── optimisation wrapper ────────────────────────
def fit_model(
    sfs,
    start_dict: Dict[str, float],         # initial parameter *dict*
    demo_model,                           # callable → demes.Graph
    experiment_config: Dict,
    *,
    sampled_params: Dict | None = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run *num_optimizations* Powell searches; return the best `top_k`
    parameter vectors and their log-likelihoods.
    """
    num_opt = experiment_config["num_optimizations"]
    top_k   = experiment_config["top_k"]
    priors  = experiment_config["priors"]

    # ---- parameter order & start vector -------------------------------
    param_names = list(start_dict.keys())
    start_vec   = np.array([start_dict[p] for p in param_names])
    print(f'Starting parameters: {start_dict}')

    # ---- sample sizes (diploid) ---------------------------------------
    sample_sizes = OrderedDict(
        (p, (n - 1) // 2)                # convert axis length → diploids
        for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    # ---- bounds from priors -------------------------------------------
    lower_bounds = [priors[p][0] for p in param_names]
    upper_bounds = [priors[p][1] for p in param_names]

    # ---- single optimisation replicate --------------------------------
    def _optimise(seed_vec: np.ndarray, tag: str):
        xopt = moments.Inference.optimize_log_lbfgsb(
            seed_vec,
            sfs,
            lambda p, n: _diffusion_sfs(
                p, demo_model, param_names, sample_sizes, experiment_config
            ),
            multinom     = False,
            verbose      = 1,
            flush_delay  = 0.0,
            maxiter      = 10_000,
            full_output  = True,
            lower_bound  = lower_bounds,
            upper_bound  = upper_bounds,
            fixed_params = [
                sampled_params.get("N0"),
                sampled_params.get("N_bottleneck"),
                None, None, None,
            ] if sampled_params else None,
        )

        p_opt = xopt[0]
        ll_val = xopt[1]

        return p_opt, ll_val

    # ---- replicate loop ------------------------------------------------
    fits: List[Tuple[np.ndarray, float]] = []
    for i in range(num_opt):
        seed_vec = np.copy(start_vec)
        # seed_vec = moments.Misc.perturb_params(start_vec, fold=0.1)
        fits.append(_optimise(seed_vec, tag=f"optim_{i:04d}"))

    # ---- top-k ---------------------------------------------------------
    fits.sort(key=lambda t: t[1], reverse=True)
    best_params = [p  for p, _ in fits[:top_k]]
    best_lls    = [ll for _, ll in fits[:top_k]]
    return best_params, best_lls

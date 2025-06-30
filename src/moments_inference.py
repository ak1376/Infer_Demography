# moments_inference.py  – now with scatter-plot utility
import matplotlib.pyplot as plt
import moments
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from matplotlib import cm, colors


# ---------------------------------------------------------------------
# diffusion helper (unchanged)
# ---------------------------------------------------------------------
def diffusion_sfs(g, parameters, sample_sizes, experiment_config):
    demography     = g
    sampled_demes  = list(sample_sizes.keys())
    sample_sizes   = [n * 2 for n in sample_sizes.values()]

    return moments.Spectrum.from_demes(
        demography,
        sample_sizes=sample_sizes,
        sampled_demes=sampled_demes,
        theta=parameters[0] * experiment_config["genome_length"]
                            * experiment_config["mutation_rate"] * 4,
    )


# ---------------------------------------------------------------------
# ❶ scatter-plot helper  ------------------------------------------------
# ---------------------------------------------------------------------
def save_scatterplots(
    true_vecs:   list[dict[str, float]],
    est_vecs:    list[dict[str, float]],
    ll_vec:      list[float],               # ← NEW!  one log-lik value per replicate
    param_names: list[str],
    outfile:     Path,
    label: str = "moments",
) -> None:
    """
    Draw true vs estimated panels, colouring each point by its log-likelihood
    (better fits = darker colours).

    Parameters
    ----------
    true_vecs   : list of dicts – ground-truth parameters (len = #replicates)
    est_vecs    : list of dicts – inferred params   (same length/order)
    ll_vec      : list of float – log-likelihood for *each* replicate
    param_names : ordered list of parameter names to show
    outfile     : where to save the .png
    label       : y-axis label prefix (default “moments”)
    """
    # colour map -------------------------------------------------------
    norm = colors.Normalize(vmin=min(ll_vec), vmax=max(ll_vec))
    cmap = cm.get_cmap("viridis")
    colour = cmap(norm(ll_vec))

    # one panel per parameter -----------------------------------------
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for i, p in enumerate(param_names):
        ax = axes[0, i]
        x = [d[p] for d in true_vecs]
        y = [d[p] for d in est_vecs]
        ax.scatter(x, y, s=20, c=colour)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.8, color="grey")
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"{label} {p}")

    # shared colourbar -------------------------------------------------
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes.ravel().tolist(), shrink=0.8,
                        label="log-likelihood")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------
# ❷ fitting routine  ---------------------------------------------------
# ---------------------------------------------------------------------
def fit_model(
    sfs,
    start,                       # REQUIRED
    g,
    experiment_config,
    sampled_params=None,
) -> tuple[list[list[float]], float]:
    """
    Fit a *moments* demographic model.

    • `start` **must** be supplied; each replicate perturbs that same vector.
    • Returns:
          best_params   – the `top_k` best parameter vectors
          best_ll       – the log-likelihood of the very best fit
    """
    if start is None:
        raise ValueError("`start` may not be None – supply an initial vector.")

    num_opt = experiment_config["num_optimizations"]
    top_k   = experiment_config["top_k"]

    sample_sizes = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    def _optimise(init_vec):
        result = moments.Inference.optimize_powell(
            init_vec,
            sfs,
            lambda p, n: diffusion_sfs(g, p, sample_sizes, experiment_config),
            multinom=False,
            verbose=0,
            flush_delay=0.0,
            maxiter=1000,
            full_output=True,
            lower_bound=[1e-6] * len(init_vec),
            fixed_params=[
                sampled_params.get("N0"),
                sampled_params.get("N_bottleneck"),
                None, None, None,
            ] if sampled_params else None,
        )
        return result[0], result[1]          # params, log-likelihood

    fits = []
    for i in tqdm(range(num_opt), desc="moments optimisations"):
        # init = moments.Misc.perturb_params(start, fold=i * 0.1)
        init = moments.Misc.perturb_params(start, fold=0.1)
        fits.append(_optimise(init))

    fits.sort(key=lambda t: t[1], reverse=True)          # best first
    best_params = [p  for p, ll in fits[:top_k]]
    best_lls    = [ll for p, ll in fits[:top_k]]
    return best_params, best_lls


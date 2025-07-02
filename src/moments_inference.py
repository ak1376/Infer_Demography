# moments_inference.py – harmonised with dadi_inference.py
# --------------------------------------------------------
# This version mirrors the parameter‑handling conventions used in
# dadi_inference.py: we pass parameter *dicts* to the demographic‑model
# builder, keep `param_names` in a single list, and let every optimiser
# replicate begin from a perturbed copy of an initial vector built from
# that dictionary.

from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import moments
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# diffusion helper (parameter‑dict aware, like dadi_inference.diffusion_sfs)
# ---------------------------------------------------------------------

def diffusion_sfs(
    init_vec: np.ndarray,
    demo_model,               # callable that takes **param_dict** → demes.Graph
    param_names: list[str],
    sample_sizes: OrderedDict,
    experiment_config: dict,
):
    """Expected SFS under the diffusion approximation (moments)."""
    # rebuild parameter dict from the optimisation vector
    p_dict = {k: v for k, v in zip(param_names, init_vec)}
    demography = demo_model(p_dict)
    sampled_demes = list(sample_sizes.keys())
    haploid_sizes = [n * 2 for n in sample_sizes.values()]

    # θ = 4 * N0 * μ * L  (using N0 as reference, as in the original code)
    theta = p_dict[param_names[0]] * 4 * experiment_config["mutation_rate"] * experiment_config["genome_length"]

    return moments.Spectrum.from_demes(
        demography,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
        theta=theta,
    )

# ---------------------------------------------------------------------
# scatter‑plot helper (unchanged)
# ---------------------------------------------------------------------

def save_scatterplots(
    true_vecs: list[dict[str, float]],
    est_vecs: list[dict[str, float]],
    ll_vec: list[float],
    param_names: list[str],
    outfile: Path,
    *,
    label: str = "moments",
) -> None:
    
    print(f'True vectors: {true_vecs}')  # debug output
    print(f'Estimated vectors: {est_vecs}')  # debug output
    """Draw one panel per parameter, coloured by log‑likelihood."""
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
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.8, color="grey")
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"{label} {p}")

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes.ravel().tolist(), shrink=0.8,
                 label="log‑likelihood")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------
# fitting routine – mirrors dadi_inference.fit_model
# ---------------------------------------------------------------------

def fit_model(
    sfs,
    start_dict: dict[str, float],      # initial *dictionary* (matches dadi)
    demo_model,                        # callable returning a demes.Graph
    experiment_config: dict,
    *,
    sampled_params: dict | None = None,
):
    """Run *num_optimizations* Powell searches and keep the top‑k fits."""

    num_opt = experiment_config["num_optimizations"]
    top_k = experiment_config["top_k"]

    param_names = list(start_dict.keys())
    start_vec = np.array([start_dict[p] for p in param_names])

    # diploid sample sizes for each deme
    sample_sizes = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    def _optimise(init_vec: np.ndarray, param_names, tag: str):
        result = moments.Inference.optimize_powell(
            init_vec,
            sfs,
            lambda p, n: diffusion_sfs(p, demo_model, param_names, sample_sizes, experiment_config),
            multinom=False,
            verbose=1,
            flush_delay=0.0,
            maxiter=5000,
            full_output=True,
            lower_bound=[1e-6] * len(init_vec),
            upper_bound=[np.inf, np.inf, np.inf, 1.0, np.inf],
            fixed_params=[
                sampled_params.get("N0"),
                sampled_params.get("N_bottleneck"),
                None, None, None,
            ] if sampled_params else None,
        )
        return result[0], result[1]

    # ── run replicates (perturb 10 % around the supplied start_dict) ────
    fits: list[tuple[np.ndarray, float]] = []
    for i in range(num_opt):
        init = start_vec.copy()
        # init = moments.Misc.perturb_params(start_vec, fold=0.1)

        fits.append(_optimise(init, param_names, tag=f"optim_{i:04d}"))

    # keep the top‑k (highest log‑likelihood)
    fits.sort(key=lambda t: t[1], reverse=True)
    best_params = [p for p, _ in fits[:top_k]]
    best_lls = [ll for _, ll in fits[:top_k]]
    return best_params, best_lls

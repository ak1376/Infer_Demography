import dadi
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def diffusion_sfs(g, parameters, sample_sizes, experiment_config):
    """Expected SFS under diffusion approximation."""
    pts_l         = [20, 30, 40]
    demography    = g
    sampled_demes = list(sample_sizes.keys())
    sample_sizes  = [n * 2 for n in sample_sizes.values()]

    return dadi.Spectrum.from_demes(
        demography,
        sample_sizes=sample_sizes,
        sampled_demes=sampled_demes,
        pts=pts_l,
    )


# ---------------------------------------------------------------------
# main fitting routine
# ---------------------------------------------------------------------
def fit_model(
    sfs,
    start,                       # <-- REQUIRED; must be a list/array
    g,
    experiment_config,
    *,
    sampled_params=None,
):
    """
    Fit a dadi demographic model.

    * `start` **must** be provided.  
      Each optimisation replicate simply perturbs that fixed vector by 10 %.

    Returns the `top_k` best parameter vectors (highest log-likelihood).
    """
    if start is None:
        raise ValueError("`start` may not be None – supply an initial vector.")

    pts_l      = [20, 30, 40]
    num_opt    = experiment_config["num_optimizations"]
    top_k      = experiment_config["top_k"]

    sample_sizes = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    # ------------------------------------------------------------
    # one optimisation run
    # ------------------------------------------------------------
    def _optimise(init_vec):
        result = dadi.Inference.optimize_log_powell(
            init_vec,
            sfs,
            lambda p, n, pts=None: diffusion_sfs(g, p, sample_sizes, experiment_config),
            pts=pts_l,
            multinom=False,
            verbose=0,
            flush_delay=0.0,
            maxiter=1000,
            full_output=True,
            lower_bound=[1e-6] * len(init_vec),   # keep guard against negatives
            fixed_params=[
                sampled_params.get("N0"),
                sampled_params.get("N_bottleneck"),
                None, None, None,
            ] if sampled_params else None,
        )
        params, loglik = result[0], result[1]
        return params, loglik

    # ------------------------------------------------------------
    # run replicates – always perturb the supplied `start`
    # ------------------------------------------------------------
    fits = []
    for i in tqdm(range(num_opt), desc="dadi optimisations"):
        # init = dadi.Misc.perturb_params(start, fold=i*0.1)
        init = dadi.Misc.perturb_params(start, fold=0.1)
        fits.append(_optimise(init))

    # ------------------------------------------------------------
    # return the top-k
    # ------------------------------------------------------------
    fits.sort(key=lambda t: t[1], reverse=True)
    best_params = [p  for p, ll in fits[:top_k]]
    best_lls    = [ll for p, ll in fits[:top_k]]
    return best_params, best_lls

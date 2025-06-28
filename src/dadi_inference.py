import dadi
from collections import OrderedDict
import numpy as np

def diffusion_sfs(g, parameters, sample_sizes, experiment_config):
    """
    Get expected SFS under diffusion approximation
    """
    pts_l = [20,30,40]
    demography = g
    sampled_demes = [p for p in sample_sizes.keys()]
    sample_sizes = [n * 2 for n in sample_sizes.values()]
    
    return dadi.Spectrum.from_demes(
        demography,
        sample_sizes=sample_sizes,
        sampled_demes=sampled_demes,
        pts=pts_l
    )

# ------ helper ---------------------------------------------------------------
def _random_start(priors: dict[str, list[float]],
                  rng:    np.random.Generator) -> list[float]:
    """Uniformly draw one parameter vector from the prior box."""
    vec = []
    for low, high in priors.values():
        v = rng.uniform(low, high)
        if isinstance(low, int) and isinstance(high, int):
            v = int(round(v))
        vec.append(v)
    return vec


# ------ main -----------------------------------------------------------------
def fit_model(
    sfs,
    start,                       # ← **may be None**
    g,
    experiment_config,
    sampled_params=None,
):
    """
    Fit a dadi demographic model.

    * If *start* is **None**, a fresh random initial vector is drawn from
      the priors **for every optimisation replicate**.
    * Otherwise the supplied *start* is perturbed by 10 %.

    Returns *top_k* best parameter vectors sorted by log-likelihood.
    """
    # bookkeeping ------------------------------------------------------
    pts_l              = [20, 30, 40]                  # projection grid
    priors             = experiment_config["priors"]
    num_opt            = experiment_config["num_optimizations"]
    top_k              = experiment_config["top_k"]
    rng                = np.random.default_rng(experiment_config.get("seed"))

    sample_sizes = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    # one optimisation run --------------------------------------------
    def _optimise(init_vec: list[float]):
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
            fixed_params=[sampled_params.get("N0"), sampled_params.get("N_bottleneck"),
                          None, None, None] if sampled_params else None,
            lower_bound=[0] * len(init_vec),  # enforce non-negativity

        )
        params, loglik = result[0], result[1]
        return params, loglik

    # run replicates ---------------------------------------------------
    fits = []
    for _ in range(num_opt):
        if start is None:
            init = _random_start(priors, rng)
        else:
            init = dadi.Misc.perturb_params(start, fold=0.1)
        fits.append(_optimise(init))

    # pick top-k -------------------------------------------------------
    fits.sort(key=lambda t: t[1], reverse=True)          # best first
    return [params for params, _ in fits[:top_k]]
import moments
from collections import OrderedDict
from tqdm import tqdm


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
# main fitting routine
# ---------------------------------------------------------------------
def fit_model(
    sfs,
    start,                       # <-- REQUIRED; no random priors
    g,
    experiment_config,
    sampled_params=None,
):
    """
    Fit a *moments* demographic model.

    * `start` is mandatory.  Every replicate perturbs that same vector
      via `moments.Misc.perturb_params(start, fold=0.1)`.
    * Returns the `top_k` parameter vectors with best log-likelihood.
    """
    if start is None:
        raise ValueError("`start` may not be None – supply an initial vector.")

    # book-keeping -----------------------------------------------------
    num_opt = experiment_config["num_optimizations"]
    top_k   = experiment_config["top_k"]

    sample_sizes = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    # ------------------------------------------------------------
    # single optimisation helper
    # ------------------------------------------------------------
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
            lower_bound=[1e-6] * len(init_vec),      # keep parameters ≥ 0
            fixed_params=[
                sampled_params.get("N0"),
                sampled_params.get("N_bottleneck"),
                None, None, None,
            ] if sampled_params else None,
        )
        params, loglik = result[0], result[1]
        return params, loglik

    # ------------------------------------------------------------
    # run replicates
    # ------------------------------------------------------------
    fits = []
    for _ in tqdm(range(num_opt), desc="moments optimisations"):
        init = moments.Misc.perturb_params(start, fold=0.1)
        fits.append(_optimise(init))

    # ------------------------------------------------------------
    # pick top-k
    # ------------------------------------------------------------
    fits.sort(key=lambda t: t[1], reverse=True)
    return [params for params, _ in fits[:top_k]]

import moments
from collections import OrderedDict
import numpy as np

def diffusion_sfs(g, parameters, sample_sizes, experiment_config):
    """
    Get expected SFS under diffusion approximation
    """
    demography = g
    sampled_demes = [p for p in sample_sizes.keys()]
    sample_sizes = [n * 2 for n in sample_sizes.values()]

    # print(f'Mutation Rate: {experiment_config["mutation_rate"]}')

    if parameters[0] <= 0:        # or any other you want to defend
        print(f"Parameters: {parameters}")
        raise ValueError("N0 went ≤ 0 – reject this step")
    
    return moments.Spectrum.from_demes(
        demography,
        sample_sizes=sample_sizes,
        sampled_demes=sampled_demes,
        theta=parameters[0] * experiment_config['genome_length'] * experiment_config['mutation_rate'] * 4,
    )

def _random_start(priors: dict[str, list[float]],
                  rng:    np.random.Generator) -> list[float]:
    """Sample one start vector uniformly from the `priors` block."""
    vec = []
    for low, high in priors.values():
        val = rng.uniform(low, high)
        if isinstance(low, int) and isinstance(high, int):
            val = int(round(val))
        vec.append(val)

    print(f"Random start vector: {vec}")
    return vec


def fit_model(
    sfs,
    start,                       # may be None  ←─────────────────────────────┐
    g,
    experiment_config,
    sampled_params=None,
) -> list[float]:
    """
    Fit demographic model to *sfs* using Powell optimisation.

    If *start* is None we sample a random initial vector from the prior
    **for each optimisation replicate** (so you'll get *num_optimizations*
    independent starts).  Otherwise we perturb the supplied *start* by 10 %.
    """
    rng = np.random.default_rng(experiment_config.get("seed"))

    sample_sizes = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    num_optimizations = experiment_config["num_optimizations"]
    top_k             = experiment_config["top_k"]
    priors            = experiment_config["priors"]

    fits: list[tuple[list[float], float]] = []

    # ------------------------------------------------------------------
    # helper to perform ONE optimisation given an initial vector
    # ------------------------------------------------------------------
    def _optimise(init_vec: list[float]) -> tuple[list[float], float]:
        result = moments.Inference.optimize_powell(
            init_vec,
            sfs,
            lambda p, n: diffusion_sfs(g, p, sample_sizes, experiment_config),
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

    # ------------------------------------------------------------------
    # run however many optimisation replicates the user asked for
    # ------------------------------------------------------------------
    for _ in range(num_optimizations):
        if start is None:
            init = _random_start(priors, rng)
        else:
            init = moments.Misc.perturb_params(start, fold=0.1)
        fits.append(_optimise(init))

    # ------------------------------------------------------------------
    # pick *top_k* by log-likelihood and return just the parameter vectors
    # ------------------------------------------------------------------
    fits.sort(key=lambda t: t[1], reverse=True)           # best first
    top_fits = [params for params, _ in fits[:top_k]]
    return top_fits
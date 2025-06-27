import moments
from collections import OrderedDict
from tqdm import tqdm

def diffusion_sfs(g, parameters, sample_sizes, experiment_config):
    """
    Get expected SFS under diffusion approximation
    """
    demography = g
    sampled_demes = [p for p in sample_sizes.keys()]
    sample_sizes = [n * 2 for n in sample_sizes.values()]
    
    return moments.Spectrum.from_demes(
        demography,
        sample_sizes=sample_sizes,
        sampled_demes=sampled_demes,
        theta=parameters[0] * experiment_config['genome_length'] * experiment_config['mutation_rate'] * 4,
    )

def fit_model(sfs, start, g, experiment_config, sampled_params=None) -> list[float]:
    """
    Fit demographic model to provided SFS
    """
    sample_sizes = OrderedDict((p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape))
    # use Powell's method because m is not differentiable at zero

    # Do num_optimizations optimizations and return the top k results (based on the log-likelihood) in a pickle file
    num_optimizations = experiment_config['num_optimizations']
    top_k = experiment_config['top_k']

    if num_optimizations > 1:
        fits = []
        for _ in tqdm(range(num_optimizations)):
            perturbed_start = moments.Misc.perturb_params(start, fold=0.1)
            fit = moments.Inference.optimize_powell(
                perturbed_start,
                sfs,
                lambda p, n: diffusion_sfs(g, p, sample_sizes, experiment_config),
                multinom=False,
                verbose=0,
                flush_delay=0.0,
                maxiter=1000,
                full_output=True,  # Get full output including log-likelihood
                fixed_params=[sampled_params['N0'], sampled_params['N_bottleneck'], None, None, None] if sampled_params else None
            )
            fit = fit[0]
            ll = fit[1]  # Log-likelihood is the second element in the tuple
            fits.append((fit, ll))
        
        # Sort by log-likelihood and return the top k fits
        fits.sort(key=lambda x: x[1], reverse=True)
        top_fits = [fit for fit, _ in fits[:top_k]]
        return top_fits

    else:
        fit = moments.Inference.optimize_powell(
            start,
            sfs,
            lambda p, n: diffusion_sfs(g, p, sample_sizes, experiment_config),
            multinom=False,
            verbose=0,
            flush_delay=0.0,
            maxiter=1000,
            fixed_params=[sampled_params['N0'], sampled_params['N_bottleneck'], None, None, None] if sampled_params else None
        )
        return fit
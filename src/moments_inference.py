import moments
import tskit 
from collections import OrderedDict
import numpy as np 

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

def fit_model(sfs, start, g, experiment_config) -> list[float]:
    """
    Fit demographic model to provided SFS
    """
    sample_sizes = OrderedDict((p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape))
    # use Powell's method because m is not differentiable at zero
    fit = moments.Inference.optimize_powell(
        start,
        sfs,
        lambda p, n: diffusion_sfs(g, p, sample_sizes, experiment_config),
        multinom=False,
        verbose=1,
        flush_delay=0.0,
        maxiter=1000,
    )
    return fit
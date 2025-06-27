import dadi
import tskit 
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

def fit_model(sfs, start, g, experiment_config) -> list[float]:
    """
    Fit demographic model to provided SFS
    """
    pts_l = [20,30,40]
    sample_sizes = OrderedDict((p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape))

    # use Powell's method because m is not differentiable at zero
    fit = dadi.Inference.optimize_log_powell(
        start,
        sfs,
        lambda p, n, pts=None: diffusion_sfs(g, p, sample_sizes, experiment_config),
        pts=pts_l,
        multinom=False,
        verbose=1,
        flush_delay=0.0,
        maxiter=1000,
        # fixed_params=[10000, 2000, None, None, None]  # Fix N0 and N_bottleneck
    )

    return fit
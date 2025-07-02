# dadi_inference.py – harmonised with moments_inference.py
# -------------------------------------------------------
# This version takes a *parameter dictionary* just like the moments helper,
# keeps a single `param_names` list, and perturbs an initial vector for each
# restart.  Both stdout and stderr from the optimiser are captured so log
# files mirror the moments workflow.

from collections import OrderedDict
import dadi
import nlopt
import numpy as np

# ---------------------------------------------------------------------
# diffusion helper (parameter‑dict aware, like moments version)
# ---------------------------------------------------------------------

def diffusion_sfs(
    init_vec: np.ndarray,
    demo_model,              # callable that takes **param_dict** → demes.Graph
    param_names: list[str],
    sample_sizes: OrderedDict,
):
    """Expected SFS under dadi’s diffusion approximation."""
    pts_l = [20, 30, 40]

    p_dict = {k: v for k, v in zip(param_names, init_vec)}

    demography = demo_model(p_dict)
    sampled_demes = list(sample_sizes.keys())
    haploid_sizes = [n * 2 for n in sample_sizes.values()]

    return dadi.Spectrum.from_demes(
        demography,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
        pts=pts_l,
    )

# ---------------------------------------------------------------------
# fitting routine – mirrors moments_inference.fit_model
# ---------------------------------------------------------------------

def fit_model(
    sfs,
    start_dict: dict[str, float],      # initial parameter *dict*
    demo_model,                       # callable returning a demes.Graph
    experiment_config: dict,
    *,
    sampled_params: dict | None = None,
):
    """Run *num_optimizations* nlopt BOBYQA searches and keep the top‑k."""

    num_opt = experiment_config["num_optimizations"]
    top_k = experiment_config["top_k"]

    param_names = list(start_dict.keys())
    start_vec = np.array([start_dict[p] for p in param_names])

    sample_sizes = OrderedDict(
        (p, (n - 1) // 2) for p, n in zip(sfs.pop_ids, sfs.shape)
    )

    def _optimise(init_vec: np.ndarray, tag: str):

        opt_params, ll_val = dadi.Inference.opt(
            init_vec,
            sfs,
            lambda p, n, pts=None: diffusion_sfs(p, demo_model, param_names, sample_sizes),
            pts=[50, 60, 70],
            algorithm=nlopt.LN_BOBYQA,
            maxeval=10000,
            verbose=1,
            lower_bound=[100, 100, 100, 1e-8, 500],
            upper_bound=[30000, 30000, 30000, 1e-4, 20000],
            fixed_params=[
                sampled_params.get("N0"),
                sampled_params.get("N_bottleneck"),
                None, None, None,
            ] if sampled_params else None,
        )

        # xopt = dadi.Inference.optimize_log_powell(
        #     init_vec,
        #     sfs,
        #     lambda p, n, pts=None: diffusion_sfs(p, demo_model, param_names, sample_sizes),
        #     pts=[20, 30, 40],
        #     lower_bound=[5000, 1000, 1000, 1e-8, 1000],
        #     upper_bound=[20000, 10000, 10000, 1e-4, 40000],
        #     multinom=False,
        #     verbose=1,
        #     flush_delay=0.0,
        #     full_output=True,
        #     maxiter=5000
        # )

        # opt_params = xopt[0]
        # ll_val = xopt[1]

        return opt_params, ll_val

    # replicate loop ----------------------------------------------------
    fits: list[tuple[np.ndarray, float]] = []
    for i in range(num_opt):
        init = dadi.Misc.perturb_params(start_vec, fold=0.1)
        fits.append(_optimise(init, tag=f"optim_{i:04d}"))

    fits.sort(key=lambda t: t[1], reverse=True)
    best_params = [p for p, _ in fits[:top_k]]
    best_lls = [ll for _, ll in fits[:top_k]]
    return best_params, best_lls

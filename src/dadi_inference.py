#!/usr/bin/env python3
"""
dadi_inference.py  – dynamic pts grid + prior-based bounds
----------------------------------------------------------
Identical to your previous version except for:

* `pts_l` is computed from the SFS at runtime;
* `lower_bound` / `upper_bound` come straight from the JSON priors.
"""

from __future__ import annotations
from collections import OrderedDict
from contextlib  import redirect_stdout, redirect_stderr
from io          import StringIO
from pathlib     import Path
import datetime

import numpy as np
import dadi
import nlopt

# ───────────────────────── helper: expected SFS ─────────────────────────
def diffusion_sfs(
    p_vec: np.ndarray,
    demo_model,                      # callable(dict) → demes.Graph
    param_names: list[str],
    sample_sizes: OrderedDict[str, int],
    pts_l: list[int],
):
    """Return the expected SFS for parameter vector `p_vec`."""
    p_dict = dict(zip(param_names, p_vec))
    graph  = demo_model(p_dict)

    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())

    return dadi.Spectrum.from_demes(
        graph,
        sample_sizes = haploid_sizes,
        sampled_demes= sampled_demes,
        pts          = pts_l,
    )

# ───────────────────────── fitting routine ──────────────────────────────
def fit_model(
    sfs,
    start_dict: dict[str, float],
    demo_model,
    experiment_config: dict,
    *,
    sampled_params: dict | None = None,
):
    num_opt   = experiment_config["num_optimizations"]
    top_k     = experiment_config["top_k"]
    priors    = experiment_config["priors"]

    # parameter order / start vector
    param_names = list(start_dict.keys())
    start_vec   = np.array([start_dict[p] for p in param_names])

    # sample sizes & pts grid
    sample_sizes = OrderedDict(
        (pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape)
    )
    n_max_hap = max(2 * n for n in sample_sizes.values())
    pts_l     = [n_max_hap, n_max_hap + 20, n_max_hap + 40]

    # bounds from priors
    lower_bounds = [priors[p][0] for p in param_names]
    upper_bounds = [priors[p][1] for p in param_names]

    # one optimisation replicate (keeps your log-file machinery intact)
    def _optimise(p0: np.ndarray, tag: str):
        log_dir  = Path(experiment_config.get("log_dir", "."))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{tag}.txt"

        buf = StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            opt_params, ll_val = dadi.Inference.opt(
                p0,
                sfs,
                lambda p, n, pts=None: diffusion_sfs(
                    p, demo_model, param_names, sample_sizes, pts_l
                ),
                pts         = pts_l,
                algorithm   = nlopt.LN_BOBYQA,
                maxeval     = 10_000,
                verbose     = 1,
                lower_bound = lower_bounds,
                upper_bound = upper_bounds,
                fixed_params=[
                    sampled_params.get("N0"),
                    sampled_params.get("N_bottleneck"),
                    None, None, None,
                ] if sampled_params else None,
            )

        log_path.write_text(
            f"# dadi optimisation {tag}\n"
            f"# started: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n"
            + buf.getvalue()
        )
        return opt_params, ll_val

    # replicate loop ----------------------------------------------------
    fits = []
    for i in range(num_opt):
        seed_vec = dadi.Misc.perturb_params(start_vec, fold=0.1)
        fits.append(_optimise(seed_vec, tag=f"optim_{i:04d}"))

    fits.sort(key=lambda t: t[1], reverse=True)
    best_params = [p  for p, _ in fits[:top_k]]
    best_lls    = [ll for _, ll in fits[:top_k]]
    return best_params, best_lls

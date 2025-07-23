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
    """
    Run **one** dadi optimisation and return a single param‑vector / LL.
    The lists in the returned dict therefore contain exactly one element.
    """
    # ─── pull out a few things from the JSON --------------------------------
    priors   = experiment_config["priors"]
    # top_k    = experiment_config.get("top_k", 1)      # keeps interface intact
    # assert top_k == 1, "With one optimisation only, TOP_K must be 1"

    # ─── parameter order / start vector -------------------------------------
    param_names = list(start_dict.keys())
    start_vec   = np.array([start_dict[p] for p in param_names])

    # ─── sample sizes & pts grid --------------------------------------------
    from collections import OrderedDict
    sample_sizes = OrderedDict(
        (pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape)
    )
    n_max_hap = max(2 * n for n in sample_sizes.values())
    pts_l     = [n_max_hap, n_max_hap + 20, n_max_hap + 40]

    # ─── bounds from priors --------------------------------------------------
    lower_bounds = [priors[p][0] for p in param_names]
    upper_bounds = [priors[p][1] for p in param_names]

    # ─── single optimisation call -------------------------------------------
    import datetime, dadi, nlopt
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO
    from pathlib import Path

    log_dir  = Path(experiment_config.get("log_dir", "."))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "optim_single.txt"

    # (optional)  small random perturbation of the midpoint start
    seed_vec = dadi.Misc.perturb_params(start_vec, fold=0.1)

    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        opt_params, ll_val = dadi.Inference.opt(
            seed_vec,
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
            ] if experiment_config['demographic_model'] == "bottleneck" else None,
        )

    log_path.write_text(
        "# dadi single optimisation\n"
        f"# finished: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n"
        + buf.getvalue()
    )

    # ─── wrap into the usual return format ----------------------------------
    best_params = [opt_params]   # lists of length 1
    best_lls    = [ll_val]
    return best_params, best_lls

#!/usr/bin/env python3
# godambe_correction_LRT/src/profile_likelihood_ld.py

"""
Logic for 1D composite-likelihood profiles of a Moments-LD fit.

For each parameter, sweep it across its bound range (log10 grid) while holding
every OTHER parameter at the fitted MLE, recompute the composite log-likelihood
against the arm's overlapping LD curve. Reuses the validated compute_J_ld
theory + likelihood machinery, so the curve matches the fit exactly.

CLI wrapper: godambe_correction_LRT/snakemake_scripts/profile_likelihood_ld.py
"""

import logging

import numpy as np
import moments

import compute_J_ld as cj                        # R_BINS, POPULATIONS, NORMALIZATION, ll_from_theory
from momentsld_inference import BOUNDS, MODELS, MODEL_FUNCS

logging.getLogger().setLevel(logging.WARNING)


def model_for(param_names):
    """Pick the demes model whose parameter set exactly matches the fit."""
    s = set(param_names)
    for name, plist in MODELS.items():
        if set(plist) == s:
            return MODEL_FUNCS[name], name
    raise ValueError(f"no model matches parameter set {sorted(s)}")


def theory_for(param_dict, model_func):
    """σD²-normalized theory LD curve for a param dict (mirrors theoretical_ld_linear)."""
    graph = model_func(param_dict)
    ref = float(param_dict["N_ANC"])
    rho_edges = 4.0 * ref * np.asarray(cj.R_BINS)
    ld_edges = moments.Demes.LD(graph, sampled_demes=cj.POPULATIONS, rho=rho_edges)
    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=cj.POPULATIONS, rho=rho_mids)
    ld_bins = [(ld_edges[i] + ld_edges[i + 1] + 4 * ld_mids[i]) / 6.0
               for i in range(len(rho_mids))]
    ld_bins.append(ld_edges[-1])
    ld_stats = moments.LD.LDstats(ld_bins, num_pops=ld_edges.num_pops,
                                  pop_ids=ld_edges.pop_ids)
    return moments.LD.Inference.sigmaD2(ld_stats)


def ll_at(param_dict, model_func, mv):
    return cj.ll_from_theory(theory_for(param_dict, model_func),
                             mv["means"], mv["varcovs"])


def compute_profile_grid(pname, mle, model_func, mv, n_grid):
    """LL vs pname, sweeping a log10 grid over BOUNDS[pname] with every other
    parameter held at the MLE. Returns (grid, lls)."""
    lo, hi = BOUNDS[pname]
    grid = np.logspace(np.log10(lo), np.log10(hi), n_grid)
    lls = []
    for val in grid:
        d = dict(mle); d[pname] = float(val)
        try:
            lls.append(ll_at(d, model_func, mv))
        except Exception:
            lls.append(np.nan)
    return grid, np.asarray(lls, float)

#!/usr/bin/env python3
# godambe_correction_LRT/src/sfs_collect_and_fim.py
"""
Logic for collecting per-start moments SFS fits (from sfs_fit_one_start.py),
picking the best, then computing the identifiability diagnostic at that MLE:
SLICE profile likelihoods (nuisance params held at the MLE while the profiled
one is swept -- reusing fit_model_realdata_scaled's built-in generate_profiles
path) and the observed Fisher information (finite-difference Hessian in the
same log10-scaled space the optimizer used), reporting eigenvalues / cond(H)
and which parameters rail against a prior bound.

NOTE: the slice profiles are SLICES, not re-optimized profiles -- a parameter
redundant with another (e.g. N_CO0 vs N_ANC) can look well-peaked in a slice
even when the true (re-optimized) profile would be flat. Cross-check against
eigenvalues/cond(H) below for that failure mode.

cond(H) > 1e8 is flagged ILL-CONDITIONED, matching the threshold used in
run_lrt.py for the LD-side diagnostic, so the two are directly comparable.

CLI wrapper: godambe_correction_LRT/snakemake_scripts/sfs_collect_and_fim.py
"""

from __future__ import annotations

import numpy as np

import numdifftools as nd

from src.inference_utils import absolute_to_scaled_params
from src.moments_inference_real import (
    fit_model_realdata_scaled,
    _base_sfs_theta1_from_scaled,
    _theta_hat_poisson_mle,
    _mask_safe_poisson_ll,
)
from sfs_fim_common import make_safe_model_func, ILL_COND_THRESHOLD, RAIL_FRAC


def compute_sfs_fim_summary(arm, model_name, sfs, cfg, param_order, model_func,
                            starts, out_dir, rel_step=1e-4):
    """Collect the per-start fits, pick the best, and compute the slice
    profiles + FIM/railing diagnostic at that MLE.

    Returns (best_fit_blob, summary) -- the wrapper writes both to disk
    (best_fit.pkl / fim_summary.json). `fit_model_realdata_scaled` itself
    writes profile_<param>.{npz,png} under out_dir/likelihood_plots_scaled/
    as a side effect of `generate_profiles=True, save_dir=out_dir`.
    """
    per_start_ll = [s["ll_hat"] for s in starts]
    best = max(starts, key=lambda s: s["ll_hat"])
    n_starts = len(starts)

    # --- 1D slice profile likelihoods at the best start (re-run that exact
    # start with save_dir set so fit_model_realdata_scaled's built-in
    # generate_profiles path writes profile_<param>.{npz,png} per free param).
    profile_cfg = dict(cfg)
    profile_cfg["num_optimizations"] = n_starts
    profile_cfg["opt_seed"] = best["seed"]
    profile_cfg["generate_profiles"] = True
    fit_model_realdata_scaled(
        sfs=sfs,
        demo_model_abs=model_func,
        experiment_config=profile_cfg,
        param_order=param_order,
        verbose=False,
        save_dir=out_dir,
    )

    # --- reconstruct the exact log10-scaled MLE vector the optimizer used ---
    p_scaled = absolute_to_scaled_params(
        best["best_abs"], N_anc_abs=best["N_anc_implied"], time_scale="2N"
    )
    x_full = np.array([np.log10(p_scaled[p]) for p in param_order])

    free_names = [p for p in param_order if p != "N_ANC"]
    free_idx = [param_order.index(p) for p in free_names]

    priors = cfg.get("_active_priors") or cfg["priors_real_data_analysis"]
    bounds_report = []
    for p in free_names:
        lo, hi = float(priors[p][0]), float(priors[p][1])
        val = p_scaled[p]
        frac = (np.log10(val) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        railed = bool(frac < RAIL_FRAC or frac > 1 - RAIL_FRAC)
        bounds_report.append(dict(param=p, value=val, lo=lo, hi=hi,
                                   frac=float(frac), railed=railed))

    sampled_demes = list(sfs.pop_ids)
    haploid_sizes = [n - 1 for n in sfs.shape]
    folded = bool(getattr(sfs, "folded", False))
    safe_model_func = make_safe_model_func(model_func)

    def loglik_full(log10_full):
        base = _base_sfs_theta1_from_scaled(
            log10_full,
            demo_model_abs=safe_model_func,
            param_names=param_order,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            folded=folded,
        )
        base_arr = np.asarray(base)
        th = _theta_hat_poisson_mle(sfs, base_arr)
        exp_arr = th * base_arr
        return _mask_safe_poisson_ll(sfs, exp_arr, eps=1e-12)

    def loglik_free(free_vec):
        full = x_full.copy()
        full[free_idx] = free_vec
        return loglik_full(full)

    ll_check = loglik_free(x_full[free_idx])

    H_fun = nd.Hessian(loglik_free, step=rel_step)
    H = H_fun(x_full[free_idx])
    info = -H
    w = np.linalg.eigvalsh(info)
    w_clip = np.clip(w, 1e-300, None)
    cond = float(np.max(w_clip) / np.min(w_clip)) if np.min(w) > 0 else float("inf")
    ill_conditioned = bool(cond > ILL_COND_THRESHOLD or np.min(w) <= 0)

    se_by_param = {}
    try:
        cov = np.linalg.inv(info)
        se = np.sqrt(np.diag(cov))
        for nm, s in zip(free_names, se):
            se_by_param[nm] = None if np.isnan(s) else float(s)
    except np.linalg.LinAlgError:
        se_by_param = {nm: None for nm in free_names}

    best_fit_blob = {
        "mode": "moments",
        "best_params": [best["best_abs"]],
        "best_ll": [float(best["ll_hat"])],
        "opt_index": [best["seed"]],
        "theta_hat": [float(best["theta_hat"])],
        "N_ANC_implied_from_theta": [float(best["N_anc_implied"])],
    }

    # --- summarize the slice profiles just written (edge-of-window check) ---
    profile_dir = out_dir / "likelihood_plots_scaled"
    slice_report = []
    for p in free_names:
        npz_path = profile_dir / f"profile_{p}.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        grid, ll = d["grid_log10"], d["ll"]
        imax = int(np.argmax(ll))
        lo_p, hi_p = float(priors[p][0]), float(priors[p][1])
        at_window_edge = imax == 0 or imax == len(grid) - 1
        at_true_bound = (grid[imax] <= np.log10(lo_p) + 1e-6 or
                          grid[imax] >= np.log10(hi_p) - 1e-6)
        slice_report.append(dict(
            param=p,
            window_lo=float(10 ** grid[0]), window_hi=float(10 ** grid[-1]),
            argmax_value=float(10 ** grid[imax]),
            at_window_edge=bool(at_window_edge),
            at_true_prior_bound=bool(at_true_bound),
        ))

    summary = {
        "arm": arm,
        "model": model_name,
        "n_starts": n_starts,
        "per_start_ll": per_start_ll,
        "best_start_seed": best["seed"],
        "best_ll": float(best["ll_hat"]),
        "sanity_ll_reconstructed": float(ll_check),
        "sanity_ll_diff": float(ll_check - best["ll_hat"]),
        "free_params": free_names,
        "bounds_report": bounds_report,
        "any_railed": bool(any(b["railed"] for b in bounds_report)),
        "eigenvalues": w.tolist(),
        "cond_H": cond,
        "ill_conditioned": ill_conditioned,
        "se_log10_scaled": se_by_param,
        "slice_profile_report": slice_report,
        "slice_profile_dir": str(profile_dir),
    }

    return best_fit_blob, summary

#!/usr/bin/env python3
# godambe_correction_LRT/scripts/sfs_collect_and_fim.py
"""
Collect per-start moments SFS fits (from sfs_fit_one_start.py), pick the best,
then compute the identifiability diagnostic at that MLE: SLICE profile
likelihoods (nuisance params held at the MLE while the profiled one is swept
-- reusing fit_model_realdata_scaled's built-in generate_profiles path) and
the observed Fisher information (finite-difference Hessian in the same
log10-scaled space the optimizer used), reporting eigenvalues / cond(H) and
which parameters rail against a prior bound.

NOTE: the slice profiles are SLICES, not re-optimized profiles -- a parameter
redundant with another (e.g. N_CO0 vs N_ANC) can look well-peaked in a slice
even when the true (re-optimized) profile would be flat. Cross-check against
eigenvalues/cond(H) below for that failure mode.

cond(H) > 1e8 is flagged ILL-CONDITIONED, matching the threshold used in
run_lrt.py for the LD-side diagnostic, so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import moments
import numdifftools as nd

from src.inference_utils import absolute_to_scaled_params
from src.moments_inference_real import (
    fit_model_realdata_scaled,
    _base_sfs_theta1_from_scaled,
    _theta_hat_poisson_mle,
    _mask_safe_poisson_ll,
)
from sfs_fim_common import make_safe_model_func, ILL_COND_THRESHOLD, RAIL_FRAC


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sfs", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--pop-ids", default="CO,FR")
    ap.add_argument("--starts", required=True, type=Path, nargs="+",
                     help="per-start pickles from sfs_fit_one_start.py")
    ap.add_argument("--rel-step", type=float, default=1e-4)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    arm, model_name = args.arm, args.model

    with open(args.config) as f:
        cfg = json.load(f)
    param_order = list(cfg["parameter_order"])
    model_func = getattr(
        importlib.import_module("src.demes_models"), f"{model_name}_model"
    )

    with open(args.sfs, "rb") as f:
        sfs = pickle.load(f)
    sfs = moments.Spectrum(sfs)
    sfs.pop_ids = [s.strip() for s in args.pop_ids.split(",")]

    starts = []
    for p in args.starts:
        with open(p, "rb") as f:
            starts.append(pickle.load(f))
    per_start_ll = [s["ll_hat"] for s in starts]
    best = max(starts, key=lambda s: s["ll_hat"])
    n_starts = len(starts)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    H_fun = nd.Hessian(loglik_free, step=args.rel_step)
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
    with open(out_dir / "best_fit.pkl", "wb") as f:
        pickle.dump(best_fit_blob, f)

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
    with open(out_dir / "fim_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{arm}/{model_name}] best ll={best['ll_hat']:.6f} (start {best['seed']})")
    print(f"[{arm}/{model_name}] cond(H)={cond:.3e}  "
          f"({'ILL-CONDITIONED' if ill_conditioned else 'ok'})")
    railed = [b["param"] for b in bounds_report if b["railed"]]
    print(f"[{arm}/{model_name}] railed params (scaled MLE vs prior bound): {railed if railed else 'none'}")
    edge_hit = [s["param"] for s in slice_report if s["at_window_edge"]]
    print(f"[{arm}/{model_name}] slice profiles peaking at window edge: {edge_hit if edge_hit else 'none'}")
    print(f"[{arm}/{model_name}] -> {out_dir}/best_fit.pkl , {out_dir}/fim_summary.json , "
          f"{profile_dir}/profile_<param>.png")


if __name__ == "__main__":
    main()

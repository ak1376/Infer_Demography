#!/usr/bin/env python3
# snakemake_scripts/momentsld_profile_and_confounding.py
#
# Post-hoc diagnostics for a completed real-data MomentsLD fit, computed
# directly from the already-aggregated best_fit.pkl + means.varcovs.pkl
# (no re-optimization needed):
#   1. 1D profile likelihoods around the best-fit point, for every parameter.
#   2. A 2D profile-likelihood surface over two named parameters (default
#      N_ANC x N_CO0), holding everything else at the best-fit point -- the
#      standard way to visualize a ridge/degeneracy between two parameters.
#   3. A scatter of per-restart best-fit values for those same two parameters
#      across every runs/run_*/inferences/{engine}/best_fit.pkl, in case the
#      structure of the fit distribution itself is informative.

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.MomentsLD_real_data import (  # noqa: E402
    compute_theoretical_ld,
    prepare_data_for_comparison,
    composite_gaussian_ll,
    load_demographic_function,
    _build_param_dict,
    load_json,
    load_pickle,
)
from src.inference_utils import profile_1d, save_profiles  # noqa: E402


def _load_engine_best_fit(path: Path):
    d = load_pickle(path)
    bp, ll = d.get("best_params"), d.get("best_ll")
    if isinstance(bp, list):
        idx = 0
        if isinstance(ll, list) and len(ll) == len(bp):
            idx = max(range(len(ll)), key=lambda i: ll[i])
        return dict(bp[idx])
    return dict(bp)


_RUN_RE = re.compile(r"^run_(\d+)$")


def _load_per_restart(runs_dir: Path, engine_subdir: str, params: List[str]) -> Dict[str, List[float]]:
    out = {p: [] for p in params}
    for d in sorted(runs_dir.iterdir()):
        if not _RUN_RE.match(d.name):
            continue
        p = d / "inferences" / engine_subdir / "best_fit.pkl"
        if not p.exists():
            continue
        try:
            blob = load_pickle(p)
        except Exception:
            continue
        bp = blob.get("best_params")
        if not isinstance(bp, dict):
            continue
        for k in params:
            if k in bp:
                out[k].append(float(bp[k]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--empirical", required=True, type=Path, help="means.varcovs.pkl")
    ap.add_argument("--best-fit-pkl", required=True, type=Path,
                     help="Aggregated best_fit.pkl (top-K) for this engine.")
    ap.add_argument("--runs-dir", required=True, type=Path,
                     help="real_data_analysis/runs directory (for the per-restart scatter).")
    ap.add_argument("--ld-engine-subdir", required=True,
                     help="e.g. MomentsLD_Chr2L_Chr3L_genmap")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pair", nargs=2, default=["N_ANC", "N_CO0"],
                     help="Two parameter names for the 2D profile + scatter.")
    ap.add_argument("--normalization", type=int, default=0)
    ap.add_argument("--profile-points", type=int, default=41)
    ap.add_argument("--profile-widen", type=float, default=0.5)
    ap.add_argument("--grid-points", type=int, default=31,
                     help="Per-axis resolution for the 2D profile grid.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_json(args.config)
    mv = load_pickle(args.empirical)
    demo_fn = load_demographic_function(cfg)

    r_bins = np.array(
        [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], dtype=float
    )
    populations = list(cfg.get("num_samples", {}).keys())
    use_scaled_units = bool(cfg.get("momentsld_use_scaled_units", False))
    priors = cfg.get("priors_real_data_analysis") if use_scaled_units else cfg.get("priors")
    param_names = list(cfg.get("parameter_order", list(priors.keys())))
    lb = np.array([float(priors[p][0]) for p in param_names], dtype=float)
    ub = np.array([float(priors[p][1]) for p in param_names], dtype=float)

    best_abs = _load_engine_best_fit(args.best_fit_pkl)
    xhat_real = np.array([best_abs[p] for p in param_names], dtype=float)
    xhat_log10 = np.log10(xhat_real)
    print("Best-fit point:", {p: v for p, v in zip(param_names, xhat_real)})

    _diag_once = {"did": True}  # suppress the one-time diagnostic log spam

    def loglik(log10_params: np.ndarray) -> float:
        theo = compute_theoretical_ld(
            log10_params,
            param_names=param_names,
            demographic_model_abs=demo_fn,
            r_bins=r_bins,
            populations=populations,
            N_ref=1.0,  # unused when use_scaled_units=False
            use_scaled_units=use_scaled_units,
            _diagnostic_once=_diag_once,
        )
        theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(
            theo, mv, normalization=args.normalization
        )
        return composite_gaussian_ll(emp_means, emp_covars, theory_arrays)

    ll_at_best = loglik(xhat_log10)
    print(f"LL at best-fit (sanity check): {ll_at_best:.4f}")

    # ---------------- 1. 1D profiles for every parameter ----------------
    print("Computing 1D profiles...")
    profiles = profile_1d(
        xhat_log10=xhat_log10,
        param_names=param_names,
        lb_full=lb,
        ub_full=ub,
        loglikelihood_fn=loglik,
        n_points=args.profile_points,
        widen=args.profile_widen,
    )
    save_profiles(
        profiles,
        args.out_dir / "profiles_1d",
        make_plots=True,
        title_prefix="MomentsLD real-data profile likelihood",
    )

    # ---------------- 2. 2D profile over the requested pair ----------------
    p1, p2 = args.pair
    i1, i2 = param_names.index(p1), param_names.index(p2)
    print(f"Computing 2D profile over ({p1}, {p2})...")

    def _grid_for(i):
        lo, hi = np.log10(lb[i]), np.log10(ub[i])
        span = hi - lo
        lo_g = max(lo, xhat_log10[i] - args.profile_widen * span)
        hi_g = min(hi, xhat_log10[i] + args.profile_widen * span)
        return np.linspace(lo_g, hi_g, args.grid_points)

    grid1 = _grid_for(i1)
    grid2 = _grid_for(i2)
    ll_surface = np.empty((len(grid2), len(grid1)))
    x = xhat_log10.copy()
    for a, g1 in enumerate(grid1):
        for b, g2 in enumerate(grid2):
            x[i1] = g1
            x[i2] = g2
            ll_surface[b, a] = loglik(x)

    np.savez(
        args.out_dir / f"profile2d_{p1}_{p2}.npz",
        grid1_log10=grid1, grid2_log10=grid2, ll=ll_surface,
        xhat1_log10=xhat_log10[i1], xhat2_log10=xhat_log10[i2],
    )

    ll_max = float(np.max(ll_surface))
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(10 ** grid1, 10 ** grid2, ll_max - ll_surface, levels=30, cmap="viridis_r")
    ax.contour(10 ** grid1, 10 ** grid2, ll_max - ll_surface, levels=[0.5, 1, 2, 4, 8], colors="white", linewidths=0.6)
    ax.scatter([10 ** xhat_log10[i1]], [10 ** xhat_log10[i2]], color="red", marker="x", s=60, label="MLE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    ax.set_title(f"2D profile likelihood: Δ log-lik (max − ll)")
    fig.colorbar(cf, ax=ax, label="Δ log-lik")
    ax.legend()
    plt.tight_layout()
    fig.savefig(args.out_dir / f"profile2d_{p1}_{p2}.png", dpi=150)
    plt.close(fig)

    # ---------------- 3. Per-restart scatter for the same pair ----------------
    print(f"Loading per-restart best-fits for scatter ({p1} vs {p2})...")
    per_restart = _load_per_restart(args.runs_dir, args.ld_engine_subdir, [p1, p2])
    v1, v2 = per_restart[p1], per_restart[p2]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(v1, v2, s=18, alpha=0.6, color="tab:green")
    ax.scatter([xhat_real[i1]], [xhat_real[i2]], color="red", marker="x", s=80, label="aggregated best fit")
    ax.axvline(lb[i1], color="gray", linestyle=":", lw=0.8)
    ax.axvline(ub[i1], color="gray", linestyle=":", lw=0.8)
    ax.axhline(lb[i2], color="gray", linestyle=":", lw=0.8)
    ax.axhline(ub[i2], color="gray", linestyle=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    ax.set_title(f"Per-restart best fits ({len(v1)} restarts): {p1} vs {p2}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(args.out_dir / f"scatter_{p1}_{p2}.png", dpi=150)
    plt.close(fig)

    print(f"All diagnostics written to {args.out_dir}")


if __name__ == "__main__":
    main()

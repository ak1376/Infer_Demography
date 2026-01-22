#!/usr/bin/env python3
"""
fisher_info_sfs.py — Observed Fisher Information (raw units) for Poisson SFS

- Engine can be 'dadi' or 'moments' (only changes how we compute expected SFS).
- Uses a model function returning a demes.Graph: --model "pkg.module:function".
- Scales expected SFS by 4*N0*mu*L (assumes first param is N0-like).

Outputs:
  <out_prefix>.fim.npy
  <out_prefix>.cov.npy          (if invertible)
  <out_prefix>.ses.json         (SEs per free param if invertible)
  <out_prefix>.summary.json     (logdet, min_eigen, cond, diag entries)
"""

from __future__ import annotations
import argparse, importlib, json, pickle, warnings
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numdifftools as nd

# ---------------------------- helpers: SFS ---------------------------------


def _build_sample_sizes_from_sfs(sfs) -> OrderedDict[str, int]:
    if hasattr(sfs, "pop_ids") and sfs.pop_ids is not None:
        return OrderedDict(
            (pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape)
        )
    pop_names = [f"pop{i}" for i in range(len(sfs.shape))]
    return OrderedDict((pop, (n - 1) // 2) for pop, n in zip(pop_names, sfs.shape))


def _auto_pts_from_sfs(sfs) -> List[int]:
    ss = _build_sample_sizes_from_sfs(sfs)
    n_max_hap = max(2 * n for n in ss.values())
    return [n_max_hap + 20, n_max_hap + 40, n_max_hap + 60]


def _scale_expected_sfs(exp_sfs, theta_vec, mu, L):
    N0 = max(float(theta_vec[0]), 1e-300)
    exp_sfs *= 4.0 * N0 * float(mu) * int(L)
    return exp_sfs


# --------------------- expected SFS (engine switch) ------------------------


def _expected_sfs_moments(theta_vec, param_names, model_func, mu, L, sample_sizes):
    import moments

    p_dict = {nm: float(v) for nm, v in zip(param_names, theta_vec)}
    graph = model_func(p_dict)
    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())
    fs = moments.Spectrum.from_demes(
        graph, sample_sizes=haploid_sizes, sampled_demes=sampled_demes
    )
    return _scale_expected_sfs(fs, theta_vec, mu, L)


def _expected_sfs_dadi(theta_vec, param_names, model_func, mu, L, sample_sizes, pts):
    import dadi

    p_dict = {nm: float(v) for nm, v in zip(param_names, theta_vec)}
    graph = model_func(p_dict)
    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())

    def _raw(_params, ns, pts_grid):
        return dadi.Spectrum.from_demes(
            graph, sample_sizes=haploid_sizes, sampled_demes=sampled_demes, pts=pts_grid
        )

    func_ex = dadi.Numerics.make_extrap_func(
        lambda p, ns, pts_grid: _raw(p, ns, pts_grid)
    )
    fs = func_ex(theta_vec, sample_sizes, pts)
    return _scale_expected_sfs(fs, theta_vec, mu, L)


def make_expected_sfs_func(engine, param_names, model_func, mu, L, sample_sizes, pts):
    engine = engine.lower()
    if engine == "dadi":
        if pts is None:
            raise ValueError(
                "dadi engine requires a 3-point pts grid (or pass 'auto')."
            )

        def f(theta_full):
            return _expected_sfs_dadi(
                theta_full, param_names, model_func, mu, L, sample_sizes, pts
            )

        return f
    elif engine == "moments":

        def f(theta_full):
            return _expected_sfs_moments(
                theta_full, param_names, model_func, mu, L, sample_sizes
            )

        return f
    else:
        raise ValueError("engine must be 'dadi' or 'moments'")


def make_poisson_loglik_sfs(sfs, expected_sfs_func, folded: bool):
    def ll_theta(theta_full: np.ndarray) -> float:
        exp_sfs = expected_sfs_func(theta_full)
        if folded:
            exp_sfs = exp_sfs.fold()
        exp_clip = np.maximum(exp_sfs, 1e-300)
        return float(np.sum(sfs * np.log(exp_clip) - exp_clip))

    return ll_theta


# ---------------------- Fisher Information (observed) ----------------------


def observed_fim_theta(
    sfs,
    param_names: List[str],
    theta_at: np.ndarray,
    model_func,
    mu: float,
    L: int,
    engine: str,
    pts: Optional[List[int]],
    fixed_params: Optional[Dict[str, float]] = None,
    rel_step: float = 1e-4,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, float]], List[int]]:
    """
    Returns (info, cov, se_dict, free_idx).
    info is kxk over FREE parameters (raw units, observed Fisher = -Hessian).
    """
    fixed_params = fixed_params or {}
    theta_at = np.asarray(theta_at, float).copy()
    theta_at = np.maximum(theta_at, 1e-300)

    sample_sizes = _build_sample_sizes_from_sfs(sfs)
    folded = bool(getattr(sfs, "folded", False))

    if engine == "dadi" and (pts == "auto" or pts is None):
        pts = _auto_pts_from_sfs(sfs)

    exp_sfs_func = make_expected_sfs_func(
        engine, param_names, model_func, mu, L, sample_sizes, pts
    )
    ll_theta = make_poisson_loglik_sfs(sfs, exp_sfs_func, folded)

    free_idx = [i for i, nm in enumerate(param_names) if nm not in fixed_params]
    fixed_idx = [i for i, nm in enumerate(param_names) if nm in fixed_params]
    if not free_idx:
        raise ValueError(
            "All parameters are fixed; FIM over free parameters is undefined."
        )

    for i in fixed_idx:
        theta_at[i] = max(float(fixed_params[param_names[i]]), 1e-300)

    def ll_free(free_vec: np.ndarray) -> float:
        full = theta_at.copy()
        full[free_idx] = np.asarray(free_vec, float)
        full = np.maximum(full, 1e-300)
        return ll_theta(full)

    step_vec = np.maximum(np.abs(theta_at[free_idx]) * rel_step, 1e-8)
    H_fun = nd.Hessian(ll_free, step=step_vec)
    H = H_fun(theta_at[free_idx])

    info = -H
    cov = None
    se_dict = None
    try:
        cov = np.linalg.inv(info)
        se = np.sqrt(np.diag(cov))
        se_dict = {param_names[i]: float(se[k]) for k, i in enumerate(free_idx)}
    except np.linalg.LinAlgError:
        warnings.warn(
            "Information matrix singular/ill-conditioned; covariance unavailable."
        )

    return info, cov, se_dict, free_idx


# ---------------------------------- CLI -----------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Observed Fisher Information for Poisson SFS (raw units)."
    )
    ap.add_argument(
        "--sfs-pkl",
        type=Path,
        required=True,
        help="Pickled observed SFS (dadi or moments Spectrum).",
    )
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON with mutation_rate and genome_length.",
    )
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model function: "pkg.module:function_name".',
    )
    ap.add_argument(
        "--param-names",
        type=str,
        required=True,
        help="Comma-separated parameter names (ordered).",
    )
    ap.add_argument(
        "--params",
        type=str,
        required=True,
        help="Comma-separated parameter values (raw units).",
    )
    ap.add_argument("--engine", choices=["dadi", "moments"], default="dadi")
    ap.add_argument(
        "--pts", type=str, default="auto", help='For dadi: "n1,n2,n3" or "auto".'
    )
    ap.add_argument("--rel-step", type=float, default=1e-4)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    # Load
    sfs = pickle.loads(args.sfs_pkl.read_bytes())
    cfg = json.loads(args.config.read_text())
    mu = float(cfg["mutation_rate"])
    L = int(cfg["genome_length"])

    # Model
    mod_path, func_name = args.model.split(":")
    model_func = getattr(importlib.import_module(mod_path), func_name)

    # Params order + values
    param_names = [x.strip() for x in args.param_names.split(",") if x.strip()]
    theta_at = np.asarray([float(x) for x in args.params.split(",")], float)
    if len(param_names) != len(theta_at):
        raise SystemExit("param-names and params length mismatch.")

    # pts parsing
    pts = "auto"
    if args.engine == "dadi":
        if args.pts and args.pts.lower() != "auto":
            xs = [int(x) for x in args.pts.split(",")]
            if len(xs) != 3:
                raise SystemExit("--pts must have 3 integers or be 'auto'.")
            pts = xs

    # Compute
    info, cov, se, free_idx = observed_fim_theta(
        sfs=sfs,
        param_names=param_names,
        theta_at=theta_at,
        model_func=model_func,
        mu=mu,
        L=L,
        engine=args.engine,
        pts=pts,
        fixed_params=None,
        rel_step=args.rel_step,
    )

    # Save arrays
    np.save(args.out_prefix.with_suffix(".fim.npy"), info)
    if cov is not None:
        np.save(args.out_prefix.with_suffix(".cov.npy"), cov)

    # Summaries
    summary = {}
    try:
        w, _ = np.linalg.eigh(info)
        w = np.asarray(w, float)
        w_clipped = np.clip(w, 1e-300, None)
        summary["logdet"] = float(np.sum(np.log(w_clipped)))
        summary["min_eigen"] = float(np.min(w))
        summary["max_eigen"] = float(np.max(w))
        summary["cond"] = float(np.max(w_clipped) / np.min(w_clipped))
    except Exception:
        pass

    if se is not None:
        summary["SE"] = se

    # Diagonal entries per free param in order
    diag = np.diag(info)
    summary["diag"] = {param_names[i]: float(diag[k]) for k, i in enumerate(free_idx)}
    args.out_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Optional SEs as a separate file (handy)
    if se is not None:
        args.out_prefix.with_suffix(".ses.json").write_text(json.dumps(se, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MomentsLD_real_data.py

Real-data MomentsLD inference that FIXES the rho-scaling reference size (N_ref),
so you can optimize in SCALED units (ratios, tau, M) without accidentally using
N_ref=1 and wrecking rho = 4*N_ref*r.

Key idea:
- Optimize in scaled ("shape") parameters:
    N_ANC   : placeholder (fixed to 1)
    N_*     : ratios r = N_*/N_ANC
    T       : tau = T_abs / (2*N_ref)
    m_*     : M = 2*N_ref*m_abs
- Convert scaled -> absolute params using fixed N_ref:
    N_*_abs = r_* * N_ref
    T_abs   = 2*N_ref*tau
    m_abs   = M / (2*N_ref)
- Compute theoretical LD with rho = 4*N_ref*r_bins

Inputs:
- --config : your experiment config JSON
- --empirical : means.varcovs.pkl (from moments.LD.Parsing.bootstrap_data)
- --outdir : where to write best_fit.pkl (+ optional profiles)
- N_ref:
    Either:
      --n-ref  (float)
    Or:
      --sfs-best-fit-pkl  (pickle containing N_ANC_implied or best_params_abs["N_ANC"])

Usage example:
  python src/MomentsLD_real_data.py \
    --config config_files/experiment_config_split_migration_growth.json \
    --empirical experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD/means.varcovs.pkl \
    --outdir experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD/inference_scaled \
    --sfs-best-fit-pkl experiments/split_migration_growth/real_data_analysis/inferences/moments/best_fit.pkl \
    --normalization 0 \
    --rtol 1e-8 \
    --verbose

Notes:
- This assumes your demographic_model in src.simulation expects ABSOLUTE params with keys
  like N_ANC, N_CO, N_FR0, N_FR1, T, m_CO_FR, m_FR_CO (as in your split_migration_growth_model).
- Uses priors_real_data_analysis for SCALED bounds (must be >0; N_ANC should be [1,1]).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import moments
import nlopt
import numdifftools as nd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_R_BINS = np.array(
    [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    dtype=float,
)
JITTER = 1e-12


# ------------------------- IO helpers -------------------------


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_pickle(path: Path) -> Any:
    return pickle.loads(path.read_bytes())


def dump_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(obj))


# ------------------------- model loading -------------------------


def load_demographic_function(config: Dict[str, Any]):
    """Load src.simulation:<model>_model"""
    demo_module = importlib.import_module("src.simulation")
    model_name = config["demographic_model"]
    if model_name == "drosophila_three_epoch":
        return getattr(demo_module, "drosophila_three_epoch")
    return getattr(demo_module, model_name + "_model")


# ------------------------- scaling logic -------------------------


def _build_param_dict(param_names: List[str], vec_real: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(param_names, vec_real)}


def scaled_to_absolute_params(
    p_scaled: Dict[str, float],
    *,
    N_ref: float,
    time_scale: str = "2N",
) -> Dict[str, float]:
    """
    Convert scaled ("shape") params to ABSOLUTE, using fixed N_ref.

    Convention (matches your SFS-scaled setup):
      - N_ANC is placeholder; set absolute N_ANC = N_ref
      - sizes N_* (except N_ANC): ratios r -> N_*_abs = r * N_ref
      - T: tau -> T_abs = 2*N_ref*tau
      - m_*: M -> m_abs = M/(2*N_ref)
    """
    if N_ref <= 0:
        raise ValueError(f"N_ref must be > 0, got {N_ref}")

    out = dict(p_scaled)
    out["N_ANC"] = float(N_ref)

    # sizes
    for k, v in list(out.items()):
        if k.startswith("N_") and k != "N_ANC":
            out[k] = float(v) * float(N_ref)

    # time
    if "T" in out:
        tau = float(out["T"])
        if time_scale != "2N":
            raise ValueError(f"Unknown time_scale={time_scale}")
        out["T"] = float(2.0 * N_ref * tau)

    # migration
    for k, v in list(out.items()):
        if k.startswith("m_"):
            M = float(v)
            out[k] = float(M) / float(2.0 * N_ref)

    return out


# ------------------------- theoretical LD -------------------------


def compute_theoretical_ld_fixed_nref(
    log10_scaled_params: np.ndarray,
    *,
    param_names: List[str],
    demographic_model_abs,
    r_bins: np.ndarray,
    populations: List[str],
    N_ref: float,
    _diagnostic_once: Dict[str, bool],
) -> moments.LD.LDstats:
    """
    Compute expected sigmaD2 using rho = 4*N_ref*r_bins, with demography built from ABSOLUTE params
    obtained by converting scaled params using the same fixed N_ref.
    """
    vec_scaled = 10 ** np.asarray(log10_scaled_params, dtype=float)
    p_scaled = _build_param_dict(param_names, vec_scaled)
    p_abs = scaled_to_absolute_params(p_scaled, N_ref=N_ref, time_scale="2N")

    graph = demographic_model_abs(p_abs)

    # FIXED rho scaling
    rho_edges = 4.0 * float(N_ref) * np.asarray(r_bins, dtype=float)

    if not _diagnostic_once.get("did", False):
        _diagnostic_once["did"] = True
        logging.info("[diagnostic] Using fixed N_ref = %.6g", N_ref)
        logging.info(
            "[diagnostic] rho_edges min/max = %.3e, %.3e", float(rho_edges.min()), float(rho_edges.max())
        )
        logging.info("[diagnostic] populations = %s", populations)

    # moments.Demes.LD returns LDstats at each rho edge
    ld_edges = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_edges)

    # Simpson-ish average inside each rho bin (your existing approach)
    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_mids)

    ld_bins = [(ld_edges[i] + ld_edges[i + 1] + 4.0 * ld_mids[i]) / 6.0 for i in range(len(rho_mids))]
    ld_bins.append(ld_edges[-1])

    ld_stats = moments.LD.LDstats(ld_bins, num_pops=ld_edges.num_pops, pop_ids=ld_edges.pop_ids)
    return moments.LD.Inference.sigmaD2(ld_stats)


# ------------------------- data prep + likelihood -------------------------


def prepare_data_for_comparison(
    theoretical_sigmaD2: moments.LD.LDstats,
    empirical_mv: Dict[str, List[np.ndarray]],
    *,
    normalization: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
      theory_arrays: list of arrays per r-bin (excluding heterozygosity)
      emp_means:    list of arrays per r-bin (excluding heterozygosity)
      emp_covars:   list of matrices per r-bin (excluding heterozygosity)
    """
    # theoretical: remove normalized LDs, then drop heterozygosity
    theory_processed = moments.LD.LDstats(
        theoretical_sigmaD2[:],
        num_pops=theoretical_sigmaD2.num_pops,
        pop_ids=theoretical_sigmaD2.pop_ids,
    )
    theory_processed = moments.LD.Inference.remove_normalized_lds(theory_processed, normalization=normalization)
    theory_arrays = [np.asarray(x) for x in theory_processed[:-1]]  # drop H

    # empirical means/covs
    emp_means = [np.asarray(x) for x in empirical_mv["means"]]
    emp_covars = [np.asarray(x) for x in empirical_mv["varcovs"]]

    # remove normalized statistics *consistently*
    emp_means, emp_covars = moments.LD.Inference.remove_normalized_data(
        emp_means,
        emp_covars,
        normalization=normalization,
        num_pops=theoretical_sigmaD2.num_pops,
    )

    # drop H
    emp_means = emp_means[:-1]
    emp_covars = emp_covars[:-1]
    return theory_arrays, emp_means, emp_covars


def composite_gaussian_ll(
    emp_means: List[np.ndarray],
    emp_covars: List[np.ndarray],
    theory_arrays: List[np.ndarray],
) -> float:
    total = 0.0
    for obs, cov, pred in zip(emp_means, emp_covars, theory_arrays):
        if obs.size == 0:
            continue
        resid = obs - pred
        covm = np.asarray(cov)
        if covm.ndim == 2 and covm.size > 1:
            covm = covm + np.eye(covm.shape[0]) * JITTER
            try:
                inv = np.linalg.inv(covm)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(covm)
            total -= 0.5 * float(resid @ inv @ resid)
        else:
            total -= 0.5 * float(resid @ resid)
    return float(total)


# ------------------------- N_ref resolution -------------------------


def resolve_n_ref(args) -> float:
    if args.n_ref is not None:
        return float(args.n_ref)

    if args.sfs_best_fit_pkl is None:
        raise ValueError("Provide --n-ref or --sfs-best-fit-pkl")

    d = load_pickle(Path(args.sfs_best_fit_pkl))

    # Try common layouts you might have saved
    if isinstance(d, dict):
        if "N_ANC_implied" in d:
            return float(d["N_ANC_implied"])
        if "best_params_abs" in d and isinstance(d["best_params_abs"], dict) and "N_ANC" in d["best_params_abs"]:
            return float(d["best_params_abs"]["N_ANC"])
        if "best_params" in d and isinstance(d["best_params"], dict) and "N_ANC" in d["best_params"]:
            return float(d["best_params"]["N_ANC"])

    raise ValueError(
        f"Could not find N_ref in {args.sfs_best_fit_pkl}. "
        "Expected keys like N_ANC_implied or best_params_abs['N_ANC']."
    )


# ------------------------- main optimize -------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--empirical", required=True, type=Path, help="means.varcovs.pkl")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--normalization", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--verbose", action="store_true")

    # N_ref options
    ap.add_argument("--n-ref", type=float, default=None, help="Fixed N_ref for rho scaling")
    ap.add_argument(
        "--sfs-best-fit-pkl",
        type=Path,
        default=None,
        help="Pickle from moments/dadi real-data SFS run containing N_ANC_implied or best_params_abs['N_ANC']",
    )

    # r bins
    ap.add_argument("--r-bins", type=str, default=None, help="Comma-separated r-bin edges; default uses module DEFAULT_R_BINS")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = load_json(args.config)
    mv = load_pickle(args.empirical)

    demo_fn = load_demographic_function(cfg)

    # r bins
    if args.r_bins is None:
        r_bins = DEFAULT_R_BINS
    else:
        r_bins = np.array([float(x) for x in args.r_bins.split(",")], dtype=float)
        if r_bins.ndim != 1 or r_bins.size < 2:
            raise ValueError("--r-bins must have at least 2 edges")

    # populations
    populations = list(cfg.get("num_samples", {}).keys())
    if not populations:
        raise ValueError("config['num_samples'] must specify populations")

    # scaled priors + parameter order
    param_names = list(cfg.get("parameter_order", list(cfg["priors_real_data_analysis"].keys())))
    priors_scaled = cfg.get("priors_real_data_analysis", None)
    if priors_scaled is None:
        raise ValueError("Need config['priors_real_data_analysis'] for scaled LD optimization")

    lb = np.array([float(priors_scaled[p][0]) for p in param_names], dtype=float)
    ub = np.array([float(priors_scaled[p][1]) for p in param_names], dtype=float)
    if np.any(lb <= 0) or np.any(ub <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb, ub) if lo <= 0 or hi <= 0]
        raise ValueError(f"All scaled bounds must be >0 (log10 opt). Bad: {bad}")

    # FIX N_ref
    N_ref = resolve_n_ref(args)

    # start = geometric midpoint (optionally jitter like your SFS pipeline)
    x0_real = 10 ** ((np.log10(lb) + np.log10(ub)) / 2.0)
    sigma = float(cfg.get("start_jitter_sigma", 0.0))
    if sigma > 0:
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        x0_real = x0_real * (10 ** rng.normal(0.0, sigma, size=x0_real.shape))
    x0_real = np.clip(x0_real, lb, ub)
    x0 = np.log10(x0_real)

    # objective
    _diag_once = {"did": False}

    def loglik(log10_params: np.ndarray) -> float:
        theo = compute_theoretical_ld_fixed_nref(
            log10_params,
            param_names=param_names,
            demographic_model_abs=demo_fn,
            r_bins=r_bins,
            populations=populations,
            N_ref=N_ref,
            _diagnostic_once=_diag_once,
        )
        theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(
            theo, mv, normalization=int(args.normalization)
        )
        return composite_gaussian_ll(emp_means, emp_covars, theory_arrays)

    # gradient (numdifftools) for nlopt LBFGS
    grad_fn = nd.Gradient(loglik, step=1e-4)

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        ll = float(loglik(x))
        if grad.size > 0:
            grad[:] = grad_fn(x)
        if args.verbose:
            vec = 10 ** np.asarray(x)
            p_scaled = _build_param_dict(param_names, vec)
            p_abs = scaled_to_absolute_params(p_scaled, N_ref=N_ref, time_scale="2N")
            # concise print
            show = ", ".join(f"{k}={p_abs[k]:.3g}" for k in param_names)
            print(f"LL = {ll:.6f} | {show} | (fixed N_ref={N_ref:.3g})")
        return ll

    opt = nlopt.opt(nlopt.LD_LBFGS, len(param_names))
    opt.set_lower_bounds(np.log10(lb))
    opt.set_upper_bounds(np.log10(ub))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(float(args.rtol))

    # run
    best_x = opt.optimize(x0)
    status = opt.last_optimize_result()
    best_ll = opt.last_optimum_value()

    best_scaled = _build_param_dict(param_names, 10 ** np.asarray(best_x))
    best_abs = scaled_to_absolute_params(best_scaled, N_ref=N_ref, time_scale="2N")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_pkl = args.outdir / "best_fit.pkl"
    dump_pickle(
        {
            "status": int(status),
            "best_ll": float(best_ll),
            "fixed_N_ref": float(N_ref),
            "best_params_scaled": best_scaled,
            "best_params_abs": best_abs,
            "param_order": param_names,
            "normalization": int(args.normalization),
            "r_bins": np.asarray(r_bins, dtype=float),
        },
        out_pkl,
    )

    logging.info("Done. status=%s  best_ll=%.6f", status, best_ll)
    logging.info("Wrote %s", out_pkl)


if __name__ == "__main__":
    main()
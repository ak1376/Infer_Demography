#!/usr/bin/env python3
"""
MomentsLD_real_data.py

Real-data MomentsLD inference in RAW (absolute) parameter space.
Optimises N_ANC, N_*, T, m_* directly in their natural units; N_ANC is
used on-the-fly as the rho anchor: rho = 4 * N_ANC * r_bins.

Inputs:
- --config  : experiment config JSON  (priors_real_data_analysis must hold
              ABSOLUTE bounds for all parameters)
- --empirical: means.varcovs.pkl (from moments.LD.Parsing.bootstrap_data)
- --outdir  : where to write best_fit.pkl

Usage example:
  python src/MomentsLD_real_data.py \
    --config config_files/experiment_config_split_migration_growth.json \
    --empirical experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD/means.varcovs.pkl \
    --outdir experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD \
    --normalization 0 \
    --verbose
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import moments
import nlopt
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from inference_utils import lhs_start_log10, profile_1d, save_profiles  # noqa: E402


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
    demo_module = importlib.import_module("src.simulation")
    model_name = config["demographic_model"]
    if model_name == "drosophila_three_epoch":
        return getattr(demo_module, "drosophila_three_epoch")
    return getattr(demo_module, model_name + "_model")


# ------------------------- theoretical LD -------------------------


def _build_param_dict(param_names: List[str], vec: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(param_names, vec)}


def compute_theoretical_ld(
    log10_abs_params: np.ndarray,
    *,
    param_names: List[str],
    demographic_model_abs,
    r_bins: np.ndarray,
    populations: List[str],
    _diagnostic_once: Dict[str, bool],
) -> moments.LD.LDstats:
    """
    Compute expected sigmaD2 in absolute parameter space.
    N_ANC from the parameter vector is used as the rho anchor:
      rho = 4 * N_ANC * r_bins
    """
    p_abs = _build_param_dict(param_names, 10 ** np.asarray(log10_abs_params, dtype=float))
    N_anc = float(p_abs["N_ANC"])

    graph = demographic_model_abs(p_abs)

    rho_edges = 4.0 * N_anc * np.asarray(r_bins, dtype=float)

    if not _diagnostic_once.get("did", False):
        _diagnostic_once["did"] = True
        logging.info("[diagnostic] N_ANC = %.6g  (rho anchor)", N_anc)
        logging.info(
            "[diagnostic] rho_edges min/max = %.3e, %.3e",
            float(rho_edges.min()), float(rho_edges.max()),
        )
        logging.info("[diagnostic] populations = %s", populations)

    ld_edges = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_edges)

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
    theory_processed = moments.LD.LDstats(
        theoretical_sigmaD2[:],
        num_pops=theoretical_sigmaD2.num_pops,
        pop_ids=theoretical_sigmaD2.pop_ids,
    )
    theory_processed = moments.LD.Inference.remove_normalized_lds(theory_processed, normalization=normalization)
    theory_arrays = [np.asarray(x) for x in theory_processed[:-1]]

    emp_means = [np.asarray(x) for x in empirical_mv["means"]]
    emp_covars = [np.asarray(x) for x in empirical_mv["varcovs"]]

    emp_means, emp_covars = moments.LD.Inference.remove_normalized_data(
        emp_means,
        emp_covars,
        normalization=normalization,
        num_pops=theoretical_sigmaD2.num_pops,
    )

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


# ------------------------- main optimize -------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--empirical", required=True, type=Path, help="means.varcovs.pkl")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--normalization", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--r-bins", type=str, default=None, help="Comma-separated r-bin edges")
    ap.add_argument(
        "--opt-seed",
        type=int,
        default=None,
        help="Run index (0-based) for LHS start selection; set by Snakemake wildcard.",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = load_json(args.config)
    if args.opt_seed is not None:
        cfg["opt_seed"] = int(args.opt_seed)
    mv = load_pickle(args.empirical)

    demo_fn = load_demographic_function(cfg)

    if args.r_bins is None:
        r_bins = DEFAULT_R_BINS
    else:
        r_bins = np.array([float(x) for x in args.r_bins.split(",")], dtype=float)
        if r_bins.ndim != 1 or r_bins.size < 2:
            raise ValueError("--r-bins must have at least 2 edges")

    populations = list(cfg.get("num_samples", {}).keys())
    if not populations:
        raise ValueError("config['num_samples'] must specify populations")

    param_names = list(cfg.get("parameter_order", list(cfg["priors"].keys())))
    priors_abs = cfg.get("priors")
    if priors_abs is None:
        raise ValueError("Need config['priors'] with absolute bounds for MomentsLD")

    lb = np.array([float(priors_abs[p][0]) for p in param_names], dtype=float)
    ub = np.array([float(priors_abs[p][1]) for p in param_names], dtype=float)
    if np.any(lb <= 0) or np.any(ub <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb, ub) if lo <= 0 or hi <= 0]
        raise ValueError(f"All bounds must be >0 for log10 optimisation. Bad: {bad}")

    x0 = lhs_start_log10(lb, ub, cfg)

    _diag_once = {"did": False}

    def loglik(log10_params: np.ndarray) -> float:
        theo = compute_theoretical_ld(
            log10_params,
            param_names=param_names,
            demographic_model_abs=demo_fn,
            r_bins=r_bins,
            populations=populations,
            _diagnostic_once=_diag_once,
        )
        theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(
            theo, mv, normalization=int(args.normalization)
        )
        return composite_gaussian_ll(emp_means, emp_covars, theory_arrays)

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        ll = float(loglik(x))
        if args.verbose:
            p_abs = _build_param_dict(param_names, 10 ** np.asarray(x))
            show = ", ".join(f"{k}={p_abs[k]:.3g}" for k in param_names)
            print(f"LL = {ll:.6f} | {show}")
        return ll

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(param_names))
    opt.set_lower_bounds(np.log10(lb))
    opt.set_upper_bounds(np.log10(ub))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(float(args.rtol))

    best_x = opt.optimize(x0)
    status = opt.last_optimize_result()
    best_ll = opt.last_optimum_value()

    best_params_abs = _build_param_dict(param_names, 10 ** np.asarray(best_x))

    if cfg.get("generate_profiles", False):
        logging.info("Computing 1-D profile likelihoods …")
        profiles = profile_1d(
            xhat_log10=np.asarray(best_x),
            param_names=param_names,
            lb_full=lb,
            ub_full=ub,
            loglikelihood_fn=loglik,
            n_points=int(cfg.get("profile_points", 41)),
            widen=float(cfg.get("profile_widen", 0.5)),
        )
        save_profiles(
            profiles,
            args.outdir / "likelihood_plots",
            make_plots=bool(cfg.get("profile_make_plots", True)),
            title_prefix="MomentsLD profile likelihood",
        )
        logging.info("Profiles written to %s/likelihood_plots/", args.outdir)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_pkl = args.outdir / "best_fit.pkl"
    dump_pickle(
        {
            "status": int(status),
            "best_ll": float(best_ll),
            "best_params_abs": best_params_abs,
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

#!/usr/bin/env python3
"""
MomentsLD_real_data.py

Real-data MomentsLD inference in SCALED parameter space.
N_ref is fixed from the moments/dadi SFS best-fit so that
rho = 4 * N_ref * r_bins is correctly anchored.

Scaled parameterisation (matches priors_real_data_analysis in the config):
  N_ANC   : fixed to 1 (placeholder; absolute value = N_ref)
  N_*     : ratio  r  = N_*/N_ref
  T       : tau    τ  = T_abs / (2 * N_ref)
  m_*     : scaled M  = 2 * N_ref * m_abs

Inputs:
- --config          : experiment config JSON
- --empirical       : means.varcovs.pkl
- --outdir          : where to write best_fit.pkl
- --sfs-best-fit-pkl: moments/dadi best_fit.pkl (to extract N_ref = N_ANC)
  OR
- --n-ref           : fixed N_ref float

Usage example:
  python src/MomentsLD_real_data.py \
    --config config_files/experiment_config_split_migration_growth.json \
    --empirical experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD/means.varcovs.pkl \
    --outdir experiments/split_migration_growth/real_data_analysis/runs/run_0/inferences/MomentsLD \
    --sfs-best-fit-pkl experiments/split_migration_growth/real_data_analysis/inferences/moments/best_fit.pkl \
    --normalization 0 \
    --opt-seed 0 \
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

from inference_utils import absolute_to_scaled_params, lhs_start_log10, profile_1d, save_profiles  # noqa: E402

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


# ------------------------- scaling -------------------------


def _build_param_dict(param_names: List[str], vec: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(param_names, vec)}


def scaled_to_absolute_params(
    p_scaled: Dict[str, float],
    *,
    N_ref: float,
    time_scale: str = "2N",
) -> Dict[str, float]:
    """Convert scaled params (ratios / tau / M) to absolute using fixed N_ref."""
    if N_ref <= 0:
        raise ValueError(f"N_ref must be > 0, got {N_ref}")
    out = dict(p_scaled)
    out["N_ANC"] = float(N_ref)
    for k, v in list(out.items()):
        if k.startswith("N_") and k != "N_ANC":
            out[k] = float(v) * float(N_ref)
    if "T" in out:
        if time_scale != "2N":
            raise ValueError(f"Unknown time_scale={time_scale}")
        out["T"] = float(2.0 * N_ref * float(out["T"]))
    for k, v in list(out.items()):
        if k.startswith("m_"):
            out[k] = float(v) / float(2.0 * N_ref)
    return out


# ------------------------- N_ref resolution -------------------------


def resolve_n_ref(args) -> float:
    if args.n_ref is not None:
        return float(args.n_ref)

    if args.sfs_best_fit_pkl is None:
        raise ValueError("Provide --n-ref or --sfs-best-fit-pkl")

    d = load_pickle(Path(args.sfs_best_fit_pkl))

    if isinstance(d, dict):
        # list written by aggregate_opts_* rules — take the first (best) entry
        if "N_ANC_implied_from_theta" in d:
            v = d["N_ANC_implied_from_theta"]
            return float(v[0] if isinstance(v, (list, tuple)) else v)
        if "N_ANC_implied" in d:
            return float(d["N_ANC_implied"])
        # best_params_abs is a plain dict (single-run output)
        if "best_params_abs" in d and isinstance(d["best_params_abs"], dict):
            return float(d["best_params_abs"]["N_ANC"])
        # best_params is a list of dicts (aggregate_opts output)
        if (
            "best_params" in d
            and isinstance(d["best_params"], list)
            and d["best_params"]
        ):
            return float(d["best_params"][0]["N_ANC"])
        # best_params is a plain dict (single-run output)
        if "best_params" in d and isinstance(d["best_params"], dict):
            return float(d["best_params"]["N_ANC"])

    raise ValueError(
        f"Could not find N_ref in {args.sfs_best_fit_pkl}. "
        "Expected keys: N_ANC_implied_from_theta, N_ANC_implied, "
        "best_params_abs['N_ANC'], or best_params[0]['N_ANC']."
    )


# ------------------------- theoretical LD -------------------------


def compute_theoretical_ld(
    log10_params: np.ndarray,
    *,
    param_names: List[str],
    demographic_model_abs,
    r_bins: np.ndarray,
    populations: List[str],
    N_ref: float,
    use_scaled_units: bool,
    _diagnostic_once: Dict[str, bool],
) -> moments.LD.LDstats:
    """Compute expected sigmaD2. Handles both scaled and absolute parameterisations."""
    vec = 10 ** np.asarray(log10_params, dtype=float)
    p_dict = _build_param_dict(param_names, vec)
    if use_scaled_units:
        p_abs = scaled_to_absolute_params(p_dict, N_ref=N_ref, time_scale="2N")
        ref = N_ref
    else:
        p_abs = p_dict
        ref = float(p_dict.get("N_ANC") or p_dict.get("N0") or next(
            (v for k, v in p_dict.items() if k.startswith("N")), N_ref
        ))

    graph = demographic_model_abs(p_abs)
    rho_edges = 4.0 * float(ref) * np.asarray(r_bins, dtype=float)

    if not _diagnostic_once.get("did", False):
        _diagnostic_once["did"] = True
        logging.info("[diagnostic] ref size for rho = %.6g", ref)
        logging.info(
            "[diagnostic] rho_edges min/max = %.3e, %.3e",
            float(rho_edges.min()),
            float(rho_edges.max()),
        )
        logging.info("[diagnostic] populations = %s", populations)

    ld_edges = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_edges)

    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_mids)

    ld_bins = [
        (ld_edges[i] + ld_edges[i + 1] + 4.0 * ld_mids[i]) / 6.0
        for i in range(len(rho_mids))
    ]
    ld_bins.append(ld_edges[-1])

    ld_stats = moments.LD.LDstats(
        ld_bins, num_pops=ld_edges.num_pops, pop_ids=ld_edges.pop_ids
    )
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
    theory_processed = moments.LD.Inference.remove_normalized_lds(
        theory_processed, normalization=normalization
    )
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
    ap.add_argument(
        "--r-bins", type=str, default=None, help="Comma-separated r-bin edges"
    )
    ap.add_argument(
        "--n-ref", type=float, default=None, help="Fixed N_ref for rho scaling"
    )
    ap.add_argument(
        "--sfs-best-fit-pkl",
        type=Path,
        default=None,
        help="moments/dadi best_fit.pkl from which N_ANC is extracted as N_ref",
    )
    ap.add_argument(
        "--opt-seed",
        type=int,
        default=None,
        help="Run index (0-based) for LHS start selection; set by Snakemake wildcard.",
    )

    args = ap.parse_args()

    # Log to both the console and a per-run text file so the iteration-to-
    # iteration optimisation progress is preserved for inspection. The file
    # always receives every INFO line (including each objective evaluation);
    # the console still honours --verbose.
    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / "optimization_log.txt"
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _file_h = logging.FileHandler(log_path, mode="w")
    _file_h.setFormatter(_fmt)
    _console_h = logging.StreamHandler()
    _console_h.setLevel(logging.INFO if args.verbose else logging.WARNING)
    _console_h.setFormatter(_fmt)
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    _root.handlers[:] = [_file_h, _console_h]
    logging.info("Optimisation log -> %s", log_path)

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

    use_scaled_units = cfg.get("momentsld_use_scaled_units", True)
    priors_key = "priors_real_data_analysis" if use_scaled_units else "priors"
    priors = cfg.get(priors_key)
    if priors is None:
        raise ValueError(f"Need config['{priors_key}']")

    param_names = list(cfg.get("parameter_order", list(priors.keys())))
    lb = np.array([float(priors[p][0]) for p in param_names], dtype=float)
    ub = np.array([float(priors[p][1]) for p in param_names], dtype=float)
    if np.any(lb <= 0) or np.any(ub <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb, ub) if lo <= 0 or hi <= 0]
        raise ValueError(f"All bounds must be >0 for log10 optimisation. Bad: {bad}")

    N_ref = resolve_n_ref(args)
    logging.info("Using N_ref = %.6g (use_scaled_units=%s)", N_ref, use_scaled_units)

    # Pin any parameters listed in fixed_parameters.
    # Value can be a number or "moments_best" (loaded from --sfs-best-fit-pkl).
    fixed_cfg = cfg.get("fixed_parameters", {})
    if fixed_cfg:
        sfs_best_abs: Dict[str, float] = {}
        if args.sfs_best_fit_pkl is not None:
            d = load_pickle(Path(args.sfs_best_fit_pkl))
            bp = d.get("best_params", [])
            sfs_best_abs = dict(bp[0]) if isinstance(bp, list) and bp else {}
        for pname, spec in fixed_cfg.items():
            if pname not in param_names:
                continue
            if spec == "moments_best":
                val_abs = sfs_best_abs.get(pname)
                if val_abs is None:
                    logging.warning("moments_best requested for %s but not found in SFS pkl; skipping", pname)
                    continue
                if use_scaled_units:
                    N_anc = float(sfs_best_abs.get("N_ANC", N_ref))
                    val = absolute_to_scaled_params(sfs_best_abs, N_anc_abs=N_anc)[pname]
                else:
                    val = float(val_abs)
            else:
                val = float(spec)
            idx = param_names.index(pname)
            lb[idx] = ub[idx] = val
            logging.info("Fixing %s = %.6g (%s)", pname, val, "scaled" if use_scaled_units else "absolute")

    x0 = lhs_start_log10(lb, ub, cfg)

    _diag_once = {"did": False}

    def loglik(log10_params: np.ndarray) -> float:
        theo = compute_theoretical_ld(
            log10_params,
            param_names=param_names,
            demographic_model_abs=demo_fn,
            r_bins=r_bins,
            populations=populations,
            N_ref=N_ref,
            use_scaled_units=use_scaled_units,
            _diagnostic_once=_diag_once,
        )
        theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(
            theo, mv, normalization=int(args.normalization)
        )
        return composite_gaussian_ll(emp_means, emp_covars, theory_arrays)

    _eval = {"n": 0}

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        ll = float(loglik(x))
        _eval["n"] += 1
        vec = 10 ** np.asarray(x)
        p_dict = _build_param_dict(param_names, vec)
        p_abs = scaled_to_absolute_params(p_dict, N_ref=N_ref) if use_scaled_units else p_dict
        show = ", ".join(f"{k}={p_abs[k]:.3g}" for k in param_names)
        logging.info("eval %4d | LL = %.6f | %s | (N_ref=%.3g)",
                     _eval["n"], ll, show, N_ref)
        return ll

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(param_names))
    opt.set_lower_bounds(np.log10(lb))
    opt.set_upper_bounds(np.log10(ub))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(float(args.rtol))

    best_x = opt.optimize(x0)
    status = opt.last_optimize_result()
    best_ll = opt.last_optimum_value()

    best_dict = _build_param_dict(param_names, 10 ** np.asarray(best_x))
    best_abs = scaled_to_absolute_params(best_dict, N_ref=N_ref) if use_scaled_units else best_dict

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
            args.outdir / "likelihood_plots_scaled",
            make_plots=bool(cfg.get("profile_make_plots", True)),
            title_prefix="MomentsLD profile likelihood (scaled)",
        )
        logging.info("Profiles written to %s/likelihood_plots_scaled/", args.outdir)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_pkl = args.outdir / "best_fit.pkl"
    dump_pickle(
        {
            "status": int(status),
            "best_ll": float(best_ll),
            "fixed_N_ref": float(N_ref),
            "use_scaled_units": use_scaled_units,
            "best_params_optimizer_space": best_dict,
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

#!/usr/bin/env python3
"""
compute_fim.py — wrapper that selects the best params and calls fisher_info_sfs

Usage (from Snakefile rule compute_fim):
  python compute_fim.py \
      --engine {wildcards.engine} \
      --fit-pkl {input.fit} \
      --sfs {input.sfs} \
      --config {params.cfg} \
      --fim-npy {output.fim} \
      --summary-json {output.summ}
"""

from __future__ import annotations
import argparse, json, pickle, importlib
from pathlib import Path
import numpy as np
import math


# import the core you pasted
from fisher_info_sfs import observed_fim_theta

def _model_import_from_cfg(cfg: dict):
    """Resolve model function from config['demographic_model'] to src.simulation:*."""
    model = cfg["demographic_model"]
    if model == "drosophila_three_epoch":
        mod_path, func_name = "src.simulation", "drosophila_three_epoch"
    else:
        # convention used elsewhere in your pipeline
        mod_path, func_name = "src.simulation", f"{model}_model"
    return getattr(importlib.import_module(mod_path), func_name)

def _param_order_from_cfg(cfg: dict) -> list[str]:
    """JSON preserves order -> use priors key order as canonical parameter order."""
    priors = cfg["priors"]
    return list(priors.keys())

def _best_params_from_engine_fit(fit_blob: dict, param_order: list[str]) -> np.ndarray:
    """
    fit_blob is aggregate_opts output:
      {'best_params': [ {name: val, ...}, ... ],
       'best_ll':     [ ll, ... ] }
    Return theta in RAW units, ordered by param_order.
    """
    lls = fit_blob.get("best_ll", [])
    idx = int(np.argmax(lls))
    pdict = fit_blob["best_params"][idx]
    return np.array([float(pdict[name]) for name in param_order], dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["dadi","moments"])
    ap.add_argument("--fit-pkl", required=True, type=Path,
                    help="engine-specific fit_params.pkl from aggregate_opts")
    ap.add_argument("--sfs", required=True, type=Path,
                    help="Pickled observed SFS (dadi or moments Spectrum)")
    ap.add_argument("--config", required=True, type=Path,
                    help="experiment config JSON (must contain priors, mutation_rate, genome_length, demographic_model)")
    ap.add_argument("--fim-npy", required=True, type=Path)
    ap.add_argument("--summary-json", required=True, type=Path)
    ap.add_argument("--rel-step", type=float, default=1e-4,
                    help="relative step for numdifftools Hessian")
    ap.add_argument("--pts", type=str, default="auto",
                    help='For dadi: "n1,n2,n3" or "auto" (default). Ignored for moments.')
    args = ap.parse_args()

    # Load inputs
    fit_blob = pickle.load(args.fit_pkl.open("rb"))
    sfs = pickle.load(args.sfs.open("rb"))
    cfg = json.loads(args.config.read_text())
    mu = float(cfg["mutation_rate"])
    L  = int(cfg["genome_length"])

    # Resolve model + parameter order, then best theta
    model_func = _model_import_from_cfg(cfg)
    param_order = _param_order_from_cfg(cfg)
    theta = _best_params_from_engine_fit(fit_blob, param_order)

    # pts handling
    pts = "auto"
    if args.engine == "dadi":
        if args.pts and args.pts.lower() != "auto":
            xs = [int(x) for x in args.pts.split(",")]
            if len(xs) != 3:
                raise SystemExit("--pts must be 3 integers or 'auto'")
            pts = xs

    # Compute FIM (observed; raw units)
    info, cov, se, free_idx = observed_fim_theta(
        sfs=sfs,
        param_names=param_order,
        theta_at=theta,
        model_func=model_func,
        mu=mu,
        L=L,
        engine=args.engine,
        pts=pts,
        fixed_params=None,
        rel_step=args.rel_step
    )

    # Write outputs
    args.fim_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.fim_npy, info)

    summary = {}
    try:
        w, _ = np.linalg.eigh(info)
        w = np.asarray(w, float)
        w_clip = np.clip(w, 1e-300, None)
        summary["logdet"] = float(np.sum(np.log(w_clip)))
        summary["min_eigen"] = float(np.min(w))
        summary["max_eigen"] = float(np.max(w))
        summary["cond"] = float(np.max(w_clip) / np.min(w_clip))
    except Exception:
        pass

    if se is not None:
        summary["SE"] = se
    diag = np.diag(info)
    summary["diag"] = {param_order[i]: float(diag[k]) for k, i in enumerate(free_idx)}

    # Sanitize NaN / Inf before JSON
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None  # JSON-safe
        else:
            return obj

    safe_summary = _sanitize(summary)
    args.summary_json.write_text(json.dumps(safe_summary, indent=2))
    print(f"✓ FIM → {args.fim_npy}")
    print(f"✓ summary → {args.summary_json}")

if __name__ == "__main__":
    main()

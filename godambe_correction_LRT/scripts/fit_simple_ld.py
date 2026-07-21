#!/usr/bin/env python3
# godambe_correction_LRT/scripts/fit_simple_ld.py

"""
STEP 2 fit: fit ONE demographic model to an arm's aggregated overlapping LD
curve (means.varcovs.pkl), via LHS MULTI-START (default 100 starts, run in
parallel), keeping the best.

Which model is selected with --model:
  split_migration_growth       NULL: CO constant   (7 params; p0 for the J step)
  split_migration_growth_both  ALT:  CO grows       (8 params; N_CO0 & N_CO1)
(aliases "null"/"alt" accepted). Both fits use the SAME overlapping means/varcovs,
so their log-likelihoods are directly comparable -> raw LRT = 2*(ll_alt - ll_null).

Single-start (geometric-mean) optimization under-converges -- exactly what we
hit on the SFS side, fixed there by LHS multi-start in refit_models_lhs.py. This
is the Moments-LD analog: diverse Latin-hypercube starts through the repo's own
lhs_start_log10, each optimized with MomentsLD_inference.optimize_parameters.

best_fit.pkl["best_params"] is that model's MLE; for the null it is the p0 the
J step embeds into the complex model.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import pickle
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference_utils import lhs_start_log10
from src.MomentsLD_inference import optimize_parameters
from src.demes_models import (
    split_migration_growth_model,
    split_migration_growth_both_model,
)
from momentsld_inference import BOUNDS, NUM_SAMPLES, NULL, ALT, MODELS, R_BINS

logging.getLogger().setLevel(logging.WARNING)

SEED = 42                                        # stable LHS grid seed

MODEL_FUNCS = {
    NULL: split_migration_growth_model,          # CO constant  (7 params)
    ALT:  split_migration_growth_both_model,     # CO grows     (8 params)
}
_ALIASES = {"null": NULL, "alt": ALT}


def resolve_model(name):
    """Model name (or 'null'/'alt' alias) -> (param_names, lb, ub, model_func)."""
    name = _ALIASES.get(name, name)
    if name not in MODEL_FUNCS:
        raise ValueError(f"unknown --model {name!r}; choose from {list(MODEL_FUNCS)}")
    names = MODELS[name]
    lb = np.array([BOUNDS[p][0] for p in names], dtype=float)
    ub = np.array([BOUNDS[p][1] for p in names], dtype=float)
    return names, lb, ub, MODEL_FUNCS[name]


def run_one_start(args):
    """One LHS start -> one optimization. (Top-level for ProcessPoolExecutor.)

    args = (opt_index, n_opt, mv, model_name).
    """
    i, n_opt, mv, model_name = args
    param_names, lb, ub, model_func = resolve_model(model_name)
    cfg = {"num_optimizations": n_opt, "seed": SEED, "opt_seed": i}
    start = 10.0 ** lhs_start_log10(lb, ub, cfg)       # LHS row i, absolute units
    opt, ll, status = optimize_parameters(
        start_values=start, lower_bounds=lb, upper_bounds=ub,
        param_names=param_names, demographic_model=model_func,
        r_bins=R_BINS, empirical_data=mv, populations=list(NUM_SAMPLES),
        normalization=0, verbose=False,
    )
    return {"best_params": dict(zip(param_names, opt)),
            "best_lls": float(ll), "status": int(status), "opt": i}


def main():
    ap = argparse.ArgumentParser(description="LHS multi-start LD fit (null or alt model).")
    ap.add_argument("--mv", type=Path, required=True,
                    help="means.varcovs.pkl from the arm's overlapping windows")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default=NULL,
                    help="model to fit: split_migration_growth (null) or "
                         "split_migration_growth_both (alt); 'null'/'alt' aliases ok "
                         "(default %(default)s)")
    ap.add_argument("--n-opt", type=int, default=100, help="LHS starts (default %(default)s)")
    ap.add_argument("--workers", type=int, default=25)
    args = ap.parse_args()

    model_name = _ALIASES.get(args.model, args.model)
    resolve_model(model_name)          # validate early
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.mv, "rb") as f:
        mv = pickle.load(f)

    print(f"fitting {model_name} to {args.mv} with {args.n_opt} LHS starts "
          f"on {args.workers} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one_start, (i, args.n_opt, mv, model_name)): i
                for i in range(args.n_opt)}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                logging.warning("start failed: %s", e)

    if not results:
        raise RuntimeError("All optimizations failed.")

    results.sort(key=lambda r: r["best_lls"], reverse=True)
    best = results[0]

    with open(args.out_dir / "best_fit.pkl", "wb") as f:
        pickle.dump(best, f)
    with open(args.out_dir / "all_starts.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"best LL = {best['best_lls']:.4f} (start {best['opt']}, "
          f"{len(results)} succeeded) -> {args.out_dir/'best_fit.pkl'}")
    for p, v in best["best_params"].items():
        print(f"  {p:8s} = {v:.4g}")


if __name__ == "__main__":
    main()

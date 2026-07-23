#!/usr/bin/env python3
# godambe_correction_LRT/scripts/sfs_fit_one_start.py
"""
Run ONE LHS-seeded moments SFS fit (one optimizer start) for one arm + model.
Each --opt-index is its own Snakemake job, mirroring the LD side's
fit_one_start_ld.py / collect_best_fit.py split. Collected + turned into the
FIM/railing diagnostic by sfs_collect_and_fim.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import moments

from src.moments_inference_real import fit_model_realdata_scaled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True,
                    help="demographic model name, e.g. split_migration_growth "
                         "(resolved to src.demes_models:<model>_model)")
    ap.add_argument("--sfs", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--pop-ids", default="CO,FR")
    ap.add_argument("--opt-index", type=int, required=True)
    ap.add_argument("--n-starts", type=int, required=True,
                    help="total starts across all jobs (sizes the shared LHS grid)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    param_order = list(cfg["parameter_order"])
    model_func = getattr(
        importlib.import_module("src.demes_models"), f"{args.model}_model"
    )

    with open(args.sfs, "rb") as f:
        sfs = pickle.load(f)
    sfs = moments.Spectrum(sfs)
    sfs.pop_ids = [s.strip() for s in args.pop_ids.split(",")]

    cfg_run = dict(cfg)
    cfg_run["num_optimizations"] = args.n_starts   # size the LHS grid to match
    cfg_run["opt_seed"] = args.opt_index

    best_abs, ll_hat, theta_hat, N_anc_implied = fit_model_realdata_scaled(
        sfs=sfs,
        demo_model_abs=model_func,
        experiment_config=cfg_run,
        param_order=param_order,
        verbose=False,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(
            seed=args.opt_index,
            best_abs=best_abs,
            ll_hat=float(ll_hat),
            theta_hat=float(theta_hat),
            N_anc_implied=float(N_anc_implied),
        ), f)

    print(f"[{args.arm}/{args.model}] start {args.opt_index}: ll={ll_hat:.6f} -> {args.out}")


if __name__ == "__main__":
    main()

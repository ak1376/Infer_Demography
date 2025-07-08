#!/usr/bin/env python3
"""
split_isolation.py – Ray-parallel, toggle-able Moments / Dadi
=============================================================

Outputs
-------
experiments/split_isolation/runs/run_0001/…
experiments/split_isolation/inferences/scatter_{moments,dadi}_vs_true.png
"""
from __future__ import annotations
import argparse, json, pickle, sys, os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import demesdraw, moments, ray

# ── project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC          = PROJECT_ROOT / "src"
EXPERIMENTS  = PROJECT_ROOT / "experiments"
sys.path.append(str(SRC))                         # import local modules

from simulation        import split_isolation_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit, save_scatterplots
from dadi_inference    import fit_model as dadi_fit

PARAM_NAMES = ["N0", "N1", "N2", "m", "t_split"]

# ── Ray initialisation (no working-dir clone, just PYTHONPATH) ─────────
ray.init(
    logging_level="ERROR",
    runtime_env={"env_vars": {"PYTHONPATH": f"{PROJECT_ROOT}/src"}},
    ignore_reinit_error=True,
    num_cpus=5,  # adjust based on your system; 5 is a good default for 8-core CPUs
)

# ── helpers ────────────────────────────────────────────────────────────
def _sample_priors(priors: Dict[str, list[float]],
                   rng: np.random.Generator) -> Dict[str, float]:
    return {k: rng.uniform(*b) for k, b in priors.items()}

def _midpoint(priors: Dict[str, list[float]]) -> Dict[str, float]:
    return {k: (lo + hi) / 2 for k, (lo, hi) in priors.items()}

def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    return [{**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
            for v, ll in zip(vecs, lls)]

# ── Ray remote tasks ───────────────────────────────────────────────────
@ray.remote
def _simulate(params: Dict[str, float],
              cfg: Dict[str, Any]) -> Tuple[Any, "moments.Spectrum"]:
    ts, _ = simulation(params, "split_isolation", cfg)
    return ts, create_SFS(ts)

@ray.remote
def _moments(sfs, cfg, start_dict, params):
    return moments_fit(
        sfs,
        start_dict=start_dict,
        demo_model=split_isolation_model,
        experiment_config={**cfg, "log_dir": f"{cfg['log_dir']}/moments"}
    )

@ray.remote
def _dadi(sfs, cfg, start_dict, params):
    return dadi_fit(
        sfs,
        start_dict=start_dict,
        demo_model=split_isolation_model,
        experiment_config={**cfg, "log_dir": f"{cfg['log_dir']}/dadi"}
    )

# ── one replicate (Ray worker) ─────────────────────────────────────────
@ray.remote
def _run_one(idx: int, params: Dict[str, float],
             cfg: Dict[str, Any], base_dir: str,
             run_mom: bool, run_dadi: bool):

    run_dir  = Path(base_dir) / f"run_{idx:04d}"
    data_dir = run_dir / "data"; data_dir.mkdir(parents=True, exist_ok=True)

    ts, sfs = ray.get(_simulate.remote(params, cfg))

    ax = demesdraw.tubes(split_isolation_model(params))
    ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_split_isolation_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "split_isolation_SFS.pkl").open("wb"))
    ts.dump(data_dir / "split_isolation_tree_sequence.trees")

    start_dict = _midpoint(cfg["priors"])
    mom_ref = dadi_ref = None
    if run_mom:  mom_ref  = _moments.remote(sfs, cfg, start_dict, params)
    if run_dadi: dadi_ref = _dadi.remote(sfs, cfg, start_dict, params)

    mom_dicts = dadi_dicts = []
    if mom_ref:
        fits, lls = ray.get(mom_ref)
        mom_dicts = _attach_ll(fits, lls)
        (run_dir / "inferences/moments").mkdir(parents=True, exist_ok=True)
        pickle.dump(mom_dicts, (run_dir / "inferences/moments/fit_params.pkl").open("wb"))
    if dadi_ref:
        fits, lls = ray.get(dadi_ref)
        dadi_dicts = _attach_ll(fits, lls)
        (run_dir / "inferences/dadi").mkdir(parents=True, exist_ok=True)
        pickle.dump(dadi_dicts, (run_dir / "inferences/dadi/fit_params.pkl").open("wb"))

    return mom_dicts, dadi_dicts, params

# ── CLI driver ─────────────────────────────────────────────────────────
def str2bool(s: str) -> bool:
    return s.lower() not in {"false", "0", "no"}

def main():
    cli = argparse.ArgumentParser("Split-isolation runner (Ray)")
    cli.add_argument("--experiment_config", required=True, type=Path)
    cli.add_argument("--num_draws", type=int)
    cli.add_argument("--run_moments", type=str2bool, default=True)
    cli.add_argument("--run_dadi",    type=str2bool, default=False)
    args = cli.parse_args()

    cfg    = json.loads(args.experiment_config.read_text())
    priors = cfg["priors"]
    draws  = args.num_draws or cfg.get("num_draws", 1)
    rng    = np.random.default_rng(cfg.get("seed"))

    runs_root = (EXPERIMENTS / cfg["demographic_model"] / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)

    futures = [_run_one.remote(i,
                               _sample_priors(priors, rng),
                               cfg, str(runs_root),
                               args.run_moments, args.run_dadi)
               for i in range(1, draws + 1)]

    all_true, all_mom, all_dadi = [], [], []
    for mom, dadi, true_par in ray.get(futures):
        if mom:  all_mom.extend(mom)
        if dadi: all_dadi.extend(dadi)
        all_true.extend([true_par] * max(len(mom), len(dadi), 1))

    inf_dir = (EXPERIMENTS / cfg["demographic_model"] / "inferences")
    inf_dir.mkdir(parents=True, exist_ok=True)

    if all_mom:
        save_scatterplots(
            all_true, all_mom,
            [d["loglik"] for d in all_mom],
            PARAM_NAMES,
            inf_dir / "scatter_moments_vs_true.png",
            label="moments",
        )
    if all_dadi:
        save_scatterplots(
            all_true, all_dadi,
            [d["loglik"] for d in all_dadi],
            PARAM_NAMES,
            inf_dir / "scatter_dadi_vs_true.png",
            label="dadi",
        )

if __name__ == "__main__" and os.getenv("RUNNING_RAY_WORKER") != "1":
    main()

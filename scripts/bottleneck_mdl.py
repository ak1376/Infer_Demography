#!/usr/bin/env python3
"""
bottleneck.py – multi-replicate runner with Ray, toggle-able Moments / Dadi
===========================================================================

Directory layout (unchanged):
    bottleneck/runs/run_0001/…
    bottleneck/inferences/scatter_moments_vs_true.png   (if run_moments)
    bottleneck/inferences/scatter_dadi_vs_true.png      (if run_dadi)
"""
from __future__ import annotations
import argparse, json, pickle, sys, os
from pathlib import Path
from typing  import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import demesdraw
import ray
import moments

# ── local & project paths ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC          = PROJECT_ROOT / "src"
EXPERIMENTS  = PROJECT_ROOT / "experiments"
sys.path.append(str(SRC))


# now the local imports
from simulation        import bottleneck_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model, save_scatterplots
from dadi_inference    import fit_model as dadi_fit_model

PARAM_NAMES = ["N0", "N_bottleneck", "N_recover",
               "t_bottleneck_start", "t_bottleneck_end"]

# ───────────────────────── helper utilities ────────────────────────────
def _sample_priors(priors: Dict[str, list[float]],
                   rng   : np.random.Generator) -> Dict[str, float]:
    d = {k: rng.uniform(*bounds) for k, bounds in priors.items()}
    # keep times in correct order
    if d["t_bottleneck_start"] <= d["t_bottleneck_end"]:
        d["t_bottleneck_start"], d["t_bottleneck_end"] = (
            d["t_bottleneck_end"], d["t_bottleneck_start"]
        )
    return d

def _midpoint_start(priors: Dict[str, list[float]]) -> Dict[str, float]:
    return {k: (lo + hi) / 2 for k, (lo, hi) in priors.items()}

def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    return [
        {**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
        for v, ll in zip(vecs, lls)
    ]

# ────────────────────────── Ray setup ──────────────────────────────────
ray.init(
    logging_level="ERROR",
    runtime_env={"env_vars": {"PYTHONPATH": f"{PROJECT_ROOT}/src"}},
    ignore_reinit_error=True,
    num_cpus=5
)


@ray.remote
def _simulate(params: Dict[str, float],
              cfg: Dict[str, Any]) -> Tuple[Any, "moments.Spectrum"]:
    ts, _ = simulation(params, "bottleneck", cfg)
    return ts, create_SFS(ts)

@ray.remote
def _moments(sfs, cfg, start_dict, params):
    return moments_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=bottleneck_model,
        experiment_config={**cfg, "log_dir": f"{cfg['log_dir']}/moments"},
        sampled_params=params,
    )

@ray.remote
def _dadi(sfs, cfg, start_dict, params):
    return dadi_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=bottleneck_model,
        experiment_config={**cfg, "log_dir": f"{cfg['log_dir']}/dadi"},
        sampled_params=params,
    )

# ─────────────────── one replicate (Ray remote) ───────────────────────
@ray.remote
def _run_one(idx: int,
             params: Dict[str, float],
             cfg: Dict[str, Any],
             base_dir: str,
             run_moments: bool,
             run_dadi: bool):
    rng = np.random.default_rng((cfg.get("seed") or 0) + idx)

    run_dir  = Path(base_dir) / f"run_{idx:04d}"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ts, sfs = ray.get(_simulate.remote(params, cfg))

    # demography figure
    ax = demesdraw.tubes(bottleneck_model(params))
    ax.set_xlabel("Time")
    ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_bottleneck_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # cache raw data
    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "bottleneck_SFS.pkl").open("wb"))
    ts.dump(data_dir / "bottleneck_tree_sequence.trees")

    start_dict = _midpoint_start(cfg["priors"])

    mom_dicts = dadi_dicts = []
    # schedule optimisers
    mom_ref = dadi_ref = None
    if run_moments:
        mom_ref = _moments.remote(sfs, cfg, start_dict, params)
    if run_dadi:
        dadi_ref = _dadi.remote(sfs, cfg, start_dict, params)

    if mom_ref:
        fits_mom, lls_mom = ray.get(mom_ref)
        mom_dicts = _attach_ll(fits_mom, lls_mom)
        mom_dir = run_dir / "inferences" / "moments"
        mom_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(mom_dicts, (mom_dir / "fit_params.pkl").open("wb"))

    if dadi_ref:
        fits_dadi, lls_dadi = ray.get(dadi_ref)
        dadi_dicts = _attach_ll(fits_dadi, lls_dadi)
        dadi_dir = run_dir / "inferences" / "dadi"
        dadi_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(dadi_dicts, (dadi_dir / "fit_params.pkl").open("wb"))

    return mom_dicts, dadi_dicts, params

# ────────────────────────── CLI entry-point ────────────────────────────
def str2bool(s: str) -> bool:           # allow =False on CLI
    return s.lower() not in {"false", "0", "no"}

def main():
    cli = argparse.ArgumentParser("Bottleneck runner (Ray)")
    cli.add_argument("--experiment_config", required=True, type=Path)
    cli.add_argument("--num_draws", type=int)
    cli.add_argument("--run_moments", type=str2bool, default=True)
    cli.add_argument("--run_dadi",    type=str2bool, default=False)
    args = cli.parse_args()

    cfg     = json.loads(args.experiment_config.read_text())
    priors  = cfg["priors"]
    draws   = args.num_draws or cfg.get("num_draws", 1)
    rng     = np.random.default_rng(cfg.get("seed"))

    base_runs = EXPERIMENTS / cfg["demographic_model"] / "runs"
    base_runs.mkdir(parents=True, exist_ok=True)

    # schedule replicates
    pending = []
    for i in range(1, draws + 1):
        params = _sample_priors(priors, rng)
        pending.append(
            _run_one.remote(i, params, cfg, str(base_runs),
                            args.run_moments, args.run_dadi)
        )

    # collect
    all_true, all_mom, all_dadi = [], [], []
    for mom, dadi, true_par in ray.get(pending):
        if mom:  all_mom.extend(mom)
        if dadi: all_dadi.extend(dadi)
        all_true.extend([true_par] * max(len(mom), len(dadi), 1))

    inf_dir = EXPERIMENTS / cfg["demographic_model"] / "inferences"
    inf_dir.mkdir(parents=True, exist_ok=True)

    if all_mom:
        save_scatterplots(
            true_vecs   = all_true,
            est_vecs    = all_mom,
            ll_vec      = [d["loglik"] for d in all_mom],
            param_names = PARAM_NAMES,
            outfile     = inf_dir / "scatter_moments_vs_true.png",
            label       = "moments",
        )

    if all_dadi:
        save_scatterplots(
            true_vecs   = all_true,
            est_vecs    = all_dadi,
            ll_vec      = [d["loglik"] for d in all_dadi],
            param_names = PARAM_NAMES,
            outfile     = inf_dir / "scatter_dadi_vs_true.png",
            label       = "dadi",
        )

if __name__ == "__main__" and os.getenv("RUNNING_RAY_WORKER") != "1":
    main()

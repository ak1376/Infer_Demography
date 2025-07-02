#!/usr/bin/env python3
"""
split_isolation.py – parallel replicates with Ray
=====================================================
Identical logic to split_isolation.py, but each replicate (simulate +
moments + dadi inference + file I/O) runs in its own Ray worker.

Usage
-----
conda/venv> pip install "ray[default]"   # one-time
conda/venv> python split_isolation_ray.py \
              --experiment_config config_files/experiment_config_split_isolation.json
"""

from __future__ import annotations
import argparse, json, pickle, sys, datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import demesdraw, moments, numpy as np
from tqdm import tqdm
import ray   
                                        # ← NEW
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # one level above scripts/
SRC_DIR      = PROJECT_ROOT / "src"       
# ─────────────────────────── local imports ──────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from simulation        import split_isolation_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model, save_scatterplots
from dadi_inference    import fit_model as dadi_fit_model

PARAM_NAMES = ["N0", "N1", "N2", "m", "t_split"]

# ───────────────────────────── helpers ──────────────────────────────────
def _sample_one(priors: Dict[str, list[float]],
                rng:    np.random.Generator) -> Dict[str, float]:
    draw: Dict[str, float] = {}
    for k, (lo, hi) in priors.items():
        v = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            v = int(round(v))
        draw[k] = v
    return draw

def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    return [
        {**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
        for v, ll in zip(vecs, lls)
    ]

# ────────────────────────── one replicate (Ray task) ────────────────────
@ray.remote
def run_one(cfg: Dict[str, Any],
            params: Dict[str, float],
            run_dir: str,                   # Path objects → str for Ray
            seed: int) -> Tuple[List[dict], List[dict]]:
    """Simulate + infer for one parameter set; returns (moments, dadi)."""
    run_dir = Path(run_dir)
    rng     = np.random.default_rng(seed)

    data_dir     = run_dir / "data";                    data_dir.mkdir(parents=True, exist_ok=True)
    mom_dir      = run_dir / "inferences" / "moments"
    dadi_dir     = run_dir / "inferences" / "dadi"
    mom_log_dir  = run_dir / "inferences" / "logs" / "moments"
    dadi_log_dir = run_dir / "inferences" / "logs" / "dadi"
    for d in (mom_dir, dadi_dir, mom_log_dir, dadi_log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. demography figure ----------------------------------------------
    g = split_isolation_model(params)
    ax = demesdraw.tubes(g); ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_split_isolation_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # 2. simulation -----------------------------------------------------
    ts, _ = simulation(params, "split_isolation", cfg)
    sfs = create_SFS(ts)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "split_isolation_SFS.pkl").open("wb"))
    ts.dump(data_dir / "split_isolation_tree_sequence.trees")

    # 3. independent start vector ---------------------------------------
    mean   = [(lo + hi) / 2 for lo, hi in cfg["priors"].values()]
    # start = mean.copy()  # use the mean of the prior as the start vector
    start  = moments.Misc.perturb_params(mean, 0.1)
    start_dict = {k: v for k, v in zip(PARAM_NAMES, mean)}

    # 4. inference -------------------------------------------------------
    fits_mom, lls_mom = moments_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=split_isolation_model,
        experiment_config={**cfg, "log_dir": str(mom_log_dir)},
    )

    fits_dadi, lls_dadi = dadi_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=split_isolation_model,
        experiment_config={**cfg, "log_dir": str(dadi_log_dir)},
    )

    mom_dicts  = _attach_ll(fits_mom,  lls_mom)
    dadi_dicts = _attach_ll(fits_dadi, lls_dadi)

    pickle.dump(mom_dicts,  (mom_dir  / "fit_params.pkl").open("wb"))
    pickle.dump(dadi_dicts, (dadi_dir / "fit_params.pkl").open("wb"))
    return mom_dicts, dadi_dicts

# ───────────────────────────── main driver ──────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("Split-isolation (Ray-parallel) runner")
    p.add_argument("--experiment_config", required=True, type=Path)
    p.add_argument("--num_draws", type=int, default=None,
                   help="How many replicates (default from JSON)")
    for name in PARAM_NAMES:                               # CLI pins
        p.add_argument(f"--{name}", type=float, default=None)
    args = p.parse_args()

    cfg      = json.loads(args.experiment_config.read_text())
    priors   = cfg["priors"]
    rng_main = np.random.default_rng(cfg.get("seed"))
    n_draw   = args.num_draws or cfg.get("num_draws", 1)

    base      = Path(cfg["demographic_model"]); base.mkdir(exist_ok=True)
    runs_root = base / "runs"; runs_root.mkdir(exist_ok=True)

    # ─── launch Ray -----------------------------------------------------
    ray.init(
        runtime_env={
            # ship the entire repo so every worker sees the same files
            "working_dir": str(PROJECT_ROOT),

            # and make “src/” importable
            "env_vars": {"PYTHONPATH": str(SRC_DIR)}
        }
    )
    futures = []
    true_vecs: list[dict[str, float]] = []

    for idx in range(1, n_draw + 1):
        params = _sample_one(priors, rng_main)     # ground truth
        for k in PARAM_NAMES:                      # CLI overrides
            manual = getattr(args, k)
            if manual is not None:
                params[k] = manual

        run_dir = runs_root / f"run_{idx:04d}"
        seed_i  = cfg.get("seed", 0) + idx

        futures.append(
            run_one.remote(cfg, params, str(run_dir), seed_i)
        )
        true_vecs.append(params)

    # gather results -----------------------------------------------------
    results = ray.get(futures)      # blocks until all replicates finish

    all_mom, all_dadi = [], []
    for (mom, dadi) in results:
        all_mom.extend(mom)
        all_dadi.extend(dadi)

    # save scatter-plots --------------------------------------------------
    inf_dir = base / "inferences"; inf_dir.mkdir(exist_ok=True)
    rep_mom   = len(all_mom)  // n_draw        # an int
    rep_dadi  = len(all_dadi) // n_draw

    save_scatterplots(
        true_vecs=true_vecs * rep_mom,         # list * int
        est_vecs=all_mom,
        ll_vec=[d["loglik"] for d in all_mom],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_moments_vs_true.png",
        label="moments",
    )

    save_scatterplots(
        true_vecs=true_vecs * rep_dadi,
        est_vecs=all_dadi,
        ll_vec=[d["loglik"] for d in all_dadi],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_dadi_vs_true.png",
        label="dadi",
    )

if __name__ == "__main__":
    main()

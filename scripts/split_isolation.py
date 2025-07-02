#!/usr/bin/env python3
"""
split_isolation.py – multi-replicate, directory-aware
=====================================================
For each replicate it

  1. samples *ground-truth* parameters from the uniform priors in the JSON
     (CLI flags may “pin” any parameter);
  2. draws an *independent* start vector from the same priors, perturbs it
     by ±10 %, and uses that as the initial guess for moments & dadi;
  3. simulates, runs inference, stores fits (+ log-lik) under
        split_isolation/runs/run_XXXX/

After all replicates finish it writes two coloured scatter-plots
(true vs estimate, coloured by log-likelihood):

    split_isolation/inferences/scatter_moments_vs_true.png
    split_isolation/inferences/scatter_dadi_vs_true.png
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import demesdraw, moments, numpy as np
from tqdm import tqdm

# ─────────────────────────── local imports ──────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from simulation        import split_isolation_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model, save_scatterplots
from dadi_inference    import fit_model as dadi_fit_model

# ───────────────────────────── constants ────────────────────────────────
PARAM_NAMES = ["N0", "N1", "N2", "m", "t_split"]

# ───────────────────────────── helpers ──────────────────────────────────
def _sample_one(priors: Dict[str, list[float]],
                rng:    np.random.Generator) -> Dict[str, float]:
    """Draw a parameter vector from uniform priors."""
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

# ───────────────────────────── one replicate ────────────────────────────
def run_one(cfg: Dict[str, Any],
            params: Dict[str, float],
            out:    Path,
            rng:    np.random.Generator) -> Tuple[List[dict], List[dict]]:
    """Simulate + infer for one param set; returns (moments, dadi) dicts."""
    data_dir     = out / "data";                    data_dir.mkdir(parents=True)
    mom_dir      = out / "inferences" / "moments"
    dadi_dir     = out / "inferences" / "dadi"
    mom_log_dir  = out / "inferences" / "logs" / "moments"
    dadi_log_dir = out / "inferences" / "logs" / "dadi"

    for d in (mom_dir, dadi_dir, mom_log_dir, dadi_log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. demography figure ------------------------------------------------
    g = split_isolation_model(params)
    ax = demesdraw.tubes(g); ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_split_isolation_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # 2. simulation -------------------------------------------------------
    ts, g_sim = simulation(params, "split_isolation", cfg)
    sfs = create_SFS(ts)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "split_isolation_SFS.pkl").open("wb"))
    ts.dump(data_dir / "split_isolation_tree_sequence.trees")

    # 3. choose a *random* start vector from the priors (+ 10 % jitter) ---
    # start_dict = _sample_one(cfg["priors"], rng)

    # start      = moments.Misc.perturb_params(
    #                 [params[p] for p in PARAM_NAMES], 0.1)
    # start_dict = {k: v for k, v in zip(PARAM_NAMES, start)}

    mean = [(a + b) / 2 for a, b in cfg['priors'].values()]
    # Convert to a dict
    start_dict = {k: v for k, v in zip(PARAM_NAMES, mean)}
    # Add some jitter
    print(f'Starting parameters: {start_dict}')  # debug output


    # 4. inference --------------------------------------------------------


    fits_mom, lls_mom = moments_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model = split_isolation_model,
        experiment_config={**cfg, "log_dir": str(mom_log_dir)}
    )
    print(f'Moments optimized parameters: {fits_mom}')  # debug output
    print(f'Moments log-likelihoods: {lls_mom}')        # debug output

    fits_dadi, lls_dadi = dadi_fit_model(
        sfs,
        start_dict=start_dict,                 # dict you already built
        demo_model=split_isolation_model,      # the *function*, not g_sim
        experiment_config={**cfg, "log_dir": str(dadi_log_dir)}
    )

    print(f'Dadi optimized parameters: {fits_dadi}')  # debug output
    print(f'Dadi log-likelihoods: {lls_dadi}')        # debug

    # mom_dicts = {}
    # dadi_dicts = {}

    mom_dicts  = _attach_ll(fits_mom,  lls_mom)
    dadi_dicts = _attach_ll(fits_dadi, lls_dadi)

    pickle.dump(mom_dicts,  (mom_dir  / "fit_params.pkl").open("wb"))
    pickle.dump(dadi_dicts, (dadi_dir / "fit_params.pkl").open("wb"))
    return mom_dicts, dadi_dicts

# ───────────────────────────── main driver ───────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("Split-isolation multi-replicate runner")
    p.add_argument("--experiment_config", required=True, type=Path)
    p.add_argument("--num_draws", type=int, default=None,
                   help="How many replicates (default from JSON)")
    # CLI pins
    for name in PARAM_NAMES:
        p.add_argument(f"--{name}", type=float, default=None)
    args = p.parse_args()

    cfg      = json.loads(args.experiment_config.read_text())
    priors   = cfg["priors"]
    rng      = np.random.default_rng(cfg.get("seed"))
    n_draw   = args.num_draws or cfg.get("num_draws", 1)

    base      = Path(cfg["demographic_model"]); base.mkdir(exist_ok=True)
    runs_root = base / "runs"; runs_root.mkdir(exist_ok=True)

    all_true, all_mom, all_dadi = [], [], []

    for idx in tqdm(range(1, n_draw + 1), desc="replicates"):
        params = _sample_one(priors, rng)           # ground truth
        print(f"Sampled params for run {idx:04d}: {params}")  # debug output
        # apply CLI overrides (“pins”)
        for k in PARAM_NAMES:
            manual = getattr(args, k)
            if manual is not None:
                params[k] = manual
        run_dir = runs_root / f"run_{idx:04d}"
        mom, dadi = run_one(cfg, params, run_dir, rng)   # ← pass rng
        all_true.extend([params] * len(dadi))
        all_mom.extend(mom)
        all_dadi.extend(dadi)

    # ─── global scatter-plots ───────────────────────────────────────────
    inf_dir = base / "inferences"; inf_dir.mkdir(exist_ok=True)
    save_scatterplots(
        true_vecs=all_true,
        est_vecs=all_mom,
        ll_vec=[d["loglik"] for d in all_mom],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_moments_vs_true.png",
        label="moments",
    )
    save_scatterplots(
        true_vecs=all_true,
        est_vecs=all_dadi,
        ll_vec=[d["loglik"] for d in all_dadi],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_dadi_vs_true.png",
        label="dadi",
    )

if __name__ == "__main__":
    main()

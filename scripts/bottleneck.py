#!/usr/bin/env python3
"""
bottleneck.py – multi-replicate, directory-aware
================================================
For every replicate it

  1. draws a parameter set from the uniform priors in the JSON,
     honouring any CLI pins;
  2. simulates, runs moments & dadi inference, stores fits (+ LL);
  3. writes everything under  bottleneck/runs/run_XXXX/ …

After all replicates finish it writes two coloured scatter-plots
(true-vs-estimate, coloured by log-likelihood) that summarise *all*
moments and dadi fits:

    bottleneck/inferences/scatter_moments_vs_true.png
    bottleneck/inferences/scatter_dadi_vs_true.png
"""
from __future__ import annotations
import argparse, json, pickle, sys, math
from pathlib import Path
from typing   import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import demesdraw, moments, numpy as np
from tqdm import tqdm

# ───────────────────────────── local imports ──────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))
from simulation      import bottleneck_model, simulation, create_SFS
from moments_inference import (
    fit_model   as moments_fit_model,
    save_scatterplots,                    # colour-by-LL helper
)
from dadi_inference    import fit_model as dadi_fit_model

# ───────────────────────────── constants ──────────────────────────────────
PARAM_NAMES = ["N0", "N_bottleneck", "N_recover",
               "t_bottleneck_start", "t_bottleneck_end"]

# ───────────────────────────── helpers ────────────────────────────────────
def _sample_one(priors: Dict[str, list[float]],
                rng:    np.random.Generator) -> Dict[str, float]:
    """Draw a single parameter vector from uniform priors."""
    d: Dict[str, float] = {}
    for k, (lo, hi) in priors.items():
        v = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            v = int(round(v))
        d[k] = v
    # ensure t_start > t_end for a valid bottleneck
    if d["t_bottleneck_start"] <= d["t_bottleneck_end"]:
        d["t_bottleneck_start"], d["t_bottleneck_end"] = (
            d["t_bottleneck_end"], d["t_bottleneck_start"])
    return d

def _attach_ll(vecs: list[np.ndarray], lls: list[float]) -> list[dict]:
    """Convert to list-of-dicts with 'loglik' added."""
    return [{**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
            for v, ll in zip(vecs, lls)]

# ───────────────────────────── one replicate ──────────────────────────────
def run_one(cfg: Dict[str, Any],
            params: Dict[str, float],
            out:    Path) -> Tuple[List[dict], List[dict]]:
    """Simulate, infer, save artefacts; return (mom_dicts, dadi_dicts)."""
    data_dir = out / "data";                 data_dir.mkdir(parents=True)
    mom_dir  = out / "inferences" / "moments"
    dadi_dir = out / "inferences" / "dadi"
    mom_dir.mkdir(parents=True)
    dadi_dir.mkdir(parents=True)

    # --- demography figure ------------------------------------------------
    g = bottleneck_model(params)
    ax = demesdraw.tubes(g); ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_bottleneck_model.png", dpi=300,
                bbox_inches="tight"); plt.close(ax.figure)

    # --- simulation -------------------------------------------------------
    ts, g_sim = simulation(params, "bottleneck", cfg)
    sfs = create_SFS(ts)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "bottleneck_SFS.pkl").open("wb"))
    ts.dump(data_dir / "bottleneck_tree_sequence.trees")

    # --- inference --------------------------------------------------------
    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], .1)
    fits_mom,  lls_mom  = moments_fit_model(sfs, start, g_sim, cfg,
                                            sampled_params=params)
    fits_dadi, lls_dadi = dadi_fit_model(   sfs, start, g_sim, cfg,
                                            sampled_params=params)

    mom_dicts  = _attach_ll(fits_mom,  lls_mom)
    dadi_dicts = _attach_ll(fits_dadi, lls_dadi)

    pickle.dump(mom_dicts,  (mom_dir  / "fit_params.pkl").open("wb"))
    pickle.dump(dadi_dicts, (dadi_dir / "fit_params.pkl").open("wb"))
    return mom_dicts, dadi_dicts

# ───────────────────────────── main driver ────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("Bottleneck multi-replicate runner")
    p.add_argument("--experiment_config", required=True, type=Path)
    p.add_argument("--num_draws", type=int, default=None,
                   help="How many replicates (default from JSON)")
    # CLI pins
    for name in PARAM_NAMES:
        p.add_argument(f"--{name}", type=float, default=None)
    args = p.parse_args()

    cfg   = json.loads(args.experiment_config.read_text())
    rng   = np.random.default_rng(cfg.get("seed"))
    n_draw = args.num_draws or cfg.get("num_draws", 1)
    priors = cfg["priors"]

    base = Path(cfg["demographic_model"]); base.mkdir(exist_ok=True)
    runs_root = base / "runs"; runs_root.mkdir(exist_ok=True)

    all_mom, all_dadi, all_true = [], [], []

    for idx in tqdm(range(1, n_draw + 1), desc="replicates"):
        params = _sample_one(priors, rng)
        # apply CLI overrides
        for k in PARAM_NAMES:
            ov = getattr(args, k);   # None or user value
            if ov is not None:
                params[k] = ov
        run_dir = runs_root / f"run_{idx:04d}"
        mom, dadi = run_one(cfg, params, run_dir)
        all_true.extend([params] * len(mom))   # one truth per fit
        all_mom.extend(mom)
        all_dadi.extend(dadi)

    # ─── overall scatter-plots ───────────────────────────────────────────
    inf_dir = base / "inferences"; inf_dir.mkdir(exist_ok=True)
    save_scatterplots(
        true_vecs=all_true,
        est_vecs=all_mom,
        ll_vec =[d["loglik"] for d in all_mom],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_moments_vs_true.png",
        label="moments",
    )
    save_scatterplots(
        true_vecs=all_true,
        est_vecs=all_dadi,
        ll_vec =[d["loglik"] for d in all_dadi],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_dadi_vs_true.png",
        label="dadi",
    )

if __name__ == "__main__":
    main()

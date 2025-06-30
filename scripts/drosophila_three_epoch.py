#!/usr/bin/env python3
"""
drosophila_three_epoch.py – multi-replicate, directory-aware
============================================================
For every replicate:

  1. Draw a parameter set from the uniform ``priors`` in the JSON,
     honouring any CLI pins;
  2. Simulate, run moments & dadi inference, store fits (+ log-lik);
  3. Write outputs under  drosophila_three_epoch/runs/run_XXXX/

Afterwards two global scatter-plots are saved (one for moments, one for
dadi) coloured by log-likelihood.
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import demesdraw, moments, numpy as np
from tqdm import tqdm

# ───────────────────────────── local imports ──────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from simulation        import drosophila_three_epoch, simulation, create_SFS
from moments_inference import (
    fit_model as moments_fit_model,
    save_scatterplots,
)
from dadi_inference    import fit_model as dadi_fit_model

# ───────────────────────────── constants ──────────────────────────────────
PARAM_NAMES: List[str] = [
    "N0",
    "AFR",
    "EUR_bottleneck",
    "EUR_recover",
    "T_AFR_expansion",
    "T_AFR_EUR_split",
    "T_EUR_expansion",
]

# ───────────────────────────── helpers ────────────────────────────────────
def _sample_one(priors: Dict[str, list[float]],
                rng:    np.random.Generator) -> Dict[str, float]:
    """Draw one parameter vector from uniform priors."""
    d: Dict[str, float] = {}
    for k, (lo, hi) in priors.items():
        v = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            v = int(round(v))
        d[k] = v
    return d


def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    """Add 'loglik' to each parameter dictionary."""
    return [
        {**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
        for v, ll in zip(vecs, lls)
    ]


# ───────────────────────────── one replicate ──────────────────────────────
def run_one(cfg: Dict[str, Any],
            params: Dict[str, float],
            out:    Path) -> Tuple[List[dict], List[dict]]:
    """Simulate + infer for one parameter set; return (moments, dadi) dicts."""
    data_dir = out / "data";                 data_dir.mkdir(parents=True)
    mom_dir  = out / "inferences" / "moments"
    dadi_dir = out / "inferences" / "dadi"
    mom_dir.mkdir(parents=True)
    dadi_dir.mkdir(parents=True)

    # 1. demography figure ------------------------------------------------
    g = drosophila_three_epoch(params)
    ax = demesdraw.tubes(g); ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_drosophila_three_epoch.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # 2. simulation -------------------------------------------------------
    ts, g_sim = simulation(params, "drosophila_three_epoch", cfg)
    sfs = create_SFS(ts)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "drosophila_three_epoch_SFS.pkl").open("wb"))
    ts.dump(data_dir / "drosophila_three_epoch_tree_sequence.trees")

    # 3. inference --------------------------------------------------------
    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], 0.1)
    fits_mom,  lls_mom  = moments_fit_model(sfs, start, g_sim, cfg)
    fits_dadi, lls_dadi = dadi_fit_model(   sfs, start, g_sim, cfg)

    mom_dicts  = _attach_ll(fits_mom,  lls_mom)
    dadi_dicts = _attach_ll(fits_dadi, lls_dadi)

    pickle.dump(mom_dicts,  (mom_dir  / "fit_params.pkl").open("wb"))
    pickle.dump(dadi_dicts, (dadi_dir / "fit_params.pkl").open("wb"))
    return mom_dicts, dadi_dicts


# ───────────────────────────── main driver ────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("Drosophila three-epoch multi-replicate runner")
    p.add_argument("--experiment_config", required=True, type=Path)
    p.add_argument("--num_draws", type=int, default=None,
                   help="Replicates to run (default from JSON)")

    # CLI “pins” – override any prior draw
    for n in PARAM_NAMES:
        p.add_argument(f"--{n}", type=float, default=None)

    args = p.parse_args()

    cfg     = json.loads(args.experiment_config.read_text())
    priors  = cfg["priors"]
    rng     = np.random.default_rng(cfg.get("seed"))
    n_draw  = args.num_draws or cfg.get("num_draws", 1)

    base      = Path(cfg["demographic_model"]); base.mkdir(exist_ok=True)
    runs_root = base / "runs";                  runs_root.mkdir(exist_ok=True)

    all_true, all_mom, all_dadi = [], [], []

    for idx in tqdm(range(1, n_draw + 1), desc="replicates"):
        params = _sample_one(priors, rng)
        # apply CLI overrides
        for k in PARAM_NAMES:
            ov = getattr(args, k)
            if ov is not None:
                params[k] = ov

        run_dir = runs_root / f"run_{idx:04d}"
        mom_dicts, dadi_dicts = run_one(cfg, params, run_dir)

        all_true.extend([params] * len(mom_dicts))
        all_mom.extend(mom_dicts)
        all_dadi.extend(dadi_dicts)

    # ─── global scatter-plots ──────────────────────────────────────────
    inf_dir = base / "inferences"; inf_dir.mkdir(exist_ok=True)

    save_scatterplots(
        true_vecs   = all_true,
        est_vecs    = all_mom,
        ll_vec      = [d["loglik"] for d in all_mom],
        param_names = PARAM_NAMES,
        outfile     = inf_dir / "scatter_moments_vs_true.png",
        label       = "moments",
    )
    save_scatterplots(
        true_vecs   = all_true,
        est_vecs    = all_dadi,
        ll_vec      = [d["loglik"] for d in all_dadi],
        param_names = PARAM_NAMES,
        outfile     = inf_dir / "scatter_dadi_vs_true.png",
        label       = "dadi",
    )


if __name__ == "__main__":
    main()

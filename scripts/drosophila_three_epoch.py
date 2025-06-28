#!/usr/bin/env python3
"""
Revised **drosophila_three_epoch.py**
------------------------------------
* Uniform‑prior sampling (see config key ``priors``) with optional CLI pins.
* Runs any number of replicates (``num_draws``) and saves each in
  ``drosophila_three_epoch/runs/run_XXXX``.
* Performs moments **and** dadi inference; stores fits and draws scatter
  summaries of estimated vs true parameters.
* Shows a tqdm progress bar.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import demesdraw
import moments
from tqdm import tqdm

# ---------------------------------------------------------------------------
# project imports (add ../src to path)
# ---------------------------------------------------------------------------

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from simulation import drosophila_three_epoch, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference import fit_model as dadi_fit_model

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

PARAM_NAMES: List[str] = [
    "N0",
    "AFR",
    "EUR_bottleneck",
    "EUR_recover",
    "T_AFR_expansion",
    "T_AFR_EUR_split",
    "T_EUR_expansion",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sample_from_priors(priors: Dict[str, List[float]], rng: np.random.Generator) -> Dict[str, float]:
    """Sample one parameter set uniformly from [low, high] for each key."""
    res: Dict[str, float] = {}
    for k, (low, high) in priors.items():
        v: float | int = rng.uniform(low, high)
        if isinstance(low, int) and isinstance(high, int):
            v = int(round(v))
        res[k] = v
    return res


# ---------------------------------------------------------------------------
# per‑replicate pipeline
# ---------------------------------------------------------------------------

def run_one(cfg: Dict[str, Any], params: Dict[str, float], out_dir: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"; data_dir.mkdir(exist_ok=True)
    mom_dir  = out_dir / "inferences" / "moments"; mom_dir.mkdir(parents=True, exist_ok=True)
    dadi_dir = out_dir / "inferences" / "dadi";    dadi_dir.mkdir(parents=True, exist_ok=True)

    # Demography figure ------------------------------------------------
    g = drosophila_three_epoch(params)
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population Size")
    plt.savefig(data_dir / "demes_drosophila_three_epoch.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # Simulate ---------------------------------------------------------
    ts, g = simulation(params, model_type="drosophila_three_epoch", experiment_config=cfg)
    SFS = create_SFS(ts)

    with open(data_dir / "sampled_params.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(data_dir / "SFS.pkl", "wb") as f:
        pickle.dump(SFS, f)
    ts.dump(data_dir / "tree_sequence.trees")

    # Start guess ------------------------------------------------------
    start = [params[p] for p in PARAM_NAMES]
    start = moments.Misc.perturb_params(start, fold=0.1)

    # Inference --------------------------------------------------------
    fit_mom = moments_fit_model(SFS, start=start, g=g, experiment_config=cfg)
    fit_dadi = dadi_fit_model(   SFS, start=start, g=g, experiment_config=cfg)

    fit_mom = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_mom]
    fit_dadi = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_dadi]

    with open(mom_dir / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_mom, f)
    with open(dadi_dir / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_dadi, f)

    return params, fit_mom[0], fit_dadi[0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drosophila three‑epoch simulations from priors")

    parser.add_argument("--experiment_config", required=True, type=Path,
                        help="Path to experiment config JSON file")
    parser.add_argument("--num_draws", type=int, default=None,
                        help="Override number of parameter draws")

    for p in PARAM_NAMES:
        parser.add_argument(f"--{p}", type=float, default=None, help=f"Fix {p} to this value")

    args = parser.parse_args()

    # Load + RNG -------------------------------------------------------
    with args.experiment_config.open() as f:
        cfg = json.load(f)

    priors = cfg["priors"]
    n_draw = args.num_draws or cfg.get("num_draws", 1)
    rng = np.random.default_rng(cfg.get("seed"))

    draws: List[Dict[str, float]] = []
    for _ in range(n_draw):
        d = sample_from_priors(priors, rng)
        for k in PARAM_NAMES:
            ov = getattr(args, k)
            if ov is not None:
                d[k] = int(ov) if isinstance(priors[k][0], int) else float(ov)
        draws.append(d)

    runs_root = Path("drosophila_three_epoch/runs"); runs_root.mkdir(parents=True, exist_ok=True)

    true_all: List[Dict[str, float]] = []
    mom_all:  List[Dict[str, float]] = []
    dadi_all: List[Dict[str, float]] = []

    for idx, ps in enumerate(tqdm(draws, total=n_draw, desc="replicates"), 1):
        run_dir = runs_root / f"run_{idx:04d}"
        tqdm.write(f"▶ replicate {idx}/{n_draw} → {run_dir}")
        t, m, d = run_one(cfg, ps, run_dir)
        true_all.append(t); mom_all.append(m); dadi_all.append(d)

    # summary plots ----------------------------------------------------
    def scatter(est: List[Dict[str, float]], label: str, out: Path):
        fig, axs = plt.subplots(1, len(PARAM_NAMES), figsize=(3 * len(PARAM_NAMES), 3))
        if len(PARAM_NAMES) == 1:
            axs = [axs]
        for i, p in enumerate(PARAM_NAMES):
            ax = axs[i]
            x = [d[p] for d in true_all]
            y = [d[p] for d in est]
            ax.scatter(x, y, s=15)
            ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.8, color="grey")
            ax.set_xlabel(f"true {p}")
            ax.set_ylabel(f"{label} {p}")
        fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)

    scatter(mom_all, "moments", runs_root / "scatter_moments_vs_true.png")
    scatter(dadi_all, "dadi",    runs_root / "scatter_dadi_vs_true.png")


if __name__ == "__main__":
    main()

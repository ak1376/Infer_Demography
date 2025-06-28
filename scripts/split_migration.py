#!/usr/bin/env python3
"""
Fully revised **split_migration.py**
-----------------------------------
* Uniform‑prior sampling driven by the experiment‑config JSON (key `priors`).
* CLI flags let you pin any parameter to a fixed value.
* Each replicate writes outputs under `<demographic_model>/runs/run_XXXX`.
* Moments and dadi inferences are stored; a scatter summary of estimated vs
  generative parameters is saved in the runs root.
* Progress bar with **tqdm**.
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

# -----------------------------------------------------------------------------
# local project imports (src directory)
# -----------------------------------------------------------------------------

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from simulation import split_migration_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference import fit_model as dadi_fit_model

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------

PARAM_NAMES: List[str] = ["N0", "N1", "N2", "m12", "m21", "t_split"]

# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------

def sample_from_priors(priors: Dict[str, List[float]], rng: np.random.Generator) -> Dict[str, float]:
    """Draw one parameter set from uniform bounds specified in *priors*."""
    draw: Dict[str, float] = {}
    for name, (low, high) in priors.items():
        val: float | int = rng.uniform(low, high)
        if isinstance(low, int) and isinstance(high, int):
            val = int(round(val))
        draw[name] = val
    return draw


# -----------------------------------------------------------------------------
# core per‑replicate pipeline
# -----------------------------------------------------------------------------

def run_one(cfg: Dict[str, Any], params: Dict[str, float], out_dir: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Run simulation + inference for a single parameter set.

    Returns (true_params, best_moments, best_dadi) for summary plotting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"; data_dir.mkdir(exist_ok=True)
    mom_dir  = out_dir / "inferences" / "moments"; mom_dir.mkdir(parents=True, exist_ok=True)
    dadi_dir = out_dir / "inferences" / "dadi";    dadi_dir.mkdir(parents=True, exist_ok=True)

    # Demography graphic ------------------------------------------------
    g = split_migration_model(params)
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population Size")
    plt.savefig(data_dir / "demes_split_migration_model.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # Simulation + SFS --------------------------------------------------
    ts, g = simulation(params, model_type="split_migration", experiment_config=cfg)
    SFS = create_SFS(ts)

    with open(data_dir / "sampled_params.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(data_dir / "SFS.pkl", "wb") as f:
        pickle.dump(SFS, f)
    ts.dump(data_dir / "tree_sequence.trees")

    # Start guess -------------------------------------------------------
    start = [params[p] for p in PARAM_NAMES]
    start = moments.Misc.perturb_params(start, fold=0.1)

    # Inference ---------------------------------------------------------
    fit_mom = moments_fit_model(SFS, start=start, g=g, experiment_config=cfg)
    fit_dadi = dadi_fit_model(   SFS, start=start, g=g, experiment_config=cfg)

    fit_mom = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_mom]
    fit_dadi = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_dadi]

    with open(mom_dir / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_mom, f)
    with open(dadi_dir / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_dadi, f)

    return params, fit_mom[0], fit_dadi[0]


# -----------------------------------------------------------------------------
# main driver
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run split‑migration simulations from priors and summarise fits")

    parser.add_argument("--experiment_config", required=True, type=Path,
                        help="Path to experiment config JSON file")
    parser.add_argument("--num_draws", type=int, default=None,
                        help="Override number of parameter draws")

    # per‑parameter CLI overrides
    for p in PARAM_NAMES:
        parser.add_argument(f"--{p}", type=float, default=None, help=f"Fix {p} to this value")

    args = parser.parse_args()

    # Load experiment config -------------------------------------------
    cfg_path: Path = args.experiment_config
    cfg: Dict[str, Any]
    with cfg_path.open() as f:
        cfg = json.load(f)

    priors = cfg["priors"]
    num_draws = args.num_draws or cfg.get("num_draws", 1)
    rng = np.random.default_rng(cfg.get("seed"))

    # Assemble draw list -----------------------------------------------
    draws: List[Dict[str, float]] = []
    for _ in range(num_draws):
        d = sample_from_priors(priors, rng)
        for n in PARAM_NAMES:
            ov = getattr(args, n)
            if ov is not None:
                d[n] = int(ov) if isinstance(priors[n][0], int) else float(ov)
        draws.append(d)

    # Prepare run directory --------------------------------------------
    runs_root = Path(f"{cfg['demographic_model']}/runs"); runs_root.mkdir(parents=True, exist_ok=True)

    true_all:  List[Dict[str, float]] = []
    mom_all:   List[Dict[str, float]] = []
    dadi_all:  List[Dict[str, float]] = []

    # Execute replicates ------------------------------------------------
    for idx, params in enumerate(tqdm(draws, total=num_draws, desc="replicates"), 1):
        run_dir = runs_root / f"run_{idx:04d}"
        tqdm.write(f"▶ replicate {idx}/{num_draws} → {run_dir}")
        t, m, d = run_one(cfg, params, run_dir)
        true_all.append(t); mom_all.append(m); dadi_all.append(d)

    # Scatter summary plots --------------------------------------------
    def scatter(estimates: List[Dict[str, float]], label: str, outfile: Path):
        fig, axs = plt.subplots(1, len(PARAM_NAMES), figsize=(3 * len(PARAM_NAMES), 3))
        if len(PARAM_NAMES) == 1:
            axs = [axs]
        for i, p in enumerate(PARAM_NAMES):
            ax = axs[i]
            x = [d[p] for d in true_all]
            y = [d[p] for d in estimates]
            ax.scatter(x, y, s=15)
            ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.8, color="grey")
            ax.set_xlabel(f"true {p}")
            ax.set_ylabel(f"{label} {p}")
        fig.tight_layout()
        fig.savefig(outfile, dpi=300)
        plt.close(fig)

    scatter(mom_all, "moments", runs_root / "scatter_moments_vs_true.png")
    scatter(dadi_all, "dadi",    runs_root / "scatter_dadi_vs_true.png")


if __name__ == "__main__":
    main()

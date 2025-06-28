#!/usr/bin/env python3
"""
Fully revised split_isolation.py
--------------------------------
* Draws any number of parameter sets from uniform priors that live in the
  experiment_config JSON (under the key ``priors``).
* CLI flags let you **pin** any individual parameter, overriding the prior.
* Each replicate's outputs are written under ``runs/run_XXXX`` so files don't
  collide.
* Keeps your original simulation + inference pipeline intact.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import demesdraw
import moments
from tqdm import tqdm

# -----------------------------------------------------------------------------
# local imports (project src directory)
# -----------------------------------------------------------------------------

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from simulation import split_isolation_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference import fit_model as dadi_fit_model

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

PARAM_NAMES = ["N0", "N1", "N2", "m", "t_split"]


def sample_from_priors(priors: Dict[str, List[float]], rng: np.random.Generator) -> Dict[str, float]:
    """Sample one parameter set from uniform [low, high] bounds.

    Int‑like bounds (both endpoints are ints) return an int.
    """
    draw: Dict[str, float] = {}
    for name, (low, high) in priors.items():
        val = rng.uniform(low, high)
        if isinstance(low, int) and isinstance(high, int):
            val = int(round(val))
        draw[name] = val
    return draw


# -----------------------------------------------------------------------------
# core pipeline (mostly your original code, just parameterised on output dir)
# -----------------------------------------------------------------------------


def run_one(experiment_config: Dict[str, Any], params: Dict[str, float], out_dir: Path) -> None:
    """Simulate + infer for one parameter set and write results inside *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    inf_dir_mom = out_dir / "inferences" / "moments"
    inf_dir_dadi = out_dir / "inferences" / "dadi"
    inf_dir_mom.mkdir(parents=True, exist_ok=True)
    inf_dir_dadi.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. demography + figure
    # ------------------------------------------------------------------
    g = split_isolation_model(params)
    ax = demesdraw.tubes(g)
    ax.set_title("Split Isolation Model")
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population Size")
    plt.savefig(data_dir / "demes_split_isolation_model.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # ------------------------------------------------------------------
    # 2. simulation + SFS
    # ------------------------------------------------------------------
    ts, g = simulation(params, model_type="split_isolation", experiment_config=experiment_config)
    SFS = create_SFS(ts)

    # persist raw outputs
    with open(data_dir / "split_isolation_sampled_params.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(data_dir / "split_isolation_SFS.pkl", "wb") as f:
        pickle.dump(SFS, f)
    ts.dump(data_dir / "split_isolation_tree_sequence.trees")

    # ------------------------------------------------------------------
    # 3. optimisation start point + perturbation
    # ------------------------------------------------------------------
    # start = [params[n] for n in PARAM_NAMES]
    # start = moments.Misc.perturb_params(start, fold=0.1)

    # start: List[float] = []
    # for p in PARAM_NAMES:
    #     low, high = experiment_config["priors"][p]
    #     mid: float | int = (low + high) / 2.0
    #     if isinstance(low, int) and isinstance(high, int):
    #         mid = int(mid)
    #     start.append(mid)

    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], fold=0.1)

    # start = moments.Misc.perturb_params(start, fold=0.1)

    # ------------------------------------------------------------------
    # 4. inference (moments + dadi)
    # ------------------------------------------------------------------
    fit_moments = moments_fit_model(SFS, start=None, g=g, experiment_config=experiment_config)
    fit_dadi    = dadi_fit_model(SFS, start=None, g=g, experiment_config=experiment_config)

    # list[dict] representation
    fit_moments = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_moments]
    fit_dadi    = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_dadi]

    with open(inf_dir_mom / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_moments, f)
    with open(inf_dir_dadi / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_dadi, f)

    # Use the *first* entry as "best" for scatter summary
    return params, fit_moments[0], fit_dadi[0]


# -----------------------------------------------------------------------------
# main driver
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run split‑isolation simulations from priors and summarise fits.")

    parser.add_argument("--experiment_config", required=True, type=Path,
                        help="Path to experiment config JSON file")
    parser.add_argument("--num_draws", type=int, default=None,
                        help="Override number of parameter sets to draw")

    for p in PARAM_NAMES:
        parser.add_argument(f"--{p}", type=float, default=None, help=f"Fix {p} to this value")

    args = parser.parse_args()

    # Load config ------------------------------------------------------
    with args.experiment_config.open() as f:
        cfg = json.load(f)

    priors = cfg["priors"]
    num_draws = args.num_draws or cfg.get("num_draws", 1)
    rng = np.random.default_rng(cfg.get("seed"))

    # Build draw list --------------------------------------------------
    draws = []
    for _ in range(num_draws):
        d = sample_from_priors(priors, rng)
        for n in PARAM_NAMES:
            override = getattr(args, n)
            if override is not None:
                d[n] = int(override) if isinstance(priors[n][0], int) else float(override)
        draws.append(d)

    # Run replicates ---------------------------------------------------
    runs_root = Path(f"{cfg['demographic_model']}/runs"); runs_root.mkdir(parents=True, exist_ok=True)

    true_all, mom_all, dadi_all = [], [], []

    for idx, pset in enumerate(tqdm(draws, total=num_draws, desc="replicates"), 1):
        run_dir = runs_root / f"run_{idx:04d}"
        print(f"▶ replicate {idx}/{num_draws} → {run_dir}")
        true_p, mom_p, dadi_p = run_one(cfg, pset, run_dir)
        true_all.append(true_p); mom_all.append(mom_p); dadi_all.append(dadi_p)

    # ------------------------------------------------------------------
    # Scatterplot summary (true vs estimated) --------------------------
    # ------------------------------------------------------------------
    def scatter_summary(estimates: List[Dict[str, float]], label: str, colour: str, outfile: Path):
        fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(3 * len(PARAM_NAMES), 3), squeeze=False)
        for i, param in enumerate(PARAM_NAMES):
            ax = axes[0, i]
            x = [d[param] for d in true_all]
            y = [d[param] for d in estimates]
            ax.scatter(x, y, marker="o")
            ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.8, color="grey")
            ax.set_xlabel(f"true {param}")
            ax.set_ylabel(f"{label} {param}")
        fig.tight_layout()
        fig.savefig(outfile, dpi=300)
        plt.close(fig)

    scatter_summary(mom_all, "moments", "C0", runs_root / "scatter_moments_vs_true.png")
    scatter_summary(dadi_all, "dadi",    "C1", runs_root / "scatter_dadi_vs_true.png")



if __name__ == "__main__":
    main()

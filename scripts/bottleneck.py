#!/usr/bin/env python3
"""
Revised **bottleneck.py** (uniform‑prior driver)
------------------------------------------------
* Uniform prior sampling driven by the JSON `priors` block.
* Optional CLI pins for any parameter.
* Any number of replicates (`num_draws`) – each saved in
  `bottleneck/runs/run_XXXX`.
* Runs **moments** and **dadi** optimisation, stores fits, and produces
  scatter plots (true vs estimated) for quick QC.
* tqdm progress bar with `tqdm.write` for clean interleaved logging.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import demesdraw
import moments
from tqdm import tqdm

# ---------------------------------------------------------------------------
# local project imports
# ---------------------------------------------------------------------------

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from simulation import bottleneck_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference   import fit_model as dadi_fit_model

# ---------------------------------------------------------------------------
# constants – keep parameter order consistent everywhere
# ---------------------------------------------------------------------------

PARAM_NAMES: List[str] = [
    "N0",
    "N_bottleneck",
    "N_recover",
    "t_bottleneck_start",
    "t_bottleneck_end",
]

# ---------------------------------------------------------------------------
# helper : draw one parameter set from uniform priors
# ---------------------------------------------------------------------------

def sample_from_priors(priors: Dict[str, List[float]], rng: np.random.Generator) -> Dict[str, float]:
    draw: Dict[str, float] = {}
    for name, (low, high) in priors.items():
        val: float | int = rng.uniform(low, high)
        if isinstance(low, int) and isinstance(high, int):
            val = int(round(val))
        draw[name] = val
    return draw

# ---------------------------------------------------------------------------
# pipeline for **one** replicate – returns (true, moments_best, dadi_best)
# ---------------------------------------------------------------------------

def run_one(cfg: Dict[str, Any], params: Dict[str, float], out_dir: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data";   data_dir.mkdir(exist_ok=True)
    mom_dir  = out_dir / "inferences" / "moments"; mom_dir.mkdir(parents=True, exist_ok=True)
    dadi_dir = out_dir / "inferences" / "dadi";    dadi_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. demography -> figure
    # ------------------------------------------------------------------
    g = bottleneck_model(params)
    if hasattr(g, "sort_events"):
        g.sort_events()                 # ensure chronological order

    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population size")
    plt.savefig(data_dir / "demes_bottleneck_model.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # ------------------------------------------------------------------
    # 2. coalescent simulation -> SFS
    # ------------------------------------------------------------------
    ts, g = simulation(params, model_type="bottleneck", experiment_config=cfg)
    SFS   = create_SFS(ts)

    with open(data_dir / "sampled_params.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(data_dir / "SFS.pkl", "wb") as f:
        pickle.dump(SFS, f)
    ts.dump(data_dir / "tree_sequence.trees")

    # ------------------------------------------------------------------
    # 3. optimisation start point (perturbed true values)
    # ------------------------------------------------------------------

    start: List[float] = []
    for p in PARAM_NAMES:
        low, high = cfg["priors"][p]
        mid: float | int = (low + high) / 2.0
        if isinstance(low, int) and isinstance(high, int):
            mid = int(mid)
        start.append(mid)

    # start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], fold=0.1)

    start = moments.Misc.perturb_params(start, fold=0.1)

    # ------------------------------------------------------------------
    # 4. inference
    # ------------------------------------------------------------------
    fit_mom  = moments_fit_model(SFS, start=start, g=g, experiment_config=cfg, sampled_params=params)
    fit_dadi = dadi_fit_model(   SFS, start=start, g=g, experiment_config=cfg, sampled_params=params)

    fit_mom  = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_mom]
    fit_dadi = [dict(zip(PARAM_NAMES, arr.tolist())) for arr in fit_dadi]

    with open(mom_dir / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_mom, f)
    with open(dadi_dir / "fit_params.pkl", "wb") as f:
        pickle.dump(fit_dadi, f)

    # use the *first* entry as the best fit for summary plots
    return params, fit_mom[0], fit_dadi[0]

# ---------------------------------------------------------------------------
# main driver – draws, replicates, summary plots
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run bottleneck simulations from priors and summarise fits")

    parser.add_argument("--experiment_config", required=True, type=Path,
                        help="Path to experiment config JSON file")
    parser.add_argument("--num_draws", type=int, default=None,
                        help="Override number of parameter draws (default: config value)")

    # per‑parameter pins (CLI overrides)
    for p in PARAM_NAMES:
        parser.add_argument(f"--{p}", type=float, default=None, help=f"Pin {p} to this value")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # config + RNG ------------------------------------------------------
    # ------------------------------------------------------------------
    with args.experiment_config.open() as f:
        cfg = json.load(f)

    priors   = cfg["priors"]
    n_draw   = args.num_draws or cfg.get("num_draws", 1)
    rng      = np.random.default_rng(cfg.get("seed"))

    # ------------------------------------------------------------------
    # draw parameter sets ----------------------------------------------
    # ------------------------------------------------------------------
    draws: List[Dict[str, float]] = []
    while len(draws) < n_draw:
        d = sample_from_priors(priors, rng)
        # ensure start > end for bottleneck timing
        if d["t_bottleneck_start"] <= d["t_bottleneck_end"]:
            d["t_bottleneck_start"], d["t_bottleneck_end"] = (
                d["t_bottleneck_end"], d["t_bottleneck_start"],
            )
        # apply CLI overrides
        for k in PARAM_NAMES:
            override = getattr(args, k)
            if override is not None:
                d[k] = int(override) if isinstance(priors[k][0], int) else float(override)
        draws.append(d)

    # ------------------------------------------------------------------
    # run replicates ----------------------------------------------------
    # ------------------------------------------------------------------
    runs_root = Path("bottleneck/runs"); runs_root.mkdir(parents=True, exist_ok=True)

    all_true, all_mom, all_dadi = [], [], []

    for idx, pset in enumerate(tqdm(draws, total=n_draw, desc="replicates"), 1):
        run_dir = runs_root / f"run_{idx:04d}"
        tqdm.write(f"▶ replicate {idx}/{n_draw} → {run_dir}")
        t, m, d = run_one(cfg, pset, run_dir)
        all_true.append(t); all_mom.append(m); all_dadi.append(d)

    # ------------------------------------------------------------------
    # scatter‑plot helper
    # ------------------------------------------------------------------
    def scatter(est: List[Dict[str, float]], label: str, outfile: Path) -> None:
        fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(3 * len(PARAM_NAMES), 3))
        if len(PARAM_NAMES) == 1:
            axes = [axes]
        for i, p in enumerate(PARAM_NAMES):
            ax = axes[i]
            x = [d[p] for d in all_true]
            y = [d[p] for d in est]
            ax.scatter(x, y, s=15)
            ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.8, color="grey")
            ax.set_xlabel(f"true {p}")
            ax.set_ylabel(f"{label} {p}")
        fig.tight_layout(); fig.savefig(outfile, dpi=300); plt.close(fig)

    scatter(all_mom,  "moments", runs_root / "scatter_moments_vs_true.png")
    scatter(all_dadi, "dadi",    runs_root / "scatter_dadi_vs_true.png")


if __name__ == "__main__":
    main()

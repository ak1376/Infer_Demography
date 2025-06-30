#!/usr/bin/env python3
"""
bottleneck.py – directory-aware, single-run version
---------------------------------------------------
All outputs are written beneath a folder whose name is taken from
`experiment_config["demographic_model"]`, so when that field is
"bottleneck" the tree looks like:

    bottleneck/
        ├── data/
        │   ├── demes_bottleneck_model.png
        │   ├── sampled_params.pkl
        │   ├── bottleneck_SFS.pkl
        │   └── bottleneck_tree_sequence.trees
        └── inferences/
            ├── moments/bottleneck_fit_params.pkl
            └── dadi/bottleneck_fit_params.pkl

The script keeps your original simulation + inference logic; only the file
layout is changed.  There is **no** prior sampling — you run a single
replicate per invocation and can override any parameter with CLI flags.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import demesdraw
import moments

# ────────────────────────────────────────────────────────────────────────────
# local project imports
# ────────────────────────────────────────────────────────────────────────────
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_PATH))

from simulation import bottleneck_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference import fit_model as dadi_fit_model

# ────────────────────────────────────────────────────────────────────────────
# constants
# ────────────────────────────────────────────────────────────────────────────
PARAM_NAMES: List[str] = [
    "N0",
    "N_bottleneck",
    "N_recover",
    "t_bottleneck_start",
    "t_bottleneck_end",
]

# ────────────────────────────────────────────────────────────────────────────
# pipeline for **one** replicate
# ────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# pipeline for **one** replicate – returns (true, moments_best, dadi_best)
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Dict[str, Any], params: Dict[str, float]) -> None:
    """Simulate demographic model, perform inference, and save artefacts."""

    mdl_name = cfg["demographic_model"]  # expected to be "bottleneck"

    # 1. directory scaffold ────────────────────────────────────────────────
    base = Path(mdl_name)  # bottleneck/
    data_dir = base / "data"
    mom_dir = base / "inferences" / "moments"
    dadi_dir = base / "inferences" / "dadi"
    for d in (data_dir, mom_dir, dadi_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. demography figure ────────────────────────────────────────────────
    g = bottleneck_model(params)
    if hasattr(g, "sort_events"):
        g.sort_events()
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population size")
    plt.savefig(data_dir / "demes_bottleneck_model.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # 3. simulation + SFS ─────────────────────────────────────────────────
    ts, g_sim = simulation(params, model_type="bottleneck", experiment_config=cfg)
    sfs = create_SFS(ts)

    (data_dir / "sampled_params.pkl").write_bytes(pickle.dumps(params))
    (data_dir / "bottleneck_SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts.dump(data_dir / "bottleneck_tree_sequence.trees")

    # 4. optimiser starting point (perturb true parameters) ──────────────
    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], fold=0.1)

    # 5. inference ────────────────────────────────────────────────────────
    fit_mom = moments_fit_model(sfs, start=start, g=g_sim, experiment_config=cfg, sampled_params=params)
    fit_dadi = dadi_fit_model(sfs, start=start, g=g_sim, experiment_config=cfg, sampled_params=params)

    # store fits as list[dict] for consistency with the rest of the repo
    pickle.dump([dict(zip(PARAM_NAMES, v.tolist())) for v in fit_mom],
                open(mom_dir / f"{mdl_name}_fit_params.pkl", "wb"))
    pickle.dump([dict(zip(PARAM_NAMES, v.tolist())) for v in fit_dadi],
                open(dadi_dir / f"{mdl_name}_fit_params.pkl", "wb"))


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Bottleneck simulation & inference (directory-aware)")
    p.add_argument("--experiment_config", required=True, type=Path,
                   help="Path to JSON experiment-config file")

    # optional CLI overrides ------------------------------------------------
    p.add_argument("--N0", type=int, default=None)
    p.add_argument("--N_bottleneck", type=int, default=None)
    p.add_argument("--N_recover", type=int, default=None)
    p.add_argument("--t_bottleneck_start", type=float, default=None)
    p.add_argument("--t_bottleneck_end", type=float, default=None)

    args = p.parse_args()

    # load config -----------------------------------------------------------
    cfg = json.loads(args.experiment_config.read_text())

    # build parameter dict (defaults match original script) -----------------
    params = {
        "N0": args.N0 or 1.0e4,
        "N_bottleneck": args.N_bottleneck or 2.0e3,
        "N_recover": args.N_recover or 5.0e3,
        "t_bottleneck_start": args.t_bottleneck_start or 300,
        "t_bottleneck_end": args.t_bottleneck_end or 100,
    }

    run_pipeline(cfg, params)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
split_isolation.py  – directory-aware version
---------------------------------------------
Each run now writes to:

    <demographic_model>/data/…
    <demographic_model>/inferences/moments/…
    <demographic_model>/inferences/dadi/…

`demographic_model` comes from the loaded experiment-config JSON.
"""

from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import demesdraw, moments
import numpy as np

# ------------------------------------------------------------------
# local imports -----------------------------------------------------
# ------------------------------------------------------------------
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from simulation import split_isolation_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference    import fit_model as dadi_fit_model

# ------------------------------------------------------------------
# helper constants --------------------------------------------------
# ------------------------------------------------------------------
PARAM_NAMES = ["N0", "N1", "N2", "m", "t_split"]


# ------------------------------------------------------------------
# main workflow -----------------------------------------------------
# ------------------------------------------------------------------
def run_pipeline(cfg: Dict[str, Any], params: Dict[str, float]) -> None:
    mdl_name = cfg["demographic_model"]

    # directory scaffold -------------------------------------------------
    base    = Path(mdl_name)                # e.g.  split_isolation/
    data_dir = base / "data"                #       split_isolation/data/
    mom_dir  = base / "inferences" / "moments"
    dadi_dir = base / "inferences" / "dadi"

    for d in (data_dir, mom_dir, dadi_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. build demography + plot ----------------------------------------
    g = split_isolation_model(params)
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population size")
    plt.savefig(data_dir / "demes_split_isolation_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # 2. simulate + SFS --------------------------------------------------
    ts, g2 = simulation(params, model_type="split_isolation", experiment_config=cfg)
    SFS    = create_SFS(ts)

    (data_dir / "sampled_params.pkl").write_bytes(pickle.dumps(params))
    (data_dir / "split_isolation_SFS.pkl").write_bytes(pickle.dumps(SFS))
    ts.dump(data_dir / "split_isolation_tree_sequence.trees")

    # 3. initial guess  --------------------------------------------------
    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], fold=0.1)

    # 4. inference -------------------------------------------------------
    fit_mom  = moments_fit_model(SFS, start=start, g=g2, experiment_config=cfg)
    fit_dadi = dadi_fit_model(   SFS, start=start, g=g2, experiment_config=cfg)

    # save fits
    pickle.dump([dict(zip(PARAM_NAMES, p.tolist())) for p in fit_mom],
                open(mom_dir  / f"{mdl_name}_fit_params.pkl", "wb"))
    pickle.dump([dict(zip(PARAM_NAMES, p.tolist())) for p in fit_dadi],
                open(dadi_dir / f"{mdl_name}_fit_params.pkl", "wb"))


# ------------------------------------------------------------------
# CLI entry-point ---------------------------------------------------
# ------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Split-isolation simulation & inference")
    p.add_argument("--experiment_config", required=True, type=Path,
                   help="Path to JSON exp-config file")

    # optional manual overrides ----------------------------------------
    p.add_argument("--N0", type=int,   default=None)
    p.add_argument("--N1", type=int,   default=None)
    p.add_argument("--N2", type=int,   default=None)
    p.add_argument("--m",  type=float, default=None)
    p.add_argument("--t_split", type=float, default=None)

    args = p.parse_args()

    # load config & build param dict ------------------------------------
    cfg = json.loads(args.experiment_config.read_text())

    params = {
        "N0": args.N0 or 1.0e4,
        "N1": args.N1 or 5.0e3,
        "N2": args.N2 or 5.0e3,
        "m":  args.m  or 1e-6,
        "t_split": args.t_split or 1.0e4
    }

    run_pipeline(cfg, params)


if __name__ == "__main__":
    main()

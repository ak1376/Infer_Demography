#!/usr/bin/env python3
"""
split_migration.py – directory-aware, single-run version
--------------------------------------------------------
Writes everything under a folder named after
`experiment_config["demographic_model"]`, e.g. `split_migration/`:

    split_migration/
        ├─ data/
        │   ├─ demes_split_migration_model.png
        │   ├─ sampled_params.pkl
        │   ├─ split_migration_SFS.pkl
        │   └─ split_migration_tree_sequence.trees
        └─ inferences/
            ├─ moments/split_migration_fit_params.pkl
            └─ dadi/split_migration_fit_params.pkl
"""

from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import demesdraw, moments

# ----------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_PATH))

from simulation import split_migration_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference    import fit_model as dadi_fit_model

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
PARAM_NAMES: List[str] = ["N0", "N1", "N2", "m12", "m21", "t_split"]

# ----------------------------------------------------------------------
# single-run pipeline
# ----------------------------------------------------------------------
def run_pipeline(cfg: Dict[str, Any], params: Dict[str, float]) -> None:
    mdl_name = cfg["demographic_model"]          # "split_migration"

    # ........ directory scaffold ......................................
    base     = Path(mdl_name)                    # split_migration/
    data_dir = base / "data"                     # split_migration/data/
    mom_dir  = base / "inferences" / "moments"
    dadi_dir = base / "inferences" / "dadi"
    for d in (data_dir, mom_dir, dadi_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ........ demography + figure .....................................
    g = split_migration_model(params)
    if hasattr(g, "sort_events"):
        g.sort_events()
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population size")
    plt.savefig(data_dir / "demes_split_migration_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # ........ simulate + SFS ..........................................
    ts, g_sim = simulation(params, model_type="split_migration",
                           experiment_config=cfg)
    sfs = create_SFS(ts)

    (data_dir / "sampled_params.pkl").write_bytes(pickle.dumps(params))
    (data_dir / "split_migration_SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts.dump(data_dir / "split_migration_tree_sequence.trees")

    # ........ optimiser start guess ...................................
    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES],
                                        fold=0.1)

    # ........ inference ...............................................
    fit_mom  = moments_fit_model(sfs, start=start, g=g_sim,
                                 experiment_config=cfg)
    fit_dadi = dadi_fit_model(   sfs, start=start, g=g_sim,
                                 experiment_config=cfg)

    pickle.dump([dict(zip(PARAM_NAMES, v.tolist())) for v in fit_mom],
                open(mom_dir / f"{mdl_name}_fit_params.pkl", "wb"))
    pickle.dump([dict(zip(PARAM_NAMES, v.tolist())) for v in fit_dadi],
                open(dadi_dir / f"{mdl_name}_fit_params.pkl", "wb"))

# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Split-migration simulation & inference")
    p.add_argument("--experiment_config", required=True, type=Path,
                   help="Path to JSON experiment-config file")

    # optional CLI overrides
    p.add_argument("--N0",      type=int,   default=None)
    p.add_argument("--N1",      type=int,   default=None)
    p.add_argument("--N2",      type=int,   default=None)
    p.add_argument("--m12",     type=float, default=None)
    p.add_argument("--m21",     type=float, default=None)
    p.add_argument("--t_split", type=float, default=None)

    args = p.parse_args()

    # ........ load config & build param dict ..........................
    cfg = json.loads(args.experiment_config.read_text())

    params = {
        "N0":      args.N0      or 1.0e4,
        "N1":      args.N1      or 5.0e3,
        "N2":      args.N2      or 5.0e3,
        "m12":     args.m12     or 1e-3,
        "m21":     args.m21     or 1e-3,
        "t_split": args.t_split or 1.0e4,
    }

    run_pipeline(cfg, params)


if __name__ == "__main__":
    main()

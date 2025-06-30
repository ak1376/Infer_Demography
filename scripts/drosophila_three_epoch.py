#!/usr/bin/env python3
"""
drosophila_three_epoch.py  – directory-aware version
----------------------------------------------------
Outputs:

    drosophila_three_epoch/
        ├─ data/
        │   ├─ demes_drosophila_three_epoch.png
        │   ├─ sampled_params.pkl
        │   ├─ drosophila_three_epoch_SFS.pkl
        │   └─ drosophila_three_epoch_tree_sequence.trees
        └─ inferences/
            ├─ moments/drosophila_three_epoch_fit_params.pkl
            └─ dadi/drosophila_three_epoch_fit_params.pkl
"""

from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import demesdraw, moments
from tqdm import tqdm  # handy if you batch calls later

# ----------------------------------------------------------------------
#  local imports
# ----------------------------------------------------------------------
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_PATH))

from simulation        import drosophila_three_epoch, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model
from dadi_inference    import fit_model as dadi_fit_model

# ----------------------------------------------------------------------
#  constants
# ----------------------------------------------------------------------
PARAM_NAMES: List[str] = [
    "N0",
    "AFR",
    "EUR_bottleneck",
    "EUR_recover",
    "T_AFR_expansion",
    "T_AFR_EUR_split",
    "T_EUR_expansion",
]

# ----------------------------------------------------------------------
#  single-run pipeline
# ----------------------------------------------------------------------
def run_pipeline(cfg: Dict[str, Any], params: Dict[str, float]) -> None:
    mdl_name = cfg["demographic_model"]                       # "drosophila_three_epoch"

    # ── directory scaffold ────────────────────────────────────────────
    base     = Path(mdl_name)
    data_dir = base / "data"
    mom_dir  = base / "inferences" / "moments"
    dadi_dir = base / "inferences" / "dadi"
    for d in (data_dir, mom_dir, dadi_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── demography + figure ───────────────────────────────────────────
    g = drosophila_three_epoch(params)
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population size")
    plt.savefig(data_dir / "demes_drosophila_three_epoch.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # ── simulate + SFS ────────────────────────────────────────────────
    ts, g_sim = simulation(params,
                           model_type="drosophila_three_epoch",
                           experiment_config=cfg)
    sfs = create_SFS(ts)

    # raw artefacts
    (data_dir / "sampled_params.pkl").write_bytes(pickle.dumps(params))
    (data_dir / "drosophila_three_epoch_SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts.dump(data_dir / "drosophila_three_epoch_tree_sequence.trees")

    # ── optimiser starting point (perturbed truth) ───────────────────
    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], fold=0.1)

    # ── inference ─────────────────────────────────────────────────────
    fit_mom  = moments_fit_model(sfs, start=start, g=g_sim, experiment_config=cfg)
    fit_dadi = dadi_fit_model(   sfs, start=start, g=g_sim, experiment_config=cfg)

    # save fits
    pickle.dump([dict(zip(PARAM_NAMES, p.tolist())) for p in fit_mom],
                open(mom_dir  / f"{mdl_name}_fit_params.pkl", "wb"))
    pickle.dump([dict(zip(PARAM_NAMES, p.tolist())) for p in fit_dadi],
                open(dadi_dir / f"{mdl_name}_fit_params.pkl", "wb"))

# ----------------------------------------------------------------------
#  CLI entry-point
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Drosophila three-epoch simulation & inference")
    p.add_argument("--experiment_config", required=True, type=Path,
                   help="Path to JSON experiment-config file")

    # optional CLI overrides
    p.add_argument("--N0",              type=int,   default=10_000)
    p.add_argument("--AFR",             type=int,   default=10_000)
    p.add_argument("--EUR_bottleneck",  type=int,   default=5_000)
    p.add_argument("--EUR_recover",     type=int,   default=5_000)
    p.add_argument("--T_AFR_expansion", type=float, default=10_000)
    p.add_argument("--T_AFR_EUR_split", type=float, default=5_000)
    p.add_argument("--T_EUR_expansion", type=float, default=2_000)

    args = p.parse_args()

    cfg = json.loads(args.experiment_config.read_text())

    params = {name: getattr(args, name) for name in PARAM_NAMES}

    run_pipeline(cfg, params)


if __name__ == "__main__":
    main()

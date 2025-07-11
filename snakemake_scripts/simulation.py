#!/usr/bin/env python3
"""Standalone simulator + cache

Generates one demographic simulation (tree‑sequence + SFS) for the chosen
model and stores all artefacts under <simulation-dir>/<simulation-number>/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import demesdraw
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulation import bottleneck_model, simulation, create_SFS  # noqa: E402

# ------------------------------------------------------------------
# parameter sampling helper
# ------------------------------------------------------------------

def sample_params(priors: Dict[str, List[float]], *,
                  rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    params = {k: float(rng.uniform(*bounds)) for k, bounds in priors.items()}
    if {"t_bottleneck_start", "t_bottleneck_end"}.issubset(params) and \
       params["t_bottleneck_start"] <= params["t_bottleneck_end"]:
        params["t_bottleneck_start"], params["t_bottleneck_end"] = (
            params["t_bottleneck_end"], params["t_bottleneck_start"])
    return params

# ------------------------------------------------------------------
# main workflow
# ------------------------------------------------------------------

def run_simulation(simulation_dir: Path, experiment_config: Path, model_type: str,
                   simulation_number: Optional[str] = None):
    cfg: Dict[str, object] = json.loads(experiment_config.read_text())
    rng = np.random.default_rng(cfg.get("seed"))

    # decide destination folder name
    if simulation_number is None:
        existing = {int(p.name) for p in simulation_dir.glob("[0-9]*") if p.is_dir()}
        simulation_number = f"{max(existing, default=0) + 1:04d}"
    out_dir = simulation_dir / simulation_number
    out_dir.mkdir(parents=True, exist_ok=True)

    # simulate
    sampled_params = sample_params(cfg["priors"])
    ts, _ = simulation(sampled_params, model_type, cfg)
    sfs   = create_SFS(ts)

    # save artefacts
    (out_dir / "sampled_params.pkl").write_bytes(pickle.dumps(sampled_params))
    (out_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts.dump(out_dir / "tree_sequence.trees")

    # plot demography
    ax = demesdraw.tubes(bottleneck_model(sampled_params))
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("N")
    plt.savefig(out_dir / f"demes_{cfg['demographic_model']}.png", dpi=300,
                bbox_inches="tight")
    plt.close(ax.figure)

    # friendly path for log message
    try:
        rel = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_dir
    print(f"✓ simulation written to {rel}")

# ------------------------------------------------------------------
# argparse entry‑point
# ------------------------------------------------------------------

def main():
    cli = argparse.ArgumentParser(description="Generate one demographic simulation")
    cli.add_argument("--simulation-dir", type=Path, required=True,
                     help="Base directory that will hold <number>/ subfolders")
    cli.add_argument("--experiment-config", type=Path, required=True,
                     help="JSON config with priors, genome length, etc.")
    cli.add_argument("--model-type", required=True,
                     choices=["bottleneck", "split_isolation", "split_migration",
                              "drosophila_three_epoch"],
                     help="Which demographic model to simulate")
    cli.add_argument("--simulation-number", type=str,
                     help="Folder name to create (e.g. '0005').  If omitted, the next free index is used.")
    args = cli.parse_args()
    run_simulation(args.simulation_dir, args.experiment_config,
                   args.model_type, args.simulation_number)


if __name__ == "__main__":
    main()

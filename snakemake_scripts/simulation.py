#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import run_one_simulation_to_dir


def main():
    cli = argparse.ArgumentParser(description="Generate one simulation (neutral or BGS, engine-aware)")
    cli.add_argument("--simulation-dir", type=Path, required=True)
    cli.add_argument("--experiment-config", type=Path, required=True)
    cli.add_argument(
        "--model-type",
        required=True,
        choices=[
            "bottleneck",
            "IM_symmetric",
            "split_migration",
            "drosophila_three_epoch",
            "split_migration_growth",
            "OOA_three_pop",
            "OOA_three_pop_gutenkunst",
        ],
    )
    cli.add_argument("--simulation-number", type=str, default=None)
    args = cli.parse_args()

    out_dir = run_one_simulation_to_dir(
        simulation_dir=args.simulation_dir,
        experiment_config_path=args.experiment_config,
        model_type=args.model_type,
        simulation_number=args.simulation_number,
    )
    print(f"✓ simulation written to {out_dir}")


if __name__ == "__main__":
    main()

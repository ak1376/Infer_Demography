#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import simulate_one_window_replicate


def main() -> None:
    cli = argparse.ArgumentParser("simulate one windowed replicate (neutral or BGS)")
    cli.add_argument("--sim-dir", required=True, type=Path, help="directory with sampled_params.pkl")
    cli.add_argument("--rep-index", required=True, type=int)
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--out-dir", required=True, type=Path)
    cli.add_argument(
        "--meta-file",
        type=Path,
        required=False,
        help="bgs.meta.json from base simulation (to reuse exact coverage for SLiM)",
    )
    cli.add_argument(
        "--seed-stride",
        type=int,
        default=10000,
        help="Seed stride: window_seed = base_seed + rep_index * seed_stride",
    )
    args = cli.parse_args()

    simulate_one_window_replicate(
        sim_dir=args.sim_dir,
        rep_index=args.rep_index,
        config_file=args.config_file,
        out_dir=args.out_dir,
        meta_file=args.meta_file,
        seed_stride=args.seed_stride,
    )


if __name__ == "__main__":
    main()

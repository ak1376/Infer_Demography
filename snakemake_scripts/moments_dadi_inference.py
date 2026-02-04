#!/usr/bin/env python3
# snakemake_scripts/moments_dadi_inference.py
# Thin CLI wrapper for dadi + moments inference (delegates to src/ runner)

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from sfs_inference_runner import run_cli

def _parse_args():
    p = argparse.ArgumentParser("CLI wrapper for dadi/moments inference (thin)")
    p.add_argument("--mode", choices=["dadi", "moments", "both"], required=True)

    p.add_argument(
        "--sfs-file",
        type=Path,
        required=True,
        help="Pickle of dadi.Spectrum | moments.Spectrum",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON experiment configuration file",
    )
    p.add_argument(
        "--model-py",
        type=str,
        required=True,
        help="module:function returning demes.Graph when called with a param dict",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Parent output directory. For --mode both writes into outdir/{dadi,moments}",
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional: Pickle or JSON file with ground truth simulation parameters",
    )
    p.add_argument(
        "--generate-profiles",
        action="store_true",
        help="Generate 1D likelihood profiles for each parameter",
    )
    p.add_argument(
        "--profile-grid-points",
        type=int,
        default=41,
        help="Number of grid points for likelihood profiles",
    )
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure project root and src/ are importable.
    # This file lives in snakemake_scripts/, so project root is parent of that.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    args.outdir.mkdir(parents=True, exist_ok=True)

    run_cli(
        mode=args.mode,
        sfs_file=args.sfs_file,
        config_file=args.config,
        model_py=args.model_py,
        outdir=args.outdir,
        generate_profiles=args.generate_profiles,
        profile_grid_points=args.profile_grid_points,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

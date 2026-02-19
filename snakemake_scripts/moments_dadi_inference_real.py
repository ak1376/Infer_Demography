#!/usr/bin/env python3
# snakemake_scripts/moments_dadi_inference_real.py
# Thin CLI wrapper for REAL-DATA SFS inference (theta-profiled, scaled params)
# Delegates to src/sfs_inference_runner_real.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: because we inserted SRC_DIR directly, imports are module-style
# (i.e. sfs_inference_runner_real.py is importable as "sfs_inference_runner_real")
from sfs_inference_runner_real import run_cli_real  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser("CLI wrapper for REAL-data dadi/moments inference (thin)")
    p.add_argument("--mode", choices=["dadi", "moments", "both"], required=True)

    p.add_argument(
        "--sfs-file",
        type=Path,
        required=True,
        help="Pickle of dadi.Spectrum | moments.Spectrum (or array-like underlying).",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON experiment configuration file (must include scaled priors for real data).",
    )
    p.add_argument(
        "--model-py",
        type=str,
        required=True,
        help="module:function returning demes.Graph when called with ABS param dict",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Parent output directory. Writes into outdir/{dadi,moments}/best_fit.pkl",
    )

    # Keep these args for symmetry / forwards compatibility (runner decides what to use)
    p.add_argument(
        "--generate-profiles",
        action="store_true",
        help="(moments only, optional) generate 1D profile curves if enabled in config too.",
    )
    p.add_argument(
        "--profile-grid-points",
        type=int,
        default=41,
        help="(moments only) number of grid points for likelihood profiles (passed via config if you use it).",
    )

    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure project root and src/ are importable (safe if called from anywhere)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    args.outdir.mkdir(parents=True, exist_ok=True)

    # NOTE:
    # - real-data runner currently ignores --generate-profiles CLI flag; it keys off config:
    #   experiment_config["generate_profiles"] == True and (moments) save_dir provided.
    # If you want CLI control, we can patch runner to plumb it through.
    if args.mode == "both":
        for m in ("moments", "dadi"):
            run_cli_real(
                sfs_file=args.sfs_file,
                config_file=args.config,
                model_py=args.model_py,
                outdir=args.outdir,
                mode=m,
                verbose=bool(args.verbose),
            )
    else:
        run_cli_real(
            sfs_file=args.sfs_file,
            config_file=args.config,
            model_py=args.model_py,
            outdir=args.outdir,
            mode=args.mode,
            verbose=bool(args.verbose),
        )


if __name__ == "__main__":
    main()

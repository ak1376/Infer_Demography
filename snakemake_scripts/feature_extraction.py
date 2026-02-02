#!/usr/bin/env python3
# snakemake_scripts/build_modeling_datasets.py
# Thin wrapper around src/feature_extraction_helpers.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-config", required=True, type=Path)
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Pass experiments/<model>/modeling (the script will create datasets/ inside)",
    )
    ap.add_argument("--tol-rel", type=float, default=1e-9)
    ap.add_argument("--tol-abs", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=6.0)
    ap.add_argument("--preview-rows", type=int, default=-1)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    from feature_extraction_helpers import build_modeling_datasets

    datasets_dir = build_modeling_datasets(
        experiment_config_path=args.experiment_config,
        out_root=args.out_dir,
        tol_rel=args.tol_rel,
        tol_abs=args.tol_abs,
        zmax=args.zmax,
        preview_rows=args.preview_rows,
    )

    print(f"✓ wrote datasets, plots, and metrics to: {datasets_dir}")


if __name__ == "__main__":
    main()

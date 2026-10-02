#!/usr/bin/env python3
# snakemake_scripts/momentsld_real_comparison_plots.py
#
# Reuses src/MomentsLD_inference.py's create_comparison_plot (the same
# function that produces empirical_vs_theoretical_comparison.pdf elsewhere
# in the pipeline) to compare fitted vs. observed LD statistics for one or
# more candidate parameter sets against the same real-data empirical
# means/covariances.

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.MomentsLD_inference import create_comparison_plot, load_config  # noqa: E402

DEFAULT_R_BINS = np.array(
    [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], dtype=float
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--empirical", required=True, type=Path, help="means.varcovs.pkl")
    ap.add_argument("--candidates", nargs="+", required=True,
                     help="label=path.pkl pairs, each .pkl a dict of ABSOLUTE param values")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    with open(args.empirical, "rb") as fh:
        mv = pickle.load(fh)

    for spec in args.candidates:
        label, path = spec.split("=", 1)
        with open(path, "rb") as fh:
            params = pickle.load(fh)
        out_path = args.out_dir / f"empirical_vs_theoretical_comparison_{label}.pdf"
        if out_path.exists():
            out_path.unlink()
        create_comparison_plot(cfg, params, mv, DEFAULT_R_BINS, out_path)
        print(f"[{label}] -> {out_path}")


if __name__ == "__main__":
    main()

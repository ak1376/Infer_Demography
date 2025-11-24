#!/usr/bin/env/snakemake_scripts/LD_inference.py

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# >>> make src importable <<<
ROOT = Path(__file__).resolve().parents[1]  # project root
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# project-local imports
from MomentsLD_inference import (
    load_config,
    load_sampled_params,
    aggregate_ld_statistics,
    create_comparison_plot,
    run_momentsld_inference,
    DEFAULT_R_BINS,
)


def _parse_args():
    p = argparse.ArgumentParser(
        "Aggregate LD stats, make comparison PDF, run Moments-LD optimisation"
    )
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="experiments/<MODEL>/simulations/<sid>",
    )
    p.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="experiments/<MODEL>/inferences/sim_<sid>/MomentsLD",
    )
    p.add_argument("--config-file", required=True, type=Path)
    p.add_argument(
        "--r-bins",
        type=str,
        default=None,
        help="Comma-separated r-bin edges. If omitted, uses DEFAULT_R_BINS",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    a = _parse_args()
    logging.basicConfig(
        level=logging.WARNING - 10 * min(a.verbose, 2),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(a.config_file)
    sampled_params = load_sampled_params(sim_dir, required=False)

    # r-bins to use for both comparison PDF and optimisation
    if a.r_bins:
        r_vec = np.array([float(x) for x in a.r_bins.split(",")], float)
    else:
        r_vec = DEFAULT_R_BINS

    # ensure output root exists
    a.output_root.mkdir(parents=True, exist_ok=True)

    # 1) aggregate LD pickles → means/varcovs/bootstrap_sets
    empirical_data = aggregate_ld_statistics(a.output_root)

    # 2) empirical vs theoretical PDF (skips if exists)
    plot_file = a.output_root / "empirical_vs_theoretical_comparison.pdf"
    create_comparison_plot(cfg, sampled_params, empirical_data, r_vec, plot_file)

    # 3) custom NLopt L-BFGS optimisation (skips if best_fit.pkl exists)
    run_momentsld_inference(cfg, empirical_data, a.output_root, r_vec, sampled_params)


if __name__ == "__main__":
    main()

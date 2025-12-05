#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Project root: /gpfs/projects/kernlab/akapoor/Infer_Demography
ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT: add the *project root* to sys.path, NOT src/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now src is a top-level package and we can import from it
from src.MomentsLD_inference import (
    run_momentsld_inference,
    aggregate_ld_statistics,
    load_sampled_params,
    load_config,
    DEFAULT_R_BINS,
    create_comparison_plot,
)



def _parse_args():
    p = argparse.ArgumentParser(
        "Aggregate LD stats, make comparison PDF, run Moments-LD optimisation"
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Simulation run directory, e.g. experiments/<MODEL>/simulations/<sid>. "
            "Optional for *real data* mode when there is no simulation and thus "
            "no sampled_params.pkl."
        ),
    )
    p.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help=(
            "experiments/<MODEL>/inferences/sim_<sid>/MomentsLD or analogous "
            "path for real data"
        ),
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

    # ------------------------------------------------------------------
    # Decide whether we're in "simulation mode" or "real data mode"
    # ------------------------------------------------------------------
    if a.run_dir is not None:
        # Simulation mode: try to load sampled_params from the sim dir
        logging.info(f"Simulation mode: loading sampled_params from {a.run_dir}")
        sampled_params = load_sampled_params(a.run_dir, required=False)
    else:
        # Real data mode: there is no sim_dir and no sampled parameters
        logging.info("Real data mode: no --run-dir given, not loading sampled_params.")
        sampled_params = None

    # r-bins to use for both comparison PDF and optimisation
    if a.r_bins:
        r_vec = np.array([float(x) for x in a.r_bins.split(",")], float)
    else:
        r_vec = DEFAULT_R_BINS

    # ensure output root exists
    a.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) aggregate LD pickles → means/varcovs/bootstrap_sets
    #    (assumes LD_stats/*.pkl live under output_root/LD_stats)
    # ------------------------------------------------------------------
    empirical_data = aggregate_ld_statistics(a.output_root)

    # ------------------------------------------------------------------
    # 2) empirical vs theoretical PDF (skips if exists)
    # ------------------------------------------------------------------
    plot_file = a.output_root / "empirical_vs_theoretical_comparison.pdf"
    create_comparison_plot(cfg, sampled_params, empirical_data, r_vec, plot_file)

    # ------------------------------------------------------------------
    # 3) custom NLopt L-BFGS optimisation (skips if best_fit.pkl exists)
    # ------------------------------------------------------------------
    run_momentsld_inference(cfg, empirical_data, a.output_root, r_vec, sampled_params)


if __name__ == "__main__":
    main()

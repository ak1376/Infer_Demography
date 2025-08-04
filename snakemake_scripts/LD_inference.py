#!/usr/bin/env python3
"""Run a full moments-LD optimisation for **one** run_XXXX directory (modular, fault-tolerant)."""

from __future__ import annotations
from pathlib import Path
import sys

# Add src/ directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import argparse
import logging
import pickle

import numpy as np

from MomentsLD_inference import (
    load_sampled_params,
    load_config,
    aggregate_ld_stats,
    write_comparison_pdf,
    run_moments_ld_optimization,
)


def infer_r_bins(ld_dir: Path) -> np.ndarray:
    """Extract r_bins as bin edges from one of the LD_stats .pkl files."""
    try:
        ld_files = sorted(ld_dir.glob("LD_stats_window_*.pkl"))
        if not ld_files:
            raise FileNotFoundError(f"No LD_stats_window_*.pkl files found in {ld_dir}")
        with open(ld_files[0], "rb") as f:
            ld_data = pickle.load(f)
        bin_tuples = ld_data["bins"]  # list of (left, right) tuples
        left_edge = bin_tuples[0][0]
        right_edges = [right for _, right in bin_tuples]
        r_bins = np.array([left_edge] + right_edges)
        return r_bins
    except Exception as e:
        raise RuntimeError(f"Could not extract r_bins: {e}")



# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    cli = argparse.ArgumentParser("moments‑LD optimise one run (modular)")
    cli.add_argument("--run-dir", type=Path, required=True,
                     help="existing experiments/.../run_XXXX folder")
    cli.add_argument("--config-file", type=Path, required=True)
    cli.add_argument("--output-root", type=Path, required=True)
    args = cli.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        run_idx = int(args.run_dir.name.split("_")[-1])
    except (IndexError, ValueError):
        logging.error("Could not parse run index from directory name: %s", args.run_dir.name)
        return

    sim_dir = args.output_root
    sim_dir.mkdir(parents=True, exist_ok=True)

    try:
        sampled_params = load_sampled_params(args.run_dir)
    except FileNotFoundError:
        logging.error("Missing sampled_params.pkl in %s", args.run_dir)
        return
    except Exception as e:
        logging.exception("Failed to load sampled parameters: %s", e)
        return

    try:
        cfg = load_config(args.config_file)
    except Exception as e:
        logging.exception("Failed to load config: %s", e)
        return

    try:
        mv = aggregate_ld_stats(sim_dir)
    except FileNotFoundError:
        logging.error("Missing LD stats directory or files for %s", sim_dir)
        return
    except Exception as e:
        logging.exception("Failed during LD aggregation: %s", e)
        return

    # Infer r_bins from LD_stats file
    try:
        ld_stats_dir = sim_dir / "LD_stats"
        r_bins = infer_r_bins(ld_stats_dir)
    except Exception as e:
        logging.exception("Failed to extract r_bins: %s", e)
        return

    # Plot comparison PDF
    try:
        write_comparison_pdf(cfg, sampled_params, mv, r_bins, sim_dir)
    except Exception as e:
        logging.warning("Failed to generate comparison plot: %s", e)

    # Optimize model
    try:
        run_moments_ld_optimization(cfg, mv, sim_dir, r_bins)
    except Exception as e:
        logging.warning("Optimization failed: %s", e)

    print(f"✓ moments-LD finished for {sim_dir.relative_to(args.output_root.parent)}")


if __name__ == "__main__":
    main()

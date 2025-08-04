#!/usr/bin/env python3
"""Run a full moments-LD optimisation for **one** run_XXXX directory (modular, fault-tolerant)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from MomentsLD_inference import (
    DEFAULT_R_BINS,
    load_sampled_params,
    load_config,
    aggregate_ld_stats,
    write_comparison_pdf,
    run_moments_ld_optimization,
)

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    cli = argparse.ArgumentParser("moments‑LD optimise one run (modular)")
    cli.add_argument("--run-dir", type=Path, required=True,
                     help="existing experiments/.../run_XXXX folder")
    cli.add_argument("--config-file", type=Path, required=True)
    cli.add_argument("--output-root", type=Path, required=True)
    args = cli.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Infer run index from directory name
    try:
        run_idx = int(args.run_dir.name.split("_")[-1])
    except (IndexError, ValueError):
        logging.error("Could not parse run index from directory name: %s", args.run_dir.name)
        return

    sim_dir = args.output_root / f"sim_{run_idx:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Load parameters
    try:
        sampled_params = load_sampled_params(args.run_dir / "data")
    except FileNotFoundError:
        logging.error("Missing sampled_params.pkl in %s", args.run_dir / "data")
        return
    except Exception as e:
        logging.exception("Failed to load sampled parameters: %s", e)
        return

    try:
        cfg = load_config(args.config_file)
    except Exception as e:
        logging.exception("Failed to load config: %s", e)
        return

    # Aggregate LD stats
    try:
        mv = aggregate_ld_stats(sim_dir)
    except FileNotFoundError:
        logging.error("Missing LD stats directory or files for %s", sim_dir)
        return
    except Exception as e:
        logging.exception("Failed during LD aggregation: %s", e)
        return

    # Plot comparison PDF
    try:
        write_comparison_pdf(cfg, sampled_params, mv, DEFAULT_R_BINS, sim_dir)
    except Exception as e:
        logging.warning("Failed to generate comparison plot: %s", e)

    # Optimize model
    try:
        run_moments_ld_optimization(cfg, mv, sim_dir, DEFAULT_R_BINS)
    except Exception as e:
        logging.warning("Optimization failed: %s", e)

    print(f"✓ moments-LD finished for {sim_dir.relative_to(args.output_root.parent)}")


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
snakemake_scripts/compute_ld_window.py

Thin wrapper called by Snakemake rule `ld_window`.

It expects these files under --sim-dir:
  windows/window_<idx>.vcf.gz
  windows/samples.txt
  windows/flat_map.txt
  windows/window_<idx>.trees   (optional, only needed for GPU path)

It writes:
  LD_stats/LD_stats_window_<idx>.pkl

Heavy lifting lives in:
  src/ld_stats.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root to Python path to enable src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.LD_stats import compute_ld_window  # you will create this module


def parse_args():
    p = argparse.ArgumentParser("compute LD stats for one window")
    p.add_argument("--sim-dir", required=True, type=Path, help=".../sim_<sid>")
    p.add_argument(
        "--window-index", required=True, type=int, help="zero-based window index"
    )
    p.add_argument(
        "--config-file", required=True, type=Path, help="experiment_config_*.json"
    )
    p.add_argument("--r-bins", required=True, help="comma-separated recomb-bin edges")
    p.add_argument(
        "--use-gpu", action="store_true", help="Enable GPU acceleration via pg_gpu"
    )
    return p.parse_args()


def main():
    args = parse_args()
    sim_dir = args.sim_dir.resolve()
    idx = args.window_index
    r_bins = np.array([float(x) for x in args.r_bins.split(",")], dtype=float)

    with open(args.config_file) as f:
        config = json.load(f)

    vcf_gz = sim_dir / "windows" / f"window_{idx}.vcf.gz"
    samples_t = sim_dir / "windows" / "samples.txt"
    rec_map_t = sim_dir / "windows" / "flat_map.txt"
    ts_file = sim_dir / "windows" / f"window_{idx}.trees"
    out_dir = sim_dir / "LD_stats"
    out_pkl = out_dir / f"LD_stats_window_{idx}.pkl"

    # sanity checks
    for path in (vcf_gz, samples_t, rec_map_t):
        if not path.exists():
            raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # idempotent
    if out_pkl.exists():
        print(f"✓ window {idx}: already computed → {out_pkl.relative_to(sim_dir)}")
        return

    stats = compute_ld_window(
        window_index=idx,
        vcf_gz=vcf_gz,
        samples_file=samples_t,
        rec_map_file=rec_map_t,
        ts_file=ts_file if ts_file.exists() else None,
        r_bins=r_bins,
        config=config,
        request_gpu=args.use_gpu,
    )

    with out_pkl.open("wb") as fh:
        pickle.dump(stats, fh)

    print(f"✓ window {idx:04d}: wrote → {out_pkl.relative_to(sim_dir)}")


if __name__ == "__main__":
    main()

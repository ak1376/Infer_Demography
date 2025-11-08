#!/usr/bin/env python3
"""Compute LD statistics for **one window** of one simulation.

Called by Snakemake rule ``ld_window``:

    python compute_ld_window.py \
        --sim-dir      MomentsLD/LD_stats/sim_0007          # same as --output-root/sim_<sid>
        --window-index 42
        --config-file  config_files/experiment_config_*DEMOGRAPHIC_MODEL*.json
        --r-bins       "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

The script expects the following files already exist in *sim-dir*:
    windows/window_<idx>.vcf.gz       (compressed VCF of the replicate)
    samples.txt                       (two‑column sample/pop table)
    flat_map.txt                      (pos \t cM map)

It writes one pickle:
    LD_stats/LD_stats_window_<idx:04d>.pkl     (moments.LD.LDstats object)
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import moments

# ------------------------------------------------------------------ CLI


def parse_args():
    p = argparse.ArgumentParser("compute LD stats for one window")
    p.add_argument(
        "--sim-dir", required=True, type=Path, help="MomentsLD/LD_stats/sim_<sid>"
    )
    p.add_argument(
        "--window-index", required=True, type=int, help="zero‑based window index"
    )
    p.add_argument(
        "--config-file",
        required=True,
        type=Path,
        help="experiment_config_bottleneck.json (not used, but keeps interface symmetrical)",
    )
    p.add_argument(
        "--r-bins",
        required=True,
        help="comma‑separated list of recombination‑bin edges, e.g. '0,1e-6,1e-5,1e-4' ",
    )
    return p.parse_args()


# ------------------------------------------------------------------ main routine


def main():
    args = parse_args()

    sim_dir = args.sim_dir.resolve()
    idx = args.window_index
    r_bins = np.array([float(x) for x in args.r_bins.split(",")])

    vcf_gz = sim_dir / "windows" / f"window_{idx}.vcf.gz"
    samples_t = sim_dir / "windows" / "samples.txt"
    rec_map_t = sim_dir / "windows" / "flat_map.txt"
    out_dir = sim_dir / "LD_stats"
    out_pkl = out_dir / f"LD_stats_window_{idx}.pkl"

    # sanity checks -------------------------------------------------
    for path in (vcf_gz, samples_t, rec_map_t):
        if not path.exists():
            raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # skip if already done (idempotent rule) ------------------------
    if out_pkl.exists():
        print(f"✓ window {idx}: already computed → {out_pkl.relative_to(sim_dir)}")
        return

    # ----------------------------------------- grab every unique pop ID
    # read unique pop IDs from the samples file
    with samples_t.open() as fh:
        pops = sorted(
            {
                line.split()[1]
                for line in fh
                if line.strip() and not line.startswith("sample")
            }
        )

    # compute LD statistics ----------------------------------------
    stats = moments.LD.Parsing.compute_ld_statistics(
        str(vcf_gz),
        rec_map_file=str(rec_map_t),
        pop_file=str(samples_t),
        pops=pops,
        r_bins=r_bins,
        report=False,
    )

    # write pickle --------------------------------------------------
    with out_pkl.open("wb") as fh:
        pickle.dump(stats, fh)

    print(f"✓ window {idx:04d}: LD stats → {out_pkl.relative_to(sim_dir)}")


if __name__ == "__main__":
    main()

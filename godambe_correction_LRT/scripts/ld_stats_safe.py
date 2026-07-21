#!/usr/bin/env python3
# godambe_correction_LRT/scripts/ld_stats_safe.py

"""
Compute LD stats for ONE window, tolerating EMPTY windows. A 0-variant window
(scikit-allel returns None genotypes and compute_ld_window crashes) gets a
sentinel {"empty": True} instead of killing the whole Snakemake run.
Aggregation skips sentinels. Drop-in for compute_ld_window.py (same CLI).
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import pickle
import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from src.LD_stats import compute_ld_window


def n_variants(vcf: Path) -> int:
    out = subprocess.run(
        f"bcftools view -H '{vcf}' 2>/dev/null | head -2 | wc -l",
        shell=True, capture_output=True, text=True,
    )
    return int((out.stdout or "0").strip() or 0)


def main():
    ap = argparse.ArgumentParser(description="Tolerant single-window LD stats.")
    ap.add_argument("--sim-dir", required=True, type=Path)
    ap.add_argument("--window-index", required=True, type=int)
    ap.add_argument("--config-file", required=True, type=Path)
    ap.add_argument("--r-bins", required=True)
    ap.add_argument("--rec-map-file", default=None, type=Path)
    args = ap.parse_args()

    sim, i = args.sim_dir, args.window_index
    vcf = sim / "windows" / f"window_{i}.vcf.gz"
    samples = sim / "windows" / "samples.txt"
    rec_map = args.rec_map_file if args.rec_map_file is not None else sim / "windows" / "flat_map.txt"
    out_dir = sim / "LD_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"LD_stats_window_{i}.pkl"

    r_bins = np.array([float(x) for x in args.r_bins.split(",")])
    with open(args.config_file) as f:
        config = json.load(f)

    if n_variants(vcf) == 0:
        print(f"window {i}: 0 variants -> sentinel (skipped)")
        stats = {"empty": True}
    else:
        stats = compute_ld_window(
            window_index=i, vcf_gz=vcf, samples_file=samples,
            rec_map_file=rec_map, r_bins=r_bins, config=config,
        )
    with out.open("wb") as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    main()

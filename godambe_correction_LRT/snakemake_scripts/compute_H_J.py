#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/compute_H_J.py
"""
Thin wrapper called by Snakemake rule `compute_H_J`.

Raw Godambe H and J for the CO-growth test, at one arm + block size.

Heavy lifting lives in:
  godambe_correction_LRT/src/compute_H_J.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import moments

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compute_H_J import compute_H_J_growth


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--block-vcf-dir", required=True, type=Path)
    ap.add_argument("--popfile", required=True, type=Path)
    ap.add_argument("--sfs", required=True, type=Path,
                     help="arm's own unfolded SFS -- the data the fits were run on")
    ap.add_argument("--simple-fit", required=True, type=Path)
    ap.add_argument("--complex-fit", required=True, type=Path)
    ap.add_argument("--n-boot-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--pop-ids", default="CO,FR")
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    with open(args.sfs, "rb") as f:
        data_sfs = pickle.load(f)
    data_sfs = moments.Spectrum(data_sfs)
    data_sfs.pop_ids = [s.strip() for s in args.pop_ids.split(",")]

    with open(args.simple_fit, "rb") as f:
        simple_fit = pickle.load(f)
    with open(args.complex_fit, "rb") as f:
        complex_fit = pickle.load(f)

    summary = compute_H_J_growth(
        arm=args.arm, block_vcf_dir=args.block_vcf_dir, popfile=args.popfile,
        data_sfs=data_sfs, simple_fit=simple_fit, complex_fit=complex_fit,
        n_boot_reps=args.n_boot_reps, seed=args.seed, eps=args.eps,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{args.arm}] H={summary['H']:.6g}  J={summary['J']:.6g}  "
          f"H/J={summary['adjust_H_over_J']:.6g}")
    print(f"[{args.arm}] D={summary['raw_D']:.6g}  D_adj={summary['D_adj']:.6g}  "
          f"p_raw={summary['p_raw']:.6g}  p_adj={summary['p_adj']:.6g}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/process_one_chunk.py
"""
Thin wrapper called by Snakemake rule `process_one_chunk`.

Runs Stage 1 (floor + crossing-distance, CO and FR separately) for ONE
chunk. Each chunk is its own Snakemake job, so the ~80 chunks that used to
run sequentially inside compute_validated_blocks.py's main loop now run in
parallel (bounded by --resources gpu=N).

Writes {"empty": true, ...} when the chunk has too few polymorphic-within-
population sites (mirrors the LD_stats_window "empty" sentinel pattern used
elsewhere in this pipeline), so the aggregation step can skip it gracefully.

Heavy lifting lives in:
  godambe_correction_LRT/src/compute_validated_blocks.py
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from compute_validated_blocks import read_popfile, process_chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--chunk-bounds-file", required=True, type=Path)
    ap.add_argument("--floor-cutoff-bp", type=int, required=True)
    ap.add_argument("--tolerance", type=float, required=True)
    ap.add_argument("--n-bins", type=int, required=True)
    ap.add_argument("--max-pairs", type=int, default=2_000_000,
                     help="cap on SNP pairs fed into the per-bin median (random subsample "
                          "above this); a dense chunk can have tens of millions of pairs, "
                          "far more than needed for a stable per-bin median")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--verbose", action="store_true", help="print per-phase timing (from_vcf, pairwise_r2+.get(), binning)")
    args = ap.parse_args()

    bounds = json.loads(args.chunk_bounds_file.read_text())
    co_samples, fr_samples = read_popfile(args.popfile)

    result = process_chunk(
        args.vcf, args.popfile, co_samples, fr_samples, args.arm,
        bounds["start"], bounds["end"], args.floor_cutoff_bp, args.tolerance, args.n_bins,
        max_pairs=args.max_pairs, verbose_timing=args.verbose,
    )
    if result is None:
        result = {"chunk_start": bounds["start"], "chunk_end": bounds["end"], "empty": True}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"chunk {bounds['start']}-{bounds['end']}: {result}")


if __name__ == "__main__":
    main()

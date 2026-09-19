#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/chunk_bounds.py
"""
Thin wrapper called by Snakemake checkpoint `chunk_bounds`.

Enumerates the (chunk_start, chunk_end) pairs Stage 1 of
compute_validated_blocks.py needs to process -- cheap (VCF span + arithmetic
only, no r2), so this can run once up front and let each chunk become its
own parallel Snakemake job (rule process_one_chunk) instead of one long
sequential loop.

Writes one JSON file per chunk: chunk_<k>.json = {"start": s, "end": e}.

Heavy lifting lives in:
  godambe_correction_LRT/src/compute_validated_blocks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from compute_validated_blocks import get_chunk_bounds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--floor-cutoff-bp", type=int, required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    chrom_start, chrom_end, chunks = get_chunk_bounds(args.vcf, args.chunk_size, args.floor_cutoff_bp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, (start, end) in enumerate(chunks):
        with open(args.out_dir / f"chunk_{k}.json", "w") as f:
            json.dump({"start": start, "end": end}, f)

    print(f"chrom spans {chrom_start:,}-{chrom_end:,}; wrote {len(chunks)} chunk bound files -> {args.out_dir}")


if __name__ == "__main__":
    main()

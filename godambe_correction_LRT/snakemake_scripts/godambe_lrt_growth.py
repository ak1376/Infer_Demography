#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/godambe_lrt_growth.py

"""
Thin wrapper called by Snakemake rules `godambe_lrt_growth` and
`godambe_lrt_growth_pooled`.

Godambe-adjusted likelihood-ratio test: does CO have exponential growth?
See the module docstring in src/godambe_lrt_growth.py for the statistical
pipeline.

Heavy lifting lives in:
  godambe_correction_LRT/src/godambe_lrt_growth.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from godambe_lrt_growth import run_godambe_lrt_growth


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, nargs="+",
                    help="Chromosome arm(s), e.g. Chr3L, or multiple (e.g. Chr2L Chr3L) to pool")
    p.add_argument("--simple-fit", type=Path, required=True, help="sfs_fit_simple/best_fit.pkl")
    p.add_argument("--complex-fit", type=Path, required=True, help="sfs_fit_complex/best_fit.pkl")
    p.add_argument("--sfs", type=Path, required=True,
                    help="unfolded.sfs.pkl for these arm(s) -- pooled/summed already if multiple arms")
    p.add_argument("--vcf", type=Path, required=True, nargs="+",
                    help="Polarized (haploid+AA) VCF(s), one per --arm, same order")
    p.add_argument("--popfile", type=Path, required=True)
    p.add_argument("--validated-blocks-bed", type=Path, nargs="*", default=[],
                    help="validated_blocks.bed from rule validated_blocks (single-arm only; "
                         "omit to skip the bonus validated-block-size report row)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True,
                    help="Per-arm cache dir for parsed per-block SFS")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--num-boot-reps", type=int, default=10_000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--block-sizes-kb", type=str, default="50,75,100,150,200,300,500",
                    help="Comma-separated candidate block sizes (kb) for the sensitivity sweep")
    p.add_argument("--score-eps", type=float, default=0.01,
                    help="Finite-difference step for the growth_CO score (J) and Hessian (H)")
    return p.parse_args()


def main():
    args = parse_args()
    block_sizes_kb = [float(x) for x in args.block_sizes_kb.split(",") if x.strip()]

    run_godambe_lrt_growth(
        arm=args.arm,
        simple_fit=args.simple_fit,
        complex_fit=args.complex_fit,
        sfs=args.sfs,
        vcf=args.vcf,
        popfile=args.popfile,
        validated_blocks_bed=args.validated_blocks_bed,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        num_boot_reps=args.num_boot_reps,
        rng_seed=args.rng_seed,
        block_sizes_kb=block_sizes_kb,
        score_eps=args.score_eps,
    )


if __name__ == "__main__":
    main()

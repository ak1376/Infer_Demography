#!/usr/bin/env python3
"""
Prune a VCF file at multiple keep-fractions.

For each fraction the original filename is preserved so that existing
LD-stats scripts can be pointed at each output directory unchanged.
The original file is also copied into an 'unpruned/' directory.

Output layout (relative to --out-dir):
  unpruned/   <original_name>.vcf.gz     <- copy of the input
  thin10/     <original_name>.vcf.gz     <- 10% of sites kept
  thin15/     <original_name>.vcf.gz
  thin20/     <original_name>.vcf.gz
  thin25/     <original_name>.vcf.gz
  thin30/     <original_name>.vcf.gz

Single VCF:
  python prune_vcf.py --vcf window_24.vcf.gz --out-dir /path/to/output

All windows in a directory (parallel):
  python prune_vcf.py --windows-dir /path/to/windows --out-dir /path/to/output --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np

THIN_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30]
SEED = 42
SUPPORT_FILES = ["samples.txt", "flat_map.txt"]


def _frac_tag(f: float) -> str:
    return f"thin{round(f * 100):02d}"


def _write_thinned(args):
    """Worker function: write one thinned VCF. Called in a process pool."""
    header, variants, n_full, frac, dest_str = args
    dest = Path(dest_str)
    if dest.exists():
        return f"  {dest.parent.parent.name}/windows/{dest.name}: already exists, skipping"

    n_keep   = max(1, round(n_full * frac))
    rng      = np.random.default_rng(SEED + round(frac * 100))
    keep_idx = np.sort(rng.choice(n_full, size=n_keep, replace=False))

    with gzip.open(str(dest), "wt") as fh:
        fh.writelines(header)
        for i in keep_idx:
            fh.write(variants[i])

    return f"  {_frac_tag(frac)}/windows/{dest.name}: kept {n_keep}/{n_full} sites ({frac:.0%})"


def prune_vcf(vcf_in: Path, out_dir: Path, workers: int = 1,
              copy_unpruned: bool = True) -> None:
    """Prune vcf_in into out_dir/{unpruned,thin*}/windows/ subdirs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    header, variants = [], []
    with gzip.open(str(vcf_in), "rt") as fh:
        for line in fh:
            (header if line.startswith("#") else variants).append(line)
    n_full = len(variants)
    fname  = vcf_in.name

    print(f"{fname}: {n_full} sites")

    src_dir = vcf_in.parent

    # Optionally copy original into unpruned/windows/
    if copy_unpruned:
        unpruned_wins = out_dir / "unpruned" / "windows"
        unpruned_wins.mkdir(parents=True, exist_ok=True)
        unpruned_dest = unpruned_wins / fname
        if unpruned_dest.exists():
            print(f"  unpruned/windows/{fname}: already exists, skipping")
        else:
            shutil.copy2(str(vcf_in), str(unpruned_dest))
            for sf in SUPPORT_FILES:
                if (src_dir / sf).exists():
                    shutil.copy2(str(src_dir / sf), str(unpruned_wins / sf))
            print(f"  unpruned/windows/{fname}: copied")

    # Prepare output windows/ dirs and copy support files
    for frac in THIN_FRACTIONS:
        wins_dir = out_dir / _frac_tag(frac) / "windows"
        wins_dir.mkdir(parents=True, exist_ok=True)
        for sf in SUPPORT_FILES:
            if (src_dir / sf).exists() and not (wins_dir / sf).exists():
                shutil.copy2(str(src_dir / sf), str(wins_dir / sf))

    # Fan out gzip-compression across workers
    tasks = [
        (header, variants, n_full, frac,
         str(out_dir / _frac_tag(frac) / "windows" / fname))
        for frac in THIN_FRACTIONS
    ]

    if workers > 1:
        with Pool(workers) as pool:
            for msg in pool.map(_write_thinned, tasks):
                print(msg)
    else:
        for task in tasks:
            print(_write_thinned(task))


def _parse_args():
    p = argparse.ArgumentParser(description="Prune a VCF at multiple keep-fractions")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--vcf",         type=Path, help="Single input VCF (gzipped)")
    grp.add_argument("--windows-dir", type=Path, help="Directory of window_*.vcf.gz files")
    p.add_argument("--out-dir",  required=True, type=Path, help="Root output directory")
    p.add_argument("--workers",  type=int, default=4,
                   help="Parallel workers for gzip compression (default: 4)")
    p.add_argument("--keep-fractions", type=str, default=None,
                   help="Comma-separated keep fractions to run, e.g. 0.15 or 0.10,0.15 "
                        "(default: all five)")
    p.add_argument("--no-unpruned", action="store_true",
                   help="Skip copying the original VCF into unpruned/ (saves disk space)")
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    out_dir = args.out_dir.resolve()

    # Override THIN_FRACTIONS if --keep-fractions specified
    if args.keep_fractions:
        THIN_FRACTIONS[:] = [float(x) for x in args.keep_fractions.split(",")]

    copy_unpruned = not args.no_unpruned

    if args.vcf:
        prune_vcf(args.vcf.resolve(), out_dir, workers=args.workers,
                  copy_unpruned=copy_unpruned)
    else:
        vcf_files = sorted(args.windows_dir.resolve().glob("window_*.vcf.gz"))
        if not vcf_files:
            raise FileNotFoundError(f"No window_*.vcf.gz in {args.windows_dir}")
        print(f"Found {len(vcf_files)} windows, processing with {args.workers} workers each\n")
        for vcf in vcf_files:
            prune_vcf(vcf, out_dir, workers=args.workers, copy_unpruned=copy_unpruned)

    print("\nDone.")

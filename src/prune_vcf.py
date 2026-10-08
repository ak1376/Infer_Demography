#!/usr/bin/env python3
"""
Prune a VCF file at multiple keep-fractions, and/or cap it at absolute
keep-counts.

Two independent pruning modes, selected by which CLI flag is used:
  --keep-fractions: keep a fixed % of sites, e.g. 0.15 -> keep 15% (random
                     subsample). Same relative thinning regardless of how
                     many sites the window started with.
  --keep-counts:    cap at an absolute number of sites, e.g. 5000 -> keep
                     min(n_full, 5000). A window that already has fewer
                     sites than the cap is copied through unchanged rather
                     than resampled -- this mode never makes a sparse
                     window sparser.

For each fraction/count the original filename is preserved so that existing
LD-stats scripts can be pointed at each output directory unchanged.
The original file is also copied into an 'unpruned/' directory.

Output layout (relative to --out-dir):
  unpruned/   <original_name>.vcf.gz     <- copy of the input
  thin10/     <original_name>.vcf.gz     <- 10% of sites kept
  thin15/     <original_name>.vcf.gz
  ...
  n5000/      <original_name>.vcf.gz     <- capped at 5000 sites
  n20000/     <original_name>.vcf.gz

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
KEEP_COUNTS: list[int] = []
SEED = 42
SUPPORT_FILES = ["samples.txt", "flat_map.txt"]


def _frac_tag(f: float) -> str:
    return f"thin{round(f * 100):02d}"


def _count_tag(n: int) -> str:
    return f"n{int(n)}"


def _write_thinned(args):
    """Worker function: write one thinned/capped VCF. Called in a process pool."""
    header, variants, n_full, tag, n_keep, seed, dest_str = args
    dest = Path(dest_str)
    if dest.exists():
        return f"  {tag}/windows/{dest.name}: already exists, skipping"

    if n_keep >= n_full:
        # Nothing to remove (count mode, window already at/below the cap) --
        # copy through unchanged rather than resampling everything.
        with gzip.open(str(dest), "wt") as fh:
            fh.writelines(header)
            fh.writelines(variants)
        return f"  {tag}/windows/{dest.name}: kept all {n_full} sites (below cap)"

    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(n_full, size=n_keep, replace=False))

    with gzip.open(str(dest), "wt") as fh:
        fh.writelines(header)
        for i in keep_idx:
            fh.write(variants[i])

    return f"  {tag}/windows/{dest.name}: kept {n_keep}/{n_full} sites ({n_keep / n_full:.0%})"


def prune_vcf(
    vcf_in: Path, out_dir: Path, workers: int = 1, copy_unpruned: bool = True
) -> None:
    """Prune vcf_in into out_dir/{unpruned,thin*}/windows/ subdirs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    header, variants = [], []
    with gzip.open(str(vcf_in), "rt") as fh:
        for line in fh:
            (header if line.startswith("#") else variants).append(line)
    n_full = len(variants)
    fname = vcf_in.name

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

    # (tag, n_keep, seed) for every requested fraction and/or absolute count.
    specs = [
        (_frac_tag(f), max(1, round(n_full * f)), SEED + round(f * 100))
        for f in THIN_FRACTIONS
    ]
    specs += [
        (_count_tag(n), min(n_full, int(n)), SEED + int(n)) for n in KEEP_COUNTS
    ]

    # Prepare output windows/ dirs and copy support files
    for tag, _, _ in specs:
        wins_dir = out_dir / tag / "windows"
        wins_dir.mkdir(parents=True, exist_ok=True)
        for sf in SUPPORT_FILES:
            if (src_dir / sf).exists() and not (wins_dir / sf).exists():
                shutil.copy2(str(src_dir / sf), str(wins_dir / sf))

    # Fan out gzip-compression across workers
    tasks = [
        (
            header,
            variants,
            n_full,
            tag,
            n_keep,
            seed,
            str(out_dir / tag / "windows" / fname),
        )
        for tag, n_keep, seed in specs
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
    grp.add_argument("--vcf", type=Path, help="Single input VCF (gzipped)")
    grp.add_argument(
        "--windows-dir", type=Path, help="Directory of window_*.vcf.gz files"
    )
    p.add_argument("--out-dir", required=True, type=Path, help="Root output directory")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for gzip compression (default: 4)",
    )
    p.add_argument(
        "--keep-fractions",
        type=str,
        default=None,
        help="Comma-separated keep fractions to run, e.g. 0.15 or 0.10,0.15 "
        "(default: all five)",
    )
    p.add_argument(
        "--keep-counts",
        type=str,
        default=None,
        help="Comma-separated absolute site-count caps, e.g. 5000 or "
        "5000,20000 -- each window is capped at min(n_full, N); a window "
        "already at or below N is copied through unchanged.",
    )
    p.add_argument(
        "--no-unpruned",
        action="store_true",
        help="Skip copying the original VCF into unpruned/ (saves disk space)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir = args.out_dir.resolve()

    # Explicit-mode override: when either flag is passed, produce ONLY what was
    # asked for (an omitted flag means none of that mode), rather than falling
    # back to the full THIN_FRACTIONS default sweep alongside it.
    if args.keep_fractions is not None or args.keep_counts is not None:
        THIN_FRACTIONS[:] = (
            [float(x) for x in args.keep_fractions.split(",")]
            if args.keep_fractions
            else []
        )
        KEEP_COUNTS[:] = (
            [int(x) for x in args.keep_counts.split(",")] if args.keep_counts else []
        )

    copy_unpruned = not args.no_unpruned

    if args.vcf:
        prune_vcf(
            args.vcf.resolve(),
            out_dir,
            workers=args.workers,
            copy_unpruned=copy_unpruned,
        )
    else:
        vcf_files = sorted(args.windows_dir.resolve().glob("window_*.vcf.gz"))
        if not vcf_files:
            raise FileNotFoundError(f"No window_*.vcf.gz in {args.windows_dir}")
        print(
            f"Found {len(vcf_files)} windows, processing with {args.workers} workers each\n"
        )
        for vcf in vcf_files:
            prune_vcf(vcf, out_dir, workers=args.workers, copy_unpruned=copy_unpruned)

    print("\nDone.")

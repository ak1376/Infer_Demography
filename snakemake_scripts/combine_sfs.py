#!/usr/bin/env python3
"""
combine_sfs.py

Sum several per-chromosome unfolded SFS pickles (moments.Spectrum) into one
combined spectrum. The SFS is a histogram over disjoint sites, so the combined
spectrum is the entry-by-entry sum of the inputs. All inputs must share the same
shape, pop_ids, and folded state.

Usage:
  python combine_sfs.py \
      --in-sfs A.unfolded.sfs.pkl B.unfolded.sfs.pkl ... \
      --output-sfs combined/autosomes.unfolded.sfs.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import moments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-sfs", nargs="+", required=True,
                   help="Per-chromosome SFS pickles to sum.")
    p.add_argument("--output-sfs", type=Path, required=True)
    p.add_argument("--in-meta", nargs="*", default=None,
                   help="Optional: per-chromosome unfolded.sfs.meta.json files "
                        "(same order as --in-sfs) to sum sequence_length across.")
    p.add_argument("--output-meta", type=Path, default=None,
                   help="Optional: combined {sequence_length, chroms} JSON, "
                        "required if --in-meta is given.")
    args = p.parse_args()

    args.output_sfs.parent.mkdir(parents=True, exist_ok=True)

    total = None
    pop_ids = None
    for path in args.in_sfs:
        with open(path, "rb") as fh:
            s = pickle.load(fh)
        print(f"{path}: shape={s.shape}, pops={s.pop_ids}, folded={s.folded}, S={s.S():.0f}")
        if total is None:
            total = s.copy()
            pop_ids = s.pop_ids
        else:
            if s.shape != total.shape:
                raise ValueError(f"{path}: shape {s.shape} != {total.shape}")
            if s.pop_ids != pop_ids:
                raise ValueError(f"{path}: pop_ids {s.pop_ids} != {pop_ids}")
            if s.folded != total.folded:
                raise ValueError(f"{path}: folded {s.folded} != {total.folded}")
            total += s

    # Corners (fixed sites) were zeroed per-chromosome; keep them masked.
    total = moments.Spectrum(total, pop_ids=pop_ids)
    total.mask[0, 0] = True
    total.mask[-1, -1] = True

    with open(args.output_sfs, "wb") as fh:
        pickle.dump(total, fh)

    print(f"\nCombined: shape={total.shape}, pops={total.pop_ids}, "
          f"folded={total.folded}, S={total.S():.0f}")
    print(f"Saved -> {args.output_sfs}")

    if args.in_meta:
        if args.output_meta is None:
            raise ValueError("--output-meta is required when --in-meta is given")
        chroms = []
        total_length = 0
        for meta_path in args.in_meta:
            with open(meta_path) as fh:
                m = json.load(fh)
            chroms.append(m["chrom"])
            total_length += int(m["sequence_length"])

        args.output_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_meta, "w") as fh:
            json.dump({"chroms": chroms, "sequence_length": total_length}, fh, indent=2)
        print(f"Combined sequence_length={total_length:,} over {chroms} -> {args.output_meta}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compute_unfolded_sfs.py

Compute a 2D unfolded SFS from a haploid-GT VCF that has an AA INFO field
(ancestral allele).  No haploid->diploid recoding needed.

Each sample contributes 1 chromosome (GT=0 -> ancestral, GT=1 -> alt).
Derived allele count per site is determined by the AA field:
  - AA == REF  ->  derived count = number of samples with GT=1
  - AA == ALT  ->  derived count = number of samples with GT=0  (flip)

Sites with any missing GT ('.') are skipped.
Sites without an AA field are skipped.

Usage:
  python compute_unfolded_sfs.py \
      --input-vcf  real_data_analysis/data/drosophila/Chr2L.polarized.vcf.gz \
      --popfile    real_data_analysis/data/drosophila/popfile.txt \
      --output-sfs real_data_analysis/data/drosophila/drosophila.unfolded.sfs.pkl \
      [--project-to N]   # optional: project each pop down to N haplotypes
"""

import argparse
import gzip
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import moments


def parse_popfile(path: Path):
    """Returns (pop_names list, sample->pop dict)."""
    sample_to_pop = {}
    pop_order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            sample, pop = parts[0], parts[1]
            sample_to_pop[sample] = pop
            if pop not in pop_order:
                pop_order.append(pop)
    return pop_order, sample_to_pop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-vcf",   type=Path, required=True,
                   help="Polarized VCF with AA INFO field (haploid GTs).")
    p.add_argument("--popfile",     type=Path, required=True,
                   help="Two-column file: sample  population")
    p.add_argument("--output-sfs",  type=Path, required=True)
    p.add_argument("--project-to",  type=int, default=None,
                   help="Project each population down to this many haplotypes.")
    args = p.parse_args()

    args.output_sfs.parent.mkdir(parents=True, exist_ok=True)

    pop_names, sample_to_pop = parse_popfile(args.popfile)
    print(f"Populations: {pop_names}")

    opener = gzip.open if str(args.input_vcf).endswith(".gz") else open

    # We'll accumulate a raw count array of shape (n_pop0+1, n_pop1+1, ...)
    # First pass: read header to get sample indices per pop
    sample_indices = defaultdict(list)  # pop -> [col indices in VCF]
    header_done = False
    vcf_samples = []

    with opener(str(args.input_vcf), "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                vcf_samples = line.rstrip("\n").split("\t")[9:]
                for i, s in enumerate(vcf_samples):
                    pop = sample_to_pop.get(s)
                    if pop is not None:
                        sample_indices[pop].append(i)
                header_done = True
                break

    n_per_pop = {pop: len(sample_indices[pop]) for pop in pop_names}
    print(f"Samples per pop: { {p: n_per_pop[p] for p in pop_names} }")

    # SFS array shape: (n_pop0+1) x (n_pop1+1)
    shape = tuple(n_per_pop[pop] + 1 for pop in pop_names)
    sfs_arr = np.zeros(shape, dtype=np.float64)

    total = skipped_missing = skipped_no_aa = skipped_no_match = kept = 0

    with opener(str(args.input_vcf), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            ref = fields[3]
            alt = fields[4]
            info = fields[7]
            gts_all = fields[9:]  # haploid: "0", "1", or "."

            total += 1

            # Parse AA from INFO
            aa = None
            for token in info.split(";"):
                if token.startswith("AA="):
                    aa = token[3:]
                    break
            if aa is None:
                skipped_no_aa += 1
                continue

            if aa == ref:
                flip = False
            elif aa == alt:
                flip = True
            else:
                skipped_no_match += 1
                continue

            # Count derived alleles per population
            derived_counts = []
            skip_site = False
            for pop in pop_names:
                idx_list = sample_indices[pop]
                n = len(idx_list)
                alt_count = 0
                for i in idx_list:
                    gt = gts_all[i]
                    if gt == ".":
                        skip_site = True
                        break
                    alt_count += int(gt)
                if skip_site:
                    break
                derived = (n - alt_count) if flip else alt_count
                derived_counts.append(derived)

            if skip_site:
                skipped_missing += 1
                continue

            kept += 1
            sfs_arr[tuple(derived_counts)] += 1

    print(f"\nSNP summary:")
    print(f"  Total sites          : {total:>10,}")
    print(f"  Kept                 : {kept:>10,}  ({100*kept/total:.1f}%)")
    print(f"  Skipped (missing GT) : {skipped_missing:>10,}")
    print(f"  Skipped (no AA)      : {skipped_no_aa:>10,}")
    print(f"  Skipped (AA mismatch): {skipped_no_match:>10,}")

    # Build moments Spectrum (unfolded)
    sfs = moments.Spectrum(sfs_arr, pop_ids=pop_names)

    # Zero out the corners (fixed sites)
    sfs[0, 0] = 0.0
    sfs[-1, -1] = 0.0

    if args.project_to is not None:
        print(f"\nProjecting each population to {args.project_to} haplotypes ...")
        sfs = sfs.project([args.project_to] * len(pop_names))

    print(f"\nSFS shape : {sfs.shape}")
    print(f"Folded    : {sfs.folded}")
    print(f"Pop IDs   : {sfs.pop_ids}")
    print(f"Total SNPs in SFS: {sfs.S():.0f}")

    with open(args.output_sfs, "wb") as fh:
        pickle.dump(sfs, fh)

    print(f"\nSaved -> {args.output_sfs}")


if __name__ == "__main__":
    main()

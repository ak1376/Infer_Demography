#!/usr/bin/env python3
"""
real_data_sfs.py: Parse VCF + popfile, construct folded SFS, and save as pickle.

Assumes input VCF has diploid-coded GTs (e.g., after haploid->diploid recode).
"""

import argparse
import pickle
from pathlib import Path
from collections import Counter, OrderedDict

import moments


def parse_args():
    p = argparse.ArgumentParser("Construct SFS from VCF and popfile")
    p.add_argument("--input-vcf", type=Path, required=True)
    p.add_argument("--popfile", type=Path, required=True)
    p.add_argument("--output-sfs", type=Path, required=True)

    # Optional overrides
    p.add_argument("--folded", action="store_true", default=True,
                   help="Construct folded SFS (recommended for unpolarized data). Default: True")
    p.add_argument("--no-folded", dest="folded", action="store_false",
                   help="Construct unfolded SFS (only if truly polarized).")

    p.add_argument("--project-chroms",
                   type=int,
                   default=None,
                   help=("Optional: project EACH population to this many chromosomes. "
                         "If omitted, uses 2 * (number of individuals in popfile) per pop."))
    return p.parse_args()


def parse_popfile(popfile_path: Path):
    """
    Returns:
      pop_names (order-preserving list)
      n_individuals_by_pop (dict pop -> count of sample IDs)
    """
    pops_in_order = []
    counts = Counter()

    with open(popfile_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sample_id, pop = parts[0], parts[1]
            counts[pop] += 1
            pops_in_order.append(pop)

    pop_names = list(OrderedDict.fromkeys(pops_in_order).keys())
    return pop_names, dict(counts)


def main():
    args = parse_args()

    pop_names, n_ind_by_pop = parse_popfile(args.popfile)

    # For your diploidGT VCF, each individual contributes 2 chromosomes.
    # So default projection per pop is 2 * (# individuals in popfile).
    if args.project_chroms is None:
        sample_sizes_dict = {pop: 2 * n_ind_by_pop[pop] for pop in pop_names}
    else:
        sample_sizes_dict = {pop: int(args.project_chroms) for pop in pop_names}

    sfs = moments.Spectrum.from_vcf(
        str(args.input_vcf),
        pop_file=str(args.popfile),
        pops=pop_names,
        sample_sizes=sample_sizes_dict,
        folded=args.folded,
    )

    # Make pop order explicit (useful downstream)
    sfs.pop_ids = pop_names

    args.output_sfs.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_sfs, "wb") as f:
        pickle.dump(sfs, f)

    print(f"Saved SFS to {args.output_sfs}")
    print(f"Pop IDs: {sfs.pop_ids}")
    print(f"Projected sample sizes (chromosomes): {sample_sizes_dict}")
    print(f"Shape: {sfs.shape}  Folded: {sfs.folded}  Sum (raw): {sfs.data.sum()}")


if __name__ == "__main__":
    main()

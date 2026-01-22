#!/usr/bin/env python3
"""
real_data_sfs.py: Parse VCF and popfile, construct SFS, and save as pickle.
"""

import argparse
import pickle
from pathlib import Path
import moments


def parse_args():
    p = argparse.ArgumentParser("Construct SFS from VCF and popfile")
    p.add_argument("--input-vcf", type=Path, required=True)
    p.add_argument("--popfile", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-sfs", type=Path, required=True)
    return p.parse_args()


def parse_popfile(popfile_path):
    pop_names = []
    with open(popfile_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            pop_names.append(fields[1])
    # Remove duplicates, preserve order
    return list(dict.fromkeys(pop_names))


def main():
    args = parse_args()
    pop_names = parse_popfile(args.popfile)
    # Projecting to 20 chromosomes per population (yielding a 21x21 SFS)
    # Using moments to parse VCF and project
    sample_sizes_dict = {pop: 20 for pop in pop_names}

    # Construct SFS from VCF using moments
    sfs = moments.Spectrum.from_vcf(
        str(args.input_vcf), pop_file=str(args.popfile), sample_sizes=sample_sizes_dict
    )
    # Save SFS as pickle
    with open(args.output_sfs, "wb") as f:
        pickle.dump(sfs, f)
    print(f"Saved SFS to {args.output_sfs}")


if __name__ == "__main__":
    main()

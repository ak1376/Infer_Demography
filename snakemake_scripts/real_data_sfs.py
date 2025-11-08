#!/usr/bin/env python3
"""
real_data_sfs.py: Parse VCF and popfile, construct SFS, and save as pickle.
"""
import argparse
import pickle
from pathlib import Path
import dadi


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
    # Count samples per population
    from collections import Counter

    pop_list = []
    with open(args.popfile) as pf:
        for line in pf:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            pop_list.append(fields[1])
    pop_counter = Counter(pop_list)
    sample_sizes = [pop_counter[pop] for pop in pop_names]
    # Construct SFS from VCF
    dd = dadi.Misc.make_data_dict_vcf(str(args.input_vcf), str(args.popfile))
    sfs = dadi.Spectrum.from_data_dict(dd, pop_names, sample_sizes)
    # Save SFS as pickle
    with open(args.output_sfs, "wb") as f:
        pickle.dump(sfs, f)
    print(f"Saved SFS to {args.output_sfs}")


if __name__ == "__main__":
    main()

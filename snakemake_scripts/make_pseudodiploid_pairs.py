#!/usr/bin/env python3
"""
Pair haploid samples into pseudo-diploids, within each population.

Each population's samples are shuffled (fixed seed) and split into disjoint
consecutive pairs, so no haploid sample is used twice and pairs never cross
populations. A population with an odd count drops its one leftover sample
(which one is random, via the shuffle).

Outputs:
  --out-pairs    TSV: name  hap1  hap2  pop   (read by recode_haploid_to_diploid.py)
  --out-popfile  "name pop" per line, same format as the haploid popfile, for
                 the MomentsLD rules that read the pseudo-diploid VCF.

Pseudo-diploid names are "<hap1>_<hap2>".
"""
import argparse
import random
from collections import OrderedDict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--popfile", required=True, type=Path,
                   help="haploid popfile (sampleID popID)")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--exclude", default="", help="comma-separated sample IDs to leave out before pairing")
    p.add_argument("--out-pairs", required=True, type=Path)
    p.add_argument("--out-popfile", required=True, type=Path)
    args = p.parse_args()

    drop = {x for x in args.exclude.split(",") if x}
    pops = OrderedDict()
    with open(args.popfile) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            if parts[0] in drop:
                print(f"[{parts[1]}] excluding {parts[0]}")
                drop.discard(parts[0])
                continue
            pops.setdefault(parts[1], []).append(parts[0])
    if drop:
        raise SystemExit(f"--exclude samples not in popfile: {sorted(drop)}")

    rng = random.Random(args.seed)
    rows = []
    for pop, samples in pops.items():
        shuffled = list(samples)
        rng.shuffle(shuffled)
        if len(shuffled) % 2:
            print(f"[{pop}] odd sample count ({len(shuffled)}); dropping {shuffled[-1]}")
            shuffled = shuffled[:-1]
        for a, b in zip(shuffled[0::2], shuffled[1::2]):
            rows.append((f"{a}_{b}", a, b, pop))
        print(f"[{pop}] {len(samples)} haploid -> {len(shuffled) // 2} pseudo-diploid")

    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pairs, "w") as f:
        f.write("name\thap1\thap2\tpop\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    with open(args.out_popfile, "w") as f:
        for name, _, _, pop in rows:
            f.write(f"{name} {pop}\n")


if __name__ == "__main__":
    main()

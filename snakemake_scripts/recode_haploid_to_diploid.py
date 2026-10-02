#!/usr/bin/env python3
"""
Combine pairs of haploid samples into pseudo-diploid individuals.

Usage:
  recode_haploid_to_diploid.py <in.vcf.gz> <out.vcf> <pairs.tsv>

pairs.tsv (from make_pseudodiploid_pairs.py) has header "name hap1 hap2 pop";
each row becomes one output sample whose GT is the phased "<hap1 GT>|<hap2 GT>"
(the phase is known exactly, so LD is computed from the two haplotypes with
use_genotypes=False). Haploid samples not listed in pairs.tsv are dropped. If
either haplotype is missing, GT is ".|.".
Only the GT field is kept (FORMAT is rewritten to "GT").
"""
import gzip
import sys

inp = sys.argv[1]    # e.g. polarized.vcf.gz
out = sys.argv[2]    # e.g. polarized.diploidGT.vcf  (NOT .gz)
pairs_tsv = sys.argv[3]

pairs = []  # (name, hap1, hap2)
with open(pairs_tsv) as f:
    next(f)  # header
    for line in f:
        name, hap1, hap2, _pop = line.rstrip("\n").split("\t")
        pairs.append((name, hap1, hap2))


def haploid_allele(gt: str) -> str:
    if gt in (".", "./.", ".|."):
        return "."
    if "/" in gt or "|" in gt:
        raise ValueError(f"expected haploid GT, got {gt!r}")
    return gt


with gzip.open(inp, "rt") as fin, open(out, "wt") as fout:
    pair_cols = None
    for line in fin:
        if line.startswith("##"):
            fout.write(line)
            continue
        if line.startswith("#CHROM"):
            header = line.rstrip("\n").split("\t")
            col = {s: i for i, s in enumerate(header)}
            missing = [s for _, a, b in pairs for s in (a, b) if s not in col]
            if missing:
                raise SystemExit(f"samples in pairs file not in VCF: {missing}")
            pair_cols = [(col[a], col[b]) for _, a, b in pairs]
            fout.write("\t".join(header[:9] + [name for name, _, _ in pairs]) + "\n")
            continue

        fields = line.rstrip("\n").split("\t")
        gt_idx = fields[8].split(":").index("GT")
        out_gts = []
        for ia, ib in pair_cols:
            a = haploid_allele(fields[ia].split(":")[gt_idx])
            b = haploid_allele(fields[ib].split(":")[gt_idx])
            out_gts.append(".|." if "." in (a, b) else f"{a}|{b}")

        fout.write("\t".join(fields[:8] + ["GT"] + out_gts) + "\n")

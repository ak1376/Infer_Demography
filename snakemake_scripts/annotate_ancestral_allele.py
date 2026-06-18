#!/usr/bin/env python3
"""
annotate_ancestral_allele.py

Reads a VCF and an ancestral-allele FASTA (DPGP MLancestor),
adds an AA INFO field to each SNP, and writes a new VCF to a
user-specified output location.  The input VCF is never touched.

For each SNP the ancestral base is looked up in the FASTA:
  - ancestral == REF  -> keep site, AA=REF  (ALT is derived)
  - ancestral == ALT  -> keep site, AA=ALT  (REF is derived; flip downstream)
  - ancestral == N or matches neither allele -> skip site

Output is bgzipped and tabix-indexed.

Usage:
  python annotate_ancestral_allele.py \
      --input-vcf  drosophila_data/data/Chr2L.vcf.gz \
      --ancestral-fasta /sietch_colab/data_share/drosophila_melanogaster/dpgp_ancestor/chr2L.q30.fa \
      --output-vcf real_data_analysis/data/drosophila/Chr2L.polarized.vcf.gz
"""

import argparse
import gzip
import subprocess
from pathlib import Path


def load_ancestral_seq(fasta_path: str) -> str:
    """Return the ancestral sequence as a 0-indexed uppercase string."""
    print(f"Loading ancestral FASTA: {fasta_path}", flush=True)
    parts = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    seq = "".join(parts)
    print(f"  {len(seq):,} bp loaded", flush=True)
    return seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-vcf",       type=Path, required=True)
    p.add_argument("--ancestral-fasta", type=Path, required=True)
    p.add_argument("--output-vcf",      type=Path, required=True,
                   help="Must end in .vcf.gz")
    args = p.parse_args()

    args.output_vcf.parent.mkdir(parents=True, exist_ok=True)

    anc_seq = load_ancestral_seq(str(args.ancestral_fasta))

    tmp_vcf = args.output_vcf.parent / (args.output_vcf.name.replace(".vcf.gz", ".tmp.vcf"))

    opener = gzip.open if str(args.input_vcf).endswith(".gz") else open

    total = kept = flipped = skipped_n = skipped_neither = 0

    with opener(str(args.input_vcf), "rt") as fin, open(tmp_vcf, "w") as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue
            if line.startswith("#CHROM"):
                fout.write('##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral allele (DPGP MLancestor)">\n')
                fout.write(line)
                continue

            fields = line.rstrip("\n").split("\t")
            pos  = int(fields[1])
            ref  = fields[3]
            alt  = fields[4]
            total += 1

            anc = anc_seq[pos - 1] if pos - 1 < len(anc_seq) else "N"

            if anc == "N":
                skipped_n += 1
                continue
            elif anc == ref:
                aa_val = ref        # ALT is derived — no flip needed
            elif anc == alt:
                aa_val = alt        # REF is derived — flip needed downstream
                flipped += 1
            else:
                skipped_neither += 1
                continue

            kept += 1
            info = fields[7]
            fields[7] = f"AA={aa_val}" if info in (".", "") else f"{info};AA={aa_val}"
            fout.write("\t".join(fields) + "\n")

    print(f"\nResults:")
    print(f"  Total SNPs            : {total:>10,}")
    print(f"  Kept (polarizable)    : {kept:>10,}  ({100*kept/total:.1f}%)")
    print(f"    ancestral == ALT    : {flipped:>10,}  (these will be flipped)")
    print(f"  Skipped (anc = N)     : {skipped_n:>10,}")
    print(f"  Skipped (no match)    : {skipped_neither:>10,}")

    print(f"\nbgzipping ...", flush=True)
    subprocess.run(["bgzip", "-f", str(tmp_vcf)], check=True)
    Path(str(tmp_vcf) + ".gz").rename(args.output_vcf)

    print(f"tabix indexing ...", flush=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(args.output_vcf)], check=True)

    print(f"\nDone -> {args.output_vcf}")


if __name__ == "__main__":
    main()

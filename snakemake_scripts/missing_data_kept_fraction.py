#!/usr/bin/env python3
"""
Fraction of a region's variable sites that survived the upstream missing-data
filter, computed from the ORIGINAL (unfiltered) DPGP2 VCF.

The VCFs this pipeline starts from (drosophila_data/data/Chr*.vcf.gz) were made
by subsetting the original DPGP2 VCF (snp-sites output of the per-genome FASTAs)
to the study samples and dropping every site where any of them has an N --
which snp-sites codes as a "*" allele. Those positions are uncallable, so they
must not count towards the sequence length L used for theta. This script counts,
inside [start, end] and among the samples in --popfile:

  kept = sites variable among the samples, none of which has a "*" (N) call
  lost = sites variable among the samples' non-N calls, at least one sample N

and writes kept / (kept + lost) to --out (JSON, with the counts).

Usage:
  python snakemake_scripts/missing_data_kept_fraction.py \
      --original-vcf drosophila_data/dpgp2_vcf/dpgp2_Chr3L_snps.vcf \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --chrom Chr3L --start 447386 --end 18392988 \
      --out real_data_analysis/data/drosophila_trimmed/Chr3L/missing_data_kept_fraction.json
"""
import argparse
import gzip
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original-vcf", required=True, type=Path)
    ap.add_argument("--popfile", required=True, type=Path,
                    help="samples the upstream filter was applied to (first column)")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--start", type=int, default=None, help="1-based, inclusive (default: whole arm)")
    ap.add_argument("--end", type=int, default=None, help="1-based, inclusive (default: whole arm)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    samples = [l.split()[0] for l in open(args.popfile) if l.strip() and not l.startswith("#")]
    lo = args.start if args.start is not None else 0
    hi = args.end if args.end is not None else float("inf")

    opener = gzip.open if str(args.original_vcf).endswith(".gz") else open
    kept = lost = 0
    with opener(args.original_vcf, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                col = {s: i for i, s in enumerate(header)}
                missing = [s for s in samples if s not in col]
                if missing:
                    raise SystemExit(f"samples not in {args.original_vcf}: {missing}")
                idx = [col[s] for s in samples]
                continue
            f = line.rstrip("\n").split("\t", max(idx) + 1)   # fields up to the last sample used
            pos = int(f[1])
            if pos < lo or pos > hi:
                continue
            alts = f[4].split(",")
            star = str(alts.index("*") + 1) if "*" in alts else None
            has_n, calls = False, set()
            for i in idx:
                g = f[i].split(":", 1)[0]
                if g == star:
                    has_n = True
                else:
                    calls.add(g)
            if len(calls) < 2:
                continue                     # not variable among the called samples
            if has_n:
                lost += 1
            else:
                kept += 1

    frac = kept / (kept + lost) if kept + lost else 1.0
    out = {
        "chrom": args.chrom,
        "region": [args.start, args.end],
        "n_samples": len(samples),
        "variable_sites_no_missing": kept,
        "variable_sites_with_missing": lost,
        "missing_data_kept_fraction": frac,
        "source_vcf": str(args.original_vcf),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{args.chrom} [{args.start}, {args.end}]: kept {kept:,}, lost {lost:,} -> {frac:.5f}  -> {args.out}")


if __name__ == "__main__":
    main()

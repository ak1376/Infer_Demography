#!/usr/bin/env python3
"""
Build one arm's haploid input VCF from the ORIGINAL DPGP2 VCF, keeping sites
where some flies have missing data.

The original VCF (snp-sites output of the per-genome FASTAs) codes a fly with
an N at a site as a "*" allele. This script, inside [start, end] and for the
samples in --popfile:

  - writes each "*" call as a missing genotype (".")
  - keeps a site if it is variable among the flies that ARE called
    (at least two different non-"*" alleles), however many flies are missing
  - keeps REF as allele 0 and lists as ALT only the non-"*" alleles those
    flies carry, with genotypes renumbered to match
  - renames CHROM to --chrom and writes ##contig length = the region span
    (end - start + 1), so compute_unfolded_sfs.py's L is the region analyzed

Positions keep their original arm coordinates (the ancestral-FASTA lookup in
annotate_ancestral_allele.py relies on that). The original VCF is only read.
Output is bgzipped and tabix-indexed.

Usage:
  python snakemake_scripts/make_real_vcf.py \
      --original-vcf drosophila_data/dpgp2_vcf/dpgp2_Chr2R_snps.vcf \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --chrom Chr2R --start 6063980 --end 20322335 \
      --out real_data_analysis/data/real_vcf/Chr2R.trim6063980-20322335.vcf.gz
"""
import argparse
import gzip
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original-vcf", required=True, type=Path)
    ap.add_argument("--popfile", required=True, type=Path, help="samples to keep (first column)")
    ap.add_argument("--chrom", required=True, help="CHROM name to write (e.g. Chr2R)")
    ap.add_argument("--start", type=int, default=None, help="1-based, inclusive (default: whole arm)")
    ap.add_argument("--end", type=int, default=None, help="1-based, inclusive (default: whole arm)")
    ap.add_argument("--out", required=True, type=Path, help="must end in .vcf.gz")
    args = ap.parse_args()

    if args.out.resolve() == args.original_vcf.resolve():
        raise SystemExit("--out must not be the original VCF")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_vcf = args.out.with_name(args.out.name.replace(".vcf.gz", ".tmp.vcf"))

    samples = [l.split()[0] for l in open(args.popfile) if l.strip() and not l.startswith("#")]
    lo = args.start if args.start is not None else 1
    hi = args.end

    opener = gzip.open if str(args.original_vcf).endswith(".gz") else open
    written = with_missing = 0
    with opener(args.original_vcf, "rt") as fin, open(tmp_vcf, "w") as fout:
        for line in fin:
            if line.startswith("##"):
                if hi is None and line.startswith("##contig="):
                    hi = int(line.split("length=")[1].split(">")[0].split(",")[0])
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                col = {s: i for i, s in enumerate(header)}
                missing = [s for s in samples if s not in col]
                if missing:
                    raise SystemExit(f"samples not in {args.original_vcf}: {missing}")
                if hi is None:
                    raise SystemExit("no ##contig length in the original VCF; pass --end")
                idx = [col[s] for s in samples]
                fout.write("##fileformat=VCFv4.2\n")
                fout.write(f"##contig=<ID={args.chrom},length={hi - lo + 1}>\n")
                fout.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
                fout.write(f"##source=make_real_vcf.py from {args.original_vcf} "
                           f"({args.chrom}:{lo}-{hi}; '*' calls written as missing)\n")
                fout.write("\t".join(header[:9] + samples) + "\n")
                continue

            f = line.rstrip("\n").split("\t", max(idx) + 1)
            pos = int(f[1])
            if pos < lo or pos > hi:
                continue
            alleles = [f[3]] + f[4].split(",")
            gts = [f[i].split(":", 1)[0] for i in idx]
            called = {g for g in gts if g != "." and alleles[int(g)] != "*"}
            if len(called) < 2:
                continue                      # not variable among the called flies
            # REF stays allele 0; ALT = the non-"*" alleles these flies carry
            used_alts = sorted(int(g) for g in called if g != "0")
            renum = {"0": "0", **{str(a): str(k + 1) for k, a in enumerate(used_alts)}}
            out_gts = [renum.get(g, ".") for g in gts]
            if "." in out_gts:
                with_missing += 1
            alt = ",".join(alleles[a] for a in used_alts)
            fout.write("\t".join([args.chrom, f[1], ".", f[3], alt, ".", ".", ".", "GT"] + out_gts) + "\n")
            written += 1

    subprocess.run(["bgzip", "-f", str(tmp_vcf)], check=True)
    Path(str(tmp_vcf) + ".gz").rename(args.out)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(args.out)], check=True)
    print(f"{args.chrom}:{lo}-{hi}: {written:,} variable sites, {with_missing:,} "
          f"({100 * with_missing / max(written, 1):.1f}%) with at least one fly missing -> {args.out}")


if __name__ == "__main__":
    main()

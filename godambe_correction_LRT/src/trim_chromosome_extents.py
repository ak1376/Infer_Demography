#!/usr/bin/env python3
"""
Trim a chromosome VCF down to its recombining interval, dropping
centromere- and telomere-proximal sequence.

Read-only on the input VCF: it is only ever read via `bcftools view`, never
modified or deleted. The trimmed region is always written to a NEW file.

Bounds match data/flies/melanogaster/clines/clineEndPts/namerica_trimmed/
recombining/{chrom}.extents (two whitespace-separated integers: start end).
Known values are baked in as a fallback (see KNOWN_EXTENTS) for chroms where
that .extents directory isn't available locally; pass --extents-dir or
--extents-file to read from the real source instead, or --start/--end to
override outright.

Usage:
  python src/trim_chromosome_extents.py \
      --vcf drosophila_data/data/Chr3L.vcf.gz --chrom Chr3L \
      --out real_data_analysis/data/trimmed_raw_vcf/Chr3L.trim447386-18392988.vcf.gz

  python src/trim_chromosome_extents.py \
      --vcf drosophila_data/data/ChrX.vcf.gz --chrom ChrX \
      --extents-dir data/flies/melanogaster/clines/clineEndPts/namerica_trimmed/recombining
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# Fallback bounds (namerica_trimmed/recombining .extents values), keyed by the
# chromosome name as it appears in this repo's VCFs (Chr2L, not chr2L).
KNOWN_EXTENTS = {
    "Chr2L": (844225, 19946732),
    "Chr2R": (6063980, 20322335),
    "Chr3L": (447386, 18392988),
    "Chr3R": (7940899, 27237549),
    "ChrX": (1036552, 20902578),
}

# chrom -> .extents filename, per the namerica_trimmed/recombining convention
# (lowercase, no capital-C "Chr" prefix).
EXTENTS_FILENAME = {
    "Chr2L": "chr2L.extents",
    "Chr2R": "chr2R.extents",
    "Chr3L": "chr3L.extents",
    "Chr3R": "chr3R.extents",
    "ChrX": "chrX.extents",
}


def read_extents(
    chrom: str,
    extents_dir: Path | None = None,
    extents_file: Path | None = None,
) -> tuple[int, int]:
    """Look up (start, end) for chrom, preferring an explicit .extents file."""
    if extents_file is not None:
        path = Path(extents_file)
    elif extents_dir is not None:
        fname = EXTENTS_FILENAME.get(chrom)
        if fname is None:
            raise ValueError(
                f"No known .extents filename for chrom {chrom!r}; pass --extents-file directly."
            )
        path = Path(extents_dir) / fname
    else:
        if chrom not in KNOWN_EXTENTS:
            raise ValueError(
                f"No bounds known for chrom {chrom!r}; pass --start/--end or --extents-file."
            )
        return KNOWN_EXTENTS[chrom]

    if not path.exists():
        raise FileNotFoundError(f".extents file not found: {path}")
    start_s, end_s = path.read_text().split()
    return int(start_s), int(end_s)


def trim_chromosome_vcf(vcf_in: Path, chrom: str, start: int, end: int, out_vcf: Path) -> None:
    """Restrict vcf_in to chrom:start-end, writing a NEW bgzipped + tabix-indexed
    VCF at out_vcf. vcf_in is opened read-only (bcftools view) and is never
    written to, moved, or deleted."""
    vcf_in = Path(vcf_in).resolve()
    out_vcf = Path(out_vcf).resolve()
    if out_vcf == vcf_in:
        raise ValueError("Refusing to write the trimmed VCF over the input VCF.")
    if not vcf_in.exists():
        raise FileNotFoundError(f"input VCF not found: {vcf_in}")
    out_vcf.parent.mkdir(parents=True, exist_ok=True)

    span = end - start + 1
    region = f"{chrom}:{start}-{end}"

    view = subprocess.Popen(["bcftools", "view", "-r", region, str(vcf_in)], stdout=subprocess.PIPE)
    sed = subprocess.Popen(
        [
            "sed",
            "-E",
            f"s/^##contig=<ID={chrom},length=[0-9]+>/##contig=<ID={chrom},length={span}>/",
        ],
        stdin=view.stdout,
        stdout=subprocess.PIPE,
    )
    view.stdout.close()
    with open(out_vcf, "wb") as fh:
        bgzip = subprocess.Popen(["bgzip", "-c"], stdin=sed.stdout, stdout=fh)
        sed.stdout.close()
        bgzip.communicate()
    sed.wait()
    view.wait()
    if view.returncode != 0 or sed.returncode != 0 or bgzip.returncode != 0:
        raise RuntimeError("bcftools | sed | bgzip pipeline failed while trimming.")

    subprocess.run(["tabix", "-f", "-p", "vcf", str(out_vcf)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True, type=Path, help="Input chromosome VCF (bgzipped). Never modified.")
    ap.add_argument("--chrom", required=True, help="Chromosome name as it appears in the VCF, e.g. Chr3L")
    ap.add_argument("--start", type=int, default=None, help="Region start (1-based, inclusive). Overrides extents lookup.")
    ap.add_argument("--end", type=int, default=None, help="Region end (1-based, inclusive). Overrides extents lookup.")
    ap.add_argument(
        "--extents-dir",
        type=Path,
        default=None,
        help="Directory of {chrom}.extents files, e.g. "
        "data/flies/melanogaster/clines/clineEndPts/namerica_trimmed/recombining",
    )
    ap.add_argument(
        "--extents-file",
        type=Path,
        default=None,
        help="Path to a single .extents file (start end) for --chrom; overrides --extents-dir",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output VCF path. Default: alongside --vcf as <chrom>.trim<start>-<end>.vcf.gz",
    )
    args = ap.parse_args()

    if args.start is not None and args.end is not None:
        start, end = args.start, args.end
    else:
        start, end = read_extents(args.chrom, args.extents_dir, args.extents_file)

    out = args.out or args.vcf.parent / f"{args.chrom}.trim{start}-{end}.vcf.gz"

    print(f"Trimming {args.vcf} ({args.chrom}) to {start}-{end} (span {end - start + 1:,} bp)")
    print(f"  input  (read-only): {args.vcf}")
    print(f"  output (new file) : {out}")
    trim_chromosome_vcf(args.vcf, args.chrom, start, end, out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import gzip
import sys

inp = sys.argv[1]   # e.g. Chr2L.vcf.gz
out = sys.argv[2]   # e.g. Chr2L.diploidGT.vcf  (NOT .gz)

def recode(gt: str) -> str:
    # haploid "0" -> "0/0", "1" -> "1/1", "." -> "./."
    if gt == ".":
        return "./."
    if gt in ("0", "1"):
        return f"{gt}/{gt}"
    return gt  # already diploid or something else

with gzip.open(inp, "rt") as fin, open(out, "wt") as fout:
    for line in fin:
        if line.startswith("#"):
            fout.write(line)
            continue

        fields = line.rstrip("\n").split("\t")

        # FORMAT column (index 8). Your file is just "GT", but keep this general.
        fmt_keys = fields[8].split(":")
        try:
            gt_idx = fmt_keys.index("GT")
        except ValueError:
            # no GT field -> write unchanged
            fout.write("\t".join(fields) + "\n")
            continue

        for i in range(9, len(fields)):
            parts = fields[i].split(":")
            if gt_idx < len(parts):
                parts[gt_idx] = recode(parts[gt_idx])
                fields[i] = ":".join(parts)

        fout.write("\t".join(fields) + "\n")

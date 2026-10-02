#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/filter_windows_near_breakpoints.py

"""
Given a set of per-window VCFs (from Snakefile rule tile_nonoverlap), decide
which windows survive a buffer around this arm's inversion breakpoints
(godambe_correction_LRT/src/inversion_breakpoints.py) and write the surviving
(and dropped) window indices to a JSON file.

Read-only on the input VCFs. Individuals/samples are never touched -- this
only decides which whole windows are kept for the downstream
bootstrap/aggregate step (Snakefile rule pool_ld_stats_excl_breakpoints).

Each window's actual span is read from its own VCF (first/last POS via
bcftools), not assumed from the tiling formula, so this is correct regardless
of how the window was tiled.

Usage:
  python filter_windows_near_breakpoints.py \
      --arm Chr3L --buffer-bp 3000000 \
      --window-vcfs .../window_0.vcf.gz .../window_1.vcf.gz ... \
      --out-json .../kept_windows_excl3000000bp.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inversion_breakpoints import overlaps_any_breakpoint  # noqa: E402


def window_index(vcf_path: Path) -> int:
    m = re.search(r"window_(\d+)\.vcf\.gz$", str(vcf_path))
    if not m:
        raise ValueError(f"Can't parse window index from {vcf_path}")
    return int(m.group(1))


def vcf_bounds(vcf_path: Path) -> tuple[int, int]:
    """First/last POS in the VCF (bcftools; matches split_vcf_windows.py's own
    get_vcf_bounds so "window span" means the same thing everywhere)."""
    first = int(subprocess.check_output(
        f"bcftools query -f '%POS\\n' '{vcf_path}' | head -n 1", shell=True).strip())
    last = int(subprocess.check_output(
        f"bcftools query -f '%POS\\n' '{vcf_path}' | tail -n 1", shell=True).strip())
    return first, last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--buffer-bp", required=True, type=int)
    ap.add_argument("--window-vcfs", required=True, nargs="+", type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    kept, dropped = [], []
    for vcf in sorted(args.window_vcfs, key=window_index):
        i = window_index(vcf)
        start, end = vcf_bounds(vcf)
        near = overlaps_any_breakpoint(start, end, args.arm, args.buffer_bp)
        (dropped if near else kept).append(i)

    result = {
        "arm": args.arm,
        "buffer_bp": args.buffer_bp,
        "n_total": len(kept) + len(dropped),
        "kept": kept,
        "dropped": dropped,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.arm}: kept {len(kept)}/{len(kept) + len(dropped)} windows "
          f"(buffer={args.buffer_bp:,} bp) -> {args.out_json}")


if __name__ == "__main__":
    main()

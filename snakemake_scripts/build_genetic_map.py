#!/usr/bin/env python3
"""
build_genetic_map.py

Turn the Comeron (2012) 100-kb crossover-rate table into a per-chromosome
cumulative genetic map in the format the LD pipeline reads:

    pos      Map(cM)
    0        0
    100000   0.000
    ...
    400000   0.023
    500000   0.047
    ...

This is exactly the notebook one-liner
    df["cum_cM"] = (df["rate_cM_per_Mb"] * 0.1).cumsum()
made reproducible: for each 100-kb window it adds rate*0.1 cM to a running
total, and writes the running total at each window boundary. Both LD backends
(moments.LD.Parsing and pg_gpu.moments_ld) interpolate SNP genetic positions
from this file, so a dense map reproduces the real recombination landscape.

The R5 tab is a *wide* layout: one 3-column block per chromosome arm, each
block holding 'Chromosome /Arm', 'Midpoint 100kb window (bp)',
'c (cM/Mb/female meiosis)'. Parsed with the standard library only (zipfile +
ElementTree) so no pandas/openpyxl is required.

Example
-------
    python build_genetic_map.py \
        --xlsx  .../Comeron_100kb_R5_R6.xlsx \
        --chrom Chr2L \
        --out   .../Chr2L/genetic_map.txt
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

ARM_LABEL = "chromosome /arm"
MID_LABEL = "midpoint"
RATE_LABEL = "cm/mb"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", required=True, type=Path)
    p.add_argument("--chrom", required=True, help="e.g. Chr2L (must match VCF naming)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--sheet-substr",
        default="R5",
        help="Substring identifying the sheet to read (default 'R5' = dm3 coords)",
    )
    p.add_argument("--window-bp", type=int, default=100_000)
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Multiply all rates by this factor. Comeron rates are cM/Mb per "
            "*female* meiosis; use 0.5 for a sex-averaged per-generation map."
        ),
    )
    return p.parse_args()


def col_to_idx(ref: str) -> int:
    """'A'->1, 'B'->2, ..., 'AA'->27 (from a cell ref like 'AB12')."""
    letters = re.match(r"[A-Z]+", ref).group()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx


def normalise_chrom(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    if low.startswith("chr"):
        low = low[3:]
    return "Chr" + low.upper()


def load_sheet_grid(xlsx: Path, sheet_substr: str):
    """Return {(col_idx, row_num): value} for the first sheet whose name
    contains sheet_substr (case-insensitive)."""
    z = zipfile.ZipFile(xlsx)

    sst = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in ss.iter(f"{{{NS}}}si"):
            sst.append("".join(t.text or "" for t in si.iter(f"{{{NS}}}t")))

    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid2tgt = {r.get("Id"): r.get("Target") for r in rels}
    chosen = None
    for s in wb.iter(f"{{{NS}}}sheet"):
        if sheet_substr.lower() in s.get("name", "").lower():
            chosen = rid2tgt[s.get(f"{{{RNS}}}id")]
            break
    if chosen is None:
        raise SystemExit(f"No sheet name contains '{sheet_substr}' in {xlsx}")

    root = ET.fromstring(z.read("xl/" + chosen.lstrip("/")))
    grid = {}
    for row in root.iter(f"{{{NS}}}row"):
        for c in row.iter(f"{{{NS}}}c"):
            v = c.find(f"{{{NS}}}v")
            if v is None:
                continue
            ref = c.get("r")
            ci = col_to_idx(ref)
            rn = int(re.search(r"\d+", ref).group())
            grid[(ci, rn)] = sst[int(v.text)] if c.get("t") == "s" else v.text
    return grid


def find_block(grid, chrom):
    """Locate (mid_col, rate_col, first_data_row) for the requested chromosome."""
    target = normalise_chrom(chrom)
    header_rows = {
        rn
        for (ci, rn), val in grid.items()
        if isinstance(val, str) and val.strip().lower() == ARM_LABEL
    }
    if not header_rows:
        raise SystemExit("Could not find any 'Chromosome /Arm' header cell.")
    header_row = min(header_rows)
    data_row0 = header_row + 1

    for (ci, rn), val in grid.items():
        if rn != header_row:
            continue
        if not (isinstance(val, str) and val.strip().lower() == ARM_LABEL):
            continue
        arm_col, mid_col, rate_col = ci, ci + 1, ci + 2
        h_mid = str(grid.get((mid_col, header_row), "")).lower()
        h_rate = str(grid.get((rate_col, header_row), "")).lower()
        if MID_LABEL not in h_mid or RATE_LABEL not in h_rate:
            continue
        arm_name = grid.get((arm_col, data_row0))
        if arm_name is not None and normalise_chrom(arm_name) == target:
            return mid_col, rate_col, data_row0
    raise SystemExit(f"Chromosome {chrom} not found in sheet.")


def main():
    args = parse_args()
    grid = load_sheet_grid(args.xlsx, args.sheet_substr)
    mid_col, rate_col, data_row0 = find_block(grid, args.chrom)

    max_row = max(rn for _, rn in grid)
    windows = []
    for rn in range(data_row0, max_row + 1):
        m = grid.get((mid_col, rn))
        r = grid.get((rate_col, rn))
        if m is None or r is None or str(m).strip() == "":
            continue
        windows.append((int(float(m)), float(r) * args.scale))
    if not windows:
        raise SystemExit(f"No data rows found for {args.chrom}.")

    windows.sort()
    half = args.window_bp // 2
    width_mb = args.window_bp / 1_000_000.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cum_cM = 0.0
    with open(args.out, "w") as out:
        out.write("pos\tMap(cM)\n")
        out.write(f"{max(windows[0][0] - half, 0)}\t0\n")  # left edge of first window
        for mid, rate in windows:
            cum_cM += rate * width_mb  # <-- the running sum ("integration")
            out.write(f"{mid + half}\t{cum_cM:.6f}\n")

    print(
        f"[map] {args.chrom}: {len(windows)} windows, "
        f"total {cum_cM:.2f} cM  ->  {args.out}"
    )


if __name__ == "__main__":
    main()

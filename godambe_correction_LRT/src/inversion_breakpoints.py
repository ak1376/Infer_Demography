#!/usr/bin/env python3
"""
Shared cosmopolitan/endemic inversion breakpoint table for D. melanogaster,
FlyBase Release 5 (dm3) coordinates -- same coordinate system as this
project's Drosophila VCFs and the Comeron_100kb map's "R5" sheet.

Single source of truth: both plot_inversion_recomb.py (visualization) and the
breakpoint-exclusion filter used by the momentsLD real-data pipeline
(godambe_correction_LRT/Snakefile) import from here, so the two never drift.
"""

from __future__ import annotations

# name -> (chrom, distal_bp, proximal_bp), R5/dm3 coordinates.
INVERSIONS = {
    "In(2L)t": ("Chr2L", 2225744, 13154180),
    "In(2R)NS": ("Chr2R", 16163839, 11278659),
    "In(3R)K": ("Chr3R", 21966092, 7576289),
    "In(3R)Mo": ("Chr3R", 24857019, 17232639),
    "In(3R)P": ("Chr3R", 20569732, 12257931),
    "In(3L)P": ("Chr3L", 3173046, 16301941),
    "In(1)A": ("ChrX", 13519769, 19473361),
    "In(1)Be": ("ChrX", 17722945, 19487744),
}

# Full arm length, R5/dm3 (from this project's VCF/FASTA headers -- see
# drosophila_data/data/*.vcf.gz ##contig lines).
CHROM_LENGTH = {
    "Chr2L": 23011544,
    "Chr2R": 21146708,
    "Chr3L": 24543557,
    "Chr3R": 27905053,
    "ChrX": 22422827,
}


def breakpoints_for_arm(chrom: str) -> list[int]:
    """All inversion breakpoint positions (bp) on this arm, distal+proximal
    pooled across every inversion that has one there (e.g. 3 arms -> 6 bp
    on Chr3R)."""
    bps = []
    for _, (c, distal, proximal) in INVERSIONS.items():
        if c == chrom:
            bps.append(distal)
            bps.append(proximal)
    return bps


def overlaps_any_breakpoint(start: int, end: int, chrom: str, buffer_bp: int) -> bool:
    """True if [start, end] comes within buffer_bp of any breakpoint on chrom
    (i.e. overlaps [bp - buffer_bp, bp + buffer_bp] for some breakpoint bp)."""
    return any(
        start < bp + buffer_bp and end > bp - buffer_bp
        for bp in breakpoints_for_arm(chrom)
    )

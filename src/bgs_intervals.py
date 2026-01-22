# src/bgs_intervals.py
"""
Background selection intervals and DFE application
Defined as functions to build and apply DFE intervals to stdpopsim Contigs.
"""

from typing import Dict, Optional
import numpy as np
import stdpopsim as sps


def _sanitize_nonoverlap(intervals: np.ndarray, L: int) -> np.ndarray:
    if intervals.size == 0:
        return intervals
    iv = intervals[np.argsort(intervals[:, 0])]
    out = []
    prev_end = -1
    for s, e in iv:
        s = int(max(0, min(s, L)))
        e = int(max(0, min(e, L)))
        if e <= s:
            continue
        if s < prev_end:
            continue
        out.append((s, e))
        prev_end = e
    return np.array(out, dtype=int) if out else np.empty((0, 2), dtype=int)


def _build_tiling_intervals(
    L: int, exon_bp: int, tile_bp: int, jitter_bp: int = 0
) -> np.ndarray:
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    if jitter_bp > 0 and len(starts) > 0:
        rng = np.random.default_rng()
        jitter = rng.integers(-jitter_bp, jitter_bp + 1, size=len(starts))
        starts = np.clip(starts + jitter, 0, max(0, L - exon_bp))
    ends = np.minimum(starts + int(exon_bp), L).astype(int)
    iv = np.column_stack([starts, ends])
    return _sanitize_nonoverlap(iv, L)


def _intervals_from_coverage(
    L: int, exon_bp: int, coverage: float, jitter_bp: int = 0
) -> np.ndarray:
    """coverage in [0,1]. If 0 → empty; if 1 → whole contig; else tiling to approximate coverage."""
    if coverage <= 0:
        return np.empty((0, 2), dtype=int)
    if coverage >= 1.0:
        return np.array([[0, int(L)]], dtype=int)
    # spacing chosen so expected selected fraction ≈ coverage
    tile_bp = max(int(exon_bp), int(round(exon_bp / float(max(coverage, 1e-12)))))
    return _build_tiling_intervals(int(L), int(exon_bp), tile_bp, jitter_bp=jitter_bp)


def _contig_from_cfg(cfg: Dict, sel: Dict):
    """
    Synthetic-only contig builder.
    Builds a stdpopsim Contig of length = cfg["genome_length"],
    with user-specified mutation_rate and recombination_rate.
    """
    sp = sps.get_species(sel.get("species", "HomSap"))

    L = float(cfg["genome_length"])
    mu = float(cfg["mutation_rate"]) if "mutation_rate" in cfg else None
    r = float(cfg["recombination_rate"]) if "recombination_rate" in cfg else None

    try:
        # Newer stdpopsim supports recombination_rate kwarg
        return sp.get_contig(
            chromosome=None,
            length=L,
            mutation_rate=mu,
            recombination_rate=r,
        )
    except TypeError:
        # Older stdpopsim doesn’t accept recombination_rate
        if r is not None:
            print(
                "[warn] This stdpopsim version ignores custom recombination_rate; "
                "using species default instead."
            )
        return sp.get_contig(
            chromosome=None,
            length=L,
            mutation_rate=mu,
        )


# CHANGED: now supports optional coverage tiling across the contig
def _apply_dfe_intervals(
    contig, sel: Dict, sampled_coverage: Optional[float] = None
) -> Dict[str, float]:
    """
    Attach DFE over intervals determined by:
      1) sampled_coverage (takes precedence; may be percent >1 or fraction <=1),
      2) sel['coverage_fraction'] or sel['coverage_percent'],
      3) sel['tile_bp'], or
      4) whole contig by default.

    Returns summary {selected_bp, selected_frac}.
    """
    sp = sps.get_species(sel.get("species", "HomSap"))
    dfe = sp.get_dfe(sel.get("dfe_id", "Gamma_K17"))

    # robust length getter
    L = int(
        getattr(contig, "length", getattr(contig, "recombination_map").sequence_length)
    )

    exon_bp = int(sel.get("exon_bp", 200))
    jitter_bp = int(sel.get("jitter_bp", 0))

    # 1) sampled_coverage overrides config if provided
    cov_frac = None
    if sampled_coverage is not None:
        # Accept either fraction in [0,1] or percent in (1,100]
        cov_frac = float(sampled_coverage)
        if cov_frac > 1.0:  # assume user passed a percent
            cov_frac = cov_frac / 100.0

    # 2) fall back to config coverage
    if cov_frac is None:
        if "coverage_fraction" in sel:
            cov_frac = float(sel["coverage_fraction"])
        elif "coverage_percent" in sel:
            cov_frac = float(sel["coverage_percent"]) / 100.0

    # Build intervals
    if cov_frac is not None:
        intervals = _intervals_from_coverage(L, exon_bp, cov_frac, jitter_bp=jitter_bp)
    elif "tile_bp" in sel and sel["tile_bp"] is not None:
        intervals = _build_tiling_intervals(
            L, exon_bp, int(sel["tile_bp"]), jitter_bp=jitter_bp
        )
    else:
        intervals = np.array([[0, L]], dtype=int)

    if intervals.size > 0:
        contig.add_dfe(intervals=intervals, DFE=dfe)

    selected_bp = int(
        np.sum((intervals[:, 1] - intervals[:, 0])) if intervals.size else 0
    )
    return dict(
        selected_bp=selected_bp,
        selected_frac=(selected_bp / float(L) if L > 0 else 0.0),
    )

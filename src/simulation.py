from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import demes
import msprime
import numpy as np
import stdpopsim as sps
import tskit
import moments


# ──────────────────────────────────
# Minimal helpers
# ──────────────────────────────────

class _ModelFromDemes(sps.DemographicModel):
    """Wrap a demes.Graph so stdpopsim engines can simulate it (bottleneck, drosophila)."""
    def __init__(self, g: demes.Graph, model_id: str = "custom_from_demes", desc: str = "custom demes"):
        model = msprime.Demography.from_demes(g)
        super().__init__(id=model_id, description=desc, long_description=desc, model=model, generation_time=1)


# Leaf-first stdpopsim models for SLiM (avoid p0=ANC extinction at split)
class _IM_Symmetric(sps.DemographicModel):
    """
    Isolation-with-migration, symmetric: YRI <-> CEU with rate m; split at time T from ANC.
    Populations are added as leaves first so p0/p1 are YRI/CEU (not ANC), avoiding zero-size errors.
    """
    def __init__(self, N0, N1, N2, T, m):
        dem = msprime.Demography()
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))
        m = float(m)
        dem.set_migration_rate(source="YRI", dest="CEU", rate=m)
        dem.set_migration_rate(source="CEU", dest="YRI", rate=m)
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])
        super().__init__(
            id="IM_sym",
            description="Isolation-with-migration, symmetric",
            long_description="ANC splits at T into YRI and CEU; symmetric migration m.",
            model=dem,
            generation_time=1,
        )


class _IM_Asymmetric(sps.DemographicModel):
    """Isolation-with-migration, asymmetric: YRI→CEU rate m12; CEU→YRI rate m21."""
    def __init__(self, N0, N1, N2, T, m12, m21):
        dem = msprime.Demography()

        # ✅ Add leaves first so that p0/p1 are extant pops, not ANC
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))

        # asymmetric migration
        dem.set_migration_rate("YRI", "CEU", float(m12))
        dem.set_migration_rate("CEU", "YRI", float(m21))

        # split backward in time
        dem.add_population_split(time=float(T),
                                 ancestral="ANC",
                                 derived=["YRI", "CEU"])

        super().__init__(
            id="IM_asym",
            description="Isolation-with-migration, asymmetric",
            long_description=(
                "ANC splits at T into YRI and CEU; asymmetric migration m12 and m21."
            ),
            model=dem,
            generation_time=1,
        )


# ──────────────────────────────────
# NEW: interval helpers for coverage-based tiling
# ──────────────────────────────────

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

def _build_tiling_intervals(L: int, exon_bp: int, tile_bp: int, jitter_bp: int = 0) -> np.ndarray:
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    if jitter_bp > 0 and len(starts) > 0:
        rng = np.random.default_rng()
        jitter = rng.integers(-jitter_bp, jitter_bp + 1, size=len(starts))
        starts = np.clip(starts + jitter, 0, max(0, L - exon_bp))
    ends = np.minimum(starts + int(exon_bp), L).astype(int)
    iv = np.column_stack([starts, ends])
    return _sanitize_nonoverlap(iv, L)

def _intervals_from_coverage(L: int, exon_bp: int, coverage: float, jitter_bp: int = 0) -> np.ndarray:
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
    r  = float(cfg["recombination_rate"]) if "recombination_rate" in cfg else None

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
            print("[warn] This stdpopsim version ignores custom recombination_rate; "
                  "using species default instead.")
        return sp.get_contig(
            chromosome=None,
            length=L,
            mutation_rate=mu,
        )

# CHANGED: now supports optional coverage tiling across the contig
def _apply_dfe_intervals(contig, sel: Dict, sampled_coverage: Optional[float] = None) -> Dict[str, float]:
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
    L = int(getattr(contig, "length", getattr(contig, "recombination_map").sequence_length))

    exon_bp   = int(sel.get("exon_bp", 200))
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
        intervals = _build_tiling_intervals(L, exon_bp, int(sel["tile_bp"]), jitter_bp=jitter_bp)
    else:
        intervals = np.array([[0, L]], dtype=int)

    if intervals.size > 0:
        contig.add_dfe(intervals=intervals, DFE=dfe)

    selected_bp = int(np.sum((intervals[:, 1] - intervals[:, 0])) if intervals.size else 0)
    return dict(selected_bp=selected_bp, selected_frac=(selected_bp / float(L) if L > 0 else 0.0))


# ──────────────────────────────────
# Your demography builders (demes)
# ──────────────────────────────────

def bottleneck_model(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    b = demes.Builder()
    b.add_deme(
        "ANC",
        epochs=[
            dict(start_size=float(sampled["N0"]),            end_time=float(sampled["t_bottleneck_start"])),
            dict(start_size=float(sampled["N_bottleneck"]), end_time=float(sampled["t_bottleneck_end"])),
            dict(start_size=float(sampled["N_recover"]),    end_time=0),
        ],
    )
    return b.resolve()


def split_isolation_model(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """Split + symmetric low migration (YRI/CEU)."""
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))
    T  = float(sampled.get("T_split", sampled.get("t_split")))
    # accept MANY possible keys; if both directions provided, average them
    m_keys = [
        "m", "m_sym", "m12", "m21", "m_YRI_CEU", "m_CEU_YRI"
    ]
    vals = [float(sampled[k]) for k in m_keys if k in sampled]
    m = float(np.mean(vals)) if vals else 0.0

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme("YRI", ancestors=["ANC"], epochs=[dict(start_size=N1)])
    b.add_deme("CEU", ancestors=["ANC"], epochs=[dict(start_size=N2)])
    if m > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m)
        b.add_migration(source="CEU", dest="YRI", rate=m)
    return b.resolve()


def split_migration_model(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Split + asymmetric migration (two rates).
    Deme names: 'YRI' and 'CEU'.
    """
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))
    T  = float(sampled.get("T_split", sampled.get("t_split")))
    m12 = float(sampled.get("m_YRI_CEU", sampled.get("m12", sampled.get("m", 0.0))))
    m21 = float(sampled.get("m_CEU_YRI", sampled.get("m21", sampled.get("m", 0.0))))

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme("YRI", ancestors=["ANC"], epochs=[dict(start_size=N1)])
    b.add_deme("CEU", ancestors=["ANC"], epochs=[dict(start_size=N2)])
    if m12 > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m12)
    if m21 > 0:
        b.add_migration(source="CEU", dest="YRI", rate=m21)
    return b.resolve()


def drosophila_three_epoch(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Two-pop Drosophila-style model:
      ANC (size N0) → split at T_AFR_EUR_split → AFR (AFR_recover)
      and EUR with a bottleneck at T_EUR_expansion then recovery to EUR_recover.
    Deme names: 'AFR' and 'EUR'.
    """
    N0              = float(sampled["N0"])
    AFR_recover     = float(sampled["AFR"])
    EUR_bottleneck  = float(sampled["EUR_bottleneck"])
    EUR_recover     = float(sampled["EUR_recover"])
    T_split         = float(sampled["T_AFR_EUR_split"])
    T_EUR_exp       = float(sampled["T_EUR_expansion"])

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T_split)])
    b.add_deme("AFR", ancestors=["ANC"], epochs=[dict(start_size=AFR_recover)])
    b.add_deme(
        "EUR", ancestors=["ANC"],
        epochs=[dict(start_size=EUR_bottleneck, end_time=T_EUR_exp),
                dict(start_size=EUR_recover, end_time=0)]
    )
    return b.resolve()


# ──────────────────────────────────
# Main entry: BGS only (SLiM via stdpopsim)
# ──────────────────────────────────

def simulation(sampled_params: Dict[str, float],
               model_type: str,
               experiment_config: Dict, sampled_coverage: float) -> Tuple[tskit.TreeSequence, demes.Graph]:
    """
    Background selection only. Uses your demes graph + stdpopsim SLiM engine.

    Config expects:
      - num_samples (keys must match deme names, e.g. {"YRI":10,"CEU":10})
      - mutation_rate, recombination_rate, genome_length (for synthetic contigs)
      - seed
      - selection:
          enabled: true
          species: "HomSap"
          dfe_id: "Gamma_K17"
          # (optional real chromosome window)
          chromosome: "chr21"
          left: 0
          right: 1e6
          genetic_map: "HapMapII_GRCh37"   # recommended for real chr
          # BGS tiling
          coverage_fraction: 0.2  # or coverage_percent: 20.0
          exon_bp: 200
          jitter_bp: 0
          # OR fixed spacing:
          # tile_bp: 5000
          # SLiM rescaling
          slim_scaling: 10.0
          slim_burn_in: 5.0
    """
    sel = experiment_config.get("selection") or {}
    if not sel.get("enabled", False):
        raise ValueError("This file only runs BGS. Set selection.enabled=true.")

    # 1) Build demes graph (kept for plotting/metadata)
    if model_type == "bottleneck":
        g = bottleneck_model(sampled_params, experiment_config)
    elif model_type == "split_isolation":
        g = split_isolation_model(sampled_params, experiment_config)   # symmetric m
    elif model_type == "split_migration":
        g = split_migration_model(sampled_params, experiment_config)   # asymmetric
    elif model_type == "drosophila_three_epoch":
        g = drosophila_three_epoch(sampled_params, experiment_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 2) Choose the SLiM-facing stdpopsim model
    if model_type == "split_isolation":
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1")))
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T  = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m  = float(sampled_params.get("m", sampled_params.get("m_YRI_CEU",
                 sampled_params.get("m12", sampled_params.get("m21", 0.0)))))
        model = _IM_Symmetric(N0, N1, N2, T, m)
    elif model_type == "split_migration":
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1")))
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T  = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m12 = float(sampled_params.get("m_YRI_CEU", sampled_params.get("m12", sampled_params.get("m", 0.0))))
        m21 = float(sampled_params.get("m_CEU_YRI", sampled_params.get("m21", sampled_params.get("m", 0.0))))
        model = _IM_Asymmetric(N0, N1, N2, T, m12, m21)
    else:
        # bottleneck & drosophila: Demes wrapper is fine
        model = _ModelFromDemes(g, model_id=f"custom_{model_type}", desc="custom demes")

    engine = experiment_config.get("engine", "slim")

    # 3) Contig + DFE intervals (coverage-aware)
    contig = _contig_from_cfg(experiment_config, sel)
    if engine == "slim":
        sel_summary = _apply_dfe_intervals(contig, sel, sampled_coverage=sampled_coverage)
    else:
        # msprime: no selection intervals
        sel_summary = dict(selected_bp=0, selected_frac=0.0)

    # 4) SLiM run
    samples = {k: int(v) for k, v in (experiment_config.get("num_samples") or {}).items()}
    base_seed = experiment_config.get("seed", None)
    if engine == "slim":
        eng = sps.get_engine("slim")
        ts = eng.simulate(
            model,
            contig,
            samples,
            slim_scaling_factor=float(sel.get("slim_scaling", 10.0)),
            slim_burn_in=float(sel.get("slim_burn_in", 5.0)),
            seed=base_seed,
        )
    elif engine == "msprime":
        eng = sps.get_engine("msprime")
        ts = eng.simulate(
            model,
            contig,
            samples,
            seed=base_seed,
        )
    else:
        raise ValueError("engine must be 'slim' or 'msprime'.")

    # Attach summary for the caller (via ts.metadata? we just return g and ts;
    # the CLI wrapper will record sel_summary into the JSON sidecar.)
    ts._bgs_selection_summary = sel_summary  # harmless, for downstream use

    return ts, g


# ──────────────────────────────────
# SFS utility
# ──────────────────────────────────

def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    """Build a moments.Spectrum using pops that have sampled individuals."""
    sample_sets: List[np.ndarray] = []
    pop_ids: List[str] = []
    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps):
            sample_sets.append(samps)
            meta = pop.metadata if isinstance(pop.metadata, dict) else {}
            pop_ids.append(meta.get("name", f"pop{pop.id}"))
    if not sample_sets:
        raise ValueError("No sampled populations found.")
    arr = ts.allele_frequency_spectrum(
        sample_sets=sample_sets, mode="site", polarised=True, span_normalise=False
    )
    sfs = moments.Spectrum(arr)
    sfs.pop_ids = pop_ids
    return sfs

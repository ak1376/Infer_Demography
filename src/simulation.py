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

    def __init__(
        self,
        g: demes.Graph,
        model_id: str = "custom_from_demes",
        desc: str = "custom demes",
    ):
        model = msprime.Demography.from_demes(g)
        super().__init__(
            id=model_id,
            description=desc,
            long_description=desc,
            model=model,
            generation_time=1,
        )


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
        # Forward-time: m12 = YRI→CEU, m21 = CEU→YRI
        # Backward-time encoding for msprime:
        dem.set_migration_rate(source="CEU", dest="YRI", rate=float(m12))  # encode YRI→CEU
        dem.set_migration_rate(source="YRI", dest="CEU", rate=float(m21))  # encode CEU→YRI


        # split backward in time
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])

        super().__init__(
            id="IM_asym",
            description="Isolation-with-migration, asymmetric",
            long_description=(
                "ANC splits at T into YRI and CEU; asymmetric migration m12 and m21."
            ),
            model=dem,
            generation_time=1,
        )

class _Bottleneck(sps.DemographicModel):
    """
    Single-population bottleneck implemented directly in msprime.Demography.
    Times in generations before present (t_start > t_end >= 0).
    """

    def __init__(self, N0, N_bottleneck, N_recover, t_bottleneck_start, t_bottleneck_end):
        t_start = float(t_bottleneck_start)
        t_end   = float(t_bottleneck_end)
        if not (t_start > t_end >= 0):
            raise ValueError("Require t_bottleneck_start > t_bottleneck_end >= 0.")

        dem = msprime.Demography()
        dem.add_population(name="ANC", initial_size=float(N0))

        # At t_start, drop to the bottleneck size
        dem.add_population_parameters_change(
            time=t_start, population="ANC", initial_size=float(N_bottleneck)
        )
        # At t_end, recover to N_recover (constant to present)
        dem.add_population_parameters_change(
            time=t_end, population="ANC", initial_size=float(N_recover)
        )

        super().__init__(
            id="bottleneck",
            description="Single-population bottleneck (N0 → N_bottleneck → N_recover).",
            long_description=(
                "One population with ancestral size N0 until t_bottleneck_start, "
                "then a bottleneck of size N_bottleneck until t_bottleneck_end, "
                "then constant size N_recover to the present."
            ),
            model=dem,
            generation_time=1,
        )

class _DrosophilaThreeEpoch(sps.DemographicModel):
    """
    Two-pop Drosophila-style three-epoch model.

    ANC (size N0) splits at T_AFR_EUR_split into:
      - AFR: constant size AFR (AFR_recover in your priors)
      - EUR: bottleneck of size EUR_bottleneck until T_EUR_expansion,
             then recovery to EUR_recover up to the present.

    Populations are added leaf-first so p0/p1 are AFR/EUR (not ANC),
    which plays nicely with SLiM’s population ordering.
    """

    def __init__(
        self,
        N0,
        AFR,
        EUR_bottleneck,
        EUR_recover,
        T_AFR_EUR_split,
        T_EUR_expansion,
    ):
        T_split = float(T_AFR_EUR_split)
        T_exp   = float(T_EUR_expansion)

        dem = msprime.Demography()

        # Leaf-first: extant pops first, then ANC
        dem.add_population(name="AFR", initial_size=float(AFR))
        dem.add_population(name="EUR", initial_size=float(EUR_bottleneck))
        dem.add_population(name="ANC", initial_size=float(N0))

        # EUR expansion (bottleneck -> recovery) at T_EUR_expansion
        dem.add_population_parameters_change(
            time=T_exp,
            population="EUR",
            initial_size=float(EUR_recover),
        )

        # Split backward in time at T_AFR_EUR_split: AFR/EUR merge into ANC
        dem.add_population_split(
            time=T_split,
            ancestral="ANC",
            derived=["AFR", "EUR"],
        )

        super().__init__(
            id="drosophila_three_epoch",
            description="Drosophila-style three-epoch AFR/EUR model",
            long_description=(
                "ANC (N0) until T_AFR_EUR_split, then split into AFR and EUR. "
                "AFR stays at AFR; EUR has a bottleneck (EUR_bottleneck) and "
                "expands at T_EUR_expansion to EUR_recover."
            ),
            model=dem,
            generation_time=1,
        )


class _SplitMigrationGrowth(sps.DemographicModel):
    """
    Custom model: CO/FR split from ANC, with growth in FR and asymmetric migration.
    """

    def __init__(self, N_CO, N_FR1, G_FR, N_ANC, m_CO_FR, m_FR_CO, T):
        # N_CO:    Effective population size of the Congolese population (constant).
        # N_FR1:   Effective population size of the French population at present (time 0).
        # G_FR:    Growth rate of the French population (exponential).
        # N_ANC:   Ancestral population size (before split).
        # m_CO_FR: Migration rate FROM Congolese TO French (forward time).
        #          (Fraction of French population replaced by Congolese migrants per generation).
        # m_FR_CO: Migration rate FROM French TO Congolese (forward time).
        #          (Fraction of Congolese population replaced by French migrants per generation).
        # T:       Time of split (generations ago).

        demogr = msprime.Demography()
        demogr.add_population(name="CO", initial_size=float(N_CO))
        demogr.add_population(name="FR", initial_size=float(N_FR1), growth_rate=float(G_FR))
        demogr.add_population(name="ANC", initial_size=float(N_ANC))
        
        # Migration Matrix M[j, k] is rate of lineages moving from j to k (backward time).
        # Lineage j->k (backward) implies Gene Flow k->j (forward).
        # We want m_CO_FR to be Forward CO->FR. This implies Lineages FR->CO.
        # So M[FR, CO] should be m_CO_FR. (Indices: CO=0, FR=1, ANC=2).
        # M[1, 0] = m_CO_FR.
        #
        # We want m_FR_CO to be Forward FR->CO. This implies Lineages CO->FR.
        # So M[CO, FR] should be m_FR_CO.
        # M[0, 1] = m_FR_CO.

        demogr.migration_matrix = np.array([
            [0,              float(m_FR_CO), 0],
            [float(m_CO_FR), 0,              0],
            [0,              0,              0]
        ])
        demogr.add_population_split(time=float(T), derived=["CO", "FR"], ancestral="ANC")

        super().__init__(
            id="split_migration_growth",
            description="Split with migration and growth (CO/FR)",
            long_description="Custom model with CO/FR split, FR growth, and asymmetric migration.",
            model=demogr,
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


# ──────────────────────────────────
# Your demography builders (demes)
# ──────────────────────────────────


def bottleneck_model(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    b = demes.Builder()
    b.add_deme(
        "ANC",
        epochs=[
            dict(
                start_size=float(sampled["N0"]),
                end_time=float(sampled["t_bottleneck_start"]),
            ),
            dict(
                start_size=float(sampled["N_bottleneck"]),
                end_time=float(sampled["t_bottleneck_end"]),
            ),
            dict(start_size=float(sampled["N_recover"]), end_time=0),
        ],
    )
    return b.resolve()


def split_isolation_model(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    """Split + symmetric low migration (YRI/CEU)."""
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))
    T = float(sampled.get("T_split", sampled.get("t_split")))
    # accept MANY possible keys; if both directions provided, average them
    m_keys = ["m", "m_sym", "m12", "m21", "m_YRI_CEU", "m_CEU_YRI"]
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


def split_migration_model(
    sampled: Dict[str, float]
) -> demes.Graph:
    """
    Split + asymmetric migration (two rates).
    Deme names: 'YRI' and 'CEU'.
    """
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))
    T = float(sampled.get("T_split", sampled.get("t_split")))
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


def drosophila_three_epoch(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    """
    Two-pop Drosophila-style model:
      ANC (size N0) → split at T_AFR_EUR_split → AFR (AFR_recover)
      and EUR with a bottleneck at T_EUR_expansion then recovery to EUR_recover.
    Deme names: 'AFR' and 'EUR'.
    """
    N0 = float(sampled["N0"])
    AFR_recover = float(sampled["AFR"])
    EUR_bottleneck = float(sampled["EUR_bottleneck"])
    EUR_recover = float(sampled["EUR_recover"])
    T_split = float(sampled["T_AFR_EUR_split"])
    T_EUR_exp = float(sampled["T_EUR_expansion"])

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T_split)])
    b.add_deme("AFR", ancestors=["ANC"], epochs=[dict(start_size=AFR_recover)])
    b.add_deme(
        "EUR",
        ancestors=["ANC"],
        epochs=[
            dict(start_size=EUR_bottleneck, end_time=T_EUR_exp),
            dict(start_size=EUR_recover, end_time=0),
        ],
    )
    return b.resolve()


def split_migration_growth_model(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    """
    Split + asymmetric migration + growth in FR.
    Deme names: 'CO' and 'FR'.
    
    Parameters:
    - N_CO:    Size of CO population (constant).
    - N_FR1:   Size of FR population at present (time 0).
    - G_FR:    Growth rate of FR population.
    - N_ANC:   Ancestral size.
    - m_CO_FR: Migration rate CO -> FR (forward time).
    - m_FR_CO: Migration rate FR -> CO (forward time).
    - T:       Split time (generations ago).
    """
    N_CO = float(sampled.get("N_CO", sampled.get("N1")))
    N_FR1 = float(sampled.get("N_FR1", sampled.get("N2")))
    N_ANC = float(sampled.get("N_ANC", sampled.get("N0")))
    m_CO_FR = float(sampled.get("m_CO_FR", 0.0))
    m_FR_CO = float(sampled.get("m_FR_CO", 0.0))
    T = float(sampled.get("T", sampled.get("T_split")))

    # Handle growth rate or start/end sizes
    # We need N_FR0 (size at split, time T) and N_FR1 (size at present, time 0)
    if "G_FR" in sampled:
        G_FR = float(sampled["G_FR"])
        # N(T) = N(0) * exp(-rT)
        N_FR0 = N_FR1 * np.exp(-G_FR * T)
    elif "N_FR0" in sampled:
        N_FR0 = float(sampled["N_FR0"])
        # G_FR not strictly needed for demes if we have start/end sizes, 
        # but good to have consistent logic if we wanted it.
    else:
        # Default no growth
        N_FR0 = N_FR1

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N_ANC, end_time=T)])
    b.add_deme("CO", ancestors=["ANC"], epochs=[dict(start_size=N_CO)])
    b.add_deme(
        "FR", 
        ancestors=["ANC"], 
        # Epoch goes from T (start) to 0 (end).
        # start_size is size at T (N_FR0). end_size is size at 0 (N_FR1).
        epochs=[dict(start_size=N_FR0, end_size=N_FR1)]
    )

    # Migration
    # m_CO_FR: Forward CO -> FR.
    # Demes uses forward-time semantics: source=Origin of genes, dest=Destination of genes.
    if m_CO_FR > 0:
        b.add_migration(source="CO", dest="FR", rate=m_CO_FR)
    
    # m_FR_CO: Forward FR -> CO.
    if m_FR_CO > 0:
        b.add_migration(source="FR", dest="CO", rate=m_FR_CO)

    return b.resolve()

def define_sps_model(model_type: str, g: demes.Graph, sampled_params: Dict[str, float]) -> sps.DemographicModel:
    """Create appropriate stdpopsim model for SLiM based on model type."""
    if model_type == "split_isolation":
        # Symmetric migration model - extract parameters
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1"))) 
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m_keys = ["m", "m_sym", "m12", "m21", "m_YRI_CEU", "m_CEU_YRI"]
        vals = [float(sampled_params[k]) for k in m_keys if k in sampled_params]
        m = float(np.mean(vals)) if vals else 0.0
        return _IM_Symmetric(N0, N1, N2, T, m)
    
    elif model_type == "split_migration":
        # Asymmetric migration model - extract parameters
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1")))
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m12 = float(sampled_params.get("m_YRI_CEU", sampled_params.get("m12", sampled_params.get("m", 0.0))))
        m21 = float(sampled_params.get("m_CEU_YRI", sampled_params.get("m21", sampled_params.get("m", 0.0))))
        return _IM_Asymmetric(N0, N1, N2, T, m12, m21)
    
    elif model_type == "drosophila_three_epoch":
        # Two-pop Drosophila three-epoch model
        N0             = float(sampled_params["N0"])
        AFR            = float(sampled_params["AFR"])
        EUR_bottleneck = float(sampled_params["EUR_bottleneck"])
        EUR_recover    = float(sampled_params["EUR_recover"])
        T_split        = float(sampled_params["T_AFR_EUR_split"])
        T_EUR_exp      = float(sampled_params["T_EUR_expansion"])

        return _DrosophilaThreeEpoch(
            N0,
            AFR,
            EUR_bottleneck,
            EUR_recover,
            T_split,
            T_EUR_exp,
            T_split,
            T_EUR_exp,
        )

    elif model_type == "split_migration_growth":
        N_CO = float(sampled_params.get("N_CO", sampled_params.get("N1")))
        N_FR1 = float(sampled_params.get("N_FR1", sampled_params.get("N2")))
        N_ANC = float(sampled_params.get("N_ANC", sampled_params.get("N0")))
        m_CO_FR = float(sampled_params.get("m_CO_FR", 0.0))
        m_FR_CO = float(sampled_params.get("m_FR_CO", 0.0))
        T = float(sampled_params.get("T", sampled_params.get("T_split")))

        if "G_FR" in sampled_params:
            G_FR = float(sampled_params["G_FR"])
        elif "N_FR0" in sampled_params:
            N_FR0 = float(sampled_params["N_FR0"])
            # G = ln(N(0)/N(T)) / T
            G_FR = np.log(N_FR1 / N_FR0) / T
        else:
            G_FR = 0.0

        return _SplitMigrationGrowth(N_CO, N_FR1, G_FR, N_ANC, m_CO_FR, m_FR_CO, T)

    else:
        # For bottleneck or any other demes-based custom model
        return _ModelFromDemes(g, model_id=f"custom_{model_type}", desc="custom demes")

# ──────────────────────────────────
# Main entry: BGS only (SLiM via stdpopsim)
# ──────────────────────────────────

def msprime_simulation(g: demes.Graph,
    experiment_config: Dict
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    samples = {pop_name: num_samples for pop_name, num_samples in experiment_config['num_samples'].items()}

    demog = msprime.Demography.from_demes(g)

    # Simulate ancestry for two populations (joint simulation)
    ts = msprime.sim_ancestry(
        samples=samples,  # Two populations
        demography=demog,
        sequence_length=experiment_config['genome_length'],
        recombination_rate=experiment_config['recombination_rate'],
        random_seed=experiment_config['seed'],
    )
    
    # Simulate mutations over the ancestry tree sequence
    ts = msprime.sim_mutations(ts, rate=experiment_config['mutation_rate'], random_seed=experiment_config['seed'])

    return ts, g

def stdpopsim_slim_simulation(g: demes.Graph,
    experiment_config: Dict, 
    sampled_coverage: float,
    model_type: str,
    sampled_params: Dict[str, float]
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    # 1) Pick model (wrap Demes for stdpopsim)
    model = define_sps_model(model_type, g, sampled_params)

    # 2) Build contig and apply DFE intervals
    sel = experiment_config.get("selection") or {}
    contig = _contig_from_cfg(experiment_config, sel)
    sel_summary = _apply_dfe_intervals(contig, sel, sampled_coverage=sampled_coverage)

    # 3) Samples
    samples = {k: int(v) for k, v in (experiment_config.get("num_samples") or {}).items()}
    base_seed = experiment_config.get("seed", None)

    # 4) Run SLiM via stdpopsim
    eng = sps.get_engine("slim")
    ts = eng.simulate(
        model,
        contig,
        samples,
        slim_scaling_factor=float(sel.get("slim_scaling", 10.0)),
        slim_burn_in=float(sel.get("slim_burn_in", 5.0)),
        seed=base_seed,
    )

    ts._bgs_selection_summary = sel_summary
    return ts, g

def simulation(
    sampled_params: Dict[str, float],
    model_type: str,
    experiment_config: Dict,
    sampled_coverage: Optional[float] = None,
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    # Build demes graph (kept for plotting/metadata)
    if model_type == "bottleneck":
        g = bottleneck_model(sampled_params, experiment_config)
    elif model_type == "split_isolation":
        g = split_isolation_model(sampled_params, experiment_config)  # symmetric m
    elif model_type == "split_migration":
        g = split_migration_model(sampled_params)  # asymmetric
    elif model_type == "drosophila_three_epoch":
        g = drosophila_three_epoch(sampled_params, experiment_config)
    elif model_type == "split_migration_growth":
        g = split_migration_growth_model(sampled_params, experiment_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    engine = str(experiment_config.get("engine", "")).lower()
    sel = experiment_config.get("selection") or {}

    if engine == "msprime":
        # Neutral path: no BGS, no coverage needed/used
        return msprime_simulation(g, experiment_config)

    if engine == "slim":
        # BGS path: require selection.enabled and a coverage
        if not sel.get("enabled", False):
            raise ValueError("engine='slim' requires selection.enabled=true in your config.")
        if sampled_coverage is None:
            raise ValueError("engine='slim' requires a non-None sampled_coverage (percent or fraction).")
        return stdpopsim_slim_simulation(g, experiment_config, sampled_coverage, model_type, sampled_params)

    raise ValueError("engine must be 'slim' or 'msprime'.")

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

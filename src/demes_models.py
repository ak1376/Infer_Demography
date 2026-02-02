# src/demes_models.py
"""
Demes demographic models
Defined as functions that accept a dict of sampled parameters (float) and return a demes.Graph.
"""

from __future__ import annotations
import math
from typing import Dict, Optional
import demes
import numpy as np


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


def split_migration_model(sampled: Dict[str, float]) -> demes.Graph:
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
    Demes equivalent of stdpopsim OutOfAfrica_2L06 (Li & Stephan 2006),
    but using your parameter names.

    Backward-time ordering should be:
      T_AFR_expansion > T_AFR_EUR_split > T_EUR_expansion > 0

    Interpretation (matching stdpopsim):
      - AFR has modern size AFR from present back to T_AFR_expansion,
        then ancestral size N0 older than that.
      - EUR has modern size EUR_recover from present back to T_EUR_expansion,
        then bottleneck size EUR_bottleneck from T_EUR_expansion back to the split.
      - At T_AFR_EUR_split, EUR merges into AFR (equivalently: EUR splits from AFR forward time).
    """

    N0 = float(sampled["N0"])  # ancestral AFR size (N_A1)
    AFR = float(sampled["AFR"])  # modern AFR size (N_A0)

    EUR_bottleneck = float(sampled["EUR_bottleneck"])  # N_E1
    EUR_recover = float(sampled["EUR_recover"])  # N_E0

    T_AFR_expansion = float(sampled["T_AFR_expansion"])  # t_A0
    T_split = float(sampled["T_AFR_EUR_split"])  # t_AE
    T_EUR_exp = float(sampled["T_EUR_expansion"])  # t_E1

    if not (T_AFR_expansion > T_split > T_EUR_exp > 0):
        raise ValueError(
            "Need T_AFR_expansion > T_AFR_EUR_split > T_EUR_expansion > 0, got: "
            f"{T_AFR_expansion=}, {T_split=}, {T_EUR_exp=}"
        )

    b = demes.Builder()

    # AFR is the root lineage (no explicit ANC deme needed)
    b.add_deme(
        "AFR",
        epochs=[
            dict(start_size=N0, end_time=T_AFR_expansion),  # older (ancestral)
            dict(start_size=AFR, end_time=0),               # recent (modern)
        ],
    )

    # EUR branches off AFR at the split time
    b.add_deme(
        "EUR",
        ancestors=["AFR"],
        start_time=T_split,
        epochs=[
            dict(start_size=EUR_bottleneck, end_time=T_EUR_exp),  # older part (near split)
            dict(start_size=EUR_recover, end_time=0),             # recent (modern)
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
        epochs=[dict(start_size=N_FR0, end_size=N_FR1)],
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


def OOA_three_pop_model_simplified(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    """
    Minimal three-pop Out-of-Africa model (YRI–CEU–CHB), Gutenkunst-style.

    Deme names:
      - 'YRI' : African population (YRI-like)
      - 'CEU' : European population
      - 'CHB' : East Asian population
      - internal ancestor demes: 'ANC', 'OOA'

    Parameters expected in `sampled` (all in generations/Ne units):
      - N_anc     / N0        : ancestral size
      - N_YRI     / N1        : present-day YRI size
      - N_OOA                 : size of the out-of-Africa bottleneck pop
      - N_CEU     / N2        : present-day CEU size
      - N_CHB                 : present-day CHB size
      - T_AFR_OOA / T1        : time of AFR vs OOA split (YRI vs non-African)
      - T_OOA_EU_AS / T2      : time of OOA -> (CEU, CHB) split

    Any missing parameters fall back to simple, reasonable defaults.
    """

    # --- sizes ---
    N_anc = float(sampled.get("N_anc", sampled.get("N0", 10_000.0)))
    N_YRI = float(sampled.get("N_YRI", sampled.get("N1", 14_000.0)))
    N_OOA = float(sampled.get("N_OOA", 2_000.0))
    N_CEU = float(sampled.get("N_CEU", sampled.get("N2", 5_000.0)))
    N_CHB = float(sampled.get("N_CHB", 5_000.0))

    # --- times (generations backwards from present) ---
    T_africa_ooa = float(
        sampled.get("T_AFR_OOA", sampled.get("T1", 2_000.0))
    )  # AFR vs OOA split
    T_ooa_eu_as = float(
        sampled.get("T_OOA_EU_AS", sampled.get("T2", 1_000.0))
    )  # CEU vs CHB split from OOA

    if not (T_africa_ooa > T_ooa_eu_as >= 0):
        raise ValueError(
            "Require T_AFR_OOA > T_OOA_EU_AS >= 0 for OutOfAfrica_3G09 model."
        )

    b = demes.Builder(time_units="generations", generation_time=1)

    # Ancestral deme up to AFR/OOA split
    b.add_deme(
        "ANC",
        epochs=[dict(start_size=N_anc, end_time=T_africa_ooa)],
    )

    # YRI (African) splits from ANC at T_africa_ooa and persists to present
    b.add_deme(
        "YRI",
        ancestors=["ANC"],
        epochs=[dict(start_size=N_YRI, end_time=0)],
    )

    # OOA deme (non-African ancestor) from AFR/OOA split to EU/AS split
    b.add_deme(
        "OOA",
        ancestors=["ANC"],
        epochs=[dict(start_size=N_OOA, end_time=T_ooa_eu_as)],
    )

    # CEU and CHB descend from OOA at T_ooa_eu_as and persist to present
    b.add_deme(
        "CEU",
        ancestors=["OOA"],
        epochs=[dict(start_size=N_CEU, end_time=0)],
    )
    b.add_deme(
        "CHB",
        ancestors=["OOA"],
        epochs=[dict(start_size=N_CHB, end_time=0)],
    )

    return b.resolve()


def OOA_three_pop_Gutenkunst(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    cfg = cfg or {}

    def getf(*keys: str, default: float) -> float:
        for k in keys:
            if k in sampled and sampled[k] is not None:
                return float(sampled[k])
        return float(default)

    # ---------------- sizes ----------------
    N_AFR_ancient = getf("N_AFR_ancient", "N_A", default=7300)
    N_AFR_recent  = getf("N_AFR_recent",  "N_AF", default=12300)
    N_OOA         = getf("N_OOA",          "N_B",  default=2100)

    # Endpoint parameterization (NO growth rates)
    N_CEU_founder = getf("N_CEU_founder", "N_EU0", default=1000)
    N_CHB_founder = getf("N_CHB_founder", "N_AS0", default=510)

    N_CEU_present = getf("N_CEU_present", "N_CEU", default=29725)
    N_CHB_present = getf("N_CHB_present", "N_CHB", default=54090)

    # ---------------- times ----------------
    T_AFR_ancient_change = getf("T_AFR_ancient_change", "T_AF", default=220e3 / 25)
    T_AFR_OOA            = getf("T_AFR_OOA",            "T_B",  default=140e3 / 25)
    T_OOA_EU_AS          = getf("T_OOA_EU_AS",          "T_EU_AS", default=21.2e3 / 25)

    if not (T_AFR_ancient_change > T_AFR_OOA > T_OOA_EU_AS >= 0):
        raise ValueError("Require T_AF > T_B > T_EU_AS >= 0")

    # ---------------- migration ----------------
    m_YRI_OOA = getf("m_YRI_OOA", "m_AF_B",  default=25e-5)
    m_YRI_CEU = getf("m_YRI_CEU", "m_AF_EU", default=3e-5)
    m_YRI_CHB = getf("m_YRI_CHB", "m_AF_AS", default=1.9e-5)
    m_CEU_CHB = getf("m_CEU_CHB", "m_EU_AS", default=9.6e-5)

    b = demes.Builder(time_units="generations", generation_time=1)

    # ---------------- ancestor ----------------
    b.add_deme(
        "ANC",
        epochs=[dict(start_size=N_AFR_ancient, end_time=T_AFR_ancient_change)],
    )

    # ---------------- YRI ----------------
    b.add_deme(
        "YRI",
        ancestors=["ANC"],
        epochs=[
            dict(start_size=N_AFR_ancient, end_time=T_AFR_OOA),
            dict(start_size=N_AFR_recent,  end_time=0),
        ],
    )

    # ---------------- OOA bottleneck ----------------
    b.add_deme(
        "OOA",
        ancestors=["YRI"],
        start_time=T_AFR_OOA,
        epochs=[dict(start_size=N_OOA, end_time=T_OOA_EU_AS)],
    )

    # ---------------- CEU / CHB (exponential via endpoints) ----------------
    b.add_deme(
        "CEU",
        ancestors=["OOA"],
        start_time=T_OOA_EU_AS,
        epochs=[
            dict(
                start_size=N_CEU_founder,
                end_size=N_CEU_present,
                end_time=0,
                size_function="exponential",
            )
        ],
    )

    b.add_deme(
        "CHB",
        ancestors=["OOA"],
        start_time=T_OOA_EU_AS,
        epochs=[
            dict(
                start_size=N_CHB_founder,
                end_size=N_CHB_present,
                end_time=0,
                size_function="exponential",
            )
        ],
    )

    # ---------------- migration ----------------
    b.add_migration(demes=["YRI", "CEU"], rate=m_YRI_CEU, start_time=T_OOA_EU_AS, end_time=0)
    b.add_migration(demes=["YRI", "CHB"], rate=m_YRI_CHB, start_time=T_OOA_EU_AS, end_time=0)
    b.add_migration(demes=["CEU", "CHB"], rate=m_CEU_CHB, start_time=T_OOA_EU_AS, end_time=0)

    b.add_migration(
        demes=["YRI", "OOA"],
        rate=m_YRI_OOA,
        start_time=T_AFR_OOA,
        end_time=T_OOA_EU_AS,
    )

    return b.resolve()

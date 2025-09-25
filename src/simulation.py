# src/simulation.py
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import demes
import msprime
import moments
import stdpopsim
import numpy as np
import tskit


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _pop_names_from_cfg(experiment_config: Optional[Dict], k: int = 2) -> List[str]:
    """
    Pull the first k population names from config['num_samples'].
    Falls back to ['N1','N2',...] if config is missing.
    """
    if experiment_config and "num_samples" in experiment_config:
        names = list(experiment_config["num_samples"].keys())
        if len(names) < k:
            raise ValueError(f"Need at least {k} pop names in config['num_samples']")
        return names[:k]
    return [f"N{i+1}" for i in range(k)]


def _anc_name_from_cfg(experiment_config: Optional[Dict]) -> str:
    """Optional override via config['ancestral_name']; else 'ANC'."""
    if experiment_config and "ancestral_name" in experiment_config:
        return str(experiment_config["ancestral_name"])
    return "ANC"


def _build_tiling_intervals(L: int, exon_bp: int = 200, tile_bp: int = 1000) -> np.ndarray:
    """
    Toy 'exon' tiling used to attach a DFE for BGS.
    Returns Nx2 array of [start, end) integer positions.
    """
    L = int(L)
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    ends = np.minimum(starts + exon_bp, L).astype(int)
    return np.column_stack([starts, ends])


def _rename_demes(g: demes.Graph, name_map: Dict[str, str]) -> demes.Graph:
    """
    Return a new demes.Graph with deme names (and references) renamed per name_map.
    name_map: dict old_name -> new_name
    """
    d = copy.deepcopy(g.asdict())

    # rename deme blocks and their ancestor references
    for deme in d["demes"]:
        if deme["name"] in name_map:
            deme["name"] = name_map[deme["name"]]
        if "ancestors" in deme and deme["ancestors"]:
            deme["ancestors"] = [name_map.get(a, a) for a in deme["ancestors"]]

    # rename migrations
    for mig in d.get("migrations", []) or []:
        if "source" in mig:
            mig["source"] = name_map.get(mig["source"], mig["source"])
        if "dest" in mig:
            mig["dest"] = name_map.get(mig["dest"], mig["dest"])

    # pulses
    for pul in d.get("pulses", []) or []:
        pul["source"] = name_map.get(pul["source"], pul["source"])
        pul["dest"]   = name_map.get(pul["dest"],   pul["dest"])

    return demes.loads(demes.dumps(d))  # validate round-trip


# ──────────────────────────────────────────────────────────────────────
# Demographic model builders (accept optional experiment_config)
# ──────────────────────────────────────────────────────────────────────

def bottleneck_model(sampled_params: Dict[str, float],
                     experiment_config: Optional[Dict] = None) -> demes.Graph:
    anc = _anc_name_from_cfg(experiment_config)
    N0, N_bottleneck, N_recover, t_start, t_end = (
        sampled_params["N0"],
        sampled_params["N_bottleneck"],
        sampled_params["N_recover"],
        sampled_params["t_bottleneck_start"],
        sampled_params["t_bottleneck_end"],
    )
    b = demes.Builder()
    b.add_deme(
        anc,
        epochs=[
            dict(start_size=N0,           end_time=t_start),
            dict(start_size=N_bottleneck, end_time=t_end),
            dict(start_size=N_recover,    end_time=0),
        ],
    )
    return b.resolve()


def split_isolation_model(sampled_params: Dict[str, float],
                          experiment_config: Optional[Dict] = None) -> demes.Graph:
    """
    Two-pop split-isolation, but read deme labels from config['num_samples'].
    Backwards compatible with old param names "N0","N1","N2","t_split".
    If you rename priors to N_<YRI>, N_<CEU>, T_split, etc., those will be used.
    """
    anc = _anc_name_from_cfg(experiment_config)
    p1, p2 = _pop_names_from_cfg(experiment_config, k=2)  # e.g. "YRI","CEU"

    # Flexible read of parameters:
    # Ancestral size
    N0 = float(
        sampled_params.get("N_anc",
            sampled_params.get("N0"))
    )
    # Leaf sizes: prefer config-labeled keys then fall back to N1/N2
    N1 = float(
        sampled_params.get(f"N_{p1}",
            sampled_params.get("N1"))
    )
    N2 = float(
        sampled_params.get(f"N_{p2}",
            sampled_params.get("N2"))
    )
    # Split time: allow T_split or t_split
    T = float(
        sampled_params.get("T_split",
            sampled_params.get("t_split"))
    )

    b = demes.Builder()
    b.add_deme(anc, epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme(p1, ancestors=[anc], epochs=[dict(start_size=N1)])
    b.add_deme(p2, ancestors=[anc], epochs=[dict(start_size=N2)])
    return b.resolve()



def split_migration_model(sampled_params: Dict[str, float],
                          experiment_config: Optional[Dict] = None) -> demes.Graph:
    """
    Two-pop split with symmetric/asymmetric migration. Reads deme labels from config.
    Accepts either m, m12/m21, or m_<p1>_<p2>.
    """
    anc = _anc_name_from_cfg(experiment_config)
    p1, p2 = _pop_names_from_cfg(experiment_config, k=2)

    N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
    N1 = float(sampled_params.get(f"N_{p1}", sampled_params.get("N1")))
    N2 = float(sampled_params.get(f"N_{p2}", sampled_params.get("N2")))
    T  = float(sampled_params.get("T_split", sampled_params.get("t_split")))

    # Migration: try the most specific names first
    m12 = sampled_params.get(f"m_{p1}_{p2}",
          sampled_params.get("m12",
          sampled_params.get("m", 0.0)))
    m21 = sampled_params.get(f"m_{p2}_{p1}",
          sampled_params.get("m21",
          sampled_params.get("m", 0.0)))
    m12 = float(m12)
    m21 = float(m21)

    b = demes.Builder()
    b.add_deme(anc, epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme(p1, ancestors=[anc], epochs=[dict(start_size=N1)])
    b.add_deme(p2, ancestors=[anc], epochs=[dict(start_size=N2)])
    if m12 > 0:
        b.add_migration(source=p1, dest=p2, rate=m12)
    if m21 > 0:
        b.add_migration(source=p2, dest=p1, rate=m21)
    return b.resolve()


def drosophila_three_epoch(sampled_params: Dict[str, float],
                           experiment_config: Optional[Dict] = None) -> demes.Graph:
    """
    D. melanogaster OutOfAfrica_2L06 with your parameter overrides and
    OPTIONAL renaming of 'AFR','EUR' to the first two names in config['num_samples'].
    """
    species = stdpopsim.get_species("DroMel")
    model = species.get_demographic_model("OutOfAfrica_2L06")

    # Parameter overrides (your original fields)
    N0               = sampled_params["N0"]
    AFR_recover      = sampled_params["AFR"]
    EUR_bottleneck   = sampled_params["EUR_bottleneck"]
    EUR_recover      = sampled_params["EUR_recover"]
    T_AFR_expansion  = sampled_params["T_AFR_expansion"]
    T_AFR_EUR_split  = sampled_params["T_AFR_EUR_split"]
    T_EUR_expansion  = sampled_params["T_EUR_expansion"]

    model.model.events[2].initial_size      = N0
    model.model.populations[0].initial_size = AFR_recover
    model.model.events[0].initial_size      = EUR_bottleneck
    model.model.populations[1].initial_size = EUR_recover
    model.model.events[2].time              = T_AFR_expansion
    model.model.events[1].time              = T_AFR_EUR_split
    model.model.events[0].time              = T_EUR_expansion

    g = model.model.to_demes()  # names: 'AFR', 'EUR'

    # Optionally rename leaf demes to config labels (first two names)
    if experiment_config and "num_samples" in experiment_config:
        cfg_names = list(experiment_config["num_samples"].keys())
        if len(cfg_names) >= 2:
            old = ["AFR", "EUR"]
            new = cfg_names[:2]
            if old != new:
                g = _rename_demes(g, dict(zip(old, new)))

    return g


# ──────────────────────────────────────────────────────────────────────
# Main simulation entry point (neutral or BGS via stdpopsim)
# ──────────────────────────────────────────────────────────────────────

def simulation(sampled_params: Dict[str, float],
               model_type: str,
               experiment_config: Dict) -> Tuple[tskit.TreeSequence, Optional[demes.Graph]]:
    """
    Simulate either:
      - Neutral ancestry+mutation via msprime using your demes builders; or
      - Background selection via stdpopsim (SLiM engine) when
        experiment_config['selection']['enabled'] is True.

    Returns:
      ts : tskit.TreeSequence
      g  : demes.Graph or None (None when using stdpopsim BGS path)
    """
    sel_cfg = (experiment_config.get("selection") or {})
    if sel_cfg.get("enabled", False):
        # --- Background selection via stdpopsim ---
        sp_id   = sel_cfg.get("species", "HomSap")
        dem_id  = sel_cfg.get("demography_id", "OutOfAfrica_3G09")
        dfe_id  = sel_cfg.get("dfe_id", "Gamma_K17")
        exon_bp = int(sel_cfg.get("exon_bp", 200))
        tile_bp = int(sel_cfg.get("tile_bp", 5000))
        Q       = float(sel_cfg.get("slim_scaling", 10.0))
        burnin  = float(sel_cfg.get("slim_burn_in", 5.0))

        sp     = stdpopsim.get_species(sp_id)
        model  = sp.get_demographic_model(dem_id)
        contig = sp.get_contig(
            chromosome=None,
            length=float(experiment_config["genome_length"]),
            mutation_rate=float(experiment_config["mutation_rate"]),
            recombination_rate=float(experiment_config["recombination_rate"]),
        )

        intervals = _build_tiling_intervals(
            experiment_config["genome_length"], exon_bp=exon_bp, tile_bp=tile_bp
        )
        dfe = sp.get_dfe(dfe_id)
        contig.add_dfe(intervals, dfe)

        # IMPORTANT: use the model's actual population names if present in config,
        # or rely on your config keys when they already match the model.
        samples = {str(k): int(v) for k, v in experiment_config["num_samples"].items()}

        ts = stdpopsim.get_engine("slim").simulate(
            model, contig, samples,
            seed=experiment_config.get("seed", None),
            slim_scaling_factor=Q,
            slim_burn_in=burnin
        )
        return ts, None  # No demes graph from msprime path

    # --- Neutral path (msprime) ---
    if model_type == "bottleneck":
        g = bottleneck_model(sampled_params, experiment_config)
    elif model_type == "split_isolation":
        g = split_isolation_model(sampled_params, experiment_config)
    elif model_type == "split_migration":
        g = split_migration_model(sampled_params, experiment_config)
    elif model_type == "drosophila_three_epoch":
        g = drosophila_three_epoch(sampled_params, experiment_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Build msprime demography and simulate
    samples = {pop_name: int(n) for pop_name, n in experiment_config["num_samples"].items()}
    demog = msprime.Demography.from_demes(g)

    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demog,
        sequence_length=float(experiment_config["genome_length"]),
        recombination_rate=float(experiment_config["recombination_rate"]),
        random_seed=int(experiment_config["seed"]),
    )
    ts = msprime.sim_mutations(
        ts, rate=float(experiment_config["mutation_rate"]),
        random_seed=int(experiment_config["seed"])
    )
    return ts, g


# ──────────────────────────────────────────────────────────────────────
# SFS builder (fixed: pop_ids match dimensionality)
# ──────────────────────────────────────────────────────────────────────

def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    """
    Generate a site frequency spectrum (moments.Spectrum) using ONLY the
    populations that actually have sampled individuals in `ts`. The
    resulting `sfs.pop_ids` has the same length as the SFS dimensionality.
    """
    # Gather (pop_index, samples) for pops that have samples
    pop_idx_with_samples: List[Tuple[int, np.ndarray]] = []
    for pop in ts.populations():
        idx = pop.id
        samp = ts.samples(population=idx)
        if len(samp) > 0:
            pop_idx_with_samples.append((idx, samp))

    if not pop_idx_with_samples:
        raise ValueError("No sampled populations found in the tree sequence.")

    # Keep a consistent order
    sample_sets = [samp for _, samp in pop_idx_with_samples]

    # Prefer metadata.name; fallback to "pop{idx}"
    pop_ids: List[str] = []
    for idx, _ in pop_idx_with_samples:
        meta = ts.population(idx).metadata
        name = meta.get("name") if isinstance(meta, dict) else None
        pop_ids.append(name if name else f"pop{idx}")

    arr = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False
    )

    sfs = moments.Spectrum(arr)
    sfs.pop_ids = pop_ids  # length equals dimensionality
    return sfs

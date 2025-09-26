# src/simulation.py
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import demes
import msprime
import moments
import numpy as np
import stdpopsim
import tskit


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _anc_name_from_cfg(experiment_config: Optional[Dict]) -> str:
    """Optional override via config['ancestral_name']; else 'ANC'."""
    if experiment_config and "ancestral_name" in experiment_config:
        return str(experiment_config["ancestral_name"])
    return "ANC"


def _cfg_pop_names(experiment_config: Optional[Dict], k: int = 2) -> List[str]:
    """
    Pull the first k population names from config['num_samples'] (for consistent labeling).
    Falls back to N1..Nk if missing.
    """
    if experiment_config and "num_samples" in experiment_config:
        names = list(experiment_config["num_samples"].keys())
        if len(names) < k:
            raise ValueError(f"Need at least {k} population names in config['num_samples']")
        return names[:k]
    return [f"N{i+1}" for i in range(k)]


def _build_tiling_intervals(L: int, exon_bp: int = 200, tile_bp: int = 1000) -> np.ndarray:
    """
    Simple exon tiling intervals for BGS.
    Returns Nx2 array of [start, end) integer positions.
    """
    L = int(L)
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    ends = np.minimum(starts + exon_bp, L).astype(int)
    return np.column_stack([starts, ends])


def _rename_demes(g: demes.Graph, name_map: Dict[str, str]) -> demes.Graph:
    """
    Return a new demes.Graph with deme names (and all references) renamed.
    """
    d = copy.deepcopy(g.asdict())

    # Demes + ancestors
    for deme in d.get("demes", []):
        old = deme["name"]
        if old in name_map:
            deme["name"] = name_map[old]
        if "ancestors" in deme and deme["ancestors"]:
            deme["ancestors"] = [name_map.get(a, a) for a in deme["ancestors"]]

    # Continuous migrations
    for mig in d.get("migrations", []) or []:
        if "source" in mig:
            mig["source"] = name_map.get(mig["source"], mig["source"])
        if "dest" in mig:
            mig["dest"] = name_map.get(mig["dest"], mig["dest"])

    # Pulses
    for pul in d.get("pulses", []) or []:
        if "source" in pul:
            pul["source"] = name_map.get(pul["source"], pul["source"])
        if "dest" in pul:
            pul["dest"] = name_map.get(pul["dest"], pul["dest"])

    return demes.loads(demes.dumps(d))  # validate


def _sps_model_from_demes(
    g: demes.Graph, model_id: str, description: str, long_description: str = ""
) -> stdpopsim.DemographicModel:
    """
    Convert a demes.Graph to a stdpopsim.DemographicModel (requires stdpopsim>=0.3).
    """
    return stdpopsim.DemographicModel.from_demes(
        g, id=model_id, description=description, long_description=long_description
    )


# ──────────────────────────────────────────────────────────────────────
# Demographic model builders (return demes.Graph)
# ──────────────────────────────────────────────────────────────────────
def bottleneck_model(sampled_params: Dict[str, float],
                     experiment_config: Optional[Dict] = None) -> demes.Graph:
    anc = _anc_name_from_cfg(experiment_config)
    N0 = float(sampled_params["N0"])
    N_b = float(sampled_params["N_bottleneck"])
    N_r = float(sampled_params["N_recover"])
    t_start = float(sampled_params["t_bottleneck_start"])  # generations ago
    t_end   = float(sampled_params["t_bottleneck_end"])    # generations ago

    b = demes.Builder()
    b.add_deme(
        anc,
        epochs=[
            dict(start_size=N0, end_time=t_start),
            dict(start_size=N_b, end_time=t_end),
            dict(start_size=N_r, end_time=0),
        ],
    )
    return b.resolve()


def split_isolation_model(sampled_params: Dict[str, float],
                          experiment_config: Optional[Dict] = None) -> demes.Graph:
    anc = _anc_name_from_cfg(experiment_config)
    p1, p2 = _cfg_pop_names(experiment_config, k=2)

    N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
    N1 = float(sampled_params.get(f"N_{p1}", sampled_params.get("N1")))
    N2 = float(sampled_params.get(f"N_{p2}", sampled_params.get("N2")))
    T  = float(sampled_params.get("T_split", sampled_params.get("t_split")))

    b = demes.Builder()
    b.add_deme(anc, epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme(p1, ancestors=[anc], epochs=[dict(start_size=N1)])
    b.add_deme(p2, ancestors=[anc], epochs=[dict(start_size=N2)])
    return b.resolve()


def split_migration_model(sampled_params: Dict[str, float],
                          experiment_config: Optional[Dict] = None) -> demes.Graph:
    anc = _anc_name_from_cfg(experiment_config)
    p1, p2 = _cfg_pop_names(experiment_config, k=2)

    N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
    N1 = float(sampled_params.get(f"N_{p1}", sampled_params.get("N1")))
    N2 = float(sampled_params.get(f"N_{p2}", sampled_params.get("N2")))
    T  = float(sampled_params.get("T_split", sampled_params.get("t_split")))
    m12 = float(sampled_params.get(f"m_{p1}_{p2}", sampled_params.get("m12", sampled_params.get("m", 0.0))))
    m21 = float(sampled_params.get(f"m_{p2}_{p1}", sampled_params.get("m21", sampled_params.get("m", 0.0))))

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
    Start from stdpopsim DroMel OutOfAfrica_2L06, apply overrides, convert to demes,
    optionally rename leaf demes to match config pop names (first two).
    """
    species = stdpopsim.get_species("DroMel")
    model = species.get_demographic_model("OutOfAfrica_2L06")

    N0               = float(sampled_params["N0"])
    AFR_recover      = float(sampled_params["AFR"])
    EUR_bottleneck   = float(sampled_params["EUR_bottleneck"])
    EUR_recover      = float(sampled_params["EUR_recover"])
    T_AFR_expansion  = float(sampled_params["T_AFR_expansion"])
    T_AFR_EUR_split  = float(sampled_params["T_AFR_EUR_split"])
    T_EUR_expansion  = float(sampled_params["T_EUR_expansion"])

    # override the underlying msprime demography embedded in stdpopsim model
    model.model.events[2].initial_size      = N0
    model.model.populations[0].initial_size = AFR_recover
    model.model.events[0].initial_size      = EUR_bottleneck
    model.model.populations[1].initial_size = EUR_recover
    model.model.events[2].time              = T_AFR_expansion
    model.model.events[1].time              = T_AFR_EUR_split
    model.model.events[0].time              = T_EUR_expansion

    g = model.model.to_demes()  # names 'AFR','EUR'

    # Optionally rename leaf demes to match config labels (first two names)
    if experiment_config and "num_samples" in experiment_config:
        cfg_names = list(experiment_config["num_samples"].keys())
        if len(cfg_names) >= 2:
            old = ["AFR", "EUR"]
            new = cfg_names[:2]
            if old != new:
                g = _rename_demes(g, dict(zip(old, new)))

    return g


# ──────────────────────────────────────────────────────────────────────
# Simulation entry (neutral via msprime; BGS via stdpopsim+SLiM using same demes)
# ──────────────────────────────────────────────────────────────────────
def _build_demes(model_type: str, sampled_params: Dict[str, float], cfg: Dict) -> demes.Graph:
    if model_type == "bottleneck":
        return bottleneck_model(sampled_params, cfg)
    if model_type == "split_isolation":
        return split_isolation_model(sampled_params, cfg)
    if model_type == "split_migration":
        return split_migration_model(sampled_params, cfg)
    if model_type == "drosophila_three_epoch":
        return drosophila_three_epoch(sampled_params, cfg)
    raise ValueError(f"Unknown model type: {model_type}")


def _samples_from_cfg_for_model(model: stdpopsim.DemographicModel, cfg: Dict) -> Dict[str, int]:
    """
    Build samples dict by intersecting cfg['num_samples'] with model population IDs.
    """
    cfg_ns = (cfg.get("num_samples") or {})
    model_ids = {p.id for p in model.populations}
    out = {k: int(v) for k, v in cfg_ns.items() if k in model_ids and int(v) > 0}
    if not out:
        raise ValueError(
            f"No overlap between num_samples keys {list(cfg_ns.keys())} "
            f"and model population IDs {sorted(model_ids)}."
        )
    return out


def _simulate_neutral(g: demes.Graph, cfg: Dict) -> tskit.TreeSequence:
    demog = msprime.Demography.from_demes(g)
    samples = {k: int(v) for k, v in (cfg.get("num_samples") or {}).items()}
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demog,
        sequence_length=float(cfg["genome_length"]),
        recombination_rate=float(cfg["recombination_rate"]),
        random_seed=int(cfg["seed"]),
    )
    ts = msprime.sim_mutations(
        ts, rate=float(cfg["mutation_rate"]), random_seed=int(cfg["seed"])
    )
    return ts


def _simulate_bgs(g: demes.Graph, cfg: Dict, sel_cfg: Dict) -> tskit.TreeSequence:
    # 1) Convert demes -> stdpopsim model (same demography as neutral)
    model_id = sel_cfg.get("model_id", "custom_from_demes")
    desc     = sel_cfg.get("model_desc", "Custom demography via demes")
    model = _sps_model_from_demes(g, model_id, desc)

    # 2) Build contig with same μ, r, L (your cfg overrides species defaults)
    sp_id   = sel_cfg.get("species", "HomSap")
    dfe_id  = sel_cfg.get("dfe_id", "Gamma_K17")
    exon_bp = int(sel_cfg.get("exon_bp", 200))
    tile_bp = int(sel_cfg.get("tile_bp", 5000))
    Q       = float(sel_cfg.get("slim_scaling", 10.0))
    burnin  = float(sel_cfg.get("slim_burn_in", 5.0))

    sp = stdpopsim.get_species(sp_id)
    contig = sp.get_contig(
        chromosome=None,
        length=float(cfg["genome_length"]),
        mutation_rate=float(cfg["mutation_rate"]),
        recombination_rate=float(cfg["recombination_rate"]),
    )

    # 3) Attach DFE tiling (Background Selection)
    intervals = _build_tiling_intervals(cfg["genome_length"], exon_bp=exon_bp, tile_bp=tile_bp)
    dfe = sp.get_dfe(dfe_id)
    contig.add_dfe(intervals, dfe)

    # 4) Samples (keys must match demes/ model pop IDs)
    samples = _samples_from_cfg_for_model(model, cfg)

    # 5) Run SLiM
    ts = stdpopsim.get_engine("slim").simulate(
        model,
        contig,
        samples,
        seed=cfg.get("seed", None),
        slim_scaling_factor=Q,
        slim_burn_in=burnin,
    )
    return ts


def simulation(sampled_params: Dict[str, float],
               model_type: str,
               experiment_config: Dict) -> Tuple[tskit.TreeSequence, Optional[demes.Graph]]:
    """
    One place to run either neutral (msprime) or BGS (SLiM via stdpopsim),
    while keeping the *same* demography built from your demes functions.
    """
    sel_cfg = (experiment_config.get("selection") or {})
    g = _build_demes(model_type, sampled_params, experiment_config)

    if sel_cfg.get("enabled", False):
        ts = _simulate_bgs(g, experiment_config, sel_cfg)
        return ts, g  # return g so you can still plot the demes graph
    else:
        ts = _simulate_neutral(g, experiment_config)
        return ts, g


# ──────────────────────────────────────────────────────────────────────
# SFS builder
# ──────────────────────────────────────────────────────────────────────
def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    """
    Build a moments.Spectrum using ONLY populations that have sampled individuals.
    sfs.pop_ids length equals dimensionality.
    """
    pop_idx_with_samples: List[Tuple[int, np.ndarray]] = []
    for pop in ts.populations():
        idx = pop.id
        samp = ts.samples(population=idx)
        if len(samp) > 0:
            pop_idx_with_samples.append((idx, samp))

    if not pop_idx_with_samples:
        raise ValueError("No sampled populations found in the tree sequence.")

    sample_sets = [samp for _, samp in pop_idx_with_samples]

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
    sfs.pop_ids = pop_ids
    return sfs

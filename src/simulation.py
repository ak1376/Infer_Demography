from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import demes
import msprime
import numpy as np
import stdpopsim as sps
import tskit
import moments

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.demes_models import (
    bottleneck_model,
    split_isolation_model,
    split_migration_model,
    drosophila_three_epoch,
    split_migration_growth_model,
    OOA_three_pop_model,
)
from src.bgs_intervals import _contig_from_cfg, _apply_dfe_intervals
from src.stdpopsim_wrappers import define_sps_model

# ──────────────────────────────────
# Main entry: BGS only (SLiM via stdpopsim)
# ──────────────────────────────────


def msprime_simulation(
    g: demes.Graph, experiment_config: Dict
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    samples = {
        pop_name: num_samples
        for pop_name, num_samples in experiment_config["num_samples"].items()
    }

    demog = msprime.Demography.from_demes(g)

    # Simulate ancestry for two populations (joint simulation)
    ts = msprime.sim_ancestry(
        samples=samples,  # Two populations
        demography=demog,
        sequence_length=experiment_config["genome_length"],
        recombination_rate=experiment_config["recombination_rate"],
        random_seed=experiment_config["seed"],
    )

    # Simulate mutations over the ancestry tree sequence
    ts = msprime.sim_mutations(
        ts,
        rate=experiment_config["mutation_rate"],
        random_seed=experiment_config["seed"],
    )

    return ts, g


def stdpopsim_slim_simulation(
    g: demes.Graph,
    experiment_config: Dict,
    sampled_coverage: float,
    model_type: str,
    sampled_params: Dict[str, float],
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    # 1) Pick model (wrap Demes for stdpopsim)
    model = define_sps_model(model_type, g, sampled_params)

    # 2) Build contig and apply DFE intervals
    sel = experiment_config.get("selection") or {}
    contig = _contig_from_cfg(experiment_config, sel)
    sel_summary = _apply_dfe_intervals(contig, sel, sampled_coverage=sampled_coverage)

    # 3) Samples
    samples = {
        k: int(v) for k, v in (experiment_config.get("num_samples") or {}).items()
    }
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
    elif model_type == "OOA_three_pop":
        g = OOA_three_pop_model(sampled_params, experiment_config)
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
            raise ValueError(
                "engine='slim' requires selection.enabled=true in your config."
            )
        if sampled_coverage is None:
            raise ValueError(
                "engine='slim' requires a non-None sampled_coverage (percent or fraction)."
            )
        return stdpopsim_slim_simulation(
            g, experiment_config, sampled_coverage, model_type, sampled_params
        )

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

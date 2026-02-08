#!/usr/bin/env python3
"""
pipeline_debugging_scripts/debugging_simulation.py

Minimal sanity check:

1) Build demes graph g (your IM_symmetric_model)
2) Simulate ONE ts with:
   - msprime direct
   - stdpopsim (msprime engine)
3) Compute observed SFS for both
4) Compute EXPECTED SFS from g via moments (no randomness)
5) Report distances:
      ||obs_msprime - expected||_F
      ||obs_stdpopsim - expected||_F

Goal: both obs SFS should be similarly close (in scale) to the same expected SFS.
They do NOT need to match each other exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json
import sys

import demes
import msprime
import moments
import numpy as np
import stdpopsim as sps
import tskit

# Make repo importable (for define_sps_model)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from src.stdpopsim_wrappers import define_sps_model  # type: ignore  # noqa: E402


# -----------------------------------------------------------------------------
# Demes graph (your model)
# -----------------------------------------------------------------------------
def IM_symmetric_model(sampled: Dict[str, float]) -> demes.Graph:
    required_keys = ["N_anc", "N_YRI", "N_CEU", "m", "T_split"]
    for k in required_keys:
        assert k in sampled, f"Missing required key: {k}"

    N0 = float(sampled["N_anc"])
    N1 = float(sampled["N_YRI"])
    N2 = float(sampled["N_CEU"])
    T = float(sampled["T_split"])
    m = float(sampled["m"])
    assert T > 0, "T_split must be > 0"

    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme(
        "YRI",
        epochs=[
            dict(start_size=N0, end_time=T),  # ancestral epoch
            dict(start_size=N1, end_time=0),  # modern epoch
        ],
    )

    b.add_deme(
        "CEU",
        ancestors=["YRI"],
        start_time=T,
        epochs=[dict(start_size=N2, end_time=0)],
    )

    if m > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m, start_time=T, end_time=0)
        b.add_migration(source="CEU", dest="YRI", rate=m, start_time=T, end_time=0)

    return b.resolve()


def pop_id(ts: tskit.TreeSequence, name: str) -> int:
    for pid in range(ts.num_populations):
        md = ts.population(pid).metadata
        if isinstance(md, (bytes, bytearray)):
            md = json.loads(md.decode())
        if isinstance(md, dict) and md.get("name") == name:
            return pid
    raise KeyError(f"Population {name!r} not found in ts metadata")


def observed_sfs(ts: tskit.TreeSequence) -> moments.Spectrum:
    YRI = pop_id(ts, "YRI")
    CEU = pop_id(ts, "CEU")
    return moments.Spectrum(
        ts.allele_frequency_spectrum(
            sample_sets=[list(ts.samples(YRI)), list(ts.samples(CEU))],
            mode="site",
            polarised=True,
            span_normalise=False,
        )
    )


def simulate_msprime_direct(
    *,
    demogr: msprime.Demography,
    sample_sizes: Dict[str, int],
    seqlen: float,
    recomb: float,
    mu: float,
    seed: int,
) -> tskit.TreeSequence:
    ts = msprime.sim_ancestry(
        samples=sample_sizes,
        sequence_length=float(seqlen),
        recombination_rate=float(recomb),
        demography=demogr,
        random_seed=int(seed),
    )
    ts = msprime.sim_mutations(ts, rate=float(mu), random_seed=int(seed) + 1)
    return ts


def simulate_stdpopsim_msprime_engine(
    *,
    g: demes.Graph,
    sample_sizes: Dict[str, int],
    seqlen: float,
    recomb: float,
    mu: float,
    seed: int,
) -> tskit.TreeSequence:
    # Wrap g as stdpopsim DemographicModel
    model = define_sps_model(g)

    # IMPORTANT: stdpopsim Species.get_contig() does NOT accept recombination_rate;
    # it uses genome/genetic map machinery. For a minimal check we:
    # - request a generic contig with length + mutation_rate
    # - then we *do not* try to force recomb here
    #
    # If you *must* match msprime_direct recomb exactly, don't use stdpopsim contigs;
    # use msprime.sim_ancestry directly (or build a RateMap).
    sp = sps.get_species("HomSap")
    contig = sp.get_contig(
        chromosome=None,
        length=float(seqlen),
        mutation_rate=float(mu),
    )

    eng = sps.get_engine("msprime")
    ts = eng.simulate(model, contig, sample_sizes, seed=int(seed))
    return ts


def main() -> None:
    # -----------------------
    # Set parameters (yours)
    # -----------------------
    N_anc = 10000
    N1 = 5000
    N2 = 5000
    T_split = 1000
    m = 1e-4

    nYRI = 5
    nCEU = 5
    sample_sizes = {"YRI": nYRI, "CEU": nCEU}

    seed = 295
    seqlen = 1e6
    recomb = 1e-8
    mu = 1e-8

    sampled_params = {
        "N_anc": N_anc,
        "N_YRI": N1,
        "N_CEU": N2,
        "T_split": T_split,
        "m": m,
    }

    # -----------------------
    # Build graph + expected SFS
    # -----------------------
    g = IM_symmetric_model(sampled_params)

    # moments expects HAPLOID sample sizes (2*n diploids)
    ns_hap = [2 * nYRI, 2 * nCEU]

    # Expected SFS (no randomness). Theta consistent with your moments usage:
    # theta = 4 * N_anc * mu * L
    theta = 4.0 * float(N_anc) * float(mu) * float(seqlen)

    S_exp = moments.Spectrum.from_demes(
        g,
        sampled_demes=["YRI", "CEU"],
        sample_sizes=ns_hap,
        theta=float(theta),
    )

    # -----------------------
    # Simulate both ways
    # -----------------------
    demogr = msprime.Demography.from_demes(g)

    ts_ms = simulate_msprime_direct(
        demogr=demogr,
        sample_sizes=sample_sizes,
        seqlen=seqlen,
        recomb=recomb,
        mu=mu,
        seed=seed,
    )
    ts_sps = simulate_stdpopsim_msprime_engine(
        g=g,
        sample_sizes=sample_sizes,
        seqlen=seqlen,
        recomb=recomb,  # kept for symmetry; not used inside stdpopsim contig here
        mu=mu,
        seed=seed,
    )

    S_ms = observed_sfs(ts_ms)
    S_sps = observed_sfs(ts_sps)

    # -----------------------
    # Compare OBSERVED vs EXPECTED
    # -----------------------
    A = np.asarray(S_ms, dtype=float)
    B = np.asarray(S_sps, dtype=float)
    E = np.asarray(S_exp, dtype=float)

    d_ms = float(np.linalg.norm(A - E))   # Frobenius norm
    d_sps = float(np.linalg.norm(B - E))

    # Helpful scale: expected total segregating sites (sum of expected SFS)
    sumE = float(E.sum())
    sumA = float(A.sum())
    sumB = float(B.sum())

    print("\nObserved vs Expected (moments.from_demes) distances:")
    print(json.dumps(
        {
            "theta": theta,
            "expected_sum": sumE,
            "msprime_obs_sum": sumA,
            "stdpopsim_obs_sum": sumB,
            "||msprime - expected||_F": d_ms,
            "||stdpopsim - expected||_F": d_sps,
            "ratio(stdpopsim/msprime)": (d_sps / d_ms) if d_ms > 0 else None,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

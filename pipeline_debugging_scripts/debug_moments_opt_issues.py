#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import moments
import nlopt
import numdifftools as nd
import demes
import demesdraw
import tskit
import sys
import stdpopsim as sps
import msprime

# Make repo importable (for define_sps_model)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


from src.stdpopsim_wrappers import define_sps_model  # type: ignore  # noqa: E402

# ───────────────────────── diffusion helper ────────────────────────────
def _diffusion_sfs(
    log_space_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    param_names: List[str],
    sampled_demes: List[str],
    haploid_sizes: List[int],
    genome_scaled_mutation_rate: float, # mu * L
    *,
    demogr: Optional[msprime.Demography] = False,  # <-- ADDED FOR msprime DEMOGRAPHY SUPPORT
    return_graph: bool = False,   # <--- ADD
):
    real_space_vec = 10 ** log_space_vec
    p_dict = {k: float(v) for k, v in zip(param_names, real_space_vec)}

    if demogr is not False: # If a demography object is provided, use it to create the graph
        demography = msprime_demography_directly(p_dict)
        graph = demography.to_demes()
    else:
        graph = demo_model(p_dict)

    muL = genome_scaled_mutation_rate
    N0 = float(p_dict[param_names[0]])
    theta = 4.0 * N0 * muL

    fs = moments.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=theta,
    )

    if return_graph:
        return fs, graph, p_dict, theta, muL   # <--- RETURN EXTRA STUFF
    return fs

def IM_asymmetric_model_no_anc(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Split + asymmetric migration (two rates), but *no separate ancestral-only deme*.
    YRI carries the ancestral epoch pre-split; CEU branches off at T_split.

    This keeps the first deme ("YRI") extant at time 0 => pop0 is extant after msprime.from_demes().

    Required keys (preferred):
      - N_anc, N_YRI, N_CEU, T_split, m_YRI_CEU, m_CEU_YRI

    Backward-compatible aliases are supported (see parsing below).
    """

    # ---- parameter parsing (with aliases) ----
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))

    T = float(sampled.get("T_split", sampled.get("t_split")))

    # directional migration rates
    m12 = float(sampled.get("m_YRI_CEU", sampled.get("m12", 0.0)))  # YRI -> CEU
    m21 = float(sampled.get("m_CEU_YRI", sampled.get("m21", 0.0)))  # CEU -> YRI

    assert T > 0, "T_split must be > 0"
    assert N0 > 0 and N1 > 0 and N2 > 0, "Population sizes must be > 0"
    assert m12 >= 0 and m21 >= 0, "Migration rates must be >= 0"

    # ---- build demes graph (IM_symmetric-style: no 'ANC' deme) ----
    b = demes.Builder(time_units="generations", generation_time=1)

    # Root extant deme YRI:
    #   - from present back to T: size N1
    #   - older than T: ancestral size N0
    b.add_deme(
        "YRI",
        epochs=[
            dict(start_size=N0, end_time=T),  # ancestral epoch (older than split)
            dict(start_size=N1, end_time=0),  # modern epoch
        ],
    )

    # CEU branches off at split time
    b.add_deme(
        "CEU",
        ancestors=["YRI"],
        start_time=T,
        epochs=[dict(start_size=N2, end_time=0)],
    )

    # Asymmetric migration AFTER split only (when both demes exist: [0, T])
    if m12 > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m12, start_time=T, end_time=0)
    if m21 > 0:
        b.add_migration(source="CEU", dest="YRI", rate=m21, start_time=T, end_time=0)

    return b.resolve()


def IM_asymmetric_model(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Split + asymmetric migration (two rates), WITH an explicit ancestral-only deme "ANC".

    Deme names: "ANC", "YRI", "CEU".

    Required keys (preferred):
      - N_anc, N_YRI, N_CEU, T_split, m_YRI_CEU, m_CEU_YRI

    Backward-compatible aliases supported:
      - N_anc: N0
      - N_YRI: N1
      - N_CEU: N2
      - T_split: t_split
      - m_YRI_CEU: m12 or m (fallback)
      - m_CEU_YRI: m21 or m (fallback)

    Notes:
      - "ANC" exists only in (T_split, ∞) and ends at T_split.
      - "YRI" and "CEU" start at T_split and persist to the present (time 0).
      - Migration is only after the split, i.e. during [0, T_split].
    """

    # ---- parameter parsing (with aliases) ----
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))

    T = float(sampled.get("T_split", sampled.get("t_split")))

    # directional migration rates
    m12 = float(sampled.get("m_YRI_CEU", sampled.get("m12", sampled.get("m", 0.0))))  # YRI -> CEU
    m21 = float(sampled.get("m_CEU_YRI", sampled.get("m21", sampled.get("m", 0.0))))  # CEU -> YRI

    assert T > 0, "T_split must be > 0"
    assert N0 > 0 and N1 > 0 and N2 > 0, "Population sizes must be > 0"
    assert m12 >= 0 and m21 >= 0, "Migration rates must be >= 0"

    # ---- build demes graph (explicit ANC root) ----
    b = demes.Builder(time_units="generations", generation_time=1)

    # Ancestor-only deme: exists only before split
    b.add_deme(
        "ANC",
        epochs=[dict(start_size=N0, end_time=T)],
    )

    # Two derived demes start at split time
    b.add_deme(
        "YRI",
        ancestors=["ANC"],
        start_time=T,
        epochs=[dict(start_size=N1, end_time=0)],
    )
    b.add_deme(
        "CEU",
        ancestors=["ANC"],
        start_time=T,
        epochs=[dict(start_size=N2, end_time=0)],
    )

    # Asymmetric migration AFTER split only (when both demes exist: [0, T])
    if m12 > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m12, start_time=T, end_time=0)
    if m21 > 0:
        b.add_migration(source="CEU", dest="YRI", rate=m21, start_time=T, end_time=0)

    return b.resolve()

def fit_model(
    sfs: moments.Spectrum,
    *,
    start_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    lb_full: np.ndarray,
    ub_full: np.ndarray,
    genome_scaled_mutation_rate: float, # mu * L
    param_names: Optional[List[str]] = None,
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-12,

) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run a **single** moments optimisation (nlopt.LD_LBFGS) in log10 space.
    Returns:
      - best_params: list with one vector of fitted REAL-space params (in param_order)
      - best_lls:    list with one max log-likelihood value
    """

    assert isinstance(sfs, moments.Spectrum), "sfs must be a moments.Spectrum"

    if np.any(lb_full <= 0) or np.any(ub_full <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb_full, ub_full) if lo <= 0 or hi <= 0]
        raise ValueError(f"All bounds must be positive for log10 optimization. Bad: {bad}")

    # ---- SFS axis / demes order ----
    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("Observed SFS has no pop_ids; cannot infer sampled_demes order.")

    haploid_sizes = [n - 1 for n in sfs.shape]

    # ---- objective: Poisson composite log-likelihood ----
    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = _diffusion_sfs(
            log_space_vec=log10_params,
            demo_model=demo_model,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            genome_scaled_mutation_rate=genome_scaled_mutation_rate,
        )
        # log(exp + eps) avoids -inf when exp has zeros
        return float(np.sum(np.log(np.asarray(exp_sfs) + eps) * np.asarray(sfs) - np.asarray(exp_sfs)))

    grad_fn = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {log10_params}")
        return ll  # maximize

    # ---- optimizer setup ----
    start_vec = np.asarray(start_vec, dtype=float)
    if start_vec.shape != (len(param_names),):
        raise ValueError(f"start_vec shape {start_vec.shape} != ({len(param_names)},)")

    opt = nlopt.opt(nlopt.LD_LBFGS, start_vec.size)
    opt.set_lower_bounds(np.log10(lb_full))
    opt.set_upper_bounds(np.log10(ub_full))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    # ---- run optimization ----
    x0 = np.log10(start_vec)
    xhat = opt.optimize(x0)

    ll_hat = loglikelihood(xhat)
    fitted_params = 10 ** xhat

    return fitted_params

# Simulation Component

def pop_id_by_name(ts: tskit.TreeSequence, name: str) -> int:
    for pop in ts.populations():
        meta = pop.metadata if isinstance(pop.metadata, dict) else {}
        if meta.get("name") == name:
            return pop.id
    raise ValueError(
        f"Population with metadata name='{name}' not found. "
        "Available names: "
        + ", ".join(
            str((p.id, (p.metadata.get('name') if isinstance(p.metadata, dict) else None)))
            for p in ts.populations()
        )
    )

def create_SFS(ts: tskit.TreeSequence, pop_names: Sequence[str] = ("YRI", "CEU")) -> moments.Spectrum:
    """
    Create a 2D site-frequency spectrum for exactly the two populations in pop_names,
    using ts population metadata 'name' field (robust to population ID ordering).
    """
    sample_sets: List[np.ndarray] = []
    for name in pop_names:
        pid = pop_id_by_name(ts, name)
        samps = ts.samples(population=pid)
        if len(samps) == 0:
            raise ValueError(f"Population '{name}' (id={pid}) has zero samples in this TS.")
        sample_sets.append(samps)

    arr = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False,
    )

    sfs = moments.Spectrum(arr)
    sfs.pop_ids = list(pop_names)

    # Debug prints (optional)
    print("create_SFS using:", list(pop_names))
    print("ts.sequence_length:", ts.sequence_length)
    print("ts.num_sites:", ts.num_sites)
    print("sum(obs_sfs):", float(np.sum(np.asarray(sfs))))
    print("sfs.shape:", sfs.shape)

    return sfs


def simulation_runner(
    g: demes.Graph, 
    samples: Dict[str, int],
    *,
    chrom_length: int = 1e8,
    mutation_rate: float = 1e-8,
    recomb_rate: float = 1e-8,
    seed: Optional[int] = 295,
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    model = define_sps_model(g) 
    contig = sps.get_species("HomSap").get_contig(
        chromosome = None,
        length=chrom_length, 
        mutation_rate=mutation_rate, 
        recombination_rate=recomb_rate,
    )
    print("contig.length:", contig.length)
    print("contig.mutation_rate:", contig.mutation_rate)
    print("contig.recombination_map.mean_rate:", contig.recombination_map.mean_rate)

    eng = sps.get_engine("msprime")
    ts = eng.simulate(
        model,
        contig,
        samples,
        seed=seed
    )
    
    return ts, model


# def msprime_demography_directly(sampled_params: Dict) -> msprime.Demography:
#     N_ANC = float(sampled_params["N_anc"])
#     N_YRI = float(sampled_params["N_YRI"])
#     N_CEU = float(sampled_params["N_CEU"])
#     T_split = float(sampled_params["T_split"])
#     m_YRI_CEU = float(sampled_params["m_YRI_CEU"])
#     m_CEU_YRI = float(sampled_params["m_CEU_YRI"])

#     demogr = msprime.Demography()
#     demogr.add_population(name="YRI", initial_size=N_YRI)
#     demogr.add_population(name="CEU", initial_size=N_CEU)
#     demogr.add_population(name="ANC", initial_size=N_ANC)

#     # migration after split only (both exist for time in [0, T_split])
#     demogr.set_migration_rate(source="YRI", dest="CEU", rate=m_YRI_CEU)
#     demogr.set_migration_rate(source="CEU", dest="YRI", rate=m_CEU_YRI)

#     demogr.add_population_split(time=T_split, ancestral="ANC", derived=["YRI", "CEU"])
#     return demogr

def msprime_demography_directly(sampled_params: Dict) -> msprime.Demography:
    """
    Build an msprime.Demography where the *parameter meanings* match demes:

      m_YRI_CEU := forward-time migration rate YRI -> CEU
      m_CEU_YRI := forward-time migration rate CEU -> YRI

    Since msprime migration rates are specified as *backward-time lineage movement*,
    we flip directions when calling set_migration_rate:

      forward-time A -> B  <=>  backward-time B -> A
    """
    N_ANC = float(sampled_params["N_anc"])
    N_YRI = float(sampled_params["N_YRI"])
    N_CEU = float(sampled_params["N_CEU"])
    T_split = float(sampled_params["T_split"])

    # These are FORWARD-TIME meanings (demes convention):
    m_YRI_CEU = float(sampled_params["m_YRI_CEU"])  # forward-time YRI -> CEU
    m_CEU_YRI = float(sampled_params["m_CEU_YRI"])  # forward-time CEU -> YRI

    demogr = msprime.Demography()
    demogr.add_population(name="YRI", initial_size=N_YRI)
    demogr.add_population(name="CEU", initial_size=N_CEU)
    demogr.add_population(name="ANC", initial_size=N_ANC)

    # Apply rates in msprime with FLIPPED direction (backward-time):
    # forward-time YRI -> CEU  ==> backward-time CEU -> YRI
    demogr.set_migration_rate(source="CEU", dest="YRI", rate=m_YRI_CEU)

    # forward-time CEU -> YRI  ==> backward-time YRI -> CEU
    demogr.set_migration_rate(source="YRI", dest="CEU", rate=m_CEU_YRI)

    # Split at T_split (backward-time: merge YRI and CEU into ANC at T_split)
    demogr.add_population_split(time=T_split, ancestral="ANC", derived=["YRI", "CEU"])

    return demogr


def simulate_msprime(demogr: msprime.Demography, seqlen: int, mutation_rate: float, recomb_rate: float, seed: Optional[int] = 295) -> tskit.TreeSequence:
    
    ts = msprime.sim_ancestry(
        samples={"YRI": 20, "CEU": 20},
        sequence_length=seqlen,
        recombination_rate=recomb_rate,
        demography=demogr,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed)

    return ts


def main():
    param_names = ["N_anc", "N_YRI", "N_CEU", "T_split", "m_YRI_CEU", "m_CEU_YRI"]
    start_vec = np.array([5000, 15000, 10000, 10000, 1e-6, 1e-5])
    lb_full = np.array([1000, 1000, 1000, 100, 1e-10, 1e-10])
    ub_full = np.array([50000, 50000, 50000, 20000, 1e-3, 1e-3])

    mu = 1e-8
    L = int(5e7)
    samples = {"YRI": 10, "CEU": 10}

    genome_scaled_mutation_rate = mu * L 
    true_sfs = _diffusion_sfs(
        log_space_vec=np.log10(start_vec),
        demo_model=IM_asymmetric_model,
        param_names=param_names,
        sampled_demes=["YRI", "CEU"],
        haploid_sizes=[20, 20],
        genome_scaled_mutation_rate=genome_scaled_mutation_rate
    )
    optim_pars = fit_model(
        sfs=true_sfs,
        start_vec=start_vec,
        demo_model=IM_asymmetric_model_no_anc,  # <-- TEST THE OTHER MODEL
        lb_full=lb_full,
        ub_full=ub_full,
        genome_scaled_mutation_rate=genome_scaled_mutation_rate,
        param_names=param_names,
        verbose=True,
    )
    np.testing.assert_allclose(optim_pars, start_vec, rtol=0.1)

    sampled_params = {name: val for name, val in zip(param_names, start_vec)}


    # -------------------------- Simulate and Inference: Stdpopsim version --------------------------
    g = IM_asymmetric_model(sampled_params)

    ts, model = simulation_runner(
        g,
        samples=samples,
        chrom_length=L,
        mutation_rate=mu
    )
    print("Simulated TS has sequence_length:", ts.sequence_length)
    print("Simulated TS has num_trees:", ts.num_trees)
    print(model.model.debug())

    sim_sfs = create_SFS(ts, pop_names=["YRI", "CEU"])

    optim_pars_sim = fit_model(
        sfs=sim_sfs,
        start_vec=start_vec,
        demo_model=IM_asymmetric_model,
        lb_full=lb_full,
        ub_full=ub_full,
        genome_scaled_mutation_rate=genome_scaled_mutation_rate,
        param_names=param_names,
        verbose=True,
    )

    # print("Optimized parameters from simulated data:\n", optim_pars_sim)
    # print("Ground Truth parameters from simulated data:\n", start_vec)
    # print("True SFS from diffusion:\n", np.round(np.log10(np.asarray(true_sfs)), 2))
    # print("Simulated SFS from ts:\n", np.round(np.log10(np.asarray(sim_sfs)), 2))

    # -------------------------- Simulate and Inference: msprime version --------------------------

    demogr = msprime_demography_directly(sampled_params)
    print(f'Demography for msprime simulation:\n{demogr.debug()}')
    ts_msprime = simulate_msprime(demogr, seqlen=L, mutation_rate=mu, recomb_rate=1e-8)
    sim_sfs_msprime = create_SFS(ts_msprime, pop_names=["YRI", "CEU"]) # Observed SFS from msprime simulation

    # Now let's create the theoretical SFS from the diffusion approximation for the same parameters
    true_sfs_msprime = _diffusion_sfs(
        log_space_vec=np.log10(start_vec),
        demo_model=IM_asymmetric_model,
        param_names=param_names,
        sampled_demes=["YRI", "CEU"],
        haploid_sizes=[20, 20],
        genome_scaled_mutation_rate=genome_scaled_mutation_rate, 
        demogr = True
    )

    print(f'True SFS from diffusion (msprime demography):\n', np.round(np.log10(np.asarray(true_sfs_msprime)), 2))
    print(f'Simulated SFS from msprime:\n', np.round(np.log10(np.asarray(sim_sfs_msprime)), 2))




if __name__ == "__main__":
    main()
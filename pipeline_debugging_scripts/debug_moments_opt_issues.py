#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import moments
import nlopt
import numdifftools as nd
import demes
import demesdraw  # unused, but keeping since you imported it
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


# ───────────────────────── msprime → demes conversion helper ─────────────────────────

def _demes_graph_from_msprime_demography(demo: msprime.Demography) -> demes.Graph:
    """
    Convert an msprime.Demography to a demes.Graph for moments.Spectrum.from_demes().
    """
    if hasattr(demo, "to_demes"):
        return demo.to_demes()
    raise RuntimeError(
        "Your msprime.Demography does not support .to_demes(). "
        "Upgrade msprime, or implement a custom conversion to demes."
    )


# ───────────────────────── diffusion helper ────────────────────────────

def _diffusion_sfs(
    log_space_vec: np.ndarray,
    *,
    param_names: List[str],
    sampled_demes: List[str],
    haploid_sizes: List[int],
    genome_scaled_mutation_rate: float,  # mu * L
    model_source: Literal["demes", "msprime"] = "demes",
    demes_builder: Optional[Callable[[Dict[str, float]], demes.Graph]] = None,
    msprime_builder: Optional[Callable[[Dict[str, float]], msprime.Demography]] = None,
    return_graph: bool = False,
):
    """
    Build expected SFS using moments diffusion approximation.

    model_source:
      - "demes": use demes_builder(p_dict) -> demes.Graph
      - "msprime": use msprime_builder(p_dict) -> msprime.Demography -> to demes.Graph
    """
    real_space_vec = 10 ** log_space_vec
    p_dict = {k: float(v) for k, v in zip(param_names, real_space_vec)}

    if model_source == "demes":
        if demes_builder is None:
            raise ValueError("model_source='demes' requires demes_builder=...")
        graph = demes_builder(p_dict)

    elif model_source == "msprime":
        if msprime_builder is None:
            raise ValueError("model_source='msprime' requires msprime_builder=...")
        demography = msprime_builder(p_dict)
        graph = _demes_graph_from_msprime_demography(demography)

    else:
        raise ValueError(f"Unknown model_source: {model_source}")

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
        return fs, graph, p_dict, theta, muL
    return fs


# ───────────────────────── Demes models ────────────────────────────

def IM_asymmetric_model_no_anc(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Split + asymmetric migration (two rates), but *no separate ancestral-only deme*.
    YRI carries the ancestral epoch pre-split; CEU branches off at T_split.
    """
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))

    T = float(sampled.get("T_split", sampled.get("t_split")))

    m12 = float(sampled.get("m_YRI_CEU", sampled.get("m12", 0.0)))  # YRI -> CEU
    m21 = float(sampled.get("m_CEU_YRI", sampled.get("m21", 0.0)))  # CEU -> YRI

    assert T > 0, "T_split must be > 0"
    assert N0 > 0 and N1 > 0 and N2 > 0, "Population sizes must be > 0"
    assert m12 >= 0 and m21 >= 0, "Migration rates must be >= 0"

    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme(
        "YRI",
        epochs=[
            dict(start_size=N0, end_time=T),
            dict(start_size=N1, end_time=0),
        ],
    )

    b.add_deme(
        "CEU",
        ancestors=["YRI"],
        start_time=T,
        epochs=[dict(start_size=N2, end_time=0)],
    )

    if m12 > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m12, start_time=T, end_time=0)
    if m21 > 0:
        b.add_migration(source="CEU", dest="YRI", rate=m21, start_time=T, end_time=0)

    return b.resolve()


def IM_asymmetric_model(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Split + asymmetric migration (two rates), WITH an explicit ancestral-only deme "ANC".
    """
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))

    T = float(sampled.get("T_split", sampled.get("t_split")))

    m12 = float(sampled.get("m_YRI_CEU", sampled.get("m12", sampled.get("m", 0.0))))
    m21 = float(sampled.get("m_CEU_YRI", sampled.get("m21", sampled.get("m", 0.0))))

    assert T > 0, "T_split must be > 0"
    assert N0 > 0 and N1 > 0 and N2 > 0, "Population sizes must be > 0"
    assert m12 >= 0 and m21 >= 0, "Migration rates must be >= 0"

    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T)])

    b.add_deme("YRI", ancestors=["ANC"], start_time=T, epochs=[dict(start_size=N1, end_time=0)])
    b.add_deme("CEU", ancestors=["ANC"], start_time=T, epochs=[dict(start_size=N2, end_time=0)])

    if m12 > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m12, start_time=T, end_time=0)
    if m21 > 0:
        b.add_migration(source="CEU", dest="YRI", rate=m21, start_time=T, end_time=0)

    return b.resolve()


# ───────────────────────── Optimization ────────────────────────────

def fit_model(
    sfs: moments.Spectrum,
    *,
    start_vec: np.ndarray,
    lb_full: np.ndarray,
    ub_full: np.ndarray,
    genome_scaled_mutation_rate: float,  # mu * L
    param_names: List[str],
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-12,
    model_source: Literal["demes", "msprime"] = "demes",
    demes_builder: Optional[Callable[[Dict[str, float]], demes.Graph]] = None,
    msprime_builder: Optional[Callable[[Dict[str, float]], msprime.Demography]] = None,
) -> np.ndarray:
    """
    Run a single moments optimisation (nlopt.LD_LBFGS) in log10 space.
    Returns fitted REAL-space params (in param_names order).
    """
    assert isinstance(sfs, moments.Spectrum), "sfs must be a moments.Spectrum"

    if np.any(lb_full <= 0) or np.any(ub_full <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb_full, ub_full) if lo <= 0 or hi <= 0]
        raise ValueError(f"All bounds must be positive for log10 optimization. Bad: {bad}")

    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("Observed SFS has no pop_ids; cannot infer sampled_demes order.")

    haploid_sizes = [n - 1 for n in sfs.shape]

    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = _diffusion_sfs(
            log_space_vec=log10_params,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            genome_scaled_mutation_rate=genome_scaled_mutation_rate,
            model_source=model_source,
            demes_builder=demes_builder,
            msprime_builder=msprime_builder,
        )
        # IMPORTANT: keep Spectrum math (mask-aware); do NOT np.asarray
        return float((np.log(exp_sfs + eps) * sfs - exp_sfs).sum())

    grad_fn = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {log10_params}")
        return ll  # maximize

    start_vec = np.asarray(start_vec, dtype=float)
    if start_vec.shape != (len(param_names),):
        raise ValueError(f"start_vec shape {start_vec.shape} != ({len(param_names)},)")

    opt = nlopt.opt(nlopt.LD_LBFGS, start_vec.size)
    opt.set_lower_bounds(np.log10(lb_full))
    opt.set_upper_bounds(np.log10(ub_full))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    x0 = np.log10(start_vec)
    xhat = opt.optimize(x0)

    fitted_params = 10 ** xhat
    return fitted_params


# ───────────────────────── Simulation helpers ────────────────────────────

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
    chrom_length: int = int(1e8),
    mutation_rate: float = 1e-8,
    recomb_rate: float = 1e-8,
    seed: Optional[int] = 295,
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    model = define_sps_model(g)
    contig = sps.get_species("HomSap").get_contig(
        chromosome=None,
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
        seed=seed,
    )
    return ts, model


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

    m_YRI_CEU = float(sampled_params["m_YRI_CEU"])  # forward-time YRI -> CEU
    m_CEU_YRI = float(sampled_params["m_CEU_YRI"])  # forward-time CEU -> YRI

    demogr = msprime.Demography()
    demogr.add_population(name="YRI", initial_size=N_YRI)
    demogr.add_population(name="CEU", initial_size=N_CEU)
    demogr.add_population(name="ANC", initial_size=N_ANC)

    # forward-time YRI -> CEU  ==> backward-time CEU -> YRI
    demogr.set_migration_rate(source="CEU", dest="YRI", rate=m_YRI_CEU)

    # forward-time CEU -> YRI  ==> backward-time YRI -> CEU
    demogr.set_migration_rate(source="YRI", dest="CEU", rate=m_CEU_YRI)

    demogr.add_population_split(time=T_split, ancestral="ANC", derived=["YRI", "CEU"])
    return demogr


def simulate_msprime(
    demogr: msprime.Demography,
    seqlen: int,
    mutation_rate: float,
    recomb_rate: float,
    samples: Dict[str, int],
    seed: Optional[int] = 295,
) -> tskit.TreeSequence:
    ts = msprime.sim_ancestry(
        samples=samples,
        sequence_length=seqlen,
        recombination_rate=recomb_rate,
        demography=demogr,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed)
    return ts


# ───────────────────────── Main ────────────────────────────

def main():
    param_names = ["N_anc", "N_YRI", "N_CEU", "T_split", "m_YRI_CEU", "m_CEU_YRI"]
    start_vec = np.array([5000, 15000, 10000, 10000, 1e-6, 1e-5])
    lb_full = np.array([1000, 1000, 1000, 100, 1e-10, 1e-10])
    ub_full = np.array([50000, 50000, 50000, 20000, 1e-3, 1e-3])

    mu = 1e-8
    L = int(5e7)
    recomb = 1e-8
    samples = {"YRI": 10, "CEU": 10}  # diploid counts for msprime; stdpopsim wrapper may interpret similarly

    genome_scaled_mutation_rate = mu * L

    # -------------------------- Diffusion sanity: DEMES source --------------------------
    true_sfs_demes = _diffusion_sfs(
        log_space_vec=np.log10(start_vec),
        param_names=param_names,
        sampled_demes=["YRI", "CEU"],
        haploid_sizes=[20, 20],
        genome_scaled_mutation_rate=genome_scaled_mutation_rate,
        model_source="demes",
        demes_builder=IM_asymmetric_model,
    )

    optim_pars_demes = fit_model(
        sfs=true_sfs_demes,
        start_vec=start_vec,
        lb_full=lb_full,
        ub_full=ub_full,
        genome_scaled_mutation_rate=genome_scaled_mutation_rate,
        param_names=param_names,
        verbose=True,
        model_source="demes",
        demes_builder=IM_asymmetric_model,
    )

    np.testing.assert_allclose(optim_pars_demes, start_vec, rtol=0.1)

    sampled_params = {name: val for name, val in zip(param_names, start_vec)}

    # # -------------------------- Simulate and inference: stdpopsim version --------------------------
    g = IM_asymmetric_model(sampled_params)

    ts, model = simulation_runner(
        g,
        samples=samples,
        chrom_length=L,
        mutation_rate=mu,
        recomb_rate=recomb,
    )
    print("Simulated TS has sequence_length:", ts.sequence_length)
    print("Simulated TS has num_trees:", ts.num_trees)
    print(model.model.debug())

    sim_sfs = create_SFS(ts, pop_names=["YRI", "CEU"])

    optim_pars_sim = fit_model(
        sfs=sim_sfs,
        start_vec=start_vec,
        lb_full=lb_full,
        ub_full=ub_full,
        genome_scaled_mutation_rate=genome_scaled_mutation_rate,
        param_names=param_names,
        verbose=True,
        model_source="demes",
        demes_builder=IM_asymmetric_model_no_anc,
    )

    print("Optimized parameters from stdpopsim-simulated data:\n", optim_pars_sim)
    print("Ground Truth parameters:\n", start_vec)

    # # -------------------------- Simulate and compare diffusion: msprime source --------------------------
    # demogr = msprime_demography_directly(sampled_params)
    # print(f"Demography for msprime simulation:\n{demogr.debug()}")

    # ts_msprime = simulate_msprime(
    #     demogr,
    #     seqlen=L,
    #     mutation_rate=mu,
    #     recomb_rate=recomb,
    #     samples={"YRI": 10, "CEU": 10},  # diploid sample sizes for msprime simulation
    #     seed=295,
    # )
    # sim_sfs_msprime = create_SFS(ts_msprime, pop_names=["YRI", "CEU"])

    # true_sfs_msprime = _diffusion_sfs(
    #     log_space_vec=np.log10(start_vec),
    #     param_names=param_names,
    #     sampled_demes=["YRI", "CEU"],
    #     haploid_sizes=[20, 20],
    #     genome_scaled_mutation_rate=genome_scaled_mutation_rate,
    #     model_source="msprime",
    #     msprime_builder=msprime_demography_directly,
    # )

    # print("True SFS from diffusion (msprime demography source):\n", np.round(np.log10(np.asarray(true_sfs_msprime)), 2))
    # print("Simulated SFS from msprime:\n", np.round(np.log10(np.asarray(sim_sfs_msprime)), 2))

    # # Optional: fit using msprime-source diffusion (so the optimizer uses msprime->demes conversion)
    # optim_pars_msprime_source = fit_model(
    #     sfs=sim_sfs_msprime,
    #     start_vec=start_vec,
    #     lb_full=lb_full,
    #     ub_full=ub_full,
    #     genome_scaled_mutation_rate=genome_scaled_mutation_rate,
    #     param_names=param_names,
    #     verbose=True,
    #     model_source="msprime",
    #     msprime_builder=msprime_demography_directly,
    # )
    # print("Optimized parameters from msprime-simulated data (msprime-source diffusion):\n", optim_pars_msprime_source)
    # print("Ground Truth parameters:\n", start_vec)


if __name__ == "__main__":
    main()

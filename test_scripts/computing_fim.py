#!/usr/bin/env python3
# Compute Poisson FIM for a split-migration model at ground-truth parameters.
# Steps:
# 1) Define demographic model
# 2) (No optimization) Use ground-truth parameters
# 3) Compute expected SFS + simulate observed SFS
# 4) Poisson log-likelihood at GT
# 5) Fisher Information via numerical Jacobian in log-parameter space

import os
import pickle
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import demes
import demesdraw
import msprime
import moments

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
outdir = "/sietch_colab/akapoor/Infer_Demography/testing_FIM_code/split_migration"
EPS = 1e-12  # for safe division/ logs

# ────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────
def split_migration_model(sampled_params: Dict[str, float]) -> demes.Graph:
    """
    Build a demes.Graph for the split-migration model with asymmetric flow.
    Params (original scale): N0, N1, N2, m12, m21, t_split
    """
    N0, N1, N2, m12, m21, t_split = (
        sampled_params["N0"],
        sampled_params["N1"],
        sampled_params["N2"],
        sampled_params["m12"],
        sampled_params["m21"],
        sampled_params["t_split"],
    )

    b = demes.Builder()
    # Ancestral deme until t_split
    b.add_deme("N0", epochs=[dict(start_size=N0, end_time=t_split)])
    # Split into N1, N2
    b.add_deme("N1", ancestors=["N0"], epochs=[dict(start_size=N1)])
    b.add_deme("N2", ancestors=["N0"], epochs=[dict(start_size=N2)])
    # Asymmetric migration
    b.add_migration(source="N1", dest="N2", rate=m12)
    b.add_migration(source="N2", dest="N1", rate=m21)
    return b.resolve()

# ────────────────────────────────────────────────────────────────────────────
# SFS helpers
# ────────────────────────────────────────────────────────────────────────────
def _diffusion_sfs(
    init_vec: np.ndarray,
    demo_model,                       # callable(param_dict) → demes.Graph
    param_names: List[str],
    sample_sizes: OrderedDict[str, int],
    experiment_config: Dict,
) -> moments.Spectrum:
    """
    Build a frequency spectrum for a given parameter vector (init_vec).
    No pts supplied — moments chooses its own integration grid.
    """
    p_dict = dict(zip(param_names, init_vec))
    graph = demo_model(p_dict)

    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())

    # theta = 4*N0*mu*L; here we use N0 as reference
    theta = (
        p_dict[param_names[0]]
        * 4.0
        * experiment_config["mutation_rate"]
        * experiment_config["genome_length"]
    )

    return moments.Spectrum.from_demes(
        graph,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
        theta=theta,
    )

def poisson_loglikelihood(observed_sfs, expected_sfs) -> float:
    """
    Poisson composite log-likelihood:
        sum_k [ x_k * log(mu_k) - mu_k ]
    Assumes RAW counts and matching polarization/folding.
    """
    expected = np.clip(np.asarray(expected_sfs, dtype=float), EPS, None)
    obs = np.asarray(observed_sfs, dtype=float)
    return float(np.sum(np.log(expected) * obs - expected))

# ────────────────────────────────────────────────────────────────────────────
# FIM helpers (numerical Jacobian in log-parameter space)
# ────────────────────────────────────────────────────────────────────────────
def _expected_sfs_flat(
    theta_vec: np.ndarray,
    param_names: List[str],
    demo_model,
    sample_sizes: OrderedDict[str, int],
    experiment_config: Dict,
) -> np.ndarray:
    """Expected SFS flattened to 1D, clipped away from 0 for stability."""
    sfs = _diffusion_sfs(
        init_vec=theta_vec,
        demo_model=demo_model,
        param_names=param_names,
        sample_sizes=sample_sizes,
        experiment_config=experiment_config,
    )
    mu = np.asarray(sfs, dtype=float).ravel()
    return np.clip(mu, EPS, None)

def fisher_information_poisson(
    theta_vec: np.ndarray,
    param_names: List[str],
    demo_model,
    sample_sizes: OrderedDict[str, int],
    experiment_config: Dict,
):
    """
    Compute Poisson FIM at theta_vec.
    Internally differentiate wrt log-parameters (phi = log theta) for stability,
    then convert back to the original parameter space.

    Returns
    -------
    I_param : (p, p) Fisher Information in original parameter space
    Cov_param : (p, p) ≈ (I_param)^{-1} (Cramér–Rao covariance)
    """
    theta_vec = np.asarray(theta_vec, dtype=float)
    phi0 = np.log(theta_vec)

    def f_phi(phi):
        return _expected_sfs_flat(np.exp(phi), param_names, demo_model, sample_sizes, experiment_config)

    # Jacobian wrt phi: shape (K, p)
    J_phi = nd.Jacobian(f_phi, method="central")(phi0)
    mu = f_phi(phi0)  # (K,)

    # I_phi = J_phi^T diag(1/mu) J_phi
    W = 1.0 / mu
    I_phi = J_phi.T @ (W[:, None] * J_phi)

    # Convert: I_phi = D * I_param * D, with D = diag(theta)
    D_inv = np.diag(1.0 / theta_vec)
    I_param = D_inv @ I_phi @ D_inv

    # Covariance ≈ inverse FIM (use tiny ridge for near-singularity)
    ridge = 1e-10 * np.eye(I_param.shape[0])
    Cov_param = np.linalg.pinv(I_param + ridge)
    return I_param, Cov_param

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(outdir, exist_ok=True)

    # Ground-truth parameter set (evaluation point)
    sampled_params = {
        "N0": 10000.0,
        "N1": 5000.0,
        "N2": 3000.0,
        "m12": 0.001,
        "m21": 0.002,
        "t_split": 2000.0,
    }

    # Plot demography
    graph = split_migration_model(sampled_params)
    ax = demesdraw.tubes(graph)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("N")
    plt.savefig(f"{outdir}/demes.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # Param order + vector at ground truth
    param_names = list(sampled_params.keys())
    init_vec = np.array([sampled_params[name] for name in param_names], dtype=float)
    print("Ground-truth parameter vector (order matches param_names):")
    print(param_names)
    print(init_vec)

    # Sample sizes (diploids) for the two derived demes
    sample_sizes = OrderedDict([("N1", 20), ("N2", 20)])  # 20 diploids each

    # Experiment config
    experiment_config = {
        "mutation_rate": 1e-8,     # per base per generation
        "genome_length": 1e7,      # total length of genome
        "recombination_rate": 1e-8,
        "seed": 42,
    }

    # Expected SFS at ground truth
    expected_sfs = _diffusion_sfs(
        init_vec,
        split_migration_model,
        param_names,
        sample_sizes,
        experiment_config,
    )
    Path(f"{outdir}/expected_sfs.pkl").write_bytes(pickle.dumps(expected_sfs))
    print(f"Saved expected SFS → {outdir}/expected_sfs.pkl")

    # Simulate observed data under the same model for a sanity check
    demog = msprime.Demography.from_demes(split_migration_model(sampled_params))
    ts = msprime.sim_ancestry(
        samples=sample_sizes,  # dict of deme name -> number of diploid samples
        demography=demog,
        sequence_length=experiment_config["genome_length"],
        recombination_rate=experiment_config["recombination_rate"],
        random_seed=experiment_config["seed"],
    )
    ts = msprime.sim_mutations(
        ts,
        rate=experiment_config["mutation_rate"],
        random_seed=experiment_config["seed"],
    )

    # Observed SFS (RAW counts; polarized)
    sample_sets = [
        ts.samples(population=pop.id)
        for pop in ts.populations()
        if len(ts.samples(population=pop.id)) > 0
    ]
    obs = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False,  # RAW counts for Poisson CL
    )
    obs = moments.Spectrum(obs)

    # Use readable pop names (drop ancestral if present)
    pop_names = [pop.metadata.get("name", f"pop{pop.id}") for pop in ts.populations()]
    if len(pop_names) > 1 and pop_names[0] == "N0":
        pop_names = pop_names[1:]
    obs.pop_ids = pop_names

    Path(f"{outdir}/SFS.pkl").write_bytes(pickle.dumps(obs))
    print(f"Saved observed SFS → {outdir}/SFS.pkl")

    # Log-likelihood at ground truth
    ll_pois = poisson_loglikelihood(observed_sfs=obs, expected_sfs=expected_sfs)
    print(f"Poisson composite log-likelihood at ground truth: {ll_pois:.3f}")

    # Fisher Information + CRLB covariance at ground truth
    I_param, Cov_param = fisher_information_poisson(
        theta_vec=init_vec,
        param_names=param_names,
        demo_model=split_migration_model,
        sample_sizes=sample_sizes,
        experiment_config=experiment_config,
    )
    se = np.sqrt(np.diag(Cov_param))

    print("\nFIM (original parameter space):")
    print(I_param)
    print("\nStd. errors at ground truth (CRLB):")
    print(dict(zip(param_names, se)))

    # Save artifacts with unambiguous names
    np.save(f"{outdir}/FIM_param.npy", I_param)
    np.save(f"{outdir}/Cov_param.npy", Cov_param)
    Path(f"{outdir}/fim_summary.pkl").write_bytes(
        pickle.dumps(
            dict(
                param_names=param_names,
                params_at_eval=init_vec,
                se=np.array(se, dtype=float),
                ll_pois=float(ll_pois),
            )
        )
    )
    print(f"\nSaved FIM → {outdir}/FIM_param.npy")
    print(f"Saved Covariance → {outdir}/Cov_param.npy")
    print(f"Saved summary → {outdir}/fim_summary.pkl")

if __name__ == "__main__":
    main()

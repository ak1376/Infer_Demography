#!/usr/bin/env python3
"""
VERY SIMPLE posterior predictive check (PPC).

This script:
1) Hard-codes predicted demographic parameters
2) Loads experiment config
3) Simulates ONE observed dataset
4) Simulates many replicate datasets under predicted parameters
5) Compares TOTAL number of segregating sites
6) Prints a simple PPC p-value
7) Plots one histogram

Nothing else.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------- Repo imports -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import simulation, create_SFS


# ----------------- SETTINGS -----------------

EXPERIMENT_CONFIG_PATH = Path(
    "/sietch_colab/akapoor/Infer_Demography/config_files/experiment_config_IM_symmetric.json"
)

MODEL_TYPE = "IM_symmetric"
POP_NAMES = ["YRI", "CEU"]

NUM_REPLICATES = 100
BASE_SEED = 1
SEED_STRIDE = 10000

# OBSERVED_MODE = "different_params"  # or "same_as_predicted"
OBSERVED_MODE = "same_as_predicted"

OUTDIR = Path("/sietch_colab/akapoor/Infer_Demography/debug_ppc_out")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ----------------- PARAMETERS -----------------

PREDICTED_PARAMS = {
    "N_anc": 1000,
    "N_YRI": 5000,
    "N_CEU": 2000,
    "m": 1e-5,
    "T_split": 5000,
}

OBSERVED_PARAMS = {
    "N_anc": 1200,
    "N_YRI": 7000,
    "N_CEU": 1500,
    "m": 5e-6,
    "T_split": 8000,
}


# ----------------- Helper -----------------

def simulate_total_snps(params: dict, experiment_config: dict, rep_index: int) -> float:
    """Simulate one dataset and return total number of segregating sites."""
    cfg = dict(experiment_config)
    base_seed = cfg.get("seed", BASE_SEED)
    cfg["seed"] = int(base_seed + rep_index * SEED_STRIDE)

    ts, _ = simulation(
        sampled_params=params,
        model_type=MODEL_TYPE,
        experiment_config=cfg,
        sampled_coverage=None,
    )

    sfs = create_SFS(ts=ts, pop_names=POP_NAMES)
    sfs = np.asarray(sfs, dtype=float)

    return sfs.sum()


# ----------------- MAIN -----------------

def main():

    with open(EXPERIMENT_CONFIG_PATH) as f:
        experiment_config = json.load(f)

    if OBSERVED_MODE == "same_as_predicted":
        obs_params = PREDICTED_PARAMS
    else:
        obs_params = OBSERVED_PARAMS

    print("=== SIMPLE PPC ===")
    print("Observed mode:", OBSERVED_MODE)
    print("Replicates:", NUM_REPLICATES)

    # Simulate observed dataset
    obs_total = simulate_total_snps(obs_params, experiment_config, rep_index=9999)

    # Simulate replicates under predicted params
    rep_totals = []
    for i in tqdm(range(NUM_REPLICATES), desc="Simulating replicates"):
        rep_totals.append(
            simulate_total_snps(PREDICTED_PARAMS, experiment_config, rep_index=i)
        )

    rep_totals = np.array(rep_totals)

    # Print summary
    print("\nObserved total SNPs:", obs_total)
    print("Replicate totals:")
    print("  min:", rep_totals.min())
    print("  median:", np.median(rep_totals))
    print("  max:", rep_totals.max())

    # Simple PPC p-value
    p_simple = np.mean(rep_totals >= obs_total)
    print("\nSimple PPC p-value:", p_simple)

    # Plot
    plt.figure()
    plt.hist(rep_totals, bins=30)
    plt.axvline(obs_total, linestyle="--")
    plt.title(f"Total SNP PPC (p={p_simple:.3f})")
    plt.xlabel("Total segregating sites")
    plt.tight_layout()
    plt.savefig(OUTDIR / "simple_ppc_total_snps.png", dpi=200)
    plt.close()

    print("\nSaved plot to:", OUTDIR / "simple_ppc_total_snps.png")


if __name__ == "__main__":
    main()
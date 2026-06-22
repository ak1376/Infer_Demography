"""
Neutral PPC for Drosophila — LD statistics.

For each of N_SIM neutral simulations, computes the theoretical LD decay
curve (σD²) directly from the sampled parameters using moments.Demes.LD,
then overlays the empirical LD decay from the real data.

No LD window simulations are needed: theoretical predictions are computed
analytically, the same way the MomentsLD optimizer evaluates the likelihood.

Usage:
    python neutral_ppc_ld_drosophila.py
"""

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import moments
import moments.LD

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.MomentsLD_inference import (
    _load_demographic_function,
    compute_theoretical_ld,
    DEFAULT_R_BINS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIM_DIR     = Path("experiments_neutral/split_migration_growth/simulations")
REAL_LD_PKL = Path("experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD/means.varcovs.pkl")
EXP_CFG     = Path("config_files/experiment_config_split_migration_growth.json")
OUT_DIR     = Path("model_calibration_drosophila")
OUT_DIR.mkdir(exist_ok=True)

N_SIM       = 100          # number of simulations to use for the envelope
R_BINS      = DEFAULT_R_BINS
NORMALIZATION = 0

# ---------------------------------------------------------------------------
# Stat labels for a 2-population model (CO, FR)
# First 3 after removing normalized stats are DD_0_0, DD_0_1, DD_1_1
# ---------------------------------------------------------------------------
STAT_LABELS = [
    r"$\sigma^2_D$ (CO–CO)",
    r"$\sigma^2_D$ (CO–FR)",
    r"$\sigma^2_D$ (FR–FR)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r_bin_midpoints(r_bins: np.ndarray) -> np.ndarray:
    return (r_bins[:-1] + r_bins[1:]) / 2.0


def theory_to_arrays(sigmaD2_ld, normalization: int = 0):
    """
    Process a theoretical σD² LDstats object into a 2D array
    (n_rbins × n_stats) matching the format used by the likelihood.
    Removes normalized statistics and drops the last bin (heterozygosity).
    """
    processed = moments.LD.Inference.remove_normalized_lds(
        sigmaD2_ld, normalization=normalization
    )
    arrays = [np.array(processed[i]) for i in range(len(processed) - 1)]
    return np.array(arrays)   # shape: (n_rbins, n_stats)


def empirical_to_arrays(mv: dict, num_pops: int, normalization: int = 0):
    """
    Process the empirical means.varcovs dict into a 2D array
    (n_rbins × n_stats) matching theory_to_arrays output.
    """
    emp_means  = [np.array(x) for x in mv["means"]]
    emp_covars = [np.array(x) for x in mv["varcovs"]]

    emp_means, _ = moments.LD.Inference.remove_normalized_data(
        emp_means, emp_covars,
        normalization=normalization,
        num_pops=num_pops,
    )
    emp_means = emp_means[:-1]   # drop heterozygosity bin
    return np.array(emp_means)   # shape: (n_rbins, n_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = json.loads(EXP_CFG.read_text())
    populations = list(cfg["num_samples"].keys())
    param_names = list(cfg["parameter_order"])

    demo_function = _load_demographic_function(cfg)

    # ---- load real data LD stats -------------------------------------------
    print("Loading observed LD statistics...")
    with REAL_LD_PKL.open("rb") as f:
        mv = pickle.load(f)

    # compute theoretical for any one param set to get num_pops
    sim_dirs = sorted(
        [d for d in SIM_DIR.iterdir() if d.is_dir()],
        key=lambda d: int(d.name),
    )[:N_SIM]

    # use first successful sim to determine num_pops
    _first_params = pickle.load(open(sim_dirs[0] / "sampled_params.pkl", "rb"))
    _log_params   = [np.log10(_first_params[n]) for n in param_names]
    _ld           = compute_theoretical_ld(_log_params, param_names, demo_function, R_BINS, populations)
    num_pops      = _ld.num_pops
    n_stats       = len(theory_to_arrays(_ld, NORMALIZATION)[0])

    obs_arrays = empirical_to_arrays(mv, num_pops, NORMALIZATION)
    print(f"  Empirical array shape: {obs_arrays.shape}  (n_rbins × n_stats)")

    # ---- compute theoretical LD for each simulation -----------------------
    print(f"Computing theoretical LD for {N_SIM} simulations...")
    all_theory = []   # list of (n_rbins × n_stats) arrays

    for i, d in enumerate(sim_dirs):
        try:
            params = pickle.load(open(d / "sampled_params.pkl", "rb"))
            log_params = [np.log10(params[n]) for n in param_names]
            sigmaD2 = compute_theoretical_ld(
                log_params, param_names, demo_function, R_BINS, populations
            )
            arr = theory_to_arrays(sigmaD2, NORMALIZATION)
            all_theory.append(arr)
        except Exception as e:
            print(f"  skipping sim {d.name}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_SIM}")

    all_theory = np.array(all_theory)   # shape: (n_sims, n_rbins, n_stats)
    print(f"  Theory array shape: {all_theory.shape}")

    # ---- plot --------------------------------------------------------------
    r_mids = r_bin_midpoints(R_BINS)
    n_plot = min(3, n_stats)            # plot first 3 stats (DD_0_0, DD_0_1, DD_1_1)

    lo  = np.percentile(all_theory, 5,  axis=0)   # (n_rbins, n_stats)
    hi  = np.percentile(all_theory, 95, axis=0)
    mn  = np.mean(all_theory,           axis=0)

    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), sharey=False)
    if n_plot == 1:
        axes = [axes]

    for col, ax in enumerate(axes):
        ax.fill_between(
            r_mids, lo[:, col], hi[:, col],
            alpha=0.3, color="steelblue", label="5–95% CI (sim)",
        )
        ax.plot(r_mids, mn[:, col],        color="steelblue", lw=1.5, label="mean sim")
        ax.plot(r_mids, obs_arrays[:, col], color="red",       lw=1.5, label="observed")
        ax.set_xscale("log")
        ax.set_xlabel("Recombination rate $r$", fontsize=10)
        ax.set_ylabel(STAT_LABELS[col], fontsize=10)
        ax.set_title(STAT_LABELS[col], fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Neutral PPC — LD decay (drosophila split_migration_growth)", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "neutral_ppc_ld.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

"""
Neutral PPC for Drosophila — LD statistics.

For each of N_SIM neutral simulations, computes the theoretical LD decay
curve (σD²) directly from the sampled parameters using moments.Demes.LD,
then overlays the empirical LD decay from the real data.

No LD window simulations are needed: theoretical predictions are computed
analytically, the same way the MomentsLD optimizer evaluates the likelihood.

Usage:
    python neutral_ppc_ld_drosophila.py --model split_migration_growth
    python neutral_ppc_ld_drosophila.py --model drosophila_three_epoch
"""

from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import moments
import moments.LD

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.MomentsLD_inference import (
    _load_demographic_function,
    compute_theoretical_ld,
    DEFAULT_R_BINS,
)

# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "split_migration_growth": {
        "config":      "config_files/experiment_config_split_migration_growth.json",
        "sim_dir":     "experiments_neutral/split_migration_growth/simulations",
        "real_ld_pkl": "experiments/split_migration_growth/real_data_analysis/inferences/MomentsLD/means.varcovs.pkl",
        "pop_labels":  ["CO", "FR"],
        "out_name":    "neutral_ppc_ld_split_migration_growth.png",
    },
    "drosophila_three_epoch": {
        "config":      "config_files/experiment_config_drosophila_three_epoch.json",
        "sim_dir":     "experiments_neutral/drosophila_three_epoch/simulations_ld_ppc",
        "real_ld_pkl": "experiments/drosophila_three_epoch/real_data_analysis/inferences/MomentsLD/means.varcovs.pkl",
        "pop_labels":  ["CO", "FR"],
        "out_name":    "neutral_ppc_ld_drosophila_three_epoch.png",
    },
}

N_SIM         = 100
R_BINS        = DEFAULT_R_BINS
NORMALIZATION = 0
OUT_DIR       = Path("model_calibration_drosophila_model")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r_bin_midpoints(r_bins: np.ndarray) -> np.ndarray:
    return (r_bins[:-1] + r_bins[1:]) / 2.0


def theory_to_arrays(sigmaD2_ld, normalization: int = 0):
    processed = moments.LD.Inference.remove_normalized_lds(
        sigmaD2_ld, normalization=normalization
    )
    arrays = [np.array(processed[i]) for i in range(len(processed) - 1)]
    return np.array(arrays)   # (n_rbins, n_stats)


def empirical_to_arrays(mv: dict, num_pops: int, normalization: int = 0):
    emp_means  = [np.array(x) for x in mv["means"]]
    emp_covars = [np.array(x) for x in mv["varcovs"]]
    emp_means, _ = moments.LD.Inference.remove_normalized_data(
        emp_means, emp_covars,
        normalization=normalization,
        num_pops=num_pops,
    )
    return np.array(emp_means[:-1])   # (n_rbins, n_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="split_migration_growth",
        help="Which demographic model to run the PPC for.",
    )
    args = parser.parse_args()

    mc = MODEL_CONFIGS[args.model]

    cfg         = json.loads(Path(mc["config"]).read_text())
    populations = list(cfg["num_samples"].keys())
    param_names = list(cfg["parameter_order"])
    pop_labels  = mc["pop_labels"]

    demo_function = _load_demographic_function(cfg)
    OUT_DIR.mkdir(exist_ok=True)

    # ---- load real data LD stats -------------------------------------------
    print("Loading observed LD statistics...")
    with Path(mc["real_ld_pkl"]).open("rb") as f:
        mv = pickle.load(f)

    # ---- collect simulation directories ------------------------------------
    sim_dirs = sorted(
        [d for d in Path(mc["sim_dir"]).iterdir() if d.is_dir()],
        key=lambda d: int(d.name),
    )[:N_SIM]

    # probe first sim to get num_pops / n_stats
    with open(sim_dirs[0] / "sampled_params.pkl", "rb") as f:
        _params = pickle.load(f)
    _log_p     = [np.log10(_params[n]) for n in param_names]
    _ld        = compute_theoretical_ld(_log_p, param_names, demo_function, R_BINS, populations)
    num_pops   = _ld.num_pops
    n_stats    = len(theory_to_arrays(_ld, NORMALIZATION)[0])

    obs_arrays = empirical_to_arrays(mv, num_pops, NORMALIZATION)
    print(f"  Empirical array shape: {obs_arrays.shape}  (n_rbins × n_stats)")

    # ---- compute theoretical LD for each simulation -----------------------
    print(f"Computing theoretical LD for {N_SIM} simulations ({args.model})...")
    all_theory = []

    for i, d in enumerate(sim_dirs):
        try:
            with open(d / "sampled_params.pkl", "rb") as f:
                params = pickle.load(f)
            log_params = [np.log10(params[n]) for n in param_names]
            sigmaD2    = compute_theoretical_ld(
                log_params, param_names, demo_function, R_BINS, populations
            )
            all_theory.append(theory_to_arrays(sigmaD2, NORMALIZATION))
        except Exception as e:
            print(f"  skipping sim {d.name}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_SIM}")

    all_theory = np.array(all_theory)   # (n_sims, n_rbins, n_stats)
    print(f"  Theory array shape: {all_theory.shape}")

    # ---- plot --------------------------------------------------------------
    r_mids = r_bin_midpoints(R_BINS)
    n_plot = min(3, n_stats)   # DD_0_0, DD_0_1, DD_1_1

    stat_labels = [
        rf"$\sigma^2_D$ ({pop_labels[0]}–{pop_labels[0]})",
        rf"$\sigma^2_D$ ({pop_labels[0]}–{pop_labels[1]})",
        rf"$\sigma^2_D$ ({pop_labels[1]}–{pop_labels[1]})",
    ]

    lo = np.percentile(all_theory, 5,  axis=0)
    hi = np.percentile(all_theory, 95, axis=0)
    mn = np.mean(all_theory,           axis=0)

    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), sharey=False)
    if n_plot == 1:
        axes = [axes]

    for col, ax in enumerate(axes):
        ax.fill_between(
            r_mids, lo[:, col], hi[:, col],
            alpha=0.3, color="steelblue", label="5–95% CI (sim)",
        )
        ax.plot(r_mids, mn[:, col],         color="steelblue", lw=1.5, label="mean sim")
        ax.plot(r_mids, obs_arrays[:, col], color="red",       lw=1.5, label="observed")
        ax.set_xscale("log")
        ax.set_xlabel("Recombination rate $r$", fontsize=10)
        ax.set_ylabel(stat_labels[col], fontsize=10)
        ax.set_title(stat_labels[col], fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(f"Neutral PPC — LD decay ({args.model})", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / mc["out_name"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

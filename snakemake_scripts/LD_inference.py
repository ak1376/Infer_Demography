#!/usr/bin/env python3
"""
Optimise moments-LD for ONE simulation folder.

CLI:
    python LD_inference.py \
        --sim-dir      experiments/<model>/simulations/67 \
        --LD_dir       experiments/<model>/inferences/sim_67/MomentsLD \
        --config-file  config_files/experiment_config_<model>.json \
        --num-windows  100 \
        --r-bins       "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

Produces in --LD_dir:
    means.varcovs.pkl
    bootstrap_sets.pkl
    empirical_vs_theoretical_comparison.pdf
    best_fit.pkl
    (optionally) fail_reason.txt
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import moments
import msprime  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# local imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation import (  # type: ignore
    bottleneck_model,
    split_isolation_model,
    split_migration_model,
    drosophila_three_epoch,
)
from Moments_LD_theoretical import split_asym_mig_MomentsLD  # type: ignore


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--sim-dir",     required=True, type=Path)
    cli.add_argument("--LD_dir",      required=True, type=Path)
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--num-windows", required=True, type=int)
    cli.add_argument("--r-bins",      required=True,
                     help="comma-separated list of r-bin edges")
    args = cli.parse_args()

    # paths
    sim_dir: Path = args.sim_dir.resolve()      # inputs from simulations
    LD_dir:  Path = args.LD_dir.resolve()       # outputs for LD inference
    ld_stats_dir  = LD_dir / "LD_stats"
    LD_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = json.loads(args.config_file.read_text())
    samp: Dict[str, Any] = pickle.load((sim_dir / "sampled_params.pkl").open("rb"))

    r_bins = np.array([float(x) for x in args.r_bins.split(",")], dtype=float)

    mv_path   = LD_dir / "means.varcovs.pkl"
    boot_path = LD_dir / "bootstrap_sets.pkl"
    pdf_path  = LD_dir / "empirical_vs_theoretical_comparison.pdf"
    best_fit_path = LD_dir / "best_fit.pkl"

    # ------------------------------------------------------------------ load LD stats
    ld_stats: Dict[int, moments.LD.LDstats] = {}
    for w in range(args.num_windows):
        p = ld_stats_dir / f"LD_stats_window_{w}.pkl"
        if not p.exists():
            # fallback to old lowercase/zero-padded convention if present
            alt = ld_stats_dir / f"ld_stats_window_{w:04d}.pkl"
            if not alt.exists():
                raise FileNotFoundError(f"Missing LD stats file for window {w}: {p} or {alt}")
            p = alt
        with p.open("rb") as fh:
            ld_stats[w] = pickle.load(fh)

    # ------------------------------------------------------------------ summary stats
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    pickle.dump(mv, mv_path.open("wb"))
    pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_stats), boot_path.open("wb"))

    # ------------------------------------------------------------------ theoretical curve
    if cfg["demographic_model"] == "bottleneck":
        demo_graph_func = bottleneck_model
    elif cfg["demographic_model"] == "split_isolation":
        demo_graph_func = split_isolation_model
    elif cfg["demographic_model"] == "split_migration":
        demo_graph_func = split_migration_model
    elif cfg["demographic_model"] == "drosophila_three_epoch":
        demo_graph_func = drosophila_three_epoch
    else:
        raise ValueError(f"Unsupported demographic model: {cfg['demographic_model']}")

    sampled_demes = sorted(cfg["num_samples"])  # e.g., ["N0"] or ["N0","N1"]
    g = demo_graph_func(samp)
    y = moments.Demes.LD(g, sampled_demes=sampled_demes, rho=4 * samp["N0"] * r_bins)
    y = moments.LD.LDstats([(a + b) / 2 for a, b in zip(y[:-2], y[1:-1])] + [y[-1]],
                           num_pops=y.num_pops, pop_ids=y.pop_ids)
    y = moments.LD.Inference.sigmaD2(y)

    # plot curves
    if cfg["demographic_model"] == "bottleneck":
        stats_to_plot = [["DD_0_0"], ["Dz_0_0_0"], ["pi2_0_0_0_0"]]
        labels = [[r"$D_0^2$"], [r"$Dz_{0,0,0}$"], [r"$\pi_{2;0,0,0,0}$"]]
    elif cfg["demographic_model"] in ["split_isolation", "split_migration", "drosophila_three_epoch"]:
        stats_to_plot = [
            ["DD_0_0"], ["DD_0_1"], ["DD_1_1"],
            ["Dz_0_0_0"], ["Dz_0_1_1"], ["Dz_1_1_1"],
            ["pi2_0_0_1_1"], ["pi2_0_1_0_1"], ["pi2_1_1_1_1"],
        ]
        labels = [
            [r"$D_0^2$"], [r"$D_0 D_1$"], [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"], [r"$Dz_{0,1,1}$"], [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"], [r"$\pi_{2;0,1,0,1}$"], [r"$\pi_{2;1,1,1,1}$"],
        ]
    else:
        raise ValueError(f"Unsupported demographic model: {cfg['demographic_model']}")

    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=r_bins,
        stats_to_plot=stats_to_plot,
        labels=labels,
        rows=3,
        plot_vcs=True,
        show=False,
        fig_size=(6, 4),
    )
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------ optimisation
    # midpoint of prior ranges
    prior_means: Dict[str, float] = {
        name: 0.5 * (low + high) for name, (low, high) in cfg["priors"].items()
    }
    print(f"Initial guess for parameters based on prior means: {prior_means}")

    model = cfg["demographic_model"]
    fixed_params = [None] * (len(prior_means))
    print(f"The length of fixed_params is {len(fixed_params)}")

    # p_guess and demo_func for the LD optimizer
    if model == "bottleneck":
        demo_func = moments.LD.Demographics1D.three_epoch
        p_guess = [
            samp["N_bottleneck"] / samp["N0"],
            prior_means["N_recover"] / prior_means["N0"],
            (prior_means["t_bottleneck_start"] - prior_means["t_bottleneck_end"]) / (2 * prior_means["N0"]),
            prior_means["t_bottleneck_end"] / (2 * prior_means["N0"]),
            samp["N0"],
        ]
        fixed_params = [p_guess[0], None, None, None, p_guess[4]]

    elif model == "split_isolation":
        demo_func = moments.LD.Demographics2D.split_mig
        p_guess = [
            prior_means["N1"] / prior_means["N0"],
            prior_means["N2"] / prior_means["N0"],
            prior_means["t_split"] / (2 * prior_means["N0"]),
            2 * prior_means["N0"] * prior_means["m"],  # m in LD model
            prior_means["N0"],
        ]

    elif model == "split_migration":
        demo_func = split_asym_mig_MomentsLD
        p_guess = [
            prior_means["N1"] / prior_means["N0"],
            prior_means["N2"] / prior_means["N0"],
            prior_means["t_split"] / (2 * prior_means["N0"]),
            prior_means["m12"],
            prior_means["m21"],
            prior_means["N0"],
        ]

    elif model == "drosophila_three_epoch":
        # TODO: choose an appropriate LD demo function
        demo_func = moments.LD.Demographics1D.three_epoch
        p_guess = [
            prior_means["AFR"],
            prior_means["EUR_recover"],
            (prior_means["T_AFR_expansion"] - prior_means["T_AFR_EUR_split"]) / (2 * prior_means["N0"]),
            prior_means["T_AFR_EUR_split"] / (2 * prior_means["N0"]),
            prior_means["N0"],
        ]

    else:
        raise ValueError(f"Need p_guess mapping for model '{model}'")

    # keys & rescale_types once
    if model == "bottleneck":
        keys = ["N_bottleneck", "N_recover", "t_bottleneck_start", "t_bottleneck_end", "N0"]
        rescale_types = ["nu", "nu", "T", "T", "Ne"]
    elif model == "split_isolation":
        keys = ["N1", "N2", "t_split", "m", "N0"]
        rescale_types = ["nu", "nu", "T", "m", "Ne"]
    elif model == "split_migration":
        keys = ["N1", "N2", "t_split", "m12", "m21", "N0"]
        rescale_types = ["nu", "nu", "T", "m12", "m21", "Ne"]
    elif model == "drosophila_three_epoch":
        keys = ["AFR", "EUR_recover", "T_AFR_expansion", "T_AFR_EUR_split", "N0"]
        rescale_types = ["nu", "nu", "T", "T", "Ne"]
    else:
        raise ValueError(f"Need mapping for model '{model}'")

    try:
        opt_params, LL = moments.LD.Inference.optimize_log_fmin(
            p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins,
            fixed_params=fixed_params, verbose=0
        )
        physical = moments.LD.Util.rescale_params(opt_params, rescale_types)
        best_fit = dict(zip(keys, physical))

    except Exception as e:
        # Graceful fallback: write placeholder so Snakemake succeeds
        print(f"[WARN] Moments-LD optimisation failed for {LD_dir}: {type(e).__name__}: {e}")
        best_fit = {k: None for k in keys}
        LL = float("nan")
        (LD_dir / "fail_reason.txt").write_text(f"{type(e).__name__}: {e}\n")

    pickle.dump({"best_params": best_fit, "best_lls": LL}, best_fit_path.open("wb"))
    print(f"✓ moments-LD finished for {LD_dir}")


if __name__ == "__main__":
    main()

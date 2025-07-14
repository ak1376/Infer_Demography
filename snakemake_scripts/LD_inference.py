#!/usr/bin/env python3
"""
Optimise moments-LD for ONE simulation folder.

CLI:
    python LD_inference.py \
        --sim-dir      MomentsLD/LD_stats/sim_7 \
        --LD_dir       MomentsLD/LD_stats/sim_7/LD_stats \
        --config-file  config_files/experiment_config_bottleneck.json \
        --num-windows  100 \
        --r-bins       "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"
Produces
    means.varcovs.pkl
    bootstrap_sets.pkl
    bottleneck_comparison.pdf
    best_fit.pkl
inside <sim-dir>.
"""
from __future__ import annotations
import argparse, json, pickle, sys, os, time
from pathlib import Path
from typing import Dict, Any

import numpy as np, moments, msprime, matplotlib
matplotlib.use("Agg")  # no X-server needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation import bottleneck_model, split_isolation_model, split_migration_model, drosophila_three_epoch



def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--sim-dir",     required=True, type=Path)
    cli.add_argument("--LD_dir",      required=True, type=Path)
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--num-windows", required=True, type=int)
    cli.add_argument("--r-bins",      required=True,
                     help="comma-separated list of r-bin edges")
    args = cli.parse_args()

    sim_dir   = args.sim_dir.resolve()
    LD_dir    = args.LD_dir.resolve()
    cfg       = json.loads(args.config_file.read_text())

    samp      = pickle.load((sim_dir / "sampled_params.pkl").open("rb"))
    r_bins    = np.array([float(x) for x in args.r_bins.split(',')])
    mv_path   = LD_dir / f"means.varcovs.pkl"
    boot_path = LD_dir / f"bootstrap_sets.pkl"

    # ----------------------------------------------------------------  load LD
    ld_stats: Dict[int, moments.LD.LDstats] = {}
    for w in range(args.num_windows):
        p = LD_dir / "LD_stats" / f"LD_stats_window_{w}.pkl"
        ld_stats[w] = pickle.load(p.open("rb"))

    # ----------------------------------------------------------------  summary
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    pickle.dump(mv, mv_path.open("wb"))
    pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_stats),
                boot_path.open("wb"))

    # ----------------------------------------------------------------  analytic

    if cfg['demographic_model'] == "bottleneck":
        demo_func = bottleneck_model
    elif cfg['demographic_model'] == "split_isolation":
        demo_func = split_isolation_model
    elif cfg['demographic_model'] == "split_migration":
        demo_func = split_migration_model
    elif cfg['demographic_model'] == "drosophila_three_epoch":
        demo_func = drosophila_three_epoch
    else:
        raise ValueError(f"Unsupported demographic model: {cfg['demographic_model']}")

    g = demo_func(samp)
    y = moments.Demes.LD(g, sampled_demes = sorted(cfg["num_samples"]),    # ["N1", "N2"],
                         rho=4 * samp["N0"] * r_bins)
    y = moments.LD.LDstats([(a+b)/2 for a, b in zip(y[:-2], y[1:-1])]
                           + [y[-1]], num_pops=y.num_pops, pop_ids=y.pop_ids)
    y = moments.LD.Inference.sigmaD2(y)
    pdf_path = LD_dir / "empirical_vs_theoretical_comparison.pdf"

    if cfg['demographic_model'] == "bottleneck":
        stats_to_plot = [
            ["DD_0_0"],
            ["Dz_0_0_0"],
            ["pi2_0_0_0_0"],
        ]
        labels = [
            [r"$D_0^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$\pi_{2;0,0,0,0}$"],
        ]

    elif cfg['demographic_model'] in ["split_isolation", "split_migration", "drosophila_three_epoch"]:
        stats_to_plot = [
            ["DD_0_0"],
            ["DD_0_1"],
            ["DD_1_1"],
            ["Dz_0_0_0"],
            ["Dz_0_1_1"],
            ["Dz_1_1_1"],
            ["pi2_0_0_1_1"],
            ["pi2_0_1_0_1"],
            ["pi2_1_1_1_1"],
        ]
        labels = [
            [r"$D_0^2$"],
            [r"$D_0 D_1$"],
            [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$Dz_{0,1,1}$"],
            [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"],
            [r"$\pi_{2;0,1,0,1}$"],
            [r"$\pi_{2;1,1,1,1}$"],
        ]
    else:
        raise ValueError(f"Unsupported demographic model: {cfg['demographic_model']}")

    # Plot LD curves
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

    fig.savefig(pdf_path, dpi=300)   # <- write the PDF
    plt.close(fig)                   # optional: free memory when running many sims


    # optimisation ---------------------------------------------------------

    # Take the mean of the prior distribution for the parameters
    # and use it as the initial guess for the optimisation

    # ------------------------------------------------------------------  initial guess from priors
    # take the midpoint of every [low, high] interval
    prior_means: dict[str, float] = {
        name: 0.5 * (low + high) for name, (low, high) in cfg["priors"].items()
    }

    print(f'Initial guess for parameters based on prior means: {prior_means}')

    model = cfg["demographic_model"]

    fixed_params = [None]*(len(prior_means))  # will be set later
    print(f'The length of fixed_params is {len(fixed_params)}')

    if model == "bottleneck":                       # ⇢ 3‑epoch, 1‑population
        # three_epoch params: (nu1, nu2, T1, T2, Ne)
        demo_func = moments.LD.Demographics1D.three_epoch
        p_guess = [
            prior_means["N_bottleneck"] / prior_means["N0"],           # ν1
            prior_means["N_recover"]    / prior_means["N0"],           # ν2
            (prior_means["t_bottleneck_start"] -                      # T1
            prior_means["t_bottleneck_end"]) / (2 * prior_means["N0"]),
            prior_means["t_bottleneck_end"] / (2 * prior_means["N0"]), # T2
            prior_means["N0"],                                         # Ne
        ]

        fixed_params = [p_guess[0], p_guess[1], None, None, None]

    elif model == "split_isolation":                # ⇢ 2‑pop, split then no mig
        demo_func = moments.LD.Demographics2D.split_mig
        p_guess = [
            prior_means["N1"]  / prior_means["N0"],                    # ν1
            prior_means["N2"]  / prior_means["N0"],                    # ν2
            prior_means["t_split"] / (2 * prior_means["N0"]),          # T_split
            2 * prior_means["N0"] * prior_means["m"],
            prior_means["N0"]                                         # Ne
        ]


    elif model == "split_migration":                # ⇢ 2‑pop, split + mig #TODO: Need to update the island model function to use t_split 
        # (nu1, nu2, T_split, m12, m21, Ne)
        p_guess = [
            prior_means["N1"]  / prior_means["N0"],                    # ν1
            prior_means["N2"]  / prior_means["N0"],                    # ν2
            prior_means["t_split"] / (2 * prior_means["N0"]),          # T_split
            prior_means["m12"],                                        # m12
            prior_means["m21"],                                        # m21
            prior_means["N0"],                                         # Ne
        ]

    elif model == "drosophila_three_epoch":         # ⇢ your stdpopsim wrapper #TODO: Need to write a custom MomentsLD function for this model
        # map however the wrapped function expects; an illustrative example:
        p_guess = [
            prior_means["AFR"],                                        # African size
            prior_means["EUR_recover"],                                # European size
            (prior_means["T_AFR_expansion"] -                          # T1
            prior_means["T_AFR_EUR_split"]) / (2 * prior_means["N0"]),
            prior_means["T_AFR_EUR_split"] / (2 * prior_means["N0"]),  # T2
            prior_means["N0"],                                         # Ne
        ]

    else:
        raise ValueError(f"Need p_guess mapping for model '{model}'")

    # TODO: Need to have a logical way to handle fixed parameters 
    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins,
        fixed_params=fixed_params, verbose=0)
    
    if cfg['demographic_model'] == "bottleneck":
        # rescale the parameters to physical units
        physical = moments.LD.Util.rescale_params(opt_params, ["nu", "nu", "T", "T", "Ne"])
        best_fit = dict(zip(["N_bottleneck", "N_recover", "t_bottleneck_start", "t_bottleneck_end", "N0"], physical))
    elif cfg['demographic_model'] == "split_isolation":
        physical = moments.LD.Util.rescale_params(opt_params, ["nu", "nu", "T", "m", "Ne"])
        best_fit = dict(zip(["N1", "N2", "t_split", "N0"], physical))
    elif cfg['demographic_model'] == "split_migration":
        physical = moments.LD.Util.rescale_params(opt_params, ["nu", "nu", "T", "m12", "m21", "Ne"])
        best_fit = dict(zip(["N1", "N2", "t_split", "m12", "m21", "N0"], physical))
    elif cfg['demographic_model'] == "drosophila_three_epoch":
        # rescale the parameters to physical units
        physical = moments.LD.Util.rescale_params(opt_params, ["nu", "nu", "T", "T", "Ne"])
        best_fit = dict(zip(["AFR", "EUR_recover", "T_AFR_expansion", "T_AFR_EUR_split", "N0"], physical))
    else:
        raise ValueError(f"Need physical rescaling for model '{cfg['demographic_model']}'")
    
    pickle.dump({"opt_params": best_fit, "loglik": LL}, (LD_dir / "best_fit.pkl").open("wb"))

    print(f"✓ moments‑LD finished for {sim_dir.name}  (LL={LL:.3f})")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

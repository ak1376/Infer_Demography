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
from simulation import bottleneck_model            # noqa: E402


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
    g = bottleneck_model(samp)
    y = moments.Demes.LD(g, sampled_demes=["N0"],
                         rho=4 * samp["N0"] * r_bins)
    y = moments.LD.LDstats([(a+b)/2 for a, b in zip(y[:-2], y[1:-1])]
                           + [y[-1]], num_pops=y.num_pops, pop_ids=y.pop_ids)
    y = moments.LD.Inference.sigmaD2(y)
    moments.LD.Plotting.plot_ld_curves_comp(
        y, mv["means"][:-1], mv["varcovs"][:-1], rs=r_bins,
        stats_to_plot=[["DD_0_0"], ["Dz_0_0_0"], ["pi2_0_0_0_0"]],
        labels=[
            [r"$D_0^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$\pi_{2;0,0,0,0}$"],   # ← one “\”, not “\\”
        ],
        rows=3, plot_vcs=True, show=False, fig_size=(6, 4),
        output=str(LD_dir / "bottleneck_comparison.pdf"),
)

    # ----------------------------------------------------------------  fit
    # Set p0 to be the mean of the prior distribution from the config file
    pri = cfg["priors"]
    mean_N0         = (pri["N0"][0]                + pri["N0"][1])                / 2
    mean_N_bot      = (pri["N_bottleneck"][0]      + pri["N_bottleneck"][1])      / 2
    mean_N_rec      = (pri["N_recover"][0]         + pri["N_recover"][1])         / 2
    mean_t_start    = (pri["t_bottleneck_start"][0] + pri["t_bottleneck_start"][1]) / 2
    mean_t_end      = (pri["t_bottleneck_end"][0]   + pri["t_bottleneck_end"][1])   / 2

    # (nu1, nu2, T1, T2, Ne) where
    #   nu1 = N_bot / N0
    #   nu2 = N_rec / N0
    #   T1  = (t_start − t_end) / (2N0)
    #   T2  =  t_end / (2N0)
    p0 = [
        mean_N_bot / mean_N0,                            # nu1
        mean_N_rec / mean_N0,                            # nu2
        (mean_t_start - mean_t_end) / (2 * mean_N0),     # T1
        mean_t_end / (2 * mean_N0),                      # T2
        mean_N0                                         # Ne (optimises Ne simultaneously)
    ]
    # p0 = [samp["N_bottleneck"]/samp["N0"], samp["N_recover"]/samp["N0"],
    #       (samp["t_bottleneck_start"]-samp["t_bottleneck_end"]) /
    #       (2*samp["N0"]), samp["t_bottleneck_end"]/(2*samp["N0"]), samp["N0"]]
    demo = moments.LD.Demographics1D.three_epoch
    opt, LL = moments.LD.Inference.optimize_log_fmin(
        p0, [mv["means"], mv["varcovs"]], [demo], rs=r_bins,
        fixed_params=[p0[0], p0[1], None, None, None], verbose=0)
    phys = moments.LD.Util.rescale_params(
        opt, ["nu", "nu", "T", "T", "Ne"])
    best = dict(zip(["N_bottleneck","N_recover",
                     "t_bottleneck_start","t_bottleneck_end","N0"], phys))
    pickle.dump({"opt_params": best, "loglik": LL},
                (LD_dir / "best_fit.pkl").open("wb"))
    
    print(f"✓ optimise moments-LD on {LD_dir.name} (LL={LL:.3f})")

if __name__ == "__main__":
    main()

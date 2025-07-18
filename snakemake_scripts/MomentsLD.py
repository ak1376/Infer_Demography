#!/usr/bin/env python3
"""Run a full moments‑LD optimisation for **one** run_XXXX directory.

Outputs are written under
    <output-root>/sim_<idx>/
        ├─ windows/window_<j>.vcf.gz               (one per replicate)
        ├─ LD_stats/ld_stats_window_<j>.pkl        (one per replicate)
        ├─ means.varcovs.pkl
        ├─ bootstrap_sets.pkl
        ├─ *DEMOGRAPHIC_MODEL*_comparison.pdf
        └─ best_fit.pkl

Usage (Snakemake will pass these flags):
    python momentsld_optimize.py \
        --run-dir        experiments/bottleneck/runs/run_0001 \
        --config-file    config_files/experiment_config_bottleneck.json \
        --output-root    MomentsLD \
        --reps           100
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import moments
import msprime
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

# ----- local imports ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run_msprime_reps(graph: moments.DemesGraph, *, L: int, u: float, r: float,
                     n: int, num_reps: int, out_dir: Path, seed: int = 42):
    win_dir = out_dir / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    demog = msprime.Demography.from_demes(graph)
    reps  = msprime.sim_ancestry({"N0": n}, demography=demog, sequence_length=L,
                                 recombination_rate=r, num_replicates=num_reps,
                                 random_seed=seed)
    for i, ts in enumerate(reps):
        ts = msprime.sim_mutations(ts, rate=u, random_seed=i + 1)
        vcf = win_dir / f"window_{i:04d}.vcf"
        with vcf.open("w") as fh:
            ts.write_vcf(fh, allow_position_zero=True)
        os.system(f"gzip -f {vcf}")


def write_samples_and_map(*, L: int, r: float, n: int, out_dir: Path):
    (out_dir / "samples.txt").write_text(
        "sample\tpop\n" + "\n".join(f"tsk_{i}\tN0" for i in range(n)) + "\n")
    (out_dir / "flat_map.txt").write_text(
        f"pos\tMap(cM)\n0\t0\n{L}\t{r * L * 100}\n")


def parse_ld_one(rep_i: int, r_bins: np.ndarray, sim_dir: Path) -> moments.LD.LDstats:
    vcf = sim_dir / "windows" / f"window_{rep_i:04d}.vcf.gz"
    stats = moments.LD.Parsing.compute_ld_statistics(
        str(vcf),
        rec_map_file=str(sim_dir / "flat_map.txt"),
        pop_file=str(sim_dir / "samples.txt"),
        pops=["N0"],
        r_bins=r_bins,
        report=False,
    )
    return stats

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    cli = argparse.ArgumentParser("moments‑LD optimise one run (serial)")
    cli.add_argument("--run-dir", type=Path, required=True,
                     help="existing experiments/bottleneck/runs/run_XXXX folder")
    cli.add_argument("--config-file", type=Path, required=True)
    cli.add_argument("--output-root", type=Path, required=True)
    cli.add_argument("--reps", type=int, default=100)
    args = cli.parse_args()

    # ----------------------------------------------------------------------
    with (args.run_dir / "data/sampled_params.pkl").open("rb") as f:
        sampled_params: Dict[str, Any] = pickle.load(f)
    cfg = json.loads(args.config_file.read_text())

    if cfg['demographic_model'] == "bottleneck":
        from simulation import bottleneck_model  as demo_model 
    elif cfg['demographic_model'] == "drosophila_three_epoch":
        from simulation import drosophila_three_epoch as demo_model  # noqa: E402
    elif cfg['demographic_model'] == "split_isolation": 
        from simulation import split_isolation_model as demo_model
    elif cfg['demographic_model'] == 'split_migration':
        from simulation import split_migration_model as demo_model


    run_idx = int(args.run_dir.name.split("_")[-1])
    sim_dir = args.output_root / f"sim_{run_idx:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    ld_dir = sim_dir / "LD_stats"
    ld_dir.mkdir(exist_ok=True)

    mean_file = sim_dir / "means.varcovs.pkl"
    boot_file = sim_dir / "bootstrap_sets.pkl"

    r_bins = np.concatenate(([0], np.logspace(-6, -3, 16)))

    # ----------------------------------------------------------------------
    if not (mean_file.exists() and boot_file.exists()):
        # fresh simulation + LD parsing ------------------------------------
        g = demo_model(sampled_params)
        run_msprime_reps(g, L=cfg["genome_length"], u=cfg["mutation_rate"],
                         r=cfg["recombination_rate"], n=cfg["num_samples"]["N0"],
                         num_reps=args.reps, out_dir=sim_dir)
        write_samples_and_map(L=cfg["genome_length"], r=cfg["recombination_rate"],
                              n=cfg["num_samples"]["N0"], out_dir=sim_dir)

        ld_stats = {}
        for i in range(args.reps):
            stat_pkl = ld_dir / f"ld_stats_window_{i:04d}.pkl"
            if stat_pkl.exists():
                ld_stats[i] = pickle.load(stat_pkl.open("rb"))
                continue
            stats = parse_ld_one(i, r_bins, sim_dir)
            pickle.dump(stats, stat_pkl.open("wb"))
            ld_stats[i] = stats

        mv = moments.LD.Parsing.bootstrap_data(ld_stats)
        pickle.dump(mv, mean_file.open("wb"))
        pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_stats), boot_file.open("wb"))
    else:
        mv = pickle.load(mean_file.open("rb"))

    # Get the population deme names from the cfg file
    sampled_demes = list(cfg['num_samples'].keys())
    print(sampled_demes)

    # ----------------------------------------------------------------------
    g = demo_model(sampled_params)
    y = moments.Demes.LD(g, sampled_demes=sampled_demes, rho=4 * sampled_params["N0"] * r_bins)
    y = moments.LD.LDstats([(yl + yr) / 2 for yl, yr in zip(y[:-2], y[1:-1])] + [y[-1]],
                            num_pops=y.num_pops, pop_ids=y.pop_ids)
    y = moments.LD.Inference.sigmaD2(y)

    pdf_path = sim_dir / "empirical_vs_theoretical_comparison.pdf"

    if cfg['demographic_model'] == "bottleneck_model":
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

    elif cfg['demographic_model'] in ["split_isolation_model", "split_migration_model", "drosophila_three_epoch"]:
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

    model = cfg["demographic_model"]

    fixed_params = [None]*len(prior_means)  # will be set later

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
            prior_means["N0"],                                         # Ne
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
        physical = moments.LD.Util.rescale_params(opt_params, ["nu", "nu", "T", "Ne"])
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
    
    pickle.dump({"best_params": best_fit, "best_lls": LL}, (sim_dir / "best_fit.pkl").open("wb"))

    print(f"✓ moments-LD finished for {sim_dir.relative_to(args.output_root.parent)}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""moments‑LD optimisation for ONE simulation_i folder (serial, no Ray).

Usage (Snakemake passes these flags):
    python optimize_momentsld.py \
        --sim-dir      experiments/bottleneck/simulations/7 \
        --config-file  config_files/experiment_config_bottleneck.json \
        --output-root  MomentsLD \
        --reps         100

The script writes exactly:
MomentsLD/
├─ sim_0007/windows/window_0000.vcf.gz  …
├─ LD_stats/sim_0007/ld_stats_window_0000.pkl …
├─ sim_0007/means.varcovs.pkl & bootstrap_sets.pkl
├─ sim_0007/bottleneck_comparison.pdf
└─ sim_0007/best_fit.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import moments
import msprime
import numpy as np

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulation import bottleneck_model  # noqa: E402

# ------------------------------------------------------------------ helpers

def simulate_windows(graph: moments.DemesGraph, *, L: int, u: float, r: float,
                     n: int, reps: int, out_dir: Path):
    """Simulate *reps* tree‑sequence windows and gzip‑VCFs under *out_dir*/windows."""
    win_dir = out_dir / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    demog = msprime.Demography.from_demes(graph)
    ts_iter = msprime.sim_ancestry({"N0": n}, demography=demog,
                                   sequence_length=L, recombination_rate=r,
                                   num_replicates=reps, random_seed=42)
    for j, ts in enumerate(ts_iter):
        ts = msprime.sim_mutations(ts, rate=u, random_seed=j + 1)
        vcf_path = win_dir / f"window_{j:04d}.vcf"
        with vcf_path.open("w") as fh:
            ts.write_vcf(fh, allow_position_zero=True)
        os.system(f"gzip -f {vcf_path}")


def write_samples_and_map(*, L: int, r: float, n: int, out_dir: Path):
    """Write samples.txt and flat_map.txt required by moments‑LD parsing."""
    (out_dir / "samples.txt").write_text(
        "sample\tpop\n" + "\n".join(f"tsk_{i}\tN0" for i in range(n)) + "\n")
    (out_dir / "flat_map.txt").write_text(
        f"pos\tMap(cM)\n0\t0\n{L}\t{r * L * 100}\n")


def compute_ld(rep_i: int, bins: np.ndarray, sim_dir: Path) -> moments.LD.LDstats:
    vcf_gz = sim_dir / "windows" / f"window_{rep_i:04d}.vcf.gz"
    return moments.LD.Parsing.compute_ld_statistics(
        str(vcf_gz),
        rec_map_file=str(sim_dir / "flat_map.txt"),
        pop_file=str(sim_dir / "samples.txt"),
        pops=["N0"],
        r_bins=bins,
        report=False,
    )

# ------------------------------------------------------------------ main

def main():
    cli = argparse.ArgumentParser("optimise moments‑LD for one simulation folder")
    cli.add_argument("--sim-dir",     type=Path, required=True,
                     help="experiments/bottleneck/simulations/<id>")
    cli.add_argument("--config-file", type=Path, required=True)
    cli.add_argument("--output-root", type=Path, required=True)
    cli.add_argument("--reps",        type=int, default=100)
    args = cli.parse_args()

    # identifiers and output layout ---------------------------------------
    sim_id   = int(args.sim_dir.name)                    # simulations/<id>
    sim_out  = args.output_root / f"sim_{sim_id:04d}"
    ld_stats = args.output_root / "LD_stats" / f"sim_{sim_id:04d}"
    sim_out.mkdir(parents=True, exist_ok=True)
    ld_stats.mkdir(parents=True, exist_ok=True)

    # global config & sampled parameters ----------------------------------
    cfg  = json.loads(args.config_file.read_text())
    samp: Dict[str, Any] = pickle.load((args.sim_dir / "sampled_params.pkl").open("rb"))

    bins     = np.concatenate(([0], np.logspace(-6, -3, 16)))
    means_p  = sim_out / "means.varcovs.pkl"
    boots_p  = sim_out / "bootstrap_sets.pkl"

    if not means_p.exists():
        # 1) simulate windows + write helper files -------------------------
        g = bottleneck_model(samp)
        simulate_windows(g, L=cfg["genome_length"], u=cfg["mutation_rate"],
                          r=cfg["recombination_rate"], n=cfg["num_samples"]["N0"],
                          reps=args.reps, out_dir=sim_out)
        write_samples_and_map(L=cfg["genome_length"], r=cfg["recombination_rate"],
                              n=cfg["num_samples"]["N0"], out_dir=sim_out)

        # 2) compute LD stats for each window -----------------------------
        ld_dict = {}
        for j in range(args.reps):
            pkl = ld_stats / f"ld_stats_window_{j:04d}.pkl"
            if pkl.exists():
                ld_dict[j] = pickle.load(pkl.open("rb"))
                continue
            stats = compute_ld(j, bins, sim_out)
            pickle.dump(stats, pkl.open("wb"))
            ld_dict[j] = stats

        mv = moments.LD.Parsing.bootstrap_data(ld_dict)
        pickle.dump(mv, means_p.open("wb"))
        pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_dict), boots_p.open("wb"))
    else:
        mv = pickle.load(means_p.open("rb"))

    # 3) analytic curve & plotting ---------------------------------------
    g = bottleneck_model(samp)
    y = moments.Demes.LD(g, sampled_demes=["N0"], rho=4 * samp["N0"] * bins)
    y = moments.LD.LDstats([(a + b) / 2 for a, b in zip(y[:-2], y[1:-1])] + [y[-1]],
                            num_pops=y.num_pops, pop_ids=y.pop_ids)
    y = moments.LD.Inference.sigmaD2(y)

    moments.LD.Plotting.plot_ld_curves_comp(
        y, mv["means"][:-1], mv["varcovs"][:-1], rs=bins,
        stats_to_plot=[["DD_0_0"], ["Dz_0_0_0"], ["pi2_0_0_0_0"]],
        labels=[[r"$D_0^2$"], [r"$Dz_{0,0,0}$"], [r"$\pi_{2;0,0,0,0}$"]],
        rows=3, plot_vcs=True, show=False, fig_size=(6, 4),
        output=str(sim_out / "bottleneck_comparison.pdf"))

    # 4) optimisation -----------------------------------------------------
    p0 = [
        samp["N_bottleneck"] / samp["N0"],
        samp["N_recover"] / samp["N0"],
        (samp["t_bottleneck_start"] - samp["t_bottleneck_end"]) / (2 * samp["N0"]),
        samp["t_bottleneck_end"] / (2 * samp["N0"]),
        samp["N0"],
    ]

    demo = moments.LD.Demographics1D.three_epoch
    opt, LL = moments.LD.Inference.optimize_log_fmin(
        p0, [mv["means"], mv["varcovs"]], [demo], rs=bins,
        fixed_params=[p0[0], p0[1], None, None, None])

    phys = moments.LD.Util.rescale_params(opt, ["nu", "nu", "T", "T", "Ne"])
    best = dict(zip(["N_bottleneck", "N_recover", "t_bottleneck_start", "t_bottleneck_end", "N0"], phys))
    pickle.dump({"opt_params": best, "loglik": LL}, (sim_out / "best_fit.pkl").open("wb"))

    print(f"✓ moments‑LD finished for simulation {sim_id:04d}")

if __name__ == "__main__":
    main()

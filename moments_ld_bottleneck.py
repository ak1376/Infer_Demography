from __future__ import annotations
import os
import time
import gzip
import numpy as np
import pickle
import msprime
import moments
import demes
import argparse, json, pickle, sys, os
from pathlib import Path
from typing  import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import demesdraw
import ray
import moments

# ── local & project paths ────────────────────────────────────────────
# Treat the folder *containing* this script as the project root
PROJECT_ROOT = Path(__file__).resolve().parent       # /sietch_colab/akapoor/Infer_Demography
SRC          = PROJECT_ROOT / "src"                  # /sietch_colab/akapoor/Infer_Demography/src

sys.path.insert(0, str(SRC))  # use insert(0, …) so it wins over site-packages

from simulation import bottleneck_model


assert msprime.__version__ >= "1"

if not os.path.isdir("./data/"):
    os.makedirs("./data/")
os.system("rm ./data/*.vcf.gz")
os.system("rm ./data/*.h5")

# Simulate replicates of tree sequences

def run_msprime_replicates(g, L, u, r, n, num_reps=100):
    demog = msprime.Demography.from_demes(g)
    tree_sequences = msprime.sim_ancestry(
        {"N0": n},
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
        num_replicates=num_reps,
        random_seed=42,
    )
    for ii, ts in enumerate(tree_sequences):
        ts = msprime.sim_mutations(ts, rate=u, random_seed=ii + 1)
        vcf_name = "./data/bottleneck.{0}.vcf".format(ii)
        with open(vcf_name, "w+") as fout:
            ts.write_vcf(fout, allow_position_zero=True)
        os.system(f"gzip {vcf_name}")

def write_samples_and_rec_map(L, r, n):
    DATA = Path("./data")
    DATA.mkdir(exist_ok=True)

    # ----- samples.txt -----
    with open(DATA / "samples.txt", "w") as f:
        f.write("sample\tpop\n")
        for i in range(n):
            f.write(f"tsk_{i}\tN0\n")      # ← pop name matches pops=["N0"]

    # ----- flat_map.txt -----
    with open(DATA / "flat_map.txt", "w") as f:
        f.write("pos\tMap(cM)\n0\t0\n")
        f.write(f"{L}\t{r * L * 100}\n")



def get_LD_stats(rep_ii, r_bins):
    vcf_file = f"./data/bottleneck.{ii}.vcf.gz"
    time1 = time.time()
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file="./data/flat_map.txt",
        pop_file="./data/samples.txt",
        pops=["N0"],
        r_bins=r_bins,
        report=False,
    )
    time2 = time.time()
    print("  finished rep", ii, "in", int(time2 - time1), "seconds")
    return ld_stats

if __name__ == "__main__":
    num_reps = 100

    with open('/sietch_colab/akapoor/Infer_Demography/experiments/bottleneck/runs/run_0001/data/sampled_params.pkl', 'rb') as f:
        sampled_params = pickle.load(f)

    with open('/sietch_colab/akapoor/Infer_Demography/config_files/experiment_config_bottleneck.json', 'r') as f:
        experiment_config = json.load(f)

    # Create the demography model
    g = bottleneck_model(sampled_params)

    # demog = msprime.Demography.from_demes(g)

    # define the bin edges
    r_bins = np.concatenate(([0], np.logspace(-6, -3, 16)))

    try:
        print("loading data if pre-computed")
        with open(f"./data/means.varcovs.bottleneck.{num_reps}_reps.bp", "rb") as fin:
            mv = pickle.load(fin)
        with open(f"./data/bootstrap_sets.bottleneck.{num_reps}_reps.bp", "rb") as fin:
            all_boot = pickle.load(fin)
    except IOError:
        print("running msprime and writing vcfs")
        run_msprime_replicates(g=g, num_reps=num_reps, L=experiment_config['genome_length'], u=experiment_config['mutation_rate'], r=experiment_config['recombination_rate'], n=experiment_config['num_samples']['N0'])

        print("writing samples and recombination map")
        write_samples_and_rec_map(L=experiment_config['genome_length'], r=experiment_config['recombination_rate'], n=experiment_config['num_samples']['N0'])

        print("parsing LD statistics")
        # Note: I usually would do this in parallel on cluster - is the slowest step
        ld_stats = {}
        for ii in range(num_reps):
            ld_stats[ii] = get_LD_stats(ii, r_bins)

        print("computing mean and varcov matrix from LD statistics sums")
        mv = moments.LD.Parsing.bootstrap_data(ld_stats)
        with open(f"./data/means.varcovs.bottleneck.{num_reps}_reps.bp", "wb+") as fout:
            pickle.dump(mv, fout)
        print(
            "computing bootstrap replicates of mean statistics (for confidence intervals"
        )
        all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats)
        with open(f"./data/bootstrap_sets.bottleneck.{num_reps}_reps.bp", "wb+") as fout:
            pickle.dump(all_boot, fout)
        os.system("rm ./data/*.vcf.gz")
        os.system("rm ./data/*.h5")

    print("computing expectations under the model")
    y = moments.Demes.LD(g, sampled_demes=["N0"], rho=4 * sampled_params['N0'] * r_bins)
    y = moments.LD.LDstats(
        [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    # plot simulated data vs expectations under the model
    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=r_bins,
        stats_to_plot = [
            ["DD_0_0"],
            ["Dz_0_0_0"],
            ["pi2_0_0_0_0"],
        ],
        labels = [
            [r"$D_0^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$\pi_{2;0,0,0,0}$"],
        ],
        #statistics=stats,
        rows=3,
        plot_vcs=True,
        show=False,
        fig_size=(6, 4),
        output="bottleneck_comparison.pdf",
    )

    print("running inference")
    # Run inference using the parsed data
    demo_func = moments.LD.Demographics1D.three_epoch
    # Set up the initial guess
    # :param params: The relative sizes and integration times of recent epochs,
    #     in genetic units: (nu1, nu2, T1, T2).
    
    p_guess = [
        sampled_params["N_bottleneck"] / sampled_params["N0"],                 # nu1  (size during the bottleneck)
        sampled_params["N_recover"]    / sampled_params["N0"],                 # nu2  (current size after recovery)
        (sampled_params["t_bottleneck_start"]                                  # T1   (length of the bottleneck)
        - sampled_params["t_bottleneck_end"]) / (2 * sampled_params["N0"]),
        sampled_params["t_bottleneck_end"] / (2 * sampled_params["N0"]),       # T2   (time since recovery began)
        sampled_params['N0']
    ]

    # p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)

    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], fixed_params=[p_guess[0], p_guess[1], None, None, None], rs=r_bins, verbose=1
    )

    physical_units = moments.LD.Util.rescale_params(
        opt_params, ["nu", "nu", "T", "T", "Ne"]
    )

    best_fit = {
        "N0": physical_units[4],
        "N_bottleneck": physical_units[0], 
        "N_recover": physical_units[1],
        "t_bottleneck_start": physical_units[2],
        "t_bottleneck_end": physical_units[3],
    }

    print(f'Optimal Parameters: {best_fit}')
    print(f'Ground Truth parameters: {sampled_params}')




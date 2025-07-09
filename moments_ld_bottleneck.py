"""Batch moments‑LD workflow for multiple bottleneck runs
-----------------------------------------------------------------
For **every** directory matching
    experiments/bottleneck/runs/run_XXXX/
this script will
1. read the sampled parameters and experiment config
2. simulate 100 replicates, parse LD statistics in parallel with Ray, and fit
   a one‑population three‑epoch model (bottleneck → recovery).
3. save all run‑specific outputs under
       experiments/bottleneck/runs/run_XXXX/inferences/momentsld/
   ├─ means.varcovs.bottleneck.N_reps.bp
   ├─ bootstrap_sets.bottleneck.N_reps.bp
   ├─ bottleneck_comparison.pdf (LD curves)
   └─ best_fit.json
4. After processing every run, combine results and draw a scatterplot
   (true vs inferred parameters, coloured by log‑likelihood).

Requirements
------------
conda install -c conda-forge "msprime>=1" moments demes pandas bottleneck ray matplotlib demesdraw
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import moments
import msprime
import numpy as np
import ray

# ───────────────────────── Project paths ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RUNS_ROOT = PROJECT_ROOT / "experiments/bottleneck/runs"

sys.path.insert(0, str(SRC_DIR))
from simulation import bottleneck_model  # noqa: E402

# ───────────────────────── Ray init (1 CPU per task) ──────────────
ray.init(num_cpus=32, ignore_reinit_error=True, log_to_driver=False,
         runtime_env={"env_vars": {"OMP_NUM_THREADS": "1"}})

# ───────────────────────── Helper functions ───────────────────────

def run_msprime_reps(graph: moments.DemesGraph, *, L: int, u: float, r: float,
                     n: int, num_reps: int, out_dir: Path, seed: int = 42):
    """Simulate *num_reps* replicates and write gz‑VCFs to *out_dir*."""
    demog = msprime.Demography.from_demes(graph)
    reps = msprime.sim_ancestry({"N0": n}, demography=demog,
                                sequence_length=L, recombination_rate=r,
                                num_replicates=num_reps, random_seed=seed)
    for i, ts in enumerate(reps):
        ts = msprime.sim_mutations(ts, rate=u, random_seed=i + 1)
        vcf = out_dir / f"bottleneck.{i}.vcf"
        with vcf.open("w") as fh:
            ts.write_vcf(fh, allow_position_zero=True)
        os.system(f"gzip -f {vcf}")


def write_samples_and_map(*, L: int, r: float, n: int, out_dir: Path):
    (out_dir / "samples.txt").write_text(
        "sample\tpop\n" + "\n".join(f"tsk_{i}\tN0" for i in range(n)) + "\n")
    (out_dir / "flat_map.txt").write_text(
        f"pos\tMap(cM)\n0\t0\n{L}\t{r * L * 100}\n")


@ray.remote
def parse_ld_remote(rep_i: int, r_bins: np.ndarray, work_dir: str):
    """Ray worker: compute LD stats for replicate *rep_i* inside *work_dir*."""
    work = Path(work_dir)
    stats = moments.LD.Parsing.compute_ld_statistics(
        str(work / f"bottleneck.{rep_i}.vcf.gz"),
        rec_map_file=str(work / "flat_map.txt"),
        pop_file=str(work / "samples.txt"),
        pops=["N0"],
        r_bins=r_bins,
        report=False,
    )
    return rep_i, stats


# ───────────────────────── scatter‑plot helper ────────────────────

def save_scatterplots(true_vecs: List[Dict[str, float]],
                      est_vecs: List[Dict[str, float]],
                      ll_vec: List[float],
                      param_names: List[str],
                      outfile: Path,
                      *, label: str = "moments") -> None:
    norm   = colors.Normalize(vmin=min(ll_vec), vmax=max(ll_vec))
    cmap   = cm.get_cmap("viridis")
    colour = cmap(norm(ll_vec))

    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for i, p in enumerate(param_names):
        ax = axes[0, i]
        ax.scatter([d[p] for d in true_vecs], [d[p] for d in est_vecs],
                   s=20, c=colour)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", lw=0.7, color="grey")
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"{label} {p}")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label="log‑likelihood")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ───────────────────────── Batch processing ───────────────────────
PARAM_NAMES = ["N_bottleneck", "N_recover", "t_bottleneck_start", "t_bottleneck_end"]
NUM_REPS = 100
r_bins = np.concatenate(([0], np.logspace(-6, -3, 16)))

all_true: List[Dict[str, float]] = []
all_est : List[Dict[str, float]] = []
all_ll  : List[float]            = []

for run_dir in sorted(RUNS_ROOT.glob("run_*")):
    print(f"\n=== Processing {run_dir.name} ===")
    data_dir = run_dir / "data"
    with (data_dir / "sampled_params.pkl").open("rb") as f:
        sampled_params: Dict[str, Any] = pickle.load(f)
    with (PROJECT_ROOT / "config_files/experiment_config_bottleneck.json").open() as f:
        cfg = json.load(f)

    g = bottleneck_model(sampled_params)

    # output dir for this run
    inf_dir = run_dir / "inferences/momentsld"
    inf_dir.mkdir(parents=True, exist_ok=True)

    mean_file = inf_dir / f"means.varcovs.bottleneck.{NUM_REPS}_reps.bp"
    boot_file = inf_dir / f"bootstrap_sets.bottleneck.{NUM_REPS}_reps.bp"

    if mean_file.exists() and boot_file.exists():
        mv = pickle.load(mean_file.open("rb"))
    else:
        run_msprime_reps(g, L=cfg["genome_length"], u=cfg["mutation_rate"],
                         r=cfg["recombination_rate"], n=cfg["num_samples"]["N0"],
                         num_reps=NUM_REPS, out_dir=inf_dir)
        write_samples_and_map(L=cfg["genome_length"], r=cfg["recombination_rate"],
                              n=cfg["num_samples"]["N0"], out_dir=inf_dir)

        futures = [parse_ld_remote.remote(i, r_bins, str(inf_dir)) for i in range(NUM_REPS)]
        ld_stats = {rep: stats for rep, stats in ray.get(futures)}
        mv = moments.LD.Parsing.bootstrap_data(ld_stats)
        pickle.dump(mv, mean_file.open("wb"))
        pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_stats), boot_file.open("wb"))
        for f in inf_dir.glob("bottleneck.*.vcf.gz"):
            f.unlink(missing_ok=True)

    # analytic expectations
    y = moments.Demes.LD(g, sampled_demes=["N0"], rho=4 * sampled_params["N0"] * r_bins)
    y = moments.LD.LDstats(
        [(yl + yr) / 2 for yl, yr in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops, pop_ids=y.pop_ids)
    y = moments.LD.Inference.sigmaD2(y)
    moments.LD.Plotting.plot_ld_curves_comp(
        y, mv["means"][:-1], mv["varcovs"][:-1], rs=r_bins,
        stats_to_plot=[["DD_0_0"], ["Dz_0_0_0"], ["pi2_0_0_0_0"]],
        labels=[[r"$D_0^2$"], [r"$Dz_{0,0,0}$"], [r"$\pi_{2;0,0,0,0}$"]],
        rows=3, plot_vcs=True, show=False, fig_size=(6, 4),
        output=str(inf_dir / "bottleneck_comparison.pdf"))

    # inference
    p_guess = [
        sampled_params["N_bottleneck"] / sampled_params["N0"],
        sampled_params["N_recover"] / sampled_params["N0"],
        (sampled_params["t_bottleneck_start"] - sampled_params["t_bottleneck_end"]) / (2 * sampled_params["N0"]),
        sampled_params["t_bottleneck_end"] / (2 * sampled_params["N0"]),
        sampled_params["N0"],
    ]
    demo_func = moments.LD.Demographics1D.three_epoch
    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins,
        fixed_params=[p_guess[0], p_guess[1], None, None, None], verbose=0)

    physical = moments.LD.Util.rescale_params(opt_params, ["nu", "nu", "T", "T", "Ne"])
    best_fit = dict(zip(["N_bottleneck", "N_recover", "t_bottleneck_start", "t_bottleneck_end", "N0"], physical))
    pickle.dump({"opt_params": best_fit, "loglik": LL}, (inf_dir / "best_fit.pkl").open("wb"))

    # collect for scatterplot
    all_true.append({k: sampled_params[k] for k in PARAM_NAMES})
    all_est.append({k: best_fit[k] for k in PARAM_NAMES})
    all_ll.append(LL)

# ───────────────────────── final scatterplot ──────────────────────
SCATTER_OUT = PROJECT_ROOT / "scatter_moments_vs_true.png"
save_scatterplots(all_true, all_est, all_ll, PARAM_NAMES, SCATTER_OUT, label="momentsld")
print(f"\nScatterplot written to {SCATTER_OUT}")

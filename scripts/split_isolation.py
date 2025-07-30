#!/usr/bin/env python3
"""
window_sim.py – windows ⇒ LD stats ⇒ comparison PDF ⇒ moments‑LD optimisation
==============================================================================

* Idempotent – skips work when outputs already exist.
* Ray‑parallel – replicate × window jobs run concurrently.
* Writes <model>_comparison.pdf **before** optimisation (true parameters).
* Optimisation only for *split_isolation* (cached in best_fit.pkl).
"""
from __future__ import annotations
import argparse, importlib, json, logging, pickle, subprocess, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import moments
import matplotlib.pyplot as plt
import ray

# ─── project paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR  = PROJECT_ROOT / "snakemake_scripts"
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SIM_SCRIPT = SCRIPTS_DIR / "simulation.py"
WIN_SCRIPT = SCRIPTS_DIR / "simulate_window.py"
LD_SCRIPT  = SCRIPTS_DIR / "compute_ld_window.py"

# ─── helpers ───────────────────────────────────────────────────────────────
def _ensure_sampled_params(cfg_file: Path, exp_root: Path, rep: int) -> None:
    sim_root = exp_root / "simulations"
    pkl      = sim_root / str(rep) / "sampled_params.pkl"
    if pkl.exists():
        return
    sim_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(SIM_SCRIPT),
            "--simulation-dir",    str(sim_root),
            "--experiment-config", str(cfg_file),
            "--model-type",        json.loads(cfg_file.read_text())["demographic_model"],
            "--simulation-number", str(rep),
        ],
        check=True,
    )

def _run_one_window_and_ld(cfg: Path, exp_root: Path, rep: int,
                           win: int, r_bins: str):
    sim_dir = exp_root / "simulations" / str(rep)
    win_dir = exp_root / "inferences" / f"sim_{rep}" / "MomentsLD" / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    if not (win_dir / f"window_{win}.vcf.gz").exists():
        subprocess.run(
            [
                sys.executable, str(WIN_SCRIPT),
                "--sim-dir", str(sim_dir),
                "--rep-index", str(win),
                "--config-file", str(cfg),
                "--out-dir", str(win_dir),
            ],
            check=True,
        )

    ld_root = exp_root / "inferences" / f"sim_{rep}" / "MomentsLD"
    ld_pkl  = ld_root / "LD_stats" / f"LD_stats_window_{win}.pkl"
    if not ld_pkl.exists():
        subprocess.run(
            [
                sys.executable, str(LD_SCRIPT),
                "--sim-dir", str(ld_root),
                "--window-index", str(win),
                "--config-file", str(cfg),
                "--r-bins", r_bins,
            ],
            check=True,
        )

def _plot_comparison(cfg_json: dict, sampled_params: dict,
                     mv: dict, r_vec: np.ndarray, out_dir: Path):
    pdf = out_dir / f"{cfg_json['demographic_model']}_comparison.pdf"
    if pdf.exists():
        return
    demo_mod  = importlib.import_module("simulation")
    demo_func = getattr(demo_mod, f"{cfg_json['demographic_model']}_model")
    graph     = demo_func(sampled_params)

    demes = list(cfg_json["num_samples"].keys())
    y     = moments.Demes.LD(graph, sampled_demes=demes,
                             rho=4*sampled_params["N0"]*r_vec)
    y     = moments.LD.LDstats(
        [(yl+yr)/2 for yl,yr in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops, pop_ids=y.pop_ids)
    y     = moments.LD.Inference.sigmaD2(y)

    stats = [
        ["DD_0_0"], ["DD_0_1"], ["DD_1_1"],
        ["Dz_0_0_0"], ["Dz_0_1_1"], ["Dz_1_1_1"],
        ["pi2_0_0_1_1"], ["pi2_0_1_0_1"], ["pi2_1_1_1_1"],
    ]
    labs  = [
        [r"$D_0^2$"], [r"$D_0D_1$"], [r"$D_1^2$"],
        [r"$Dz_{0,0,0}$"], [r"$Dz_{0,1,1}$"], [r"$Dz_{1,1,1}$"],
        [r"$\pi_{2;0,0,1,1}$"], [r"$\pi_{2;0,1,0,1}$"],
        [r"$\pi_{2;1,1,1,1}$"],
    ]
    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y, mv["means"][:-1], mv["varcovs"][:-1],
        rs=r_vec, stats_to_plot=stats, labels=labs,
        rows=3, plot_vcs=True, show=False, fig_size=(6, 4))
    fig.savefig(pdf, dpi=300); plt.close(fig)
    logging.info("PDF written → %s", pdf.name)

def _aggregate_and_optimise(cfg: Path, exp_root: Path, rep: int, r_bins: str):
    root   = exp_root / "inferences" / f"sim_{rep}" / "MomentsLD"
    statsD = root / "LD_stats"
    mean   = root / "means.varcovs.pkl"
    boot   = root / "bootstrap_sets.pkl"
    best   = root / "best_fit.pkl"

    ld_stats = {int(p.stem.split("_")[-1]): pickle.loads(p.read_bytes())
                for p in statsD.glob("LD_stats_window_*.pkl")}
    if not ld_stats:
        logging.warning("rep %s: no LD pickles – skip aggregation", rep)
        return

    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    pickle.dump(mv, mean.open("wb"))
    pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_stats), boot.open("wb"))

    cfg_json = json.loads(cfg.read_text())
    r_vec    = np.array([float(x) for x in r_bins.split(',')])
    params   = pickle.loads(
        (exp_root / "simulations" / str(rep) / "sampled_params.pkl").read_bytes())
    _plot_comparison(cfg_json, params, mv, r_vec, root)

    if cfg_json["demographic_model"] != "split_isolation" or best.exists():
        return

    priors = cfg_json["priors"]; pm = {k:(lo+hi)/2 for k,(lo,hi) in priors.items()}
    p0 = [pm["N1"]/pm["N0"], pm["N2"]/pm["N0"],
          pm["t_split"]/(2*pm["N0"]), pm["m"], pm["N0"]]

    opt, ll = moments.LD.Inference.optimize_log_lbfgsb(
        p0, [mv["means"], mv["varcovs"]],
        [moments.LD.Demographics2D.split_mig],
        rs=r_vec, verbose=1)
    best_phys = dict(zip(["N1","N2","t_split","m","N0"],
                         moments.LD.Util.rescale_params(opt, ["nu","nu","T","m","Ne"])))
    pickle.dump({"best_params": best_phys, "best_lls": ll}, best.open("wb"))
    logging.info("rep %s: optimisation finished (LL=%.2f)", rep, ll)

# ─── CLI / main ────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser("windows ⇒ LD ⇒ PDF ⇒ optimisation; Ray‑parallel")
    p.add_argument("-c","--config", required=True, type=Path)
    p.add_argument("-e","--exp-root", required=True, type=Path)
    p.add_argument("-r","--rep-index", type=int, nargs="+", default=[0],
                   help="Replicate indices, e.g. -r 0 1 2")
    p.add_argument("-n","--window-index", type=int, nargs="+", default=[0],
                   help="Window indices, e.g. -n 0 1 2 …")
    p.add_argument("--r-bins", required=True)
    p.add_argument("--ray-address", default=None)
    p.add_argument("--no-ray", action="store_true")
    p.add_argument("-v","--verbose", action="count", default=0)
    return p.parse_args()

def main():
    a = _parse_args()
    logging.basicConfig(
        level=logging.WARNING - 10*min(a.verbose,2),
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # ensure sampled_params for every replicate
    for rep in a.rep_index:
        _ensure_sampled_params(a.config, a.exp_root, rep)

    # ---------- window & LD tasks -----------------------------------------
    if a.no_ray:
        for rep in a.rep_index:
            for win in a.window_index:
                _run_one_window_and_ld(a.config, a.exp_root, rep, win, a.r_bins)
    else:
        ray.init(address=a.ray_address, ignore_reinit_error=True)
        tasks = [ray.remote(_run_one_window_and_ld).remote(
                    a.config, a.exp_root, rep, win, a.r_bins)
                 for rep in a.rep_index for win in a.window_index]
        ray.get(tasks)
        ray.shutdown()

    # ---------- aggregation & optimisation per replicate ------------------
    for rep in a.rep_index:
        _aggregate_and_optimise(a.config, a.exp_root, rep, a.r_bins)

# ─── entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

import json
import logging
import pickle
import importlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import moments


# ─── Constants ─────────────────────────────────────────────────────────────
DEFAULT_R_BINS = np.concatenate(([0.0], np.logspace(-6, -3, 16)))


# ─── Loaders ───────────────────────────────────────────────────────────────
def load_sampled_params(run_dir: Path) -> Dict[str, Any]:
    pkl = run_dir / "sampled_params.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"sampled_params.pkl missing in {pkl.parent}")
    return pickle.loads(pkl.read_bytes())


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


# ─── Aggregation ───────────────────────────────────────────────────────────
def aggregate_ld_stats(out_dir: Path) -> Dict[str, Any]:
    print(f'Out Dir: {out_dir}')
    means = out_dir / "means.varcovs.pkl"
    boots = out_dir / "bootstrap_sets.pkl"

    if means.exists() and boots.exists():
        return pickle.loads(means.read_bytes())

    ld_dir = out_dir / "LD_stats"
    print(f'Aggregating LD stats from {ld_dir}')
    ld_stats = {
        int(p.stem.split("_")[-1]): pickle.loads(p.read_bytes())
        for p in ld_dir.glob("LD_stats_window_*.pkl")
    }
    if not ld_stats:
        raise RuntimeError(f"No LD pickle files found in {ld_dir}")

    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    pickle.dump(mv, means.open("wb"))
    pickle.dump(moments.LD.Parsing.get_bootstrap_sets(ld_stats), boots.open("wb"))
    logging.info("Aggregated %d LD pickles", len(ld_stats))
    return mv


# ─── Plotting ──────────────────────────────────────────────────────────────
def write_comparison_pdf(cfg: dict, sampled_params: dict, mv: dict,
                         r_vec: np.ndarray, out_dir: Path) -> None:
    print(f'Writing comparison PDF to {out_dir}')
    pdf = out_dir / f"empirical_vs_theoretical_comparison.pdf"
    if pdf.exists():
        return

    demo_mod = importlib.import_module("simulation")
    demo_func = getattr(demo_mod, cfg["demographic_model"] + "_model")
    graph = demo_func(sampled_params)
    demes = list(cfg["num_samples"].keys())

    y = moments.Demes.LD(graph, sampled_demes=demes, rho=4 * sampled_params["N0"] * r_vec)
    y = moments.LD.LDstats(
        [(yl + yr) / 2 for yl, yr in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    stats = [
        ["DD_0_0"], ["DD_0_1"], ["DD_1_1"],
        ["Dz_0_0_0"], ["Dz_0_1_1"], ["Dz_1_1_1"],
        ["pi2_0_0_1_1"], ["pi2_0_1_0_1"], ["pi2_1_1_1_1"],
    ]
    labels = [
        [r"$D_0^2$"], [r"$D_0D_1$"], [r"$D_1^2$"],
        [r"$Dz_{0,0,0}$"], [r"$Dz_{0,1,1}$"], [r"$Dz_{1,1,1}$"],
        [r"$\pi_{2;0,0,1,1}$"], [r"$\pi_{2;0,1,0,1}$"], [r"$\pi_{2;1,1,1,1}$"],
    ]

    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y, mv["means"][:-1], mv["varcovs"][:-1],
        rs=r_vec, stats_to_plot=stats, labels=labels,
        rows=3, plot_vcs=True, show=False, fig_size=(6, 4),
    )
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    logging.info("Written comparison PDF %s", pdf.name)


# ─── Optimisation ──────────────────────────────────────────────────────────
def run_moments_ld_optimization(cfg: dict, mv: dict,
                                out_dir: Path, r_vec: np.ndarray) -> None:
    best_pkl = out_dir / "best_fit.pkl"
    if best_pkl.exists():
        logging.info("best_fit.pkl already exists - skipping optimisation")
        return

    priors = cfg["priors"]
    pm = {k: 0.5 * (lo + hi) for k, (lo, hi) in priors.items()}

    model = cfg["demographic_model"]

    if model == "split_isolation":
        demo_func = moments.LD.Demographics2D.split_mig
        p0 = [
            pm["N1"] / pm["N0"],
            pm["N2"] / pm["N0"],
            pm["t_split"] / (2 * pm["N0"]),
            2 * pm["m"] * pm["N0"],
            pm["N0"],
        ]
        keys = ["N1", "N2", "t_split", "m", "N0"]
        rtypes = ["nu", "nu", "T", "m", "Ne"]
    elif model == "bottleneck":
        demo_func = moments.LD.Demographics1D.three_epoch
        p0 = [
            pm["N_bottleneck"] / pm["N0"],
            pm["N_recover"] / pm["N0"],
            (pm["t_bottleneck_start"] - pm["t_bottleneck_end"]) / (2 * pm["N0"]),
            pm["t_bottleneck_end"] / (2 * pm["N0"]),
            pm["N0"],
        ]
        keys = ["N_bottleneck", "N_recover", "t_bottleneck_start", "t_bottleneck_end", "N0"]
        rtypes = ["nu", "nu", "T", "T", "Ne"]
    else:
        logging.warning("Optimisation for model '%s' not implemented - writing placeholder", model)
        pickle.dump({"best_params": {}, "best_lls": float("nan")}, best_pkl.open("wb"))
        return

    try:
        opt, ll = moments.LD.Inference.optimize_log_lbfgsb(
            p0, [mv["means"], mv["varcovs"]], [demo_func], rs=r_vec, verbose=1,
        )
        phys = moments.LD.Util.rescale_params(opt, rtypes)
        best = dict(zip(keys, phys))
    except Exception as exc:
        logging.warning("Optimisation failed: %s: %s", type(exc).__name__, exc)
        best = {k: None for k in keys}
        ll = float("nan")
        (out_dir / "fail_reason.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    pickle.dump({"best_params": best, "best_lls": ll}, best_pkl.open("wb"))
    logging.info("Optimisation finished - LL = %.2f", ll)

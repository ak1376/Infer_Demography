#!/usr/bin/env python3
"""
Real-data MomentsLD fit vs. data, in the empirical_vs_theoretical_comparison.pdf
layout (moments.LD.Plotting.plot_ld_curves_comp): data = dashed mean with a
+/- 1.96 SE band, model = solid line from the fit's best parameters.

The model curve is computed with the fitting code itself
(src/MomentsLD_real_data.py: compute_theoretical_ld, prepare_data_for_comparison,
composite_gaussian_ll), and the log-likelihood is recomputed from it and
printed next to the stored best_ll as a consistency check.

Works on a single restart's best_fit.pkl (best_params is a dict) or the
aggregated one (best_params is a top-K list; the best entry is plotted).

Usage:
  python snakemake_scripts/plot_momentsld_fit_real.py \
      --empirical <REAL_LD_ROOT>/Chr3L/means.varcovs.pkl \
      --best-fit  <fit dir>/best_fit.pkl \
      --config    config_files/experiment_config_<model>.json \
      --r-bins    "0,1e-06,...,0.001" \
      --out       <fit dir>/fit_comparison.pdf
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
from MomentsLD_real_data import (  # noqa: E402
    compute_theoretical_ld, prepare_data_for_comparison, composite_gaussian_ll, load_demographic_function,
)
from plot_ld_decay_real import STATS_TO_PLOT, LABELS, ROWS  # noqa: E402  same panels as the data-only plot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--empirical", required=True, type=Path)
    ap.add_argument("--best-fit", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--r-bins", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    with open(args.empirical, "rb") as fh:
        mv = pickle.load(fh)
    with open(args.best_fit, "rb") as fh:
        fit = pickle.load(fh)

    params = fit["best_params"]
    ll_stored = fit["best_ll"]
    if isinstance(params, list):                       # aggregated top-K: plot the best
        params, ll_stored = params[0], ll_stored[0]
    names = list(fit.get("param_order") or cfg["parameter_order"])
    r_bins = np.array([float(x) for x in args.r_bins.split(",")])
    pops = list(cfg["num_samples"].keys())

    # best_params are absolute, so N_ANC is the rho reference -- the same
    # reference the fit used (absolute mode: N_ANC; scaled mode: N_ref = N_ANC).
    theory = compute_theoretical_ld(
        np.log10([params[p] for p in names]), param_names=names,
        demographic_model_abs=load_demographic_function(cfg), r_bins=r_bins,
        populations=pops, N_ref=float(params["N_ANC"]), use_scaled_units=False,
        _diagnostic_once={"did": True},
    )
    theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(theory, mv, normalization=0)
    ll = composite_gaussian_ll(emp_means, emp_covars, theory_arrays)
    print(f"log-likelihood: recomputed {ll:.4f}, stored {float(ll_stored):.4f}")

    fig = moments.LD.Plotting.plot_ld_curves_comp(
        theory, mv["means"][:-1], mv["varcovs"][:-1], rs=r_bins,
        stats_to_plot=STATS_TO_PLOT, labels=LABELS, rows=ROWS,
        plot_vcs=True, show=False, fig_size=(6, 4),
    )
    short = ", ".join(f"{p}={params[p]:.3g}" for p in names)
    fig.suptitle(f"LL = {ll:.2f}   |   {short}", fontsize=5)
    fig.subplots_adjust(top=0.9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

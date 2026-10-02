#!/usr/bin/env python3
"""
Slice through a fitted MomentsLD model: vary ONE parameter over a log grid,
hold every other parameter at the best-fit value (no re-optimization), and
overlay the resulting model LD curves on the empirical ones.

Panels: sigma_D^2 CO, Dz CO, D_CO*D_FR, Dz_{CO,FR,FR} (model curve per grid
value, coloured low -> high, empirical mean +/- SE in black, best fit dashed),
plus the composite log-likelihood along the slice.

Usage:
  python snakemake_scripts/plot_momentsld_param_slice.py \
      --config config_files/experiment_config_split_migration_growth_both.json \
      --empirical real_data_analysis/data/drosophila_trimmed/ld/<ld_name>/Chr3L/means.varcovs.pkl \
      --best-fit experiments/<model>/real_trimmed/Chr3L/inferences/<engine>/best_fit.pkl \
      --param N_CO0 --grid-min-param N_CO1 --grid-max 3e7 --n-grid 9 \
      --out experiments/<model>/real_trimmed/Chr3L/inferences/<engine>/slice_N_CO0
"""
from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.MomentsLD_real_data import (compute_theoretical_ld, composite_gaussian_ll,  # noqa: E402
                                     load_demographic_function, prepare_data_for_comparison)

PANELS = [("DD_0_0", r"$\sigma_D^2$ CO"), ("Dz_0_0_0", r"$Dz$ CO"),
          ("DD_0_1", r"$D_{CO}D_{FR}$"), ("Dz_0_1_1", r"$Dz_{CO,FR,FR}$")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--empirical", required=True)
    ap.add_argument("--best-fit", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--grid-min", type=float, default=None)
    ap.add_argument("--grid-min-param", default=None, help="use this fitted parameter's value as the grid minimum")
    ap.add_argument("--grid-max", type=float, required=True)
    ap.add_argument("--n-grid", type=int, default=9)
    ap.add_argument("--out", required=True, help="output prefix (.png, .tsv)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mv = pickle.load(open(args.empirical, "rb"))
    fit = pickle.load(open(args.best_fit, "rb"))
    best = dict(fit["best_params"][0] if isinstance(fit["best_params"], list) else fit["best_params"])
    names = list(fit.get("param_order") or cfg["priors"])
    lo = best[args.grid_min_param] if args.grid_min_param else args.grid_min
    grid = np.logspace(np.log10(lo), np.log10(args.grid_max), args.n_grid)

    ld_names = mv["stats"][0]
    bins = np.array(mv["bins"]); nb = len(bins)
    r_edges = np.concatenate([bins[:, 0], bins[-1:, 1]])
    x = np.sqrt(np.maximum(bins[:, 0], bins[:, 1] / 10) * bins[:, 1])
    theo_names = moments.LD.Util.moment_names(2)[0]
    demo = load_demographic_function(cfg)
    pops = list(cfg["num_samples"].keys())

    def model(params):
        theo = compute_theoretical_ld(np.log10([params[n] for n in names]), param_names=names,
                                      demographic_model_abs=demo, r_bins=r_edges, populations=pops,
                                      N_ref=1.0, use_scaled_units=False, _diagnostic_once={"did": True})
        t, em, ec = prepare_data_for_comparison(theo, mv, normalization=0)
        return theo, composite_gaussian_ll(em, ec, t)

    curves, lls = [], []
    for v in grid:
        p = dict(best); p[args.param] = float(v)
        theo, ll = model(p)
        curves.append(theo); lls.append(ll)
    best_theo, best_ll = model(best)

    with open(f"{args.out}.tsv", "w") as f:
        f.write(f"{args.param}\tlog_likelihood\n")
        for v, ll in zip(grid, lls):
            f.write(f"{v:.6g}\t{ll:.4f}\n")

    cmap = plt.get_cmap("viridis")
    cols = [cmap(i / (len(grid) - 1)) for i in range(len(grid))]
    fig, axes = plt.subplots(1, len(PANELS) + 1, figsize=(4.3 * (len(PANELS) + 1), 4.6))
    for ax, (st, lab) in zip(axes, PANELS):
        k, kt = ld_names.index(st), theo_names.index(st)
        for theo, c, v in zip(curves, cols, grid):
            ax.plot(x, [theo[b][kt] for b in range(nb)], color=c, lw=1.4, label=f"{v:.3g}")
        ax.plot(x, [best_theo[b][kt] for b in range(nb)], color="tab:red", ls="--", lw=1.4,
                label=f"best fit ({best[args.param]:.3g})")
        m = [mv["means"][b][k] for b in range(nb)]
        se = [np.sqrt(np.asarray(mv["varcovs"][b])[k, k]) for b in range(nb)]
        ax.errorbar(x, m, yerr=se, fmt="o", color="black", ms=3, capsize=2, zorder=5, label="data ± SE")
        ax.set_xscale("log"); ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(lab); ax.set_xlabel("r (Comeron map)")
    axes[0].legend(title=args.param, fontsize=7, title_fontsize=8)
    a = axes[-1]
    a.plot(grid, lls, "o-", color="black")
    for v, ll, c in zip(grid, lls, cols):
        a.plot(v, ll, "o", color=c, ms=8)
    a.plot(best[args.param], best_ll, "x", color="tab:red", ms=10, mew=2, label="best fit")
    a.set_xscale("log"); a.set_xlabel(f"{args.param}"); a.set_ylabel("composite log-likelihood")
    a.set_title("fit along the slice"); a.legend(fontsize=8)
    others = ", ".join(f"{n}={best[n]:.3g}" for n in names if n != args.param)
    fig.suptitle(f"Vary {args.param} only (others fixed at best fit: {others})", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{args.out}.png", dpi=120)
    print(f"-> {args.out}.png\n-> {args.out}.tsv")
    for v, ll in zip(grid, lls):
        print(f"  {args.param}={v:10.3g}  LL={ll:10.2f}")


if __name__ == "__main__":
    main()

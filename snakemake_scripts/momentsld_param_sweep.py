#!/usr/bin/env python3
"""
Which parameters move which MomentsLD curves?

Given reference parameters for the configured demographic model -- a real-data
best fit, or hand-picked values -- sweep each parameter over a grid (every
other parameter held at its reference value) and
compute the model's EXPECTED LD statistics with the same calculation the fits
use (src.MomentsLD_real_data.compute_theoretical_ld: sigma_D^2-normalized,
normalized by pop-0 pi2, rho = 4 * N_ANC * r in absolute units).  No
simulation noise -- these are the model's expectations.

Outputs in --out-dir:
  sweep_<param>.png          3x3 panels (sigma_D^2, Dz, pi2 for CO / FR / cross),
                             one curve per grid value (low -> high), reference dashed red,
                             optional empirical data in black
  where_<param>.png          statistic x r-bin heatmap: log10 fold-change of the
                             statistic between the lowest and highest grid value
                             (where along r this parameter acts, and in which direction)
  sensitivity_overview.png   parameter x statistic heatmap: median over r bins of
                             |log10 fold-change| across the sweep
  sweep_loglik.tsv           (with --empirical) composite LL vs the data at each grid value
  sweep_curves.tsv           long table: param, grid_value, is_reference, statistic, r_bin_lo, r_bin_hi, value
  reference.json             the reference values used (and their label)

Usage:
  python snakemake_scripts/momentsld_param_sweep.py \
      --config config_files/experiment_config_split_migration_growth_both.json \
      --reference-from-fit experiments/<model>/real_trimmed/Chr3L/inferences/<engine>/best_fit.pkl \
      --n-grid 7 --workers 8 \
      --out-dir experiments/<model>/param_sweeps/<label>

  # hand-picked values instead (e.g. a simulation's true parameters), sweeping ref/10 .. ref*10:
  python snakemake_scripts/momentsld_param_sweep.py --config <cfg> --reference params.json \
      --reference-label "simulated truth" --fold 10 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import textwrap
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.MomentsLD_real_data import (compute_theoretical_ld, composite_gaussian_ll,  # noqa: E402
                                     load_demographic_function, prepare_data_for_comparison)

# Same edges as the Snakefile's R_BINS_STR.
DEFAULT_R_BINS = ("0,1e-06,1.58489e-06,2.51189e-06,3.98107e-06,6.30957e-06,1e-05,1.58489e-05,2.51189e-05,"
                  "3.98107e-05,6.30957e-05,0.0001,0.000158489,0.000251189,0.000398107,0.000630957,0.001")
PANELS = [("DD_0_0", r"$\sigma_D^2$ CO"), ("DD_0_1", r"$D_{CO}D_{FR}$"), ("DD_1_1", r"$\sigma_D^2$ FR"),
          ("Dz_0_0_0", r"$Dz$ CO"), ("Dz_0_1_1", r"$Dz_{CO,FR,FR}$"), ("Dz_1_1_1", r"$Dz$ FR"),
          ("pi2_0_0_1_1", r"$\pi_2$ CO,CO,FR,FR"), ("pi2_0_1_0_1", r"$\pi_2$ CO,FR,CO,FR"), ("pi2_1_1_1_1", r"$\pi_2$ FR")]

_CTX = {}


def _init(cfg, names, r_edges, pops, mv=None):
    _CTX.update(demo=load_demographic_function(cfg), names=names, r_edges=r_edges, pops=pops,
                tn=moments.LD.Util.moment_names(2)[0], mv=mv)


def _curves(params):
    """Expected statistic curves for one parameter set -> {stat: array over r bins}."""
    c = _CTX
    th = compute_theoretical_ld(np.log10([params[n] for n in c["names"]]), param_names=c["names"],
                                demographic_model_abs=c["demo"], r_bins=c["r_edges"], populations=c["pops"],
                                N_ref=1.0, use_scaled_units=False, _diagnostic_once={"did": True})
    nb = len(c["r_edges"]) - 1
    out = {s: np.array([th[b][c["tn"].index(s)] for b in range(nb)]) for s, _ in PANELS}
    if c["mv"] is not None:   # composite log-likelihood against the data, same as the fit's objective
        t, em, ec = prepare_data_for_comparison(th, c["mv"], normalization=0)
        out["__ll__"] = float(composite_gaussian_ll(em, ec, t))
    return out


def _load_reference(args, names):
    if args.reference_from_fit:
        d = pickle.load(open(args.reference_from_fit, "rb"))
        ref = dict(d["best_params"][0] if isinstance(d["best_params"], list) else d["best_params"])
    else:
        ref = json.load(open(args.reference))
    missing = [n for n in names if n not in ref]
    if missing:
        raise SystemExit(f"reference is missing parameters: {missing}")
    return {n: float(ref[n]) for n in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference", help="JSON file {param: value}")
    g.add_argument("--reference-from-fit", help="best_fit.pkl whose best_params are the reference")
    ap.add_argument("--reference-label", default=None,
                    help='name shown in plots (default: "best fit" with --reference-from-fit, else "reference")')
    ap.add_argument("--params", default=None, help="comma-separated subset to sweep (default: all)")
    ap.add_argument("--n-grid", type=int, default=7)
    ap.add_argument("--fold", type=float, default=None, help="grid = reference/F .. reference*F (default: prior range)")
    ap.add_argument("--range", action="append", default=[], metavar="NAME:LO:HI", help="override one grid range")
    ap.add_argument("--r-bins", default=DEFAULT_R_BINS)
    ap.add_argument("--empirical", default=None, help="optional means.varcovs.pkl to overlay")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    names = list(cfg["priors"])
    pops = list(cfg["num_samples"].keys())
    ref = _load_reference(args, names)
    ref_label = args.reference_label or ("best fit" if args.reference_from_fit else "reference")
    sweep = [p for p in (args.params.split(",") if args.params else names)]
    r_edges = np.array([float(x) for x in args.r_bins.split(",")])
    nb = len(r_edges) - 1
    x = np.sqrt(np.maximum(r_edges[:-1], r_edges[1:] / 10) * r_edges[1:])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump({"label": ref_label, "source": args.reference_from_fit or args.reference, "params": ref},
              open(out / "reference.json", "w"), indent=2)

    overrides = {s.split(":")[0]: (float(s.split(":")[1]), float(s.split(":")[2])) for s in args.range}
    grids = {}
    for p in sweep:
        if p in overrides:
            lo, hi = overrides[p]
        elif args.fold:
            lo, hi = ref[p] / args.fold, ref[p] * args.fold
        else:
            lo, hi = map(float, cfg["priors"][p])
        grids[p] = np.logspace(np.log10(lo), np.log10(hi), args.n_grid)

    jobs = [("__reference__", None, dict(ref))]
    for p in sweep:
        for v in grids[p]:
            q = dict(ref); q[p] = float(v); jobs.append((p, float(v), q))
    mv_emp = pickle.load(open(args.empirical, "rb")) if args.empirical else None
    print(f"{len(jobs)} model evaluations ({len(sweep)} parameters x {args.n_grid} grid values + reference)")
    with Pool(args.workers, initializer=_init, initargs=(cfg, names, r_edges, pops, mv_emp)) as pool:
        results = pool.map(_curves, [j[2] for j in jobs])
    ref_c = results[0]
    by_param = {p: [] for p in sweep}
    for (p, v, _), res in zip(jobs[1:], results[1:]):
        by_param[p].append((v, res))

    emp = None
    if args.empirical:
        mv = pickle.load(open(args.empirical, "rb"))
        en = mv["stats"][0]
        emp = {s: (np.array([mv["means"][b][en.index(s)] for b in range(nb)]),
                   np.array([np.sqrt(np.asarray(mv["varcovs"][b])[en.index(s), en.index(s)]) for b in range(nb)]))
               for s, _ in PANELS}

    with open(out / "sweep_curves.tsv", "w") as f:
        f.write("param\tgrid_value\tis_reference\tstatistic\tr_bin_lo\tr_bin_hi\tvalue\n")
        for st, _ in PANELS:
            for b in range(nb):
                f.write(f"reference\tnan\t1\t{st}\t{r_edges[b]:.6g}\t{r_edges[b + 1]:.6g}\t{ref_c[st][b]:.8g}\n")
        for p in sweep:
            for v, res in by_param[p]:
                for st, _ in PANELS:
                    for b in range(nb):
                        f.write(f"{p}\t{v:.6g}\t0\t{st}\t{r_edges[b]:.6g}\t{r_edges[b + 1]:.6g}\t{res[st][b]:.8g}\n")

    if mv_emp is not None:
        with open(out / "sweep_loglik.tsv", "w") as f:
            f.write(f"param\tgrid_value\tlog_likelihood\n")
            f.write(f"reference\tnan\t{ref_c['__ll__']:.4f}\n")
            for p in sweep:
                for v, res in by_param[p]:
                    f.write(f"{p}\t{v:.6g}\t{res['__ll__']:.4f}\n")
    ref_txt = "\n".join(textwrap.wrap(f"others held at {ref_label}: " + ", ".join(f"{n}={ref[n]:.3g}" for n in names), 110))
    overview = np.zeros((len(sweep), len(PANELS)))
    cmap = plt.get_cmap("viridis")
    for pi, p in enumerate(sweep):
        runs = by_param[p]
        cols = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]
        # --- curves ---
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
        for ax, (st, lab) in zip(axes.flat, PANELS):
            for (v, res), c in zip(runs, cols):
                ax.plot(x, res[st], color=c, lw=1.3,
                        label=f"{v:.3g}" + (f"   LL {res['__ll__']:.0f}" if "__ll__" in res else ""))
            ax.plot(x, ref_c[st], color="tab:red", lw=2.2, ls="--",
                    label=f"{ref_label} {ref[p]:.3g}" + (f"   LL {ref_c['__ll__']:.0f}" if "__ll__" in ref_c else ""))
            if emp:
                ax.errorbar(x, emp[st][0], yerr=emp[st][1], fmt="o", color="black", ms=4.5, capsize=2.5, elinewidth=1.2, zorder=5,
                            label="data")
            ax.set_xscale("log"); ax.axhline(0, color="grey", lw=0.5); ax.set_title(lab)
        for ax in axes[-1]:
            ax.set_xlabel("r")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title=p, loc="center right", fontsize=8, title_fontsize=9, frameon=False)
        fig.suptitle(f"Sweep of {p} ({runs[0][0]:.3g} → {runs[-1][0]:.3g})\n{ref_txt}", fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.86, 0.95]); fig.savefig(out / f"sweep_{p}.png", dpi=110); plt.close(fig)
        # --- where along r (signed log10 fold change, lowest -> highest grid value) ---
        lo_c, hi_c = runs[0][1], runs[-1][1]
        W = np.array([np.log10(np.abs(hi_c[st]) + 1e-12) - np.log10(np.abs(lo_c[st]) + 1e-12) for st, _ in PANELS])
        overview[pi] = np.median(np.abs(W), axis=1)
        lim = max(np.nanmax(np.abs(W)), 1e-3)
        fig, ax = plt.subplots(figsize=(10, 4.6))
        im = ax.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_yticks(range(len(PANELS))); ax.set_yticklabels([lab for _, lab in PANELS])
        ax.set_xticks(range(nb)); ax.set_xticklabels([f"{e:.1e}" for e in r_edges[1:]], rotation=60, fontsize=7)
        ax.set_xlabel("r bin (upper edge)")
        fig.colorbar(im, ax=ax, label=f"log10(high / low)\n{p}: {runs[0][0]:.3g} → {runs[-1][0]:.3g}")
        ax.set_title(f"Where {p} acts along r\nred = statistic rises as {p} increases, blue = falls", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"where_{p}.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(sweep) + 2))
    im = ax.imshow(overview, cmap="magma_r", aspect="auto")
    ax.set_yticks(range(len(sweep))); ax.set_yticklabels(sweep)
    ax.set_xticks(range(len(PANELS))); ax.set_xticklabels([lab for _, lab in PANELS], rotation=30, ha="right")
    for i in range(len(sweep)):
        for j in range(len(PANELS)):
            ax.text(j, i, f"{overview[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if overview[i, j] > overview.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="median over r of\n|log10 fold-change| across the sweep")
    ax.set_title("Which parameter moves which statistic\n(whole sweep range; 0.3 ≈ 2×, 1 = 10×)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "sensitivity_overview.png", dpi=120); plt.close(fig)
    print(f"-> {out}/sensitivity_overview.png, sweep_<param>.png, where_<param>.png, sweep_curves.tsv")


if __name__ == "__main__":
    main()

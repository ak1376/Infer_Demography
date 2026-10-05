#!/usr/bin/env python3
"""
Compare a real-data moments or dadi best fit against the observed 2-population SFS.

The model's expected SFS is rebuilt exactly as the real-data fit computes it,
for whichever engine made the fit (the pickle's "mode"): Spectrum.from_demes at
theta=1 for the fitted demography -- moments directly (src/moments_inference_real.py),
dadi on the same pts_l grids with the same Richardson extrapolation
(src/dadi_inference_real.py) -- times the fitted theta_hat. Because theta is profiled, the
model's total SNP count always equals the observed one, so every panel
compares shape only -- observed bars vs. model dots, in SNP counts:

  top row    : each population's own spectrum (marginal SFS); SNP categories
               (private to each population / shared / fixed in one)
  bottom row : the spectrum of SNPs private to each population; F_ST of the
               observed vs. model SFS, and the fitted params

Also writes the numbers behind the figure (F_ST, log-likelihood, params, SNP
categories) to <out>.json next to the PNG.

Works for a single optimization run's best_fit.pkl ({best_params: dict,
best_ll: float, theta_hat: float}) and for the aggregated top-K best_fit.pkl
(lists of those, the highest best_ll entry is plotted).

Usage:
  python snakemake_scripts/plot_sfs_fit_real.py \
      --fit-pkl  experiments/<model>/real_trimmed/inferences/{moments,dadi}/best_fit.pkl \
      --sfs      real_data_analysis/data/drosophila_trimmed/combined/autosomes.unfolded.sfs.pkl \
      --config   config_files/<experiment_config>.json \
      --model-py demes_models:<model>_model \
      --out      experiments/<model>/real_trimmed/inferences/{moments,dadi}/sfs_fit.png
"""
import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
import numpy as np

# Same import paths as moments_dadi_inference_real.py, so --model-py specs
# like "demes_models:<model>_model" (a module in src/) resolve identically.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Palette (dataviz reference instance, light mode)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OBS_COLOR = "#2a78d6"
MODEL_COLOR = INK


def load_best(fit_pkl: Path):
    """(engine, params dict, ll, theta_hat, opt_index or None) -- top entry for top-K pickles."""
    with open(fit_pkl, "rb") as fh:
        d = pickle.load(fh)
    engine = d.get("mode", "moments")
    params, ll, theta = d["best_params"], d["best_ll"], d.get("theta_hat")
    if isinstance(params, list):
        k = int(np.argmax(ll))
        opt = d.get("opt_index", [None] * len(ll))[k]
        return engine, params[k], float(ll[k]), float(theta[k]), opt
    return engine, params, float(ll), float(theta), None


def expected_sfs(engine, graph, obs_fs, pops, ns, cfg, theta_hat):
    """The fit's expected SFS: engine's theta=1 spectrum for `graph`, times theta_hat."""
    if engine == "dadi":
        import dadi
        from src.dadi_inference_real import _choose_pts_l
        pts_l = _choose_pts_l(dadi.Spectrum(obs_fs), cfg)
        func_ex = dadi.Numerics.make_extrap_func(
            lambda _p, ns_local, pts: dadi.Spectrum.from_demes(
                graph, sampled_demes=pops, sample_sizes=list(ns_local), pts=pts))
        base = func_ex(None, tuple(ns), pts_l)
    else:
        base = moments.Spectrum.from_demes(graph, sampled_demes=pops, sample_sizes=ns, theta=1.0)
    return np.asarray(base.data, dtype=float) * theta_hat


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def obs_vs_model(ax, x, obs, model, title, xlabel, xticklabels=None):
    ax.bar(x, obs, width=0.72, color=OBS_COLOR, edgecolor=SURFACE, linewidth=2,
           label="observed", zorder=2)
    ax.plot(x, model, color=MODEL_COLOR, linewidth=1.5, marker="o", markersize=6,
            markerfacecolor=SURFACE, markeredgewidth=1.6, label="best-fit model", zorder=3)
    style(ax)
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    ax.set_ylabel("SNPs", color=INK_2, fontsize=9.5)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fit-pkl", required=True, type=Path)
    ap.add_argument("--sfs", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--model-py", required=True, help="module:function returning a demes graph")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="", help="text prepended to the figure title")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    mod, fn = args.model_py.split(":")
    model_func = getattr(importlib.import_module(mod), fn)

    with open(args.sfs, "rb") as fh:
        obs_fs = pickle.load(fh)
    obs = np.asarray(obs_fs.data, dtype=float)
    pops = list(cfg["num_samples"].keys())       # same pop order the fit uses
    ns = [n - 1 for n in obs.shape]

    engine, params, ll, theta_hat, opt = load_best(args.fit_pkl)
    graph = model_func({k: float(v) for k, v in params.items()})
    model = expected_sfs(engine, graph, obs_fs, pops, ns, cfg, theta_hat)

    # Monomorphic corners carry no information and are excluded by the fit.
    keep = np.ones_like(obs, dtype=bool)
    keep[0, 0] = keep[-1, -1] = False
    obs, model = np.where(keep, obs, 0.0), np.where(keep, model, 0.0)
    # Weir & Cockerham F_ST (moments.Spectrum.Fst), monomorphic corners masked.
    fst = {}
    for name, arr in (("observed", obs), ("model", model)):
        sp = moments.Spectrum(arr, pop_ids=pops)
        sp.mask[0, 0] = sp.mask[-1, -1] = True
        fst[name] = float(sp.Fst())
    print(f"F_ST: observed {fst['observed']:.4f}, model {fst['model']:.4f}")
    print(f"total SNPs: observed {obs.sum():,.0f}, model {model.sum():,.0f} "
          f"(ratio {model.sum() / obs.sum():.4f}; ~1 by construction of theta_hat)")

    n1, n2 = ns
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.28})

    # Each population's own spectrum, segregating in that population only.
    for k, ax in enumerate(axes[0, :2]):
        n = ns[k]
        o, m = obs.sum(axis=1 - k), model.sum(axis=1 - k)
        i = np.arange(1, n)
        obs_vs_model(ax, i, o[1:n], m[1:n], f"{pops[k]}: its own spectrum",
                     f"number of the {n} {pops[k]} flies carrying the new allele")
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")

    # SNP categories.
    seg1, seg2 = slice(1, n1), slice(1, n2)
    cats = {
        f"only in {pops[0]}": lambda a: a[seg1, 0].sum(),
        f"only in {pops[1]}": lambda a: a[0, seg2].sum(),
        "shared": lambda a: a[seg1, seg2].sum(),
        "fixed in one\npopulation": lambda a: a.sum() - a[seg1, 0].sum() - a[0, seg2].sum() - a[seg1, seg2].sum(),
    }
    x = np.arange(len(cats))
    obs_vs_model(axes[0, 2], x, [f(obs) for f in cats.values()], [f(model) for f in cats.values()],
                 "Where the SNPs are", "", xticklabels=list(cats))
    axes[0, 2].lines[0].set_linestyle("none")   # categories aren't ordered -- dots only

    # Spectra of SNPs private to one population.
    obs_vs_model(axes[1, 0], np.arange(1, n1), obs[seg1, 0], model[seg1, 0],
                 f"SNPs only in {pops[0]}", f"number of the {n1} {pops[0]} flies carrying the new allele")
    obs_vs_model(axes[1, 1], np.arange(1, n2), obs[0, seg2], model[0, seg2],
                 f"SNPs only in {pops[1]}", f"number of the {n2} {pops[1]} flies carrying the new allele")

    # Fitted parameters.
    ax = axes[1, 2]
    ax.axis("off")
    order = cfg.get("parameter_order", list(params))
    lines = [f"{p:<10s} {float(params[p]):>12.4g}" for p in order if p in params]
    fst_lines = [f"{'observed':<10s} {fst['observed']:>12.4f}", f"{'model':<10s} {fst['model']:>12.4f}"]
    ax.text(0, 1, "F_ST (how different CO and FR are)\n\n" + "\n".join(fst_lines)
            + "\n\n\nBest-fit parameters\n\n" + "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=10, color=INK_2)

    who = f"opt {opt}, " if opt is not None else ""
    fig.suptitle(
        f"{args.label + '  ·  ' if args.label else ''}{engine} best fit vs. observed SFS  ·  "
        f"{who}log-likelihood {ll:,.1f}  ·  {obs.sum():,.0f} SNPs",
        x=0.01, ha="left", y=0.97, color=INK, fontsize=13)
    fig.text(0.01, 0.925, "Bars = observed, dots = what the best-fit model predicts. "
             "Total SNPs match by construction (θ is fitted), so the panels compare shape.",
             ha="left", color=MUTED, fontsize=9.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved -> {args.out}")

    # Numbers behind the figure, next to it: <out stem>.json
    summary = {
        "engine": engine, "fit_pkl": str(args.fit_pkl), "sfs": str(args.sfs), "opt_index": opt,
        "best_ll": ll, "theta_hat": theta_hat, "best_params": {k: float(v) for k, v in params.items()},
        "fst_observed": fst["observed"], "fst_model": fst["model"],
        "snps_observed": float(obs.sum()), "snps_model": float(model.sum()),
        "categories_observed": {k.replace("\n", " "): float(f(obs)) for k, f in cats.items()},
        "categories_model": {k.replace("\n", " "): float(f(model)) for k, f in cats.items()},
    }
    out_json = args.out.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()

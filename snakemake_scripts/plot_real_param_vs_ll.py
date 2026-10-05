#!/usr/bin/env python3
"""
Where the best optimization runs of each real-data experiment put each fitted
parameter: one panel per parameter, one row per experiment, one dot per run.

Only each experiment's --top best runs (by log-likelihood, ranked within that
experiment -- LLs from different data aren't comparable) are shown; the best
run is the large dot. A tight cluster means the good runs agree on the value; a
spread means the data don't pin it down; rows centered in different places
mean the experiments disagree.

Reads each run's sfs_fit.json (written by plot_sfs_fit_real.py).

Usage:
  python snakemake_scripts/plot_real_param_vs_ll.py \
      --experiment "all flies=experiments/<model>/real_trimmed/Chr3L/runs" \
      --experiment "carriers excluded=experiments/<model>/real_trimmed_excl-FR217-FR361/Chr3L/runs" \
      --engine moments --out params_vs_ll.png
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Palette (dataviz reference instance, light mode); slots 1-3 validated (worst CVD dE 9.2;
# slot 3 is below 3:1 contrast, relieved by the text label on every row)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_runs(runs_dir: str, engine: str):
    rows = []
    for p in sorted(glob.glob(f"{runs_dir}/run_*/inferences/{engine}/sfs_fit.json")):
        with open(p) as fh:
            j = json.load(fh)
        rows.append((j["best_ll"], j["best_params"]))
    if not rows:
        raise SystemExit(f"no {engine} sfs_fit.json under {runs_dir}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", action="append", required=True,
                    help='"label=path/to/runs" (repeatable, max 3)')
    ap.add_argument("--engine", default="moments")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top", type=int, default=10, help="best runs to show per experiment")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    if len(args.experiment) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} experiments")

    exps = []
    for spec in args.experiment:
        label, runs_dir = spec.split("=", 1)
        rows = load_runs(runs_dir, args.engine)
        ll = np.array([r[0] for r in rows])
        exps.append((label, ll.max() - ll, [r[1] for r in rows]))
    params = list(exps[0][2][0])

    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, (0.8 + 0.65 * len(exps)) * nrow + 0.6), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.9, "wspace": 0.12})
    axes = np.atleast_1d(axes).ravel()
    ys = np.arange(len(exps))[::-1]            # first experiment on top
    for ax, p in zip(axes, params):
        for (label, gap, prs), color, y in zip(exps, COLORS, ys):
            top = np.argsort(gap)[:args.top]   # best runs first
            x = np.array([prs[i][p] for i in top], dtype=float)
            ax.scatter(x[1:], np.full(len(x) - 1, y), s=40, color=color, alpha=0.55,
                       edgecolor=SURFACE, linewidth=1, zorder=3)
            ax.scatter(x[:1], [y], s=150, color=color, edgecolor=INK, linewidth=1.4, zorder=4)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=8))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2g}".replace("e+0", "e").replace("e-0", "e-")))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylim(-0.7, len(exps) - 0.3)
        ax.set_yticks(ys)
        ax.set_yticklabels([e[0] for e in exps] if ax in axes[::ncol] else [])
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="y", length=0, labelcolor=INK_2, labelsize=10)
        ax.tick_params(axis="x", colors=MUTED, labelcolor=INK_2, labelsize=8.5)
        ax.xaxis.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(p, loc="left", color=INK, fontsize=11.5, fontweight="bold")
    for ax in axes[len(params):]:
        ax.axis("off")

    fig.suptitle(args.title or f"{args.engine}: where the {args.top} best runs put each parameter",
                 x=0.01, ha="left", y=1.06, color=INK, fontsize=13)
    fig.text(0.01, 0.965, f"Large dot = best run; small dots = the next {args.top - 1} best runs "
             "(by log-likelihood, within each experiment). Tight cluster = the runs agree; "
             "spread out = the data don't pin this parameter down.",
             ha="left", color=MUTED, fontsize=9.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

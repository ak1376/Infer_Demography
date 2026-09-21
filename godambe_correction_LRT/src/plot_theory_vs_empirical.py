#!/usr/bin/env python3
# godambe_correction_LRT/src/plot_theory_vs_empirical.py

"""
Overlay the fitted model's THEORETICAL LD-decay curve on the empirical curve
(+ per-window spread), for the three representative two-point stats
(within-CO, within-FR, cross CO-FR). Answers: does the best-fit demographic
model actually reproduce the observed decay shape, or is it a poor fit
despite a "good" likelihood relative to the null?

Usage:
  python godambe_correction_LRT/src/plot_theory_vs_empirical.py \
      --fit godambe_correction_LRT/real_arms/Chr3L/momentsld/fit_co_grow/best_fit.pkl \
      --mv godambe_correction_LRT/real_arms/Chr3L/momentsld/means.varcovs.diagonly.pkl \
      --ld-stats-dir godambe_correction_LRT/real_arms/Chr3L/momentsld/LD_stats \
      --title "Chr3L co_grow_from_anc" \
      --out godambe_correction_LRT/real_arms/Chr3L/momentsld/fit_co_grow/theory_vs_empirical.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_likelihood_ld import model_for, theory_for, r_bins_from_mv  # noqa: E402
from bootstrap_ld import load_windows, average_ld_structure  # noqa: E402
import compute_J_ld as cj  # noqa: E402

FOCUS_LD_STATS = ["DD_0_0", "DD_1_1", "DD_0_1"]


def bin_midpoints(bins):
    return np.array([(lo + hi) / 2.0 for lo, hi in bins])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True, type=Path)
    ap.add_argument("--mv", required=True, type=Path)
    ap.add_argument("--ld-stats-dir", required=True, type=Path)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    best = pickle.load(args.fit.open("rb"))
    mle = {k: float(v) for k, v in best["best_params"].items()}
    mv = pickle.load(args.mv.open("rb"))
    model_func, model_name = model_for(list(mle))
    r_bins = r_bins_from_mv(mv)

    theory = theory_for(mle, model_func, r_bins)
    ld_names = mv["stats"][0]
    focus_idx = [ld_names.index(s) for s in FOCUS_LD_STATS]

    windows = load_windows(args.ld_stats_dir)
    win_list = list(windows.values())
    h_names = win_list[0]["stats"][1]
    bins = win_list[0]["bins"]
    x = bin_midpoints(bins)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, stat, k in zip(axes, FOCUS_LD_STATS, focus_idx):
        for wid, w in windows.items():
            normed = average_ld_structure([w], ld_names, h_names)
            curve = np.array([normed[b][k] for b in range(len(bins))])
            ax.plot(x, curve, color="tab:blue", alpha=0.2, lw=1)
        emp_mean = np.array([mv["means"][b][k] for b in range(len(bins))])
        theory_curve = np.array([theory[b][k] for b in range(len(bins))])
        ax.plot(x, emp_mean, color="black", lw=2.5, label="empirical mean")
        ax.plot(x, theory_curve, color="tab:red", lw=2.5, ls="--", label="fitted model (theory)")
        ax.set_xscale("log")
        ax.set_xlabel("recombination distance r (bin midpoint)")
        ax.set_ylabel(f"{stat} / {cj.NORMALIZATION if False else 'pi2_0_0_0_0'}")
        ax.set_title(stat)
        ax.legend(fontsize=8)
    fig.suptitle(f"Theoretical vs. empirical LD decay -- {args.title} ({model_name}, LL={best['best_lls']:.2f})")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

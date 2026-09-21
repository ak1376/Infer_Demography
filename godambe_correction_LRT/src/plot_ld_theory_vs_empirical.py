#!/usr/bin/env python3
# godambe_correction_LRT/src/plot_ld_theory_vs_empirical.py

"""
Theoretical-vs-empirical LD-curve comparison plot for one arm/model fit from
the godambe momentsld/ pipeline (fit_simple.py's model zoo, NOT
src/simulation.py's -- reusing MomentsLD_inference.create_comparison_plot
as-is would silently load the wrong demographic model).

r_bins are derived from the empirical means.varcovs.pkl's own 'bins' entry
(same fix as fit_simple_ld.py's _r_bins_from_mv), not a hardcoded default --
the empirical side already has the right bins; only the theoretical side
needs to be told to match them.

Usage:
  python godambe_correction_LRT/src/plot_ld_theory_vs_empirical.py \
      --mv godambe_correction_LRT/real_arms/Chr3L/momentsld/means.varcovs.diagonly.pkl \
      --best-fit godambe_correction_LRT/real_arms/Chr3L/momentsld/fit_simple/best_fit.pkl \
      --model split_migration_growth \
      --out godambe_correction_LRT/real_arms/Chr3L/momentsld/theory_vs_empirical_simple.pdf
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import moments
from src.MomentsLD_inference import compute_theoretical_ld
from fit_simple_ld import resolve_model, _r_bins_from_mv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mv", required=True, type=Path, help="means.varcovs.pkl used for the fit")
    ap.add_argument("--best-fit", required=True, type=Path, help="fit_<model>/best_fit.pkl")
    ap.add_argument("--model", required=True, help="split_migration_growth(_both) or co_const_at_anc/co_grow_from_anc")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.mv, "rb") as f:
        mv = pickle.load(f)
    with open(args.best_fit, "rb") as f:
        fit = pickle.load(f)
    params = fit["best_params"] if "best_params" in fit and isinstance(fit["best_params"], dict) else fit

    param_names, lb, ub, demo_function = resolve_model(args.model)
    r_bins = _r_bins_from_mv(mv)
    populations = mv.get("pops", ["CO", "FR"])

    log_params = [np.log10(params[name]) for name in param_names]
    theoretical_ld = compute_theoretical_ld(log_params, param_names, demo_function, r_bins, populations)

    emp_means = mv["means"][:-1]   # drop H, matching create_comparison_plot's convention
    emp_covars = mv["varcovs"][:-1]
    r_vec_plot = r_bins

    if len(emp_means) != len(theoretical_ld) - 1:
        raise ValueError(
            f"bin count mismatch after fix: {len(emp_means)} empirical vs "
            f"{len(theoretical_ld) - 1} theoretical LD bins -- something upstream still disagrees on r_bins"
        )

    stats_to_plot = [
        ["DD_0_0"], ["DD_0_1"], ["DD_1_1"],
        ["Dz_0_0_0"], ["Dz_0_1_1"], ["Dz_1_1_1"],
        ["pi2_0_0_1_1"], ["pi2_0_1_0_1"], ["pi2_1_1_1_1"],
    ]
    labels = [
        [r"$D_0^2$"], [r"$D_0 D_1$"], [r"$D_1^2$"],
        [r"$Dz_{0,0,0}$"], [r"$Dz_{0,1,1}$"], [r"$Dz_{1,1,1}$"],
        [r"$\pi_{2;0,0,1,1}$"], [r"$\pi_{2;0,1,0,1}$"], [r"$\pi_{2;1,1,1,1}$"],
    ]

    fig = moments.LD.Plotting.plot_ld_curves_comp(
        theoretical_ld, emp_means, emp_covars,
        rs=r_vec_plot, stats_to_plot=stats_to_plot, labels=labels,
        rows=3, plot_vcs=True, show=False, fig_size=(6, 4),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

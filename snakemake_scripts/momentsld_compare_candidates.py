#!/usr/bin/env python3
# snakemake_scripts/momentsld_compare_candidates.py
#
# Overlay two candidate parameter sets' theoretical LD-decay curves against
# the same empirical means/covariances, for direct visual comparison (e.g.
# a "best" vs. "next-best, N_ANC pinned" candidate).

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import moments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.MomentsLD_real_data import (  # noqa: E402
    compute_theoretical_ld,
    load_demographic_function,
    load_json,
    load_pickle,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--empirical", required=True, type=Path)
    ap.add_argument("--candidates", nargs="+", required=True,
                     help="label=path.pkl pairs, e.g. best=candidate_best.pkl upper_anc=candidate_upper_anc.pkl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--normalization", type=int, default=0)
    args = ap.parse_args()

    cfg = load_json(args.config)
    mv = load_pickle(args.empirical)
    demo_fn = load_demographic_function(cfg)
    r_bins = np.array(
        [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3], dtype=float
    )
    populations = list(cfg.get("num_samples", {}).keys())
    param_names = list(cfg.get("parameter_order"))

    colors = {"best": "tab:green", "upper_anc": "tab:red"}
    default_colors = ["tab:green", "tab:red", "tab:blue", "tab:orange"]

    candidates: Dict[str, Dict[str, float]] = {}
    for i, spec in enumerate(args.candidates):
        label, path = spec.split("=", 1)
        candidates[label] = pickle.load(open(path, "rb"))

    _diag_once = {"did": True}
    theory_by_label = {}
    for label, params in candidates.items():
        log10_params = np.log10(np.array([params[p] for p in param_names], dtype=float))
        theo = compute_theoretical_ld(
            log10_params,
            param_names=param_names,
            demographic_model_abs=demo_fn,
            r_bins=r_bins,
            populations=populations,
            N_ref=1.0,
            use_scaled_units=False,
            _diagnostic_once=_diag_once,
        )
        theory_processed = moments.LD.LDstats(theo[:], num_pops=theo.num_pops, pop_ids=theo.pop_ids)
        theory_processed = moments.LD.Inference.remove_normalized_lds(
            theory_processed, normalization=args.normalization
        )
        theory_by_label[label] = [np.asarray(x) for x in theory_processed[:-1]]  # drop het term

    emp_means = [np.asarray(x) for x in mv["means"]]
    emp_covars = [np.asarray(x) for x in mv["varcovs"]]
    emp_means, emp_covars = moments.LD.Inference.remove_normalized_data(
        emp_means, emp_covars, normalization=args.normalization, num_pops=theo.num_pops
    )
    emp_means = emp_means[:-1]
    emp_covars = emp_covars[:-1]

    stat_names_full = moments.LD.Util.moment_names(theo.num_pops)[0]
    _del_idx = stat_names_full.index(
        "pi2_{0}_{0}_{0}_{0}".format(args.normalization)
    )
    stat_names = [s for i, s in enumerate(stat_names_full) if i != _del_idx]
    n_bins = len(emp_means)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    r_mid = r_mid[:n_bins] if len(r_mid) >= n_bins else np.arange(n_bins)

    n_stats = len(stat_names)
    ncols = 4
    nrows = -(-n_stats // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for s_idx, stat in enumerate(stat_names):
        ax = axes[s_idx]
        obs = np.array([emp_means[b][s_idx] for b in range(n_bins)])
        err = np.array([np.sqrt(max(emp_covars[b][s_idx, s_idx], 0)) if emp_covars[b].ndim == 2
                        else np.sqrt(max(emp_covars[b][s_idx], 0)) for b in range(n_bins)])
        ax.errorbar(r_mid, obs, yerr=err, fmt="o", color="black", ms=3, lw=1, capsize=2, label="empirical")
        for j, (label, arrs) in enumerate(theory_by_label.items()):
            pred = np.array([arrs[b][s_idx] for b in range(n_bins)])
            ax.plot(r_mid, pred, "-", color=default_colors[j % len(default_colors)], label=label)
        ax.set_xscale("log")
        ax.set_title(stat, fontsize=9)
        ax.tick_params(labelsize=7)

    for i in range(n_stats, len(axes)):
        axes[i].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("Empirical vs theoretical LD statistics: candidate comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

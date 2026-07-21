#!/usr/bin/env python3
# godambe_correction_LRT/scripts/compute_J_ld.py

"""
Hand-rolled J for the LD growth test.

J = variance across bootstrap replicates of the log-likelihood's slope in the
CO-growth direction, evaluated at growth = 0.

"Growth" is not an explicit parameter -- it's whether the two CO sizes differ.
So we promote N_CO1 (present-day CO size) to the growth knob and hold N_CO0
(split-time CO size) fixed at the null value. "growth = 0" is the point where
N_CO1 == N_CO0 == null N_CO; the slope is a finite-difference wiggle of N_CO1
around that point.

For each replicate:  slope = dLL/dN_CO1 at N_CO1 = N_CO0, using that replicate's
LD curve as the data and the FIXED full-data varcovs as the weighting.
Then  J = mean(slope**2)  (the slopes are ~mean-zero at the optimum).

  scores  = the per-replicate slopes (one number per bootstrap replicate)
  J       = mean(scores**2), a single number = how much the slopes scatter

Built to be re-run across non-overlapping window sizes: use --tag to label a run
and --out to collect J vs window size.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import moments

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from src.MomentsLD_inference import (
    prepare_data_for_comparison,     # normalization removal + alignment
    compute_composite_likelihood,    # the Gaussian LD log-likelihood
)
from src.demes_models import split_migration_growth_both_model  # ALT model

logging.getLogger().setLevel(logging.WARNING)  # silence the DEBUG spam

GC = ROOT / "godambe_correction_LRT"

# MUST match the r_bins the LD stats were computed with.
R_BINS = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5])
POPULATIONS = ["CO", "FR"]
NORMALIZATION = 0

# Ground truth (constant CO) -- the clean choice for p0 in a calibration study.
TRUTH = {
    "N_ANC": 2.0e6, "N_CO": 2.0e6, "N_FR0": 2.0e6, "N_FR1": 2.0e6,
    "T": 4.0e5, "m_CO_FR": 1.0e-4, "m_FR_CO": 1.0e-4,
}


def build_alt_params(null_params: dict, n_co1: float) -> dict:
    """ALT-model params: N_CO0 held at null N_CO, N_CO1 = the growth knob."""
    return {
        "N_ANC":   null_params["N_ANC"],
        "N_CO0":   null_params["N_CO"],   # split-time CO size, held fixed
        "N_CO1":   n_co1,                  # present CO size -- the growth axis
        "N_FR0":   null_params["N_FR0"],
        "N_FR1":   null_params["N_FR1"],
        "T":       null_params["T"],
        "m_CO_FR": null_params["m_CO_FR"],
        "m_FR_CO": null_params["m_FR_CO"],
    }


def theoretical_ld_linear(alt_param_dict: dict):
    """compute_theoretical_ld in LINEAR space (no 10** step), so any N_CO1 works.

    Mirrors src/MomentsLD_inference.py::compute_theoretical_ld exactly, incl.
    Simpson-binning and appending the heterozygosity block (ld_edges[-1]).
    """
    graph = split_migration_growth_both_model(alt_param_dict)
    ref_size = alt_param_dict["N_ANC"]
    rho_edges = 4.0 * ref_size * np.asarray(R_BINS)
    ld_edges = moments.Demes.LD(graph, sampled_demes=POPULATIONS, rho=rho_edges)
    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=POPULATIONS, rho=rho_mids)
    ld_bins = [(ld_edges[i] + ld_edges[i + 1] + 4 * ld_mids[i]) / 6.0
               for i in range(len(rho_mids))]
    ld_bins.append(ld_edges[-1])
    ld_stats = moments.LD.LDstats(ld_bins, num_pops=ld_edges.num_pops,
                                  pop_ids=ld_edges.pop_ids)
    # σD² normalization -- WITHOUT this the theory is raw coalescent-unit LD
    # (~1e12) and dwarfs the σD² empirical curve (~0.1). Matches
    # compute_theoretical_ld's final step.
    return moments.LD.Inference.sigmaD2(ld_stats)


def ll_from_theory(theory, curve, varcovs) -> float:
    """Gaussian LD log-likelihood of a PRECOMPUTED theory curve vs `curve`,
    weighted by `varcovs`. `curve`/`varcovs` keep the norm stat.
    """
    theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(
        theory, {"means": curve, "varcovs": varcovs}, NORMALIZATION
    )
    return compute_composite_likelihood(emp_means, emp_covars, theory_arrays)


def _growth_theories(null_params: dict, eps_frac: float):
    """The two model curves at N_CO1 = N_CO0 +/- step. They depend ONLY on the
    parameters (not on any replicate), so compute once and reuse everywhere.
    """
    N_CO = float(null_params["N_CO"])
    step = eps_frac * N_CO
    theory_plus  = theoretical_ld_linear(build_alt_params(null_params, N_CO + step))
    theory_minus = theoretical_ld_linear(build_alt_params(null_params, N_CO - step))
    return theory_plus, theory_minus, step


def growth_score(curve, varcovs, null_params: dict, eps_frac: float = 0.01,
                 theories=None) -> float:
    """Slope dLL/dN_CO1 at N_CO1 = N_CO0 (= growth 0), other params fixed.

    Centered finite difference; step is a fraction of N_CO (a size ~millions),
    so it's numerically safe (unlike nudging a growth RATE from 0). Pass
    precomputed `theories` (from _growth_theories) to avoid recomputing the
    replicate-independent model curves.
    """
    if theories is None:
        theories = _growth_theories(null_params, eps_frac)
    theory_plus, theory_minus, step = theories
    ll_plus  = ll_from_theory(theory_plus, curve, varcovs)
    ll_minus = ll_from_theory(theory_minus, curve, varcovs)
    return (ll_plus - ll_minus) / (2.0 * step)


def compute_J(all_boot, full_varcovs, null_params, eps_frac: float = 0.01):
    """J = mean over replicates of the squared growth score.

    Returns (J, scores): scores is one slope per replicate; J = mean(scores**2).
    """
    theories = _growth_theories(null_params, eps_frac)   # compute ONCE, reuse
    scores = np.array([
        growth_score(rep, full_varcovs, null_params, eps_frac, theories)
        for rep in all_boot
    ])
    J = float(np.mean(scores ** 2))
    return J, scores


def main():
    ap = argparse.ArgumentParser(description="Hand-rolled LD Godambe J (growth direction).")
    ap.add_argument("--nonoverlap-dir", type=Path, default=GC / "ld" / "nonoverlap")
    ap.add_argument("--eps-frac", type=float, default=0.01,
                    help="finite-difference step as a fraction of N_CO (default %(default)s)")
    ap.add_argument("--tag", type=str, default="nonoverlap",
                    help="label for this run (e.g. the window size), for the J sweep")
    ap.add_argument("--out", type=Path, default=GC / "ld" / "J_results.pkl",
                    help="append J for this tag into this pickle")
    ap.add_argument("--null-fit", type=Path, default=None,
                    help="best_fit.pkl to load p0 (best_params) from. If omitted, "
                         "uses the hardcoded ground-truth TRUTH (simulation only).")
    args = ap.parse_args()

    if args.null_fit is not None:
        with open(args.null_fit, "rb") as f:
            null_params = pickle.load(f)["best_params"]   # fitted SIMPLE-model p0
    else:
        null_params = dict(TRUTH)   # p0 = ground-truth null (simulation only)

    with open(args.nonoverlap_dir / "means.varcovs.pkl", "rb") as f:
        mv = pickle.load(f)
    with open(args.nonoverlap_dir / "bootstrap_sets.pkl", "rb") as f:
        all_boot = pickle.load(f)
    full_varcovs = mv["varcovs"]

    print(f"p0 = ground-truth null (N_CO = {null_params['N_CO']:.0f})")
    print(f"{len(all_boot)} bootstrap replicates, eps_frac = {args.eps_frac}")

    # sanity: the score on the actual (full) data curve
    s_data = growth_score(mv["means"], full_varcovs, null_params, args.eps_frac)
    print(f"growth score on the full data curve: {s_data:.6g}")

    J, scores = compute_J(all_boot, full_varcovs, null_params, args.eps_frac)

    print("\n--- per-replicate growth slopes (scores) ---")
    print(f"  mean = {scores.mean():.6g}   (should be ~0: replicates push both ways)")
    print(f"  std  = {scores.std():.6g}")
    print(f"  min  = {scores.min():.6g}")
    print(f"  max  = {scores.max():.6g}")
    print(f"\nJ = mean(scores^2) = {J:.6g}")

    # collect J vs tag (window size) for the sweep
    results = {}
    if args.out.exists():
        with open(args.out, "rb") as f:
            results = pickle.load(f)
    results[args.tag] = {"J": J, "scores": scores, "n_boot": len(all_boot),
                         "eps_frac": args.eps_frac}
    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved J for tag '{args.tag}' -> {args.out}")


if __name__ == "__main__":
    main()

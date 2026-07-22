#!/usr/bin/env python3
# godambe_correction_LRT/scripts/momentsld_inference.py

"""
Moments-LD point-estimate step for the growth LRT, on the OVERLAPPING windows.

Does three things, reusing the repo's simulation Moments-LD code
(src/MomentsLD_inference.py):

  0. Aggregate the per-window LD stats into means + varcovs, and plot the
     EMPIRICAL vs THEORETICAL LD curves at the TRUE (constant-CO) parameters
     -- a data sanity check (create_comparison_plot).
  1. Fit the NULL model  (split_migration_growth      -> CO constant).
  2. Fit the ALT  model  (split_migration_growth_both -> CO grows).

Then reports the raw (uncorrected) LRT = 2*(ll_alt - ll_null).

IMPORTANT
---------
* Uses the TRIMMED r_bins the LD stats were computed with, NOT DEFAULT_R_BINS.
* Runs on ld/overlap (the point-estimate windows). The means/varcovs here come
  from overlapping windows -- fine for the point estimate, but do NOT use the
  bootstrap_sets.pkl written here for the Godambe J; that must come from the
  NON-overlapping tiles in a later step.
* Optimizes in ABSOLUTE units (momentsld_use_scaled_units=False), matching the
  simulation Moments-LD workflow.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from src.MomentsLD_inference import (
    aggregate_ld_statistics,
    create_comparison_plot,
    run_momentsld_inference,
)
from src.demes_models import (
    split_migration_growth_model,
    split_migration_growth_both_model,
)

GC = ROOT / "godambe_correction_LRT"

# MUST match the --r-bins used to compute the LD stats (compute_ld_window).
R_BINS = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5])

NUM_SAMPLES = {"CO": 10, "FR": 9}

# Absolute-unit bounds. Size upper bound widened 10^6.5 -> 10^7.5 (31.6M) after
# N_ANC and N_CO0 railed at 10^6.5 on Chr2L: gives a decade of headroom so we can
# tell "bound was too tight" (params move interior) from "genuinely unidentifiable"
# (params rail at the new bound too). 10^7.5 is a diagnostic ceiling, not a
# biological claim -- Drosophila Ne is ~1-4M.
BOUNDS = {
    "N_ANC":   [3162, 31622776],
    "N_CO":    [3162, 31622776],
    "N_CO0":   [3162, 31622776],
    "N_CO1":   [3162, 31622776],
    "N_FR0":   [3162, 31622776],
    "N_FR1":   [3162, 31622776],
    "T":       [1000, 1000000],
    "m_CO_FR": [1e-08, 0.001],
    "m_FR_CO": [1e-08, 0.001],
}

# Ground truth the data were simulated under (constant CO -> single N_CO).
TRUTH = {
    "N_ANC": 2.0e6, "N_CO": 2.0e6, "N_FR0": 2.0e6, "N_FR1": 2.0e6,
    "T": 4.0e5, "m_CO_FR": 1.0e-4, "m_FR_CO": 1.0e-4,
}

MODELS = {
    "split_migration_growth":       # NULL: CO constant (free N_CO)
        ["N_ANC", "N_CO", "N_FR0", "N_FR1", "T", "m_CO_FR", "m_FR_CO"],
    "split_migration_growth_both":  # ALT:  CO grows (free N_CO0 AND N_CO1)
        ["N_ANC", "N_CO0", "N_CO1", "N_FR0", "N_FR1", "T", "m_CO_FR", "m_FR_CO"],
}
NULL = "split_migration_growth"
ALT  = "split_migration_growth_both"

# ---------------------------------------------------------------------------
# Reparameterized, IDENTIFIABLE CO-growth test (Option A).
#
# The original ALT frees N_CO0 (CO size just after the split) as its own
# parameter. But N_CO0 sits at ~the same time as N_ANC (the pre-split trunk),
# on the SAME lineage (CO is the trunk), so LD cannot separate them -> the two
# are redundant -> the Godambe H matrix is singular (cond(H) ~ 1e25, negative
# adjust). Fix: TIE N_CO0 = N_ANC, so CO continues at the ancestral size through
# the split and then grows/declines to N_CO1. N_ANC stays FREE and becomes the
# SOLE deep-time size (so it is now well identified).
#
#   CO_CONST (null): CO constant at N_ANC (N_CO = N_ANC)      -- 6 free params
#   CO_GROW  (alt) : CO -> N_CO1, starting from N_ANC         -- 7 free params
#   df = 1 (freeing N_CO1 from N_ANC).
#
# We optimize the SIZE N_CO1 (not a rate G_CO, whose null value 0 blows up
# finite differences via exp(-eps*T) at large T); report G_CO = ln(N_CO1/N_ANC)/T
# as a derived quantity.
CO_CONST = "co_const_at_anc"
CO_GROW  = "co_grow_from_anc"
MODELS[CO_CONST] = ["N_ANC", "N_FR0", "N_FR1", "T", "m_CO_FR", "m_FR_CO"]
MODELS[CO_GROW]  = ["N_ANC", "N_CO1", "N_FR0", "N_FR1", "T", "m_CO_FR", "m_FR_CO"]


def co_const_at_anc_model(sampled, cfg=None):
    """NULL: CO constant at the ancestral size (no post-split CO change)."""
    d = dict(sampled)
    d["N_CO"] = d["N_ANC"]
    return split_migration_growth_model(d, cfg)


def co_grow_from_anc_model(sampled, cfg=None):
    """ALT: CO grows/declines exponentially from N_ANC to N_CO1."""
    d = dict(sampled)
    d["N_CO0"] = d["N_ANC"]          # tie the deep-time CO size to the ancestral size
    return split_migration_growth_both_model(d, cfg)


# single source of truth: model name -> demes-graph builder
MODEL_FUNCS = {
    NULL:     split_migration_growth_model,
    ALT:      split_migration_growth_both_model,
    CO_CONST: co_const_at_anc_model,
    CO_GROW:  co_grow_from_anc_model,
}


def make_config(model_name, generate_profiles, verbose):
    """Build the config dict run_momentsld_inference expects for one model."""
    param_order = MODELS[model_name]
    return {
        "demographic_model": model_name,
        "priors": {p: BOUNDS[p] for p in param_order},
        "num_samples": NUM_SAMPLES,
        "ld_normalization": 0,
        "momentsld_use_scaled_units": False,   # absolute units, like the sim workflow
        "generate_profiles": generate_profiles,
        "ld_verbose": verbose,
    }


def truth_for(model_name):
    """Truth params keyed to a model's parameter names (N_CO0 = N_CO1 = N_CO)."""
    if model_name == NULL:
        return dict(TRUTH)
    d = {k: v for k, v in TRUTH.items() if k != "N_CO"}
    d["N_CO0"] = TRUTH["N_CO"]
    d["N_CO1"] = TRUTH["N_CO"]
    return d


def main():
    ap = argparse.ArgumentParser(description="Moments-LD inference (null + alt) + raw LRT.")
    ap.add_argument("--ld-root", type=Path, default=GC / "ld" / "overlap",
                    help="dir containing LD_stats/ (default %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=GC / "ld" / "moments_ld_inference")
    ap.add_argument("--profiles", action=argparse.BooleanOptionalAction, default=True,
                    help="generate 1D likelihood profiles per model (default: on)")
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                    help="print every optimizer LL evaluation (default: off)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 0. aggregate LD stats (means + varcovs) from the overlapping windows ---
    mv = aggregate_ld_statistics(args.ld_root)
    n_stats = len(mv["means"])
    print(f"Aggregated LD stats from {args.ld_root} "
          f"({n_stats} statistic groups, r_bins={list(R_BINS)}).")

    # --- 0b. empirical vs theoretical LD curves at the TRUE (null) parameters ---
    plot_path = args.out_dir / "ld_curves_comparison.pdf"
    create_comparison_plot(
        make_config(NULL, generate_profiles=False, verbose=False),
        truth_for(NULL), mv, R_BINS, plot_path,
    )
    print(f"Empirical-vs-theoretical LD plot -> {plot_path}")

    # --- 1 & 2. fit both models to the same LD curve ---
    best = {}
    for model in (NULL, ALT):
        results_dir = args.out_dir / model
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Moments-LD inference: {model} ===")
        run_momentsld_inference(
            make_config(model, generate_profiles=args.profiles, verbose=args.verbose),
            mv, results_dir, R_BINS, sampled_params=truth_for(model),
        )
        with open(results_dir / "best_fit.pkl", "rb") as f:
            best[model] = pickle.load(f)

        r = best[model]
        print(f"  LL = {r['best_lls']:.4f}   (status {r['status']})")
        for p, v in r["best_params"].items():
            tru = truth_for(model).get(p)
            tag = f"  (truth {tru:.4g})" if tru is not None else ""
            print(f"    {p:8s} = {v:12.4g}{tag}")

    # --- raw LRT ---
    ll_null = best[NULL]["best_lls"]
    ll_alt  = best[ALT]["best_lls"]
    lrt = 2.0 * (ll_alt - ll_null)
    print("\n" + "=" * 60)
    print(f"LL(null={NULL})      = {ll_null:.4f}")
    print(f"LL(alt ={ALT}) = {ll_alt:.4f}")
    print(f"raw LRT = 2*(ll_alt - ll_null) = {lrt:.4f}   (df = 1, UNCORRECTED)")
    print("Next: Godambe correction using the NON-overlapping tiles.")

    summary = {
        "ll_null": ll_null, "ll_alt": ll_alt, "lrt_uncorrected": lrt, "df": 1,
        "best_null": best[NULL], "best_alt": best[ALT],
        "r_bins": R_BINS, "ld_root": str(args.ld_root),
    }
    out_pkl = args.out_dir / "lrt_summary_ld.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(summary, f)
    print(f"Saved -> {out_pkl}")


if __name__ == "__main__":
    main()

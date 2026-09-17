#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/run_lrt.py

"""
Standalone diagnostic (not called by any Snakemake rule): Godambe-corrected
composite-likelihood LRT for CO growth, one arm. See the module docstring in
src/run_lrt.py for the statistical pipeline and the two supported
parameterizations (--alt-model).

Heavy lifting lives in:
  godambe_correction_LRT/src/run_lrt.py
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import compute_J_ld as cj                        # R_BINS, POPULATIONS, NORMALIZATION, ll_from_theory
from momentsld_inference import ALT, CO_GROW
from run_lrt import (
    TESTS, make_model_func, embed_null, build_all_boot,
    validate_model_func, diagnose_godambe, restricted_adjust,
    NORMALIZATION,
)


def main():
    ap = argparse.ArgumentParser(description="Godambe-corrected CO-growth LRT for one arm.")
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--alt-model", default=CO_GROW, choices=[CO_GROW, ALT],
                    help="which alt model / parameterization (default %(default)s)")
    ap.add_argument("--blocksize", type=int, default=100_000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--null-fit", type=Path, default=None,
                    help="override null best_fit.pkl (default: <arm>/fit_<null_tag>/best_fit.pkl)")
    ap.add_argument("--alt-fit", type=Path, default=None,
                    help="override alt best_fit.pkl (default: <arm>/fit_<alt_tag>/best_fit.pkl)")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    spec = TESTS[args.alt_model]
    arm = args.arm_dir
    model_func, alt_names = make_model_func(args.alt_model)

    null_path = args.null_fit or (arm / f"fit_{spec['null_tag']}" / "best_fit.pkl")
    null = pickle.load(null_path.open("rb"))
    null_params = null["best_params"]
    ll_null = float(null["best_lls"])
    p0, alt_dict, nested_indices = embed_null(null_params, args.alt_model)

    print(f"arm            : {arm.name}")
    print(f"test           : null={spec['null_model']}  ->  alt={args.alt_model}")
    print(f"p0 ({len(p0)}-vec, +Ne): {['%.4g' % v for v in p0]}")
    print(f"nested_indices : {nested_indices}  (df={len(nested_indices)})\n")

    ok = validate_model_func(model_func, args.alt_model, p0, alt_dict)
    if args.validate_only:
        return
    if not ok:
        raise SystemExit("model_func validation FAILED -- fix scaling first.")

    alt_path = args.alt_fit or (arm / f"fit_{spec['alt_tag']}" / "best_fit.pkl")
    if not alt_path.exists():
        print(f"\n[waiting] {alt_path} not present yet.")
        return
    alt = pickle.load(alt_path.open("rb"))
    p_alt = alt["best_params"]
    ll_alt = float(alt["best_lls"])

    mv = pickle.load((arm / "overlap" / "means.varcovs.pkl").open("rb"))
    ld_dir = arm / "sweep" / f"bs{args.blocksize}" / "LD_stats"
    all_boot, nn = build_all_boot(ld_dir, args.n_boot, args.seed)
    print(f"\nall_boot: {len(all_boot)} replicates from {nn} tiles @ {args.blocksize} bp")

    # full-matrix version (moments LD) -- shown only to document the breakdown
    cond, tr_full = diagnose_godambe(model_func, all_boot, p0, mv["means"], mv["varcovs"],
                                     args.eps, np.asarray(cj.R_BINS), NORMALIZATION, pass_Ne=False)
    # restricted version (SFS-style) -- the one we actually use
    c, tr = restricted_adjust(
        model_func, all_boot, p0, nested_indices, mv["means"], mv["varcovs"],
        args.eps, np.asarray(cj.R_BINS), NORMALIZATION, pass_Ne=False)

    D = 2.0 * (ll_alt - ll_null)
    D_adj = c * D
    df = len(nested_indices)
    p_two = float(chi2.sf(D_adj, df=df)) if D_adj > 0 else 1.0
    p_one = 0.5 * p_two

    # growth direction (for co_grow_from_anc: N_CO1 vs N_ANC; else N_CO1 vs N_CO0)
    n1 = float(p_alt.get("N_CO1"))
    n0 = float(p_alt.get("N_CO0", p_alt.get("N_ANC")))
    grew = n1 > n0
    G_CO = np.log(n1 / n0) / float(p_alt["T"])

    print("\n" + "=" * 64)
    print(f"ll_null={ll_null:.4f}  ll_alt={ll_alt:.4f}")
    print(f"raw   D      = 2*(ll_alt-ll_null) = {D:.4f}")
    print(f"adjust c     = df/trace(J H^-1)   = {c:.4g}   (restricted; SFS-style)"
          + ("   [INVALID: adjust<=0]" if c <= 0 else ""))
    print(f"corrected    D_adj = c*D          = {D_adj:.4f}")
    print(f"CO size      : {n0:.4g} -> N_CO1={n1:.4g}  ({'GREW' if grew else 'shrank/flat'}, "
          f"G_CO={G_CO:.3g}/gen)")
    print(f"p-value (two-sided chi2_{df})       = {p_two:.4g}")
    print(f"p-value (one-sided, growth test)  = {p_one:.4g}"
          + ("" if grew else "   [MLE is not growth -> growth not supported]"))
    print("=" * 64)

    out = arm / f"lrt_result_{spec['alt_tag']}.pkl"
    with out.open("wb") as f:
        pickle.dump({"alt_model": args.alt_model, "D": D, "adjust": float(c),
                     "D_adj": D_adj, "cond_H": float(cond), "trace_JHinv": tr,
                     "p_two_sided": p_two, "p_one_sided": p_one,
                     "ll_null": ll_null, "ll_alt": ll_alt, "grew": grew,
                     "G_CO": float(G_CO), "blocksize": args.blocksize}, f)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

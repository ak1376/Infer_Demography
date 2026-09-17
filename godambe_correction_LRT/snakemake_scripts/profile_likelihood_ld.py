#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/profile_likelihood_ld.py

"""
Standalone diagnostic (not called by any Snakemake rule): 1D composite-
likelihood profiles for a Moments-LD fit.

Reads a best_fit.pkl (null or complex; the model is inferred from the
parameter names) and the arm's overlap/means.varcovs.pkl, then plots LL vs
each parameter -- one separate figure per parameter (+ a combined panel).

Diagnostic reading:
  * a clear interior peak            -> identifiable
  * monotonic rise to a bound        -> railed / unidentifiable in that direction
  * flat                             -> no information on that parameter
A dashed vertical line marks the MLE; a horizontal line marks LL_max - 1.92
(the ~95% CI cut for 1 df, where it applies).

Heavy lifting lives in:
  godambe_correction_LRT/src/profile_likelihood_ld.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from profile_likelihood_ld import model_for, ll_at, compute_profile_grid
from momentsld_inference import MODELS


def main():
    ap = argparse.ArgumentParser(description="1D composite-likelihood profiles for an LD fit.")
    ap.add_argument("--fit", type=Path, required=True, help="best_fit.pkl (null or complex)")
    ap.add_argument("--mv", type=Path, required=True, help="overlap/means.varcovs.pkl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-grid", type=int, default=25, help="points per profile (default %(default)s)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = pickle.load(args.fit.open("rb"))
    mle = {k: float(v) for k, v in best["best_params"].items()}
    mv = pickle.load(args.mv.open("rb"))
    model_func, model_name = model_for(list(mle))
    param_names = MODELS[model_name]

    ll_mle = ll_at(mle, model_func, mv)
    print(f"model = {model_name}   LL(MLE) = {ll_mle:.4f}")

    n = len(param_names)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)

    for idx, pname in enumerate(param_names):
        grid, lls = compute_profile_grid(pname, mle, model_func, mv, args.n_grid)
        lo, hi = grid[0], grid[-1]

        at_bound = (abs(mle[pname] - lo) / lo < 1e-3) or (abs(mle[pname] - hi) / hi < 1e-3)

        # per-parameter separate figure
        f1, a1 = plt.subplots(figsize=(5, 3.6))
        for ax in (a1, axes[idx // ncol][idx % ncol]):
            ax.plot(grid, lls, "-o", ms=3, lw=1)
            ax.axvline(mle[pname], color="red", ls="--", lw=1,
                       label=f"MLE={mle[pname]:.3g}" + ("  (RAILED)" if at_bound else ""))
            ax.axhline(ll_mle - 1.92, color="gray", ls=":", lw=1, label="LL_max - 1.92")
            ax.set_xscale("log")
            ax.set_xlabel(pname); ax.set_ylabel("composite LL")
            ax.set_title(f"{pname}" + ("  [railed]" if at_bound else ""), fontsize=10)
            ax.legend(fontsize=7)
        f1.tight_layout()
        out1 = args.out_dir / f"profile_{pname}.png"
        f1.savefig(out1, dpi=130); plt.close(f1)
        flag = "  <-- RAILED" if at_bound else ""
        print(f"  {pname:8s}: LL range [{np.nanmin(lls):.1f}, {np.nanmax(lls):.1f}]"
              f"  MLE={mle[pname]:.4g}{flag}  -> {out1.name}")

    # blank any unused panels
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"1D likelihood profiles ({model_name})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    combined = args.out_dir / "profiles_all.png"
    fig.savefig(combined, dpi=130); plt.close(fig)
    print(f"\ncombined panel -> {combined}")
    print(f"separate plots -> {args.out_dir}/profile_<param>.png")


if __name__ == "__main__":
    main()

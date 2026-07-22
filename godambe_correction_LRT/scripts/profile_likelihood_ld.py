#!/usr/bin/env python3
# godambe_correction_LRT/scripts/profile_likelihood_ld.py

"""
1D composite-likelihood profiles for a Moments-LD fit.

For each parameter, sweep it across its bound range (log10 grid) while holding
every OTHER parameter at the fitted MLE, recompute the composite log-likelihood
against the arm's overlapping LD curve, and plot LL vs that parameter -- one
separate figure per parameter (+ a combined panel).

Reads a best_fit.pkl (null or complex; the model is inferred from the parameter
names) and the arm's overlap/means.varcovs.pkl. Reuses the validated
compute_J_ld theory + likelihood machinery, so the curve matches the fit exactly.

Diagnostic reading:
  * a clear interior peak            -> identifiable
  * monotonic rise to a bound        -> railed / unidentifiable in that direction
  * flat                             -> no information on that parameter
A dashed vertical line marks the MLE; a horizontal line marks LL_max - 1.92
(the ~95% CI cut for 1 df, where it applies).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_J_ld as cj                        # R_BINS, POPULATIONS, NORMALIZATION, ll_from_theory
from momentsld_inference import BOUNDS, MODELS, MODEL_FUNCS

logging.getLogger().setLevel(logging.WARNING)


def model_for(param_names):
    """Pick the demes model whose parameter set exactly matches the fit."""
    s = set(param_names)
    for name, plist in MODELS.items():
        if set(plist) == s:
            return MODEL_FUNCS[name], name
    raise ValueError(f"no model matches parameter set {sorted(s)}")


def theory_for(param_dict, model_func):
    """σD²-normalized theory LD curve for a param dict (mirrors theoretical_ld_linear)."""
    graph = model_func(param_dict)
    ref = float(param_dict["N_ANC"])
    rho_edges = 4.0 * ref * np.asarray(cj.R_BINS)
    ld_edges = moments.Demes.LD(graph, sampled_demes=cj.POPULATIONS, rho=rho_edges)
    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=cj.POPULATIONS, rho=rho_mids)
    ld_bins = [(ld_edges[i] + ld_edges[i + 1] + 4 * ld_mids[i]) / 6.0
               for i in range(len(rho_mids))]
    ld_bins.append(ld_edges[-1])
    ld_stats = moments.LD.LDstats(ld_bins, num_pops=ld_edges.num_pops,
                                  pop_ids=ld_edges.pop_ids)
    return moments.LD.Inference.sigmaD2(ld_stats)


def ll_at(param_dict, model_func, mv):
    return cj.ll_from_theory(theory_for(param_dict, model_func),
                             mv["means"], mv["varcovs"])


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
        lo, hi = BOUNDS[pname]
        grid = np.logspace(np.log10(lo), np.log10(hi), args.n_grid)
        lls = []
        for val in grid:
            d = dict(mle); d[pname] = float(val)
            try:
                lls.append(ll_at(d, model_func, mv))
            except Exception:
                lls.append(np.nan)
        lls = np.asarray(lls, float)

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

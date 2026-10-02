#!/usr/bin/env python3
"""Overlay the current best MomentsLD fit on the empirical LD curves (works mid-run).

Usage: plot_ld_fit_overlay.py <ld_dir with means.varcovs.pkl> <fit dir containing runs/run_*/inferences/MomentsLD> <config.json> <out.pdf>
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import moments
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.MomentsLD_real_data import (compute_theoretical_ld, composite_gaussian_ll,
                                     load_demographic_function, prepare_data_for_comparison)

ld_dir, fit_dir, cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
fit_subdir = sys.argv[5] if len(sys.argv) > 5 else "MomentsLD"      # e.g. MomentsLD_100kb_genmap_phased
label = sys.argv[6] if len(sys.argv) > 6 else "Chr3L pseudo-diploid, 100 kb windows"
cfg = json.load(open(cfg_path))
mv = pickle.load(open(ld_dir / "means.varcovs.pkl", "rb"))
ld_names, h_names = mv["stats"]
bins = np.array(mv["bins"])
r_edges = np.concatenate([bins[:, 0], bins[-1:, 1]])
mids = np.sqrt(np.maximum(bins[:, 0], bins[:, 1] / 10) * bins[:, 1])

# best finished start so far
fits = []
for f in glob.glob(str(fit_dir / f"runs/run_*/inferences/{fit_subdir}/best_fit.pkl")):
    d = pickle.load(open(f, "rb"))
    fits.append((float(np.ravel(d["best_ll"])[0]), f, d))
# Also consider still-running starts: their optimization_log.txt lists every
# evaluated point (params printed to 3 significant figures, so the recomputed
# LL for such a point differs slightly from the logged one).
import re
n_logs = 0
for f in glob.glob(str(fit_dir / f"runs/run_*/inferences/{fit_subdir}/optimization_log.txt")):
    evs = re.findall(r"LL = (-?[\d.]+) \| (.*?) \| \(N_ref", open(f).read())
    if not evs:
        continue
    n_logs += 1
    ll, ptxt = max(evs, key=lambda t: float(t[0]))
    p = {k: float(v) for k, v in (kv.split("=") for kv in ptxt.split(", "))}
    fits.append((float(ll), f + " (in progress, 3 s.f.)",
                 {"param_order": list(cfg["priors"]), "best_params": p, "use_scaled_units": False,
                  "fixed_N_ref": 1.0, "normalization": 0}))
print(f"{n_logs} starts with logged evaluations")
# The aggregated fit (inferences/<subdir>/best_fit.pkl), when it exists, is the final answer: use it.
agg = fit_dir / "inferences" / fit_subdir / "best_fit.pkl"
if agg.exists():
    d = pickle.load(open(agg, "rb"))
    fits.append((float(np.ravel(d["best_ll"])[0]) + 1e6, str(agg), d))   # always ranks first
# prefer an exact finished best_fit.pkl over a 3-s.f. log point when LLs tie (within 0.1)
fits.sort(key=lambda t: -(round(t[0], 1) + (0.01 if t[1].endswith("best_fit.pkl") else 0)))
best_ll, best_path, best = fits[0]
if best_ll > 1e5:
    best_ll -= 1e6
if isinstance(best.get("best_params"), list):          # aggregated top-K schema -> rank 0
    best = {**best, "best_params": best["best_params"][0]}
best.setdefault("param_order", list(cfg["priors"]))
best.setdefault("use_scaled_units", bool(cfg.get("momentsld_use_scaled_units", True)))
best.setdefault("fixed_N_ref", 1.0)
best.setdefault("normalization", 0)
names = list(best["param_order"])
params = best["best_params"]
print(f"{len(fits)} candidate points; best LL {best_ll:.3f} from {best_path}")

theo = compute_theoretical_ld(
    np.log10([params[n] for n in names]), param_names=names,
    demographic_model_abs=load_demographic_function(cfg), r_bins=r_edges,
    populations=list(cfg["num_samples"].keys()), N_ref=float(best.get("fixed_N_ref") or 1.0),
    use_scaled_units=bool(best["use_scaled_units"]), _diagnostic_once={"did": True},
)
t_arr, e_m, e_c = prepare_data_for_comparison(theo, mv, normalization=int(best["normalization"]))
ll_check = composite_gaussian_ll(e_m, e_c, t_arr)
print(f"recomputed LL {ll_check:.3f} (stored {best_ll:.3f})")

theo_names = moments.LD.Util.moment_names(2)[0]
PANELS = [
    ("DD_0_0", r"$\sigma_D^2$ CO"), ("DD_0_1", r"$D_{CO}D_{FR}$"), ("DD_1_1", r"$\sigma_D^2$ FR"),
    ("Dz_0_0_0", r"$Dz$ CO"), ("Dz_0_1_1", r"$Dz_{CO,FR,FR}$"), ("Dz_1_1_1", r"$Dz$ FR"),
    ("pi2_0_0_1_1", r"$\pi_2$ CO,CO,FR,FR"), ("pi2_0_1_0_1", r"$\pi_2$ CO,FR,CO,FR"), ("pi2_1_1_1_1", r"$\pi_2$ FR"),
]
nb = len(bins)
ptxt = ", ".join(f"{n}={params[n]:.3g}" for n in names)
header = (f"{label} — best fit over {n_logs} starts, "
          f"LL={best_ll:.2f}\n{ptxt}")

with PdfPages(out) as pdf:
    for resid in (False, True):
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
        for ax, (stat, label) in zip(axes.flat, PANELS):
            i = ld_names.index(stat)
            m = np.array([mv["means"][k][i] for k in range(nb)])
            se = np.array([np.sqrt(np.asarray(mv["varcovs"][k])[i, i]) for k in range(nb)])
            model = np.array([theo[k][theo_names.index(stat)] for k in range(nb)])
            if resid:
                ax.errorbar(mids, (m - model) / se, fmt="o", color="black", ms=3)
                ax.axhline(0, color="tab:red", lw=1)
                for y in (-2, 2):
                    ax.axhline(y, color="grey", ls="--", lw=0.7)
                ax.set_ylabel("(data − model) / SE")
            else:
                ax.errorbar(mids, m, yerr=se, fmt="o", color="black", ms=3, capsize=2, label="data ± SE")
                ax.plot(mids, model, color="tab:red", lw=1.5, label="model")
                ax.axhline(0, color="grey", lw=0.5)
            ax.set_xscale("log")
            ax.set_title(label)
        for ax in axes[-1]:
            ax.set_xlabel("r (Comeron map)")
        if not resid:
            axes[0, 0].legend(fontsize=8)
        fig.suptitle(header + ("\nstandardized residuals" if resid else ""), fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        if not resid:
            fig.savefig(out.with_suffix(".png"), dpi=110)
        plt.close(fig)
print(f"-> {out}")

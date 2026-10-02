#!/usr/bin/env python3
"""
Where does a parameter act on the LD curves?  For each --param, recompute all
nine plotted MomentsLD statistics at best-fit x each --factor (every other
parameter held at the best fit, no re-optimization) and overlay them on the
empirical curves.  One 3x3 figure per parameter, plus a TSV of the % change
per statistic and r bin between the lowest and highest factor.

Usage:
  python snakemake_scripts/plot_momentsld_param_sensitivity.py \
      --config config_files/experiment_config_*.json \
      --empirical .../ld/<ld_name>/Chr3L/means.varcovs.pkl \
      --best-fit .../inferences/<engine>/best_fit.pkl \
      --param T:0.5,1,2 --param m_CO_FR:0.1,1,10 --param m_FR_CO:0.1,1,10 \
      --out-dir .../inferences/<engine>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.MomentsLD_real_data import compute_theoretical_ld, load_demographic_function  # noqa: E402

PANELS = [("DD_0_0", r"$\sigma_D^2$ CO"), ("DD_0_1", r"$D_{CO}D_{FR}$"), ("DD_1_1", r"$\sigma_D^2$ FR"),
          ("Dz_0_0_0", r"$Dz$ CO"), ("Dz_0_1_1", r"$Dz_{CO,FR,FR}$"), ("Dz_1_1_1", r"$Dz$ FR"),
          ("pi2_0_0_1_1", r"$\pi_2$ CO,CO,FR,FR"), ("pi2_0_1_0_1", r"$\pi_2$ CO,FR,CO,FR"), ("pi2_1_1_1_1", r"$\pi_2$ FR")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--empirical", required=True)
    ap.add_argument("--best-fit", required=True)
    ap.add_argument("--param", action="append", required=True, help="NAME:f1,f2,... multiplicative factors")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mv = pickle.load(open(args.empirical, "rb"))
    fit = pickle.load(open(args.best_fit, "rb"))
    best = dict(fit["best_params"][0] if isinstance(fit["best_params"], list) else fit["best_params"])
    names = list(cfg["priors"])
    demo = load_demographic_function(cfg)
    pops = list(cfg["num_samples"].keys())
    ld_names = mv["stats"][0]
    bins = np.array(mv["bins"]); nb = len(bins)
    r_edges = np.concatenate([bins[:, 0], bins[-1:, 1]])
    x = np.sqrt(np.maximum(bins[:, 0], bins[:, 1] / 10) * bins[:, 1])
    tn = moments.LD.Util.moment_names(2)[0]
    out_dir = Path(args.out_dir)

    def curves(p):
        th = compute_theoretical_ld(np.log10([p[n] for n in names]), param_names=names, demographic_model_abs=demo,
                                    r_bins=r_edges, populations=pops, N_ref=1.0, use_scaled_units=False,
                                    _diagnostic_once={"did": True})
        return {s: np.array([th[b][tn.index(s)] for b in range(nb)]) for s, _ in PANELS}

    rows = []
    for spec in args.param:
        pname, facs = spec.split(":")
        facs = [float(f) for f in facs.split(",")]
        runs = []
        for f in facs:
            p = dict(best); p[pname] = best[pname] * f
            runs.append((f, p[pname], curves(p)))
        cols = plt.get_cmap("coolwarm")(np.linspace(0, 1, len(facs)))
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
        for ax, (st, lab) in zip(axes.flat, PANELS):
            k = ld_names.index(st)
            for (f, v, c), col in zip(runs, cols):
                ax.plot(x, c[st], color=col, lw=2 if f == 1 else 1.4, ls="-" if f == 1 else "--",
                        label=f"{pname} = {v:.3g}" + (" (best fit)" if f == 1 else f" (×{f:g})"))
            m = [mv["means"][b][k] for b in range(nb)]
            se = [np.sqrt(np.asarray(mv["varcovs"][b])[k, k]) for b in range(nb)]
            ax.errorbar(x, m, yerr=se, fmt="o", color="black", ms=3, capsize=2, zorder=5)
            lo_c, hi_c = runs[0][2][st], runs[-1][2][st]
            pct = 100 * (hi_c - lo_c) / np.maximum(np.abs(runs[[f for f, _, _ in runs].index(1.0)][2][st]), 1e-12)
            ax.set_title(f"{lab}   (max change {np.max(np.abs(pct)):.0f}%)", fontsize=10)
            ax.set_xscale("log"); ax.axhline(0, color="grey", lw=0.5)
            rows.append((pname, st, pct))
        for ax in axes[-1]:
            ax.set_xlabel("r (Comeron map)")
        axes[0, 0].legend(fontsize=7)
        others = ", ".join(f"{n}={best[n]:.3g}" for n in names if n != pname)
        fig.suptitle(f"Effect of {pname} alone (others fixed: {others}); black = data ± SE", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / f"sensitivity_{pname}.png", dpi=110); plt.close(fig)
        print(f"-> {out_dir / f'sensitivity_{pname}.png'}")

    with open(out_dir / "sensitivity_summary.tsv", "w") as f:
        f.write("param\tstatistic\t" + "\t".join(f"bin{b}_pct_change" for b in range(nb)) + "\n")
        for pname, st, pct in rows:
            f.write(f"{pname}\t{st}\t" + "\t".join(f"{v:.2f}" for v in pct) + "\n")
    show = [0, 3, 6, 9, 12, 15]
    print("\n% change (lowest -> highest factor, relative to best fit) at r bins", show)
    for pname, st, pct in rows:
        if np.max(np.abs(pct)) >= 5:
            print(f"  {pname:8s} {st:12s} {np.round(pct[show]).astype(int)}")


if __name__ == "__main__":
    main()

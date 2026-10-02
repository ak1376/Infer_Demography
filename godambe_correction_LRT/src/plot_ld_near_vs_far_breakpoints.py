#!/usr/bin/env python3
"""
Split the 100 kb LD windows into "near" (within --buffer-bp of an inversion
breakpoint) and "far", and compare them two ways, per population:

  <out>_longrange.png : long-range r^2 per window along the arm (mean r^2 with
      windows >= 1 Mb away, from the haploid VCF), near windows shaded, with
      the near / far averages printed on each panel.
  <out>_decay.png     : the MomentsLD statistics (sigma_D^2, Dz, per pop and
      cross-pop) aggregated separately over near and far windows, from the
      per-window LD_stats pkls -- i.e. the same curves MomentsLD fits.

Usage:
  python godambe_correction_LRT/src/plot_ld_near_vs_far_breakpoints.py \
      --vcf real_data_analysis/data/drosophila_trimmed/Chr3L/polarized.vcf.gz \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --ld-dir real_data_analysis/data/drosophila_trimmed/ld/Chr3L_100kb_genmap_phased/Chr3L \
      --inversion "In(3L)P" --buffer-bp 3000000 \
      --out real_data_analysis/data/drosophila_trimmed/ld_viz/Chr3L_In3LP/near_vs_far_3Mb
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inversion_breakpoints import INVERSIONS  # noqa: E402
from plot_r2_around_inversion import read_haploid_vcf, standardize, window_bounds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--ld-dir", required=True, help="dir with windows/ and LD_stats/ (one arm)")
    ap.add_argument("--inversion", default="In(3L)P")
    ap.add_argument("--buffer-bp", type=int, default=3_000_000)
    ap.add_argument("--far-bp", type=int, default=1_000_000)
    ap.add_argument("--snps-per-window", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chrom, b1, b2 = INVERSIONS[args.inversion]
    bps = sorted([b1, b2])
    ld_dir = Path(args.ld_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    bounds = window_bounds(str(ld_dir / "windows"))
    starts = np.array([b[0] for b in bounds]); ends = np.array([b[1] for b in bounds])
    mids = (starts + ends) / 2
    near = np.array([any(s < bp + args.buffer_bp and e > bp - args.buffer_bp for bp in bps)
                     for s, e in bounds])
    buf_mb = args.buffer_bp / 1e6
    print(f"{len(bounds)} windows: {near.sum()} within {buf_mb:g} Mb of a breakpoint, {(~near).sum()} farther")

    # ---------------- 1. long-range r^2 along the arm ----------------
    pos, samples, H = read_haploid_vcf(args.vcf)
    pop_of = dict(line.split()[:2] for line in open(args.popfile) if line.strip())
    pops = sorted(set(pop_of[s] for s in samples if s in pop_of))
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(len(pops), 1, figsize=(14, 3.4 * len(pops)), sharex=True)
    for ax, pop in zip(np.atleast_1d(axes), pops):
        cols = [i for i, s in enumerate(samples) if pop_of.get(s) == pop]
        h = H[:, cols]; n = len(cols)
        mac = np.minimum(h.sum(1), n - h.sum(1))
        keep = (h >= 0).all(1) & (mac >= 2)
        p_pop, h_pop = pos[keep], h[keep]
        idx = []
        for s, e in bounds:
            ii = np.where((p_pop >= s) & (p_pop <= e))[0]
            if len(ii) > args.snps_per_window:
                ii = np.sort(rng.choice(ii, args.snps_per_window, replace=False))
            idx.append(ii)
        lab = np.concatenate([np.full(len(ii), w) for w, ii in enumerate(idx)])
        Z = standardize(h_pop[np.concatenate(idx)])
        R2 = (Z @ Z.T / n) ** 2
        oh = np.zeros((len(lab), len(bounds)), np.float32); oh[np.arange(len(lab)), lab] = 1
        M = (oh.T @ R2 @ oh) / np.maximum(np.outer(oh.sum(0), oh.sum(0)), 1)
        far = np.abs(mids[:, None] - mids[None, :]) >= args.far_bp
        lr = np.array([M[i, far[i]].mean() for i in range(len(bounds))])
        chance = 1 / (n - 1)

        for s, e in zip(starts[near], ends[near]):
            ax.axvspan(s / 1e6, e / 1e6, color="tab:red", alpha=0.10, lw=0)
        ax.plot(mids / 1e6, lr, color="tab:blue", lw=1.4)
        ax.axhline(chance, color="grey", ls=":", lw=1.2)
        ax.text(mids[-1] / 1e6, chance, "  chance", va="center", fontsize=8, color="grey")
        for bp in bps:
            ax.axvline(bp / 1e6, color="black", ls="--", lw=1)
        ax.set_title(f"{pop} ({n} chromosomes)   —   average long-range r²:  "
                     f"within {buf_mb:g} Mb of a breakpoint = {lr[near].mean():.3f},   "
                     f"farther = {lr[~near].mean():.3f},   chance = {chance:.3f}", loc="left", fontsize=10)
        ax.set_ylabel("long-range r²")
    np.atleast_1d(axes)[-1].set_xlabel(f"{chrom} position (Mb)")
    fig.suptitle(f"How linked is each 100 kb window to windows ≥ {args.far_bp / 1e6:g} Mb away?  "
                 f"(red shading = within {buf_mb:g} Mb of an {args.inversion} breakpoint, dashed)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}_longrange.png", dpi=130); plt.close(fig)

    # ---------------- 2. MomentsLD decay curves, near vs far ----------------
    stats = {i: pickle.load(open(ld_dir / "LD_stats" / f"LD_stats_window_{i}.pkl", "rb")) for i in range(len(bounds))}
    groups = {f"within {buf_mb:g} Mb of breakpoint ({near.sum()} windows)": [i for i in range(len(bounds)) if near[i]],
              f"farther ({(~near).sum()} windows)": [i for i in range(len(bounds)) if not near[i]],
              f"all ({len(bounds)} windows)": list(range(len(bounds)))}
    mvs = {g: moments.LD.Parsing.bootstrap_data({j: stats[i] for j, i in enumerate(ii)}) for g, ii in groups.items()}
    names = stats[0]["stats"][0]
    rb = np.array(stats[0]["bins"])
    x = np.sqrt(np.maximum(rb[:, 0], rb[:, 1] / 10) * rb[:, 1])
    PANELS = [("DD_0_0", r"$\sigma_D^2$ CO"), ("DD_1_1", r"$\sigma_D^2$ FR"), ("DD_0_1", r"$D_{CO}D_{FR}$"),
              ("Dz_0_0_0", r"$Dz$ CO"), ("Dz_1_1_1", r"$Dz$ FR"), ("Dz_0_1_1", r"$Dz_{CO,FR,FR}$")]
    colors = ["tab:red", "tab:blue", "tab:gray"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, (st, lab) in zip(axes.flat, PANELS):
        k = names.index(st)
        for (g, mv), c, dx in zip(mvs.items(), colors, (0.96, 1.0, 1.04)):
            m = np.array([mv["means"][b][k] for b in range(len(rb))])
            se = np.array([np.sqrt(mv["varcovs"][b][k, k]) for b in range(len(rb))])
            ax.errorbar(x * dx, m, yerr=se, fmt="o-", ms=3, lw=1.2, capsize=2, color=c, label=g,
                        alpha=0.6 if c == "tab:gray" else 1)
        ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.5); ax.set_title(lab)
    for ax in axes[-1]:
        ax.set_xlabel("r (Comeron map)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{chrom}: the LD curves MomentsLD fits, split by distance from {args.inversion} breakpoints "
                 f"(mean ± SE, normalized by CO $\\pi_2$)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}_decay.png", dpi=120); plt.close(fig)
    print(f"-> {out}_longrange.png\n-> {out}_decay.png")


if __name__ == "__main__":
    main()

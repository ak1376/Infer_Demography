#!/usr/bin/env python3
"""
r^2 around an inversion's breakpoints, per population, from the haploid VCF.

Each haploid DPGP sample is one real chromosome, so r^2 is computed directly
from haplotypes (no pseudo-diploid pairing involved).

Two figures per run:
  1. <out>_windows.png : window x window mean r^2 across the arm, one panel per
     population, using the SAME 100 kb windows as the LD analysis (bounds read
     from --windows-dir/window_*.vcf.gz). An inversion segregating in a
     population shows up as a block of elevated r^2 linking distant windows
     inside the inverted region (strongest near breakpoint x breakpoint).
  2. <out>_zoom.png : SNP x SNP r^2 within +/- --zoom-bp of each breakpoint,
     per population, with window edges marked.
  3. <out>_lines.png : per window, along the arm, one panel per population:
       long-range r^2 = mean r^2 with every window >= --far-bp away
       local r^2      = mean r^2 of SNP pairs 10-100 kb apart within the window

Per population, SNPs are kept only if the minor allele is carried by >= 2
chromosomes in that population (singletons carry no LD information).
With ~10 chromosomes per population, unlinked SNPs still have r^2 ~ 1/(n-1)
by chance, so the background is not ~0; compare against it.

Usage:
  python godambe_correction_LRT/src/plot_r2_around_inversion.py \
      --vcf real_data_analysis/data/drosophila_trimmed/Chr3L/polarized.vcf.gz \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --windows-dir real_data_analysis/data/drosophila_trimmed/ld/Chr3L_100kb_genmap_phased/Chr3L/windows \
      --inversion "In(3L)P" \
      --out real_data_analysis/data/drosophila_trimmed/ld_viz/Chr3L_In3LP/r2
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inversion_breakpoints import INVERSIONS  # noqa: E402


def read_haploid_vcf(path):
    """-> positions (n_sites,), sample names, haplotypes (n_sites, n_samples) int8 (-1 = missing)."""
    pos, rows, samples = [], [], None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            fld = line.rstrip("\n").split("\t")
            gt_i = fld[8].split(":").index("GT")
            gts = [x.split(":")[gt_i] for x in fld[9:]]
            rows.append([-1 if g == "." else int(g) for g in gts])
            pos.append(int(fld[1]))
    return np.array(pos), samples, np.array(rows, dtype=np.int8)


def window_bounds(windows_dir):
    """(start, end) of every window_*.vcf.gz, by index, from first/last POS."""
    files = sorted(glob.glob(os.path.join(windows_dir, "window_*.vcf.gz")),
                   key=lambda p: int(re.search(r"window_(\d+)", p).group(1)))
    bounds = []
    for fp in files:
        first = last = None
        with gzip.open(fp, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = int(line.split("\t", 2)[1])
                first = p if first is None else first
                last = p
        bounds.append((first, last))
    return bounds


def standardize(h):
    """h: (n_snps, n_haps) 0/1 -> rows centered/scaled so r = Z @ Z.T / n_haps."""
    h = h.astype(np.float32)
    mu = h.mean(axis=1, keepdims=True)
    sd = h.std(axis=1, keepdims=True)
    return (h - mu) / sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--inversion", default="In(3L)P")
    ap.add_argument("--snps-per-window", type=int, default=100)
    ap.add_argument("--zoom-bp", type=int, default=150_000)
    ap.add_argument("--far-bp", type=int, default=1_000_000,
                    help="min distance between windows for the long-range line")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    chrom, bp1, bp2 = INVERSIONS[args.inversion]
    bps = sorted([bp1, bp2])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pos, samples, H = read_haploid_vcf(args.vcf)
    pop_of = dict(line.split()[:2] for line in open(args.popfile) if line.strip())
    pops = sorted(set(pop_of[s] for s in samples if s in pop_of))
    bounds = window_bounds(args.windows_dir)
    starts = np.array([b[0] for b in bounds]); ends = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(args.seed)
    print(f"{len(pos)} sites, {len(bounds)} windows, pops {pops}, {args.inversion} breakpoints {bps}")

    fig_w, axw = plt.subplots(1, len(pops), figsize=(7.5 * len(pops), 6.5))
    fig_z, axz = plt.subplots(len(pops), len(bps), figsize=(6.5 * len(bps), 6 * len(pops)))
    axz = np.atleast_2d(axz)
    fig_l, axl = plt.subplots(len(pops), 1, figsize=(14, 3.6 * len(pops)), sharex=True)
    axl = np.atleast_1d(axl)

    for pi, pop in enumerate(pops):
        cols = [i for i, s in enumerate(samples) if pop_of.get(s) == pop]
        h = H[:, cols]
        ok = (h >= 0).all(axis=1)
        mac = np.minimum(h.sum(axis=1), len(cols) - h.sum(axis=1))
        keep = ok & (mac >= 2)
        p_pop, h_pop = pos[keep], h[keep]
        n = len(cols)
        bg = 1.0 / (n - 1)

        # ---- 1. window x window mean r^2 ----
        idx_per_win = []
        for s, e in bounds:
            ii = np.where((p_pop >= s) & (p_pop <= e))[0]
            if len(ii) > args.snps_per_window:
                ii = np.sort(rng.choice(ii, args.snps_per_window, replace=False))
            idx_per_win.append(ii)
        allidx = np.concatenate(idx_per_win)
        lab = np.concatenate([np.full(len(ii), w) for w, ii in enumerate(idx_per_win)])
        Z = standardize(h_pop[allidx])
        R2 = (Z @ Z.T / n) ** 2
        nw = len(bounds)
        onehot = np.zeros((len(lab), nw), np.float32); onehot[np.arange(len(lab)), lab] = 1
        sums = onehot.T @ R2 @ onehot
        cnt = np.outer(onehot.sum(0), onehot.sum(0))
        M = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        np.fill_diagonal(M, np.nan)  # within-window mean is dominated by short-range LD
        ax = axw[pi] if len(pops) > 1 else axw
        mid = (starts + ends) / 2 / 1e6
        im = ax.imshow(M, origin="lower", cmap="magma", vmin=bg * 0.8, vmax=np.nanpercentile(M, 99.5),
                       extent=[mid[0], mid[-1], mid[0], mid[-1]], aspect="equal", interpolation="nearest")
        for b in bps:
            ax.axvline(b / 1e6, color="cyan", ls="--", lw=0.8); ax.axhline(b / 1e6, color="cyan", ls="--", lw=0.8)
        ax.set_title(f"{pop} (n={n} chromosomes): mean r² between 100 kb windows\n"
                     f"dashed = {args.inversion} breakpoints; chance level ≈ {bg:.2f}")
        ax.set_xlabel(f"{chrom} position (Mb)"); ax.set_ylabel(f"{chrom} position (Mb)")
        fig_w.colorbar(im, ax=ax, shrink=0.8, label="mean r²")

        # ---- 3. line plot along the arm ----
        mids_bp = (starts + ends) / 2
        far = np.abs(mids_bp[:, None] - mids_bp[None, :]) >= args.far_bp
        long_r2 = np.array([np.nanmean(M[i, far[i]]) for i in range(nw)])
        local_r2 = np.full(nw, np.nan)
        for w, (s, e) in enumerate(bounds):
            ii = np.where((p_pop >= s) & (p_pop <= e))[0]
            if len(ii) < 10:
                continue
            Zw = standardize(h_pop[ii])
            r2w = (Zw @ Zw.T / n) ** 2
            d = np.abs(p_pop[ii][:, None] - p_pop[ii][None, :])
            m = (d >= 10_000) & (d <= 100_000)
            local_r2[w] = r2w[m].mean() if m.any() else np.nan
        a = axl[pi]
        a.plot(mids_bp / 1e6, local_r2, color="tab:orange", lw=1.2, label="local: pairs 10–100 kb apart, within window")
        a.plot(mids_bp / 1e6, long_r2, color="tab:blue", lw=1.5, label=f"long-range: with windows ≥ {args.far_bp / 1e6:g} Mb away")
        a.axhline(bg, color="grey", ls=":", lw=1, label=f"chance level (1/(n−1) = {bg:.2f})")
        for b in bps:
            a.axvline(b / 1e6, color="black", ls="--", lw=0.9)
        a.axvspan(bps[0] / 1e6, bps[1] / 1e6, color="tab:green", alpha=0.07, label=f"{args.inversion} (inside breakpoints)")
        a.set_ylabel("mean r²")
        a.set_title(f"{pop} (n={n} chromosomes)", loc="left")
        a.legend(fontsize=8, loc="upper right", ncol=2)

        # ---- 2. SNP x SNP zoom around each breakpoint ----
        for bi, b in enumerate(bps):
            sel = np.where((p_pop >= b - args.zoom_bp) & (p_pop <= b + args.zoom_bp))[0]
            Zs = standardize(h_pop[sel])
            r2 = (Zs @ Zs.T / n) ** 2
            x = p_pop[sel] / 1e6
            a = axz[pi, bi]
            a.pcolormesh(x, x, r2, cmap="magma", vmin=0, vmax=1, shading="nearest", rasterized=True)
            a.axvline(b / 1e6, color="cyan", ls="--", lw=1); a.axhline(b / 1e6, color="cyan", ls="--", lw=1)
            for s in starts[(starts > b - args.zoom_bp) & (starts < b + args.zoom_bp)]:
                a.axvline(s / 1e6, color="white", lw=0.4, alpha=0.6); a.axhline(s / 1e6, color="white", lw=0.4, alpha=0.6)
            a.set_aspect("equal")
            a.set_title(f"{pop}: SNP×SNP r², ±{args.zoom_bp // 1000} kb around {b:,}\n"
                        f"({len(sel)} SNPs; white = 100 kb window edges)")
            a.set_xlabel("Mb"); a.set_ylabel("Mb")

    fig_w.suptitle(f"{chrom}: long-range LD and the {args.inversion} breakpoints (haploid data)")
    fig_w.tight_layout(); fig_w.savefig(f"{out}_windows.png", dpi=130)
    axl[-1].set_xlabel(f"{chrom} position (Mb), 100 kb windows")
    fig_l.suptitle(f"{chrom}: LD along the chromosome, per population (dashed = {args.inversion} breakpoints)")
    fig_l.tight_layout(); fig_l.savefig(f"{out}_lines.png", dpi=130)
    fig_z.suptitle(f"{chrom}: fine-scale r² at the {args.inversion} breakpoints")
    fig_z.tight_layout(); fig_z.savefig(f"{out}_zoom.png", dpi=110)
    print(f"-> {out}_windows.png\n-> {out}_zoom.png\n-> {out}_lines.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Mean within-window r^2 along a chromosome, per population, from the haploid VCF.

  1. Chunk the chromosome into non-overlapping windows: bounds read from
     --windows-dir/window_*.vcf.gz (the same windows as an LD run) if given,
     else tiled every --window-bp from the first SNP.
  2. Per population, keep SNPs whose minor allele is on >= 2 chromosomes.
  3. In each window, compute r^2 for every pair of kept SNPs.
  4. Average those r^2 -> one value per window.
  5. Plot against window position, with the breakpoints of every inversion
     on --chrom (from inversion_breakpoints.py) and the no-LD baseline
     (~1/(n-1) with n chromosomes).

Usage:
  python godambe_correction_LRT/src/plot_window_mean_r2.py \
      --vcf real_data_analysis/data/drosophila_trimmed/Chr3L/polarized.vcf.gz \
      --popfile real_data_analysis/data/drosophila/popfile.txt \
      --windows-dir real_data_analysis/data/drosophila_trimmed/ld/Chr3L_100kb_genmap_phased/Chr3L/windows \
      --chrom Chr3L --buffer-bp 3000000 \
      --out real_data_analysis/data/drosophila_trimmed/ld_viz/window_mean_r2/Chr3L
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inversion_breakpoints import INVERSIONS  # noqa: E402
from plot_r2_around_inversion import read_haploid_vcf, standardize, window_bounds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popfile", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--windows-dir", default=None, help="reuse an LD run's window bounds")
    ap.add_argument("--window-bp", type=int, default=100_000, help="tile size when --windows-dir is not given")
    ap.add_argument("--exclude", default="", help="comma-separated sample IDs to drop (e.g. inversion carriers)")
    ap.add_argument("--buffer-bp", type=int, default=3_000_000, help="shade windows this close to a breakpoint")
    ap.add_argument("--out", required=True, help="output prefix (.png and .tsv written)")
    args = ap.parse_args()

    chrom = args.chrom
    invs = {name: sorted([a, b]) for name, (c, a, b) in INVERSIONS.items() if c == chrom}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    pos, samples, H = read_haploid_vcf(args.vcf)
    drop = {x for x in args.exclude.split(",") if x}
    if drop:
        missing = drop - set(samples)
        if missing:
            raise SystemExit(f"--exclude samples not in VCF: {sorted(missing)}")
        kept = [i for i, s in enumerate(samples) if s not in drop]
        samples, H = [samples[i] for i in kept], H[:, kept]
        print(f"excluded {sorted(drop)}")
    lo, hi = int(pos.min()), int(pos.max())
    bps = sorted(bp for v in invs.values() for bp in v if lo <= bp <= hi)   # breakpoints inside the data
    pop_of = dict(line.split()[:2] for line in open(args.popfile) if line.strip())
    pops = sorted(set(pop_of[s] for s in samples if s in pop_of))
    if args.windows_dir:                                           # step 1
        bounds = window_bounds(args.windows_dir)
    else:
        bounds = [(s, min(s + args.window_bp - 1, hi)) for s in range(lo, hi + 1, args.window_bp)]
    mids = np.array([(s + e) / 2 for s, e in bounds])
    near = np.array([any(s < bp + args.buffer_bp and e > bp - args.buffer_bp for bp in bps) for s, e in bounds])

    rows = {}
    for pop in pops:
        cols = [i for i, s in enumerate(samples) if pop_of.get(s) == pop]
        h = H[:, cols]; n = len(cols)
        mac = np.minimum(h.sum(1), n - h.sum(1))
        keep = (h >= 0).all(1) & (mac >= 2)                        # step 2
        p_pop, h_pop = pos[keep], h[keep]
        mean_r2, nsnp = [], []
        for s, e in bounds:
            ii = np.where((p_pop >= s) & (p_pop <= e))[0]
            nsnp.append(len(ii))
            if len(ii) < 2:
                mean_r2.append(np.nan); continue
            Z = standardize(h_pop[ii])
            r2 = (Z @ Z.T / n) ** 2                                # step 3: all SNP pairs
            iu = np.triu_indices(len(ii), k=1)
            mean_r2.append(r2[iu].mean())                          # step 4
        rows[pop] = (np.array(mean_r2), np.array(nsnp), 1 / (n - 1), n)

    with open(f"{out}.tsv", "w") as f:
        f.write("window\tstart\tend\tnear_breakpoint\t" + "\t".join(f"{p}_mean_r2\t{p}_n_snps" for p in pops) + "\n")
        for w, (s, e) in enumerate(bounds):
            f.write(f"{w}\t{s}\t{e}\t{int(near[w])}\t" + "\t".join(f"{rows[p][0][w]:.5f}\t{rows[p][1][w]}" for p in pops) + "\n")

    fig, axes = plt.subplots(len(pops), 1, figsize=(14, 3.4 * len(pops)), sharex=True, sharey=True)   # step 5
    for ax, pop in zip(np.atleast_1d(axes), pops):
        r2, _, chance, n = rows[pop]
        for (s, e), nb in zip(bounds, near):
            if nb:
                ax.axvspan(s / 1e6, e / 1e6, color="tab:red", alpha=0.08, lw=0)
        ax.plot(mids / 1e6, r2, color="black", lw=1.2, marker="o", ms=2)
        ax.axhline(chance, color="grey", ls=":", lw=1.2)
        for (name, (a, b)), col in zip(invs.items(), ["tab:red", "tab:purple", "tab:green", "tab:orange"]):
            for bp in (a, b):
                if lo <= bp <= hi:
                    ax.axvline(bp / 1e6, color=col, ls="--", lw=1.1,
                               label=name if (ax is np.atleast_1d(axes)[0] and bp == min(x for x in (a, b) if lo <= x <= hi)) else None)
        ax.set_ylabel("mean r² within window")
        ax.set_title(f"{pop} ({n} chromosomes):  mean over windows = {np.nanmean(r2):.3f}   |   "
                     f"within {args.buffer_bp / 1e6:g} Mb of a breakpoint = {np.nanmean(r2[near]):.3f},  "
                     f"farther = {np.nanmean(r2[~near]):.3f}   |   no-LD baseline (dotted) = {chance:.3f}",
                     loc="left", fontsize=10)
    np.atleast_1d(axes)[-1].set_xlabel(f"{chrom} position (Mb), non-overlapping {args.window_bp // 1000 if not args.windows_dir else 100} kb windows")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8, title="inversion breakpoints (dashed)")
    outside = [f"{name} {bp:,}" for name, v in invs.items() for bp in v if not (lo <= bp <= hi)]
    note = f"; outside trimmed region, not shown: {', '.join(outside)}" if outside else ""
    if drop:
        note += f"; EXCLUDED: {', '.join(sorted(drop))}"
    fig.suptitle(f"{chrom}: average r² between all SNP pairs within each window  "
                 f"(shaded = within {args.buffer_bp / 1e6:g} Mb of a breakpoint{note})", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=130)
    print(f"-> {out}.png\n-> {out}.tsv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compute_unfolded_sfs.py

Compute a 2D unfolded SFS from a haploid-GT VCF that has an AA INFO field
(ancestral allele).  No haploid->diploid recoding needed.

Each sample contributes 1 chromosome (GT=0 -> ancestral, GT=1 -> alt).
Derived allele count per site is determined by the AA field:
  - AA == REF  ->  derived count = number of samples with GT=1
  - AA == ALT  ->  derived count = number of samples with GT=0  (flip)

Sites with any missing GT ('.') are skipped.
Sites without an AA field are skipped.

sequence_length (the L in theta = 4*mu*L*N_ANC) starts as the region length
from the VCF's ##contig header. With --unpolarized-vcf (the same region before
annotate_ancestral_allele dropped sites with no usable ancestral base), it is
scaled by (SNPs kept in the SFS / SNPs before polarization), on the assumption
that sequence is lost in the same proportion as SNPs.

With --output-png it also plots each population's own (marginal) spectrum
against the neutral constant-size expectation (proportional to 1/i), plus the
joint SFS as a log-scaled heatmap.

Usage:
  python compute_unfolded_sfs.py \
      --input-vcf  real_data_analysis/data/drosophila/Chr2L.polarized.vcf.gz \
      --popfile    real_data_analysis/data/drosophila/popfile.txt \
      --output-sfs real_data_analysis/data/drosophila/drosophila.unfolded.sfs.pkl \
      [--unpolarized-vcf <pre-polarization VCF>]   # optional: scale L by kept SNP fraction
      [--project-to N]   # optional: project each pop down to N haplotypes
      [--exclude FR217,FR361]   # optional: leave these samples out
      [--output-png unfolded.sfs.png]   # optional: marginal + joint SFS plot
"""

import argparse
import gzip
import json
import re
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import moments

_CONTIG_RE = re.compile(r"^##contig=<ID=([^,]+),length=(\d+)>")


def parse_contig_length(vcf_path: Path, opener) -> Tuple[str, int]:
    """Read the VCF header and return the (chrom, length) from its ##contig line.

    This is the actual physical sequence length the VCF's own header claims
    for this contig -- used downstream as the L in theta = 4*mu*L*N_ANC for
    real-data inference, so it stays in sync with whichever VCF actually
    produced the SFS instead of a hand-maintained config value. Per-chromosome
    VCFs in this pipeline carry exactly one ##contig line.
    """
    with opener(str(vcf_path), "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                break
            m = _CONTIG_RE.match(line)
            if m:
                return m.group(1), int(m.group(2))
    raise ValueError(f"No ##contig header found in {vcf_path}")


# Palette (dataviz reference instance, light mode)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POP_COLORS = ["#2a78d6", "#eb6834"]      # categorical slots 1-2, validated (CVD dE 24.7)
BLUES = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_marginal(ax, counts, pop, color, other):
    n = len(counts) - 1
    i = np.arange(1, n)
    seg = counts[1:n]
    prop = seg / seg.sum()
    neutral = (1 / i) / np.sum(1 / i)
    ax.bar(i, prop, width=0.72, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.plot(i, neutral, color=INK_2, linestyle="--", linewidth=1.5, marker="o",
            markersize=4, markerfacecolor=SURFACE, zorder=3)
    ax.annotate("neutral, constant size (∝ 1/i)", xy=(i[1], neutral[1]),
                xytext=(14, 10), textcoords="offset points", ha="left",
                fontsize=8.5, color=INK_2)
    style(ax)
    ax.set_xticks(i)
    ax.set_xlabel(f"derived allele count in {pop} (of {n})", color=INK_2, fontsize=10)
    ax.set_ylabel("share of segregating sites", color=INK_2, fontsize=10)
    ax.set_title(f"{pop}: {int(seg.sum()):,} segregating sites", loc="left",
                 color=INK, fontsize=11, fontweight="bold", pad=20)
    ax.text(0, 1.015, f"not shown: {int(counts[0]):,} absent, {int(counts[n]):,} fixed in {pop} "
                      f"(segregating only in {other})",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def plot_sfs(fs, chrom: str, L: float, region_length: float, out: Path):
    """Marginal spectra vs. neutral 1/i, plus the joint SFS heatmap."""
    pops = list(getattr(fs, "pop_ids", None) or ["pop0", "pop1"])
    data = np.asarray(fs.data, dtype=float)

    fig = plt.figure(figsize=(15, 4.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.32)
    for k in range(2):
        marginal = np.asarray(fs.marginalize([1 - k]).data, dtype=float)
        plot_marginal(fig.add_subplot(gs[0, k]), marginal, pops[k], POP_COLORS[k], pops[1 - k])

    # Joint SFS: rows = pop 0 derived count, columns = pop 1 derived count.
    ax = fig.add_subplot(gs[0, 2])
    joint = np.ma.masked_where(data <= 0, data)
    joint[0, 0] = np.ma.masked                       # monomorphic corners carry no information
    joint[-1, -1] = np.ma.masked
    cmap = BLUES.copy()
    cmap.set_bad(SURFACE)
    im = ax.imshow(joint, origin="lower", cmap=cmap, aspect="auto",
                   norm=LogNorm(vmin=max(1, joint.min()), vmax=joint.max()))
    ax.set_xlabel(f"derived allele count in {pops[1]}", color=INK_2, fontsize=10)
    ax.set_ylabel(f"derived allele count in {pops[0]}", color=INK_2, fontsize=10)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("joint SFS (sites per cell, log scale)", loc="left", color=INK,
                 fontsize=11, fontweight="bold", pad=20)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8.5)
    cb.outline.set_visible(False)

    fst = float(fs.Fst()) if hasattr(fs, "Fst") else float("nan")
    thetas = [float(fs.marginalize([1 - k]).Watterson_theta()) / L for k in range(2)]
    fig.suptitle(
        f"{chrom} unfolded SFS  ·  {int(fs.S()):,} segregating sites  ·  Fst = {fst:.3f}  ·  "
        f"Watterson θ/bp: {pops[0]} {thetas[0]:.2e}, {pops[1]} {thetas[1]:.2e}  "
        f"(L = {L / 1e6:.2f} Mb effective of {region_length / 1e6:.2f} Mb)",
        x=0.01, ha="left", y=1.02, color=INK, fontsize=11.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved -> {out}")


def parse_popfile(path: Path):
    """Returns (pop_names list, sample->pop dict)."""
    sample_to_pop = {}
    pop_order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            sample, pop = parts[0], parts[1]
            sample_to_pop[sample] = pop
            if pop not in pop_order:
                pop_order.append(pop)
    return pop_order, sample_to_pop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-vcf",   type=Path, required=True,
                   help="Polarized VCF with AA INFO field (haploid GTs).")
    p.add_argument("--popfile",     type=Path, required=True,
                   help="Two-column file: sample  population")
    p.add_argument("--output-sfs",  type=Path, required=True)
    p.add_argument("--output-meta", type=Path, default=None,
                   help="Optional: write {chrom, sequence_length, n_sites_*} JSON here, "
                        "sequence_length taken from the VCF's own ##contig header.")
    p.add_argument("--unpolarized-vcf", type=Path, default=None,
                   help="Optional: the same region before polarization. sequence_length is "
                        "scaled by (SNPs kept / SNPs in this VCF).")
    p.add_argument("--project-to",  type=int, default=None,
                   help="Project each population down to this many haplotypes.")
    p.add_argument("--exclude",     default="",
                   help="Comma-separated sample IDs to leave out of the SFS.")
    p.add_argument("--output-png",  type=Path, default=None,
                   help="Optional: write the marginal + joint SFS plot here.")
    args = p.parse_args()

    args.output_sfs.parent.mkdir(parents=True, exist_ok=True)

    pop_names, sample_to_pop = parse_popfile(args.popfile)
    drop = {s for s in args.exclude.split(",") if s}
    if drop - set(sample_to_pop):
        raise SystemExit(f"--exclude samples not in popfile: {sorted(drop - set(sample_to_pop))}")
    for s in sorted(drop):
        print(f"[{sample_to_pop.pop(s)}] excluding {s}")
    print(f"Populations: {pop_names}")

    opener = gzip.open if str(args.input_vcf).endswith(".gz") else open

    chrom, sequence_length = parse_contig_length(args.input_vcf, opener)
    print(f"Contig from VCF header: {chrom}  length={sequence_length:,}")

    # We'll accumulate a raw count array of shape (n_pop0+1, n_pop1+1, ...)
    # First pass: read header to get sample indices per pop
    sample_indices = defaultdict(list)  # pop -> [col indices in VCF]
    header_done = False
    vcf_samples = []

    with opener(str(args.input_vcf), "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                vcf_samples = line.rstrip("\n").split("\t")[9:]
                for i, s in enumerate(vcf_samples):
                    pop = sample_to_pop.get(s)
                    if pop is not None:
                        sample_indices[pop].append(i)
                header_done = True
                break

    n_per_pop = {pop: len(sample_indices[pop]) for pop in pop_names}
    print(f"Samples per pop: { {p: n_per_pop[p] for p in pop_names} }")

    # SFS array shape: (n_pop0+1) x (n_pop1+1)
    shape = tuple(n_per_pop[pop] + 1 for pop in pop_names)
    sfs_arr = np.zeros(shape, dtype=np.float64)

    total = skipped_missing = skipped_no_aa = skipped_no_match = kept = 0

    with opener(str(args.input_vcf), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            ref = fields[3]
            alt = fields[4]
            info = fields[7]
            gts_all = fields[9:]  # haploid: "0", "1", or "."

            total += 1

            # Parse AA from INFO
            aa = None
            for token in info.split(";"):
                if token.startswith("AA="):
                    aa = token[3:]
                    break
            if aa is None:
                skipped_no_aa += 1
                continue

            if aa == ref:
                flip = False
            elif aa == alt:
                flip = True
            else:
                skipped_no_match += 1
                continue

            # Count derived alleles per population
            derived_counts = []
            skip_site = False
            for pop in pop_names:
                idx_list = sample_indices[pop]
                n = len(idx_list)
                alt_count = 0
                for i in idx_list:
                    gt = gts_all[i]
                    if gt == ".":
                        skip_site = True
                        break
                    alt_count += int(gt)
                if skip_site:
                    break
                derived = (n - alt_count) if flip else alt_count
                derived_counts.append(derived)

            if skip_site:
                skipped_missing += 1
                continue

            kept += 1
            sfs_arr[tuple(derived_counts)] += 1

    print(f"\nSNP summary:")
    print(f"  Total sites          : {total:>10,}")
    print(f"  Kept                 : {kept:>10,}  ({100*kept/total:.1f}%)")
    print(f"  Skipped (missing GT) : {skipped_missing:>10,}")
    print(f"  Skipped (no AA)      : {skipped_no_aa:>10,}")
    print(f"  Skipped (AA mismatch): {skipped_no_match:>10,}")

    # Build moments Spectrum (unfolded)
    sfs = moments.Spectrum(sfs_arr, pop_ids=pop_names)

    # Zero out the corners (fixed sites)
    sfs[0, 0] = 0.0
    sfs[-1, -1] = 0.0

    if args.project_to is not None:
        print(f"\nProjecting each population to {args.project_to} haplotypes ...")
        sfs = sfs.project([args.project_to] * len(pop_names))

    print(f"\nSFS shape : {sfs.shape}")
    print(f"Folded    : {sfs.folded}")
    print(f"Pop IDs   : {sfs.pop_ids}")
    print(f"Total SNPs in SFS: {sfs.S():.0f}")

    with open(args.output_sfs, "wb") as fh:
        pickle.dump(sfs, fh)

    print(f"\nSaved -> {args.output_sfs}")

    region_length = sequence_length
    if args.output_meta is not None:
        args.output_meta.parent.mkdir(parents=True, exist_ok=True)
        n_before = None
        if args.unpolarized_vcf is not None:
            with gzip.open(args.unpolarized_vcf, "rt") as fh:
                n_before = sum(1 for line in fh if line[0] != "#")
            sequence_length = int(round(region_length * kept / n_before))
            print(f"Effective L = {region_length:,} x {kept:,}/{n_before:,} = {sequence_length:,}")
        meta = {
            "chrom": chrom,
            "sequence_length": sequence_length,
            "region_length": region_length,
            "n_sites_before_polarization": n_before,
            "kept_fraction": None if n_before is None else kept / n_before,
            "source_vcf": str(args.input_vcf),
            "n_sites_total": total,
            "n_sites_kept": kept,
            "n_sites_skipped_missing_gt": skipped_missing,
            "n_sites_skipped_no_aa": skipped_no_aa,
            "n_sites_skipped_aa_mismatch": skipped_no_match,
        }
        with open(args.output_meta, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"Saved -> {args.output_meta}")

    if args.output_png is not None:
        plot_sfs(sfs, chrom, sequence_length, region_length, args.output_png)


if __name__ == "__main__":
    main()

# windowed_summary_statistics.py

"""
Windowed summary statistics + between-block autocorrelation for the real
Drosophila data.

Goal: check whether non-overlapping genomic windows ("blocks") behave as
approximately independent replicates. If they do, the per-window summary
statistics should show little to no autocorrelation across lags. Strong
autocorrelation means neighbouring windows still share linkage information
and the blocks are too small to be treated as independent bootstrap units.

Pipeline
--------
1. Load the real Drosophila polarized VCF -> SNP positions + per-population
   allele counts.
2. Partition the chromosome into `--n-windows` equal-width, non-overlapping
   windows.
3. For each window and each population compute:
       - pi          (nucleotide diversity, per site)
       - Tajima's D
   and for each window compute the between-population:
       - Fst         (Hudson)
4. Plot each statistic along the chromosome (one panel per statistic).
5. Compute the autocorrelation at lags 1..K for each statistic and each
   population, and plot it with a white-noise significance band. Little to
   no autocorrelation => windows look independent.

Usage
-----
    python windowed_summary_statistics.py --n-windows 100
    python windowed_summary_statistics.py --n-windows 500 --max-lag 30
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import allel

ROOT = Path("/sietch_colab/akapoor/Infer_Demography")
REAL_VCF = str(ROOT / "real_data_analysis/data/drosophila/Chr2L/polarized.vcf.gz")
POPFILE = str(ROOT / "real_data_analysis/data/drosophila/popfile.txt")
OUT_DIR = ROOT / "figures"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_popfile(popfile_path):
    """Returns (pop_order, sample_to_pop) from a whitespace-separated
    <sample> <population> file."""
    sample_to_pop = {}
    pop_order = []
    with open(popfile_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sample, pop = parts[0], parts[1]
            sample_to_pop[sample] = pop
            if pop not in pop_order:
                pop_order.append(pop)
    return pop_order, sample_to_pop


def load_allele_counts(vcf_path, popfile_path):
    """Read the haploid VCF and return (pos, ac, pop_names, n_samples).

    pos         : (n_sites,) sorted int64 SNP positions.
    ac[pop]     : (n_sites, 2) int32 allele-count array [n_ref, n_alt] for
                  the population's samples -- the input format scikit-allel's
                  windowed_* functions expect.
    pop_names   : population labels in popfile order.
    n_samples   : {pop: n_haploid_samples}.

    pi, Tajima's D and Hudson Fst are all invariant to ancestral/derived
    polarization (they depend only on allele frequencies), so we count the
    ALT allele directly rather than re-polarizing with the AA field.
    """
    pop_names, sample_to_pop = _parse_popfile(popfile_path)

    opener = gzip.open if vcf_path.endswith(".gz") else open

    sample_indices = {pop: [] for pop in pop_names}
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                vcf_samples = line.rstrip("\n").split("\t")[9:]
                for i, s in enumerate(vcf_samples):
                    pop = sample_to_pop.get(s)
                    if pop is not None:
                        sample_indices[pop].append(i)
                break

    n_samples = {pop: len(idx) for pop, idx in sample_indices.items()}

    pos_list = []
    alt_counts = {pop: [] for pop in pop_names}
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            gts = fields[9:]
            pos_list.append(int(fields[1]))
            for pop in pop_names:
                alt_counts[pop].append(sum(int(gts[i]) for i in sample_indices[pop]))

    pos = np.array(pos_list, dtype=np.int64)
    ac = {}
    for pop in pop_names:
        alt = np.array(alt_counts[pop], dtype=np.int32)
        n = n_samples[pop]
        ac[pop] = np.stack([n - alt, alt], axis=1).astype(np.int32)

    return pos, ac, pop_names, n_samples


# ---------------------------------------------------------------------------
# Windowing + statistics
# ---------------------------------------------------------------------------
def make_windows(pos, n_windows):
    """Return (windows, centers) for `n_windows` equal-width, non-overlapping
    windows spanning [pos.min(), pos.max()].

    windows : (n_windows, 2) int array of inclusive [start, stop] bounds, the
              format scikit-allel's windowed_* functions accept.
    centers : (n_windows,) window midpoints (bp).
    """
    start, stop = int(pos.min()), int(pos.max())
    edges = np.linspace(start, stop, n_windows + 1)
    starts = np.floor(edges[:-1]).astype(np.int64)
    stops = np.floor(edges[1:]).astype(np.int64)
    # Make windows non-overlapping: each window is [start, stop - 1], last one
    # keeps the final position.
    stops[:-1] -= 1
    windows = np.stack([starts, stops], axis=1)
    centers = 0.5 * (starts + stops)
    return windows, centers


def windowed_stats(pos, ac, pop_names, windows):
    """Compute per-window pi and Tajima's D for each population and Hudson Fst
    between the first two populations.

    Returns a dict:
        stats['pi'][pop]       -> (n_windows,) per-site pi
        stats['tajima_d'][pop] -> (n_windows,) Tajima's D
        stats['fst']           -> (n_windows,) Hudson Fst (pop0 vs pop1)
        stats['n_sites'][pop]  -> (n_windows,) segregating-site count
    """
    stats = {"pi": {}, "tajima_d": {}, "n_sites": {}}

    for pop in pop_names:
        pi, _, n_bases, counts = allel.windowed_diversity(pos, ac[pop], windows=windows)
        D, _, _ = allel.windowed_tajima_d(pos, ac[pop], windows=windows)
        stats["pi"][pop] = np.asarray(pi, dtype=float)
        stats["tajima_d"][pop] = np.asarray(D, dtype=float)
        stats["n_sites"][pop] = np.asarray(counts)

    if len(pop_names) >= 2:
        p0, p1 = pop_names[0], pop_names[1]
        fst, _, _ = allel.windowed_hudson_fst(pos, ac[p0], ac[p1], windows=windows)
        stats["fst"] = np.asarray(fst, dtype=float)

    return stats


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------
def autocorrelation(series, max_lag):
    """NaN-robust autocorrelation of an ordered 1D series at lags 1..max_lag.

    For each lag k we correlate the series with a copy shifted by k windows,
    using only positions where both entries are finite. The result is
    normalized so lag 0 == 1. Windows with NaN (too few sites to define the
    statistic) are simply skipped for the pairs they belong to.

    Returns (lags, acf) with lags = [1..max_lag].
    """
    x = np.asarray(series, dtype=float)
    finite = np.isfinite(x)
    # Mean/variance from all finite values, used as the lag-0 normalizer.
    mu = np.nanmean(x)
    var = np.nanmean((x[finite] - mu) ** 2)

    lags = np.arange(1, max_lag + 1)
    acf = np.full(len(lags), np.nan)
    if var <= 0 or finite.sum() < 3:
        return lags, acf

    for i, k in enumerate(lags):
        a = x[:-k]
        b = x[k:]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() < 3:
            continue
        acf[i] = np.mean((a[both] - mu) * (b[both] - mu)) / var

    return lags, acf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_stats_along_genome(centers, stats, pop_names, n_windows, out_path):
    """One panel per statistic: pi, Tajima's D, Fst along the chromosome."""
    have_fst = "fst" in stats
    n_panels = 3 if have_fst else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True)

    x = centers / 1e6  # Mb
    colors = {"CO": "tab:blue", "FR": "tab:orange"}

    ax = axes[0]
    for pop in pop_names:
        ax.plot(x, stats["pi"][pop], marker="o", ms=2, lw=1,
                color=colors.get(pop), label=pop)
    ax.set_ylabel(r"$\pi$ (per site)")
    ax.set_title(f"Per-window nucleotide diversity  ({n_windows} windows)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for pop in pop_names:
        ax.plot(x, stats["tajima_d"][pop], marker="o", ms=2, lw=1,
                color=colors.get(pop), label=pop)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax.set_ylabel("Tajima's D")
    ax.set_title("Per-window Tajima's D")
    ax.legend(fontsize=8)

    if have_fst:
        ax = axes[2]
        ax.plot(x, stats["fst"], marker="o", ms=2, lw=1, color="tab:green",
                label=f"{pop_names[0]} vs {pop_names[1]}")
        ax.set_ylabel(r"$F_{ST}$ (Hudson)")
        ax.set_title("Per-window Fst")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Chromosome position (Mb)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_autocorrelation(stats, pop_names, n_windows, max_lag, out_path):
    """One panel per statistic; each panel overlays the ACF for each
    population (for Fst there is a single between-population series). A shaded
    band marks the +/-1.96/sqrt(N) white-noise (no-autocorrelation) envelope."""
    have_fst = "fst" in stats
    n_panels = 3 if have_fst else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.0 * n_panels), sharex=True)

    colors = {"CO": "tab:blue", "FR": "tab:orange"}

    def _band(ax, n_eff):
        if n_eff and n_eff > 0:
            b = 1.96 / np.sqrt(n_eff)
            ax.axhspan(-b, b, color="gray", alpha=0.15,
                       label="95% white-noise band")

    # pi
    ax = axes[0]
    for pop in pop_names:
        lags, acf = autocorrelation(stats["pi"][pop], max_lag)
        ax.plot(lags, acf, marker="o", ms=3, color=colors.get(pop), label=pop)
    _band(ax, np.isfinite(stats["pi"][pop_names[0]]).sum())
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ACF")
    ax.set_title(r"Autocorrelation of per-window $\pi$")
    ax.legend(fontsize=8)

    # Tajima's D
    ax = axes[1]
    for pop in pop_names:
        lags, acf = autocorrelation(stats["tajima_d"][pop], max_lag)
        ax.plot(lags, acf, marker="o", ms=3, color=colors.get(pop), label=pop)
    _band(ax, np.isfinite(stats["tajima_d"][pop_names[0]]).sum())
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation of per-window Tajima's D")
    ax.legend(fontsize=8)

    # Fst
    if have_fst:
        ax = axes[2]
        lags, acf = autocorrelation(stats["fst"], max_lag)
        ax.plot(lags, acf, marker="o", ms=3, color="tab:green",
                label=f"{pop_names[0]} vs {pop_names[1]}")
        _band(ax, np.isfinite(stats["fst"]).sum())
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("ACF")
        ax.set_title("Autocorrelation of per-window Fst")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Lag (windows)")
    fig.suptitle(f"Between-block autocorrelation  ({n_windows} windows)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Window-count sweep
# ---------------------------------------------------------------------------
def sweep_window_counts(pos, ac, pop_names, window_counts, out_path):
    """Pick a pragmatic bootstrap block size from a sweep of window counts.

    Decision rule (kept deliberately simple):
      * HARD-PASS statistics -- Tajima's D in each population and Fst -- must
        have |lag-1 ACF| < band, where band = 1.96 / sqrt(n_windows).
      * WARNING statistics -- pi in each population -- are reported but never
        block a candidate. Raw pi carries chromosome-scale diversity structure
        that keeps its ACF high at every block size, so requiring it to pass
        would reject everything.
      * A candidate PASSES if all hard-pass statistics are inside the band.
        Among passing candidates, choose the coarsest partition (largest
        blocks = smallest n_windows).
      * If none pass, fall back to the candidate that minimizes the maximum
        standardized excess over the hard-pass stats,
            score = max(|ACF_stat| / band),
        and report that no candidate fully passed.

    Returns (chosen_n_windows, fully_passed: bool).
    """
    have_fst = len(pop_names) >= 2

    hard_labels = [f"D_{pop}" for pop in pop_names] + (["Fst"] if have_fst else [])
    warn_labels = [f"pi_{pop}" for pop in pop_names]
    all_labels = warn_labels + hard_labels  # display order: pi first, then hard

    span = int(pos.max() - pos.min())

    # rows[n] -> {label: lag1_acf}
    rows = {}
    for n in window_counts:
        windows, _ = make_windows(pos, n)
        stats = windowed_stats(pos, ac, pop_names, windows)
        vals = {}
        for pop in pop_names:
            _, acf_pi = autocorrelation(stats["pi"][pop], 1)
            _, acf_d = autocorrelation(stats["tajima_d"][pop], 1)
            vals[f"pi_{pop}"] = acf_pi[0]
            vals[f"D_{pop}"] = acf_d[0]
        if have_fst:
            _, acf_f = autocorrelation(stats["fst"], 1)
            vals["Fst"] = acf_f[0]
        rows[n] = vals

    def hard_pass(n):
        band = 1.96 / np.sqrt(n)
        return all(np.isfinite(rows[n][lbl]) and abs(rows[n][lbl]) < band
                   for lbl in hard_labels)

    def hard_score(n):
        """Max standardized excess over hard-pass stats (lower is better)."""
        band = 1.96 / np.sqrt(n)
        vals = [abs(rows[n][lbl]) / band for lbl in hard_labels
                if np.isfinite(rows[n][lbl])]
        return max(vals) if vals else np.inf

    def pi_fail(n):
        band = 1.96 / np.sqrt(n)
        return any(np.isfinite(rows[n][lbl]) and abs(rows[n][lbl]) >= band
                   for lbl in warn_labels)

    # Print table
    header = ("  n_windows      bp/win   " +
              "  ".join(f"{lbl:>8s}" for lbl in all_labels) +
              "     band   hard_pass  pi_warn")
    print("\nLag-1 autocorrelation sweep")
    print("  hard-pass stats: " + ", ".join(hard_labels))
    print("  warning stats:   " + ", ".join(warn_labels) + " (reported only)")
    print(header)
    for n in window_counts:
        band = 1.96 / np.sqrt(n)
        cells = "  ".join(f"{rows[n][lbl]:>+8.3f}" for lbl in all_labels)
        hp = "yes" if hard_pass(n) else "no"
        pw = "WARN" if pi_fail(n) else "ok"
        print(f"  {n:>9d}  {span / n:>10,.0f}   {cells}   {band:>+6.3f}"
              f"     {hp:>4s}     {pw}")

    # Choose
    passing = [n for n in window_counts if hard_pass(n)]
    if passing:
        chosen = min(passing)  # smallest n_windows -> largest blocks
        fully_passed = True
    else:
        chosen = min(window_counts, key=hard_score)  # best-effort fallback
        fully_passed = False

    # Plot lag-1 |ACF| vs window count, hard-pass solid, warning dashed.
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = np.array(window_counts, dtype=float)
    for lbl in hard_labels:
        y = np.array([abs(rows[n][lbl]) for n in window_counts])
        ax.plot(ns, y, marker="o", ms=4, lw=1.6, label=f"{lbl} (hard)")
    for lbl in warn_labels:
        y = np.array([abs(rows[n][lbl]) for n in window_counts])
        ax.plot(ns, y, marker="s", ms=4, lw=1.2, ls="--", alpha=0.7,
                label=f"{lbl} (warn)")
    ax.plot(ns, 1.96 / np.sqrt(ns), color="black", ls=":", lw=1.5,
            label="95% white-noise band")
    ax.axvline(chosen, color="red", ls="-", lw=1.5, alpha=0.7,
               label=f"chosen = {chosen}" + ("" if fully_passed else " (best effort)"))
    ax.set_xscale("log")
    ax.set_xlabel("num_windows")
    ax.set_ylabel("|lag-1 autocorrelation|")
    ax.set_title("Between-block dependence vs. window count\n"
                 "(hard-pass stats must dip below the band; pi is warning-only)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # ---- Decision + interpretation --------------------------------------
    win_bp = span / chosen
    print("\n" + "=" * 68)
    if fully_passed:
        print(f"CHOSEN bootstrap block size (hard-pass: {', '.join(hard_labels)}):")
    else:
        print("No candidate passed all hard-pass statistics.")
        print(f"Best-effort choice (minimizes max |ACF|/band over hard stats, "
              f"score={hard_score(chosen):.2f}):")
    print(f"    n_windows   = {chosen}")
    print(f"    block size  = {win_bp:,.0f} bp  ({win_bp / 1e6:.3f} Mb)")
    print("    --> pass this window size to your LRT_adjust script.")

    if pi_fail(chosen):
        band = 1.96 / np.sqrt(chosen)
        pi_str = ", ".join(f"{lbl}={rows[chosen][lbl]:+.3f}" for lbl in warn_labels)
        print(f"\n  pi WARNING at the chosen block size ({pi_str}; band=+/-{band:.3f}):")
        print("    pi lag-1 autocorrelation remains outside the white-noise band.")
    print("\n  Interpretation (cautious): this is a pragmatic first-pass")
    print("  bootstrap block size. Persistent pi autocorrelation reflects")
    print("  broad-scale diversity heterogeneity along the chromosome, so the")
    print("  Godambe LRT results should be interpreted with that caveat.")
    print("=" * 68)

    return chosen, fully_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-windows", type=int, default=100,
                   help="Number of equal-width non-overlapping windows.")
    p.add_argument("--max-lag", type=int, default=20,
                   help="Maximum lag (in windows) for the autocorrelation.")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep over several window counts and report the "
                        "smallest count (largest blocks) whose lag-1 "
                        "autocorrelation is inside the white-noise band for "
                        "every statistic.")
    p.add_argument("--sweep-counts", type=int, nargs="+",
                   default=[10, 20, 30, 50, 75, 100, 150, 200, 300, 500],
                   help="Candidate window counts for --sweep (ascending).")
    p.add_argument("--vcf", default=REAL_VCF)
    p.add_argument("--popfile", default=POPFILE)
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Step 1: load
    print("Loading allele counts from VCF ...")
    pos, ac, pop_names, n_samples = load_allele_counts(args.vcf, args.popfile)
    print(f"  {len(pos):,} sites, populations: {pop_names}")
    for pop in pop_names:
        print(f"    {pop}: {n_samples[pop]} haploid samples")

    if args.sweep:
        counts = sorted(set(args.sweep_counts))
        sweep_window_counts(
            pos, ac, pop_names, counts,
            OUT_DIR / "windowed_stats_acf_sweep.png",
        )
        return

    # Step 2: windows
    max_lag = min(args.max_lag, args.n_windows - 1)
    windows, centers = make_windows(pos, args.n_windows)
    span = int(pos.max() - pos.min())
    print(f"  {args.n_windows} windows over {span:,} bp "
          f"(~{span / args.n_windows:,.0f} bp/window)")

    # Step 3: per-window stats
    print("Computing windowed statistics ...")
    stats = windowed_stats(pos, ac, pop_names, windows)

    # Report how many windows are usable (finite) per statistic
    for pop in pop_names:
        n_pi = np.isfinite(stats["pi"][pop]).sum()
        n_d = np.isfinite(stats["tajima_d"][pop]).sum()
        print(f"    {pop}: pi finite in {n_pi}/{args.n_windows}, "
              f"Tajima's D finite in {n_d}/{args.n_windows} windows")
    if "fst" in stats:
        n_f = np.isfinite(stats["fst"]).sum()
        print(f"    Fst finite in {n_f}/{args.n_windows} windows")

    # Step 4: plot stats along the genome
    plot_stats_along_genome(
        centers, stats, pop_names, args.n_windows,
        OUT_DIR / f"windowed_stats_n{args.n_windows}.png",
    )

    # Step 5: autocorrelation
    plot_autocorrelation(
        stats, pop_names, args.n_windows, max_lag,
        OUT_DIR / f"windowed_stats_autocorr_n{args.n_windows}.png",
    )

    # Print lag-1 autocorrelation as a quick numeric summary
    print("\nLag-1 autocorrelation (want ~0 for independent blocks):")
    for pop in pop_names:
        _, acf_pi = autocorrelation(stats["pi"][pop], max_lag)
        _, acf_d = autocorrelation(stats["tajima_d"][pop], max_lag)
        print(f"    {pop}: pi={acf_pi[0]:+.3f}   Tajima's D={acf_d[0]:+.3f}")
    if "fst" in stats:
        _, acf_f = autocorrelation(stats["fst"], max_lag)
        print(f"    Fst: {acf_f[0]:+.3f}")
    b = 1.96 / np.sqrt(args.n_windows)
    print(f"    (white-noise 95% band at this window count: +/-{b:.3f})")


if __name__ == "__main__":
    main()

# find_ld_blocks_Chr3L.py
"""
Find LD block boundaries on Chr3L via a bridging-score scan, to pick a
bootstrap block size for bootstrapping_for_LRT_Chr3L.py.

Chr3L has ~497k SNPs -- an exact SNP x SNP r^2 matrix only fits in memory
for a modest number of SNPs at a time, not the whole chromosome at once. So
this processes Chr3L in --chunk-size chunks (default 300kb, sized to stay
well within GPU memory even in the densest SNP regions), and within each
chunk uses pg_gpu's own bridging-score method (see pg_gpu's `ld_blocks`
tutorial) to find where LD breaks down: scan along the chromosome, and at
each point compare the SNPs just to its left against the SNPs just to its
right. A dip means LD doesn't span that point -- a real block boundary.
The gaps between consecutive boundaries are the empirically-found LD
blocks; their size distribution is the answer to "how big should my
bootstrap blocks be."

Runs on GPU (pg_gpu's pairwise_r2() forces GPU internally). This machine's
GPUs are often shared with other users' large jobs -- check
`nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv` and pass
--gpu-device accordingly.

Two modes, so you don't have to re-run the expensive GPU pass just to
change a plot:

  1. Compute + plot (default): runs the full per-chunk GPU scan, saves
     results to <out-dir>/results.pkl, then makes all plots.
       python find_ld_blocks_Chr3L.py --out-dir figures/ld_blocks_Chr3L

  2. Plot only: skip the GPU pass entirely and re-plot from a saved
     results.pkl (e.g. after tweaking MAX_BRIDGE_SCORE's downstream
     analysis, or just re-styling a figure).
       python find_ld_blocks_Chr3L.py --out-dir figures/ld_blocks_Chr3L \\
           --from-pickle figures/ld_blocks_Chr3L/results.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DROSO_DIR = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis/data/drosophila"
DEFAULT_VCF = f"{DROSO_DIR}/Chr3L/polarized.diploidGT.vcf.gz"  # needs diploid GT (0/0, 1/1, ...)
DEFAULT_POPFILE = f"{DROSO_DIR}/popfile.txt"

# Fixed categorical color per window size (never re-cycled -- window=75 is
# always blue, 150 always orange, 300 always aqua, across every figure).
# All 8 validated palette slots, so a broader sweep still gets a distinct
# color per window size.
WINDOW_COLOR_RAMP = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRID_COLOR = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"


# --------------------------------------------------------------------------
# Bridging-score algorithm (adapted from pg_gpu's own ld_blocks tutorial)
# --------------------------------------------------------------------------

def bridging_score(r2, window):
    """score[k] = mean r^2 between SNPs [k-window, k) and [k, k+window).

    High score = still linked across k; a dip means LD breaks down there.
    Edges within `window` of either end of the chunk get NaN (no full
    window on one side).
    """
    n = r2.shape[0]
    score = np.full(n, np.nan)
    for k in range(window, n - window):
        score[k] = r2[k - window:k, k:k + window].mean()
    return score


def find_block_boundaries(score, max_score, min_separation):
    """SNP indices of local minima in `score` at or below max_score."""
    finite = np.isfinite(score)
    inverted = -score.copy()
    inverted[~finite] = -np.inf
    peaks, _ = find_peaks(inverted, height=-max_score, distance=min_separation)
    return peaks


def outlier_stats(sizes):
    """Tukey's rule (matches whis=1.5 in the boxplots below): count + % of
    values beyond 1.5xIQR past Q1/Q3, and the upper fence itself."""
    q1, q3 = np.percentile(sizes, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = (sizes < lower) | (sizes > upper)
    n_outliers = int(is_outlier.sum())
    return n_outliers, 100 * n_outliers / len(sizes), upper


# --------------------------------------------------------------------------
# GPU compute pass
# --------------------------------------------------------------------------

def run_gpu_scan(vcf, popfile, window_sizes, chunk_size, max_bridge_score,
                  min_separation, test_length, gpu_device):
    import cupy as cp
    from pg_gpu import HaplotypeMatrix

    cp.cuda.Device(gpu_device).use()
    mempool = cp.get_default_memory_pool()

    full_hm = HaplotypeMatrix.from_vcf(vcf)
    all_positions = np.asarray(full_hm.positions)
    chrom_start = int(all_positions.min())
    chrom_end = int(all_positions.max())
    del full_hm

    loop_end = chrom_end if test_length is None else min(chrom_end, chrom_start + test_length)
    print(f"Chr3L spans {chrom_start:,}-{chrom_end:,} ({(chrom_end - chrom_start) / 1e6:.1f} Mb)")
    print(f"processing {chrom_start:,}-{loop_end:,} ({(loop_end - chrom_start) / 1e6:.1f} Mb)")

    all_boundaries_bp = {w: [] for w in window_sizes}
    chunk_log = []

    for chunk_start in range(chrom_start, loop_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, loop_end)
        region = f"Chr3L:{chunk_start}-{chunk_end}"

        try:
            region_hm = HaplotypeMatrix.from_vcf(vcf, region=region)
        except ValueError:
            chunk_log.append((chunk_start, chunk_end, 0))
            continue

        region_hm.load_pop_file(popfile)
        region_hm.transfer_to_gpu()

        r2 = region_hm.pairwise_r2().get()  # real, exact SNP x SNP matrix -- computed once, reused below
        positions = region_hm.positions
        if isinstance(positions, cp.ndarray):
            positions = positions.get()
        n_snps = region_hm.num_variants

        del region_hm
        mempool.free_all_blocks()  # release this chunk's GPU memory before the next one loads

        counts = {}
        for w in window_sizes:
            score = bridging_score(r2, w)
            boundary_idx = find_block_boundaries(score, max_bridge_score, min_separation)
            all_boundaries_bp[w].extend(positions[boundary_idx].tolist())
            counts[w] = len(boundary_idx)

        chunk_log.append((chunk_start, chunk_end, n_snps))
        print(f"{region}: {n_snps:,} SNPs -> boundaries "
              + ", ".join(f"w={w}:{counts[w]}" for w in window_sizes))

    print(f"\ntotal boundaries across {len(chunk_log)} chunks: "
          + ", ".join(f"w={w}: {len(all_boundaries_bp[w])}" for w in window_sizes))

    return {
        "chrom_start": chrom_start,
        "chrom_end": chrom_end,
        "loop_end": loop_end,
        "all_positions": all_positions,
        "all_boundaries_bp": all_boundaries_bp,
        "chunk_log": chunk_log,
        "window_sizes": window_sizes,
        "max_bridge_score": max_bridge_score,
        "min_separation": min_separation,
    }


def run_region_scan(vcf, popfile, region, window_sizes, gpu_device):
    """Real bridging-score curves (not just boundaries) for one region, for plotting."""
    import cupy as cp
    from pg_gpu import HaplotypeMatrix

    cp.cuda.Device(gpu_device).use()
    mempool = cp.get_default_memory_pool()

    region_hm = HaplotypeMatrix.from_vcf(vcf, region=region)
    region_hm.load_pop_file(popfile)
    region_hm.transfer_to_gpu()

    r2 = region_hm.pairwise_r2().get()
    positions = region_hm.positions
    if isinstance(positions, cp.ndarray):
        positions = positions.get()
    n_snps = region_hm.num_variants

    del region_hm
    mempool.free_all_blocks()

    return {"region": region, "n_snps": n_snps, "positions": positions, "r2": r2}


# --------------------------------------------------------------------------
# Derived stats (pure CPU/numpy -- re-run freely from a saved pickle)
# --------------------------------------------------------------------------

def compute_block_stats(results):
    chrom_start, loop_end = results["chrom_start"], results["loop_end"]
    all_positions = results["all_positions"]
    window_sizes = results["window_sizes"]

    block_sizes_kb, snp_density, n_snps_per_block = {}, {}, {}
    for w in window_sizes:
        boundaries_bp = np.array(sorted(set(results["all_boundaries_bp"][w])))
        edges = np.concatenate(([chrom_start], boundaries_bp, [loop_end]))
        sizes = np.diff(edges) / 1000
        block_sizes_kb[w] = sizes

        snp_counts = np.diff(np.searchsorted(all_positions, edges))
        n_snps_per_block[w] = snp_counts
        snp_density[w] = snp_counts / sizes

    results["block_sizes_kb"] = block_sizes_kb
    results["snp_density"] = snp_density
    results["n_snps_per_block"] = n_snps_per_block
    return results


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def make_plots(results, region_data, out_dir, zoom_snps=500):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_sizes = results["window_sizes"]
    colors = dict(zip(window_sizes, WINDOW_COLOR_RAMP))
    block_sizes_kb = results["block_sizes_kb"]
    snp_density = results["snp_density"]

    # ---- histogram of block sizes ----
    fig, axes = plt.subplots(1, len(window_sizes), figsize=(5 * len(window_sizes), 4), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), window_sizes):
        ax.hist(block_sizes_kb[w], bins=60, color=colors[w], alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"window={w}", color=INK)
        ax.set_xlabel("LD block size (kb)")
        _style_axis(ax)
    np.atleast_1d(axes)[0].set_ylabel("count")
    fig.suptitle("Chr3L: LD block sizes by bridging-score window size", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "block_size_histogram.png", dpi=150)
    plt.close(fig)

    # ---- block-size boxplot, with outlier annotations ----
    box_width = max(7, 1.0 * len(window_sizes))
    positions_x = list(range(1, len(window_sizes) + 1))
    fig, ax = plt.subplots(figsize=(box_width, 6))
    bp = ax.boxplot([block_sizes_kb[w] for w in window_sizes], vert=True, whis=1.5,
                     positions=positions_x, widths=0.5, patch_artist=True,
                     medianprops=dict(color=INK, linewidth=1.6),
                     whiskerprops=dict(color=MUTED, linewidth=1.2),
                     capprops=dict(color=MUTED, linewidth=1.2),
                     flierprops=dict(marker="o", markersize=4, markerfacecolor="none",
                                      markeredgecolor=MUTED, alpha=0.6))
    for patch, w in zip(bp["boxes"], window_sizes):
        patch.set_facecolor(colors[w])
        patch.set_alpha(0.25)
        patch.set_edgecolor(colors[w])
        patch.set_linewidth(1.3)
    ax.set_yscale("log")
    ax.set_xticks(positions_x)
    ax.set_xticklabels([str(w) for w in window_sizes])
    ax.set_xlabel("window size (SNPs)")
    ax.set_ylabel("LD block size (kb, log scale)")
    ax.set_title("Chr3L: LD block size distribution by window size", fontweight="bold", color=INK)
    _style_axis(ax)
    # outlier % centered above each box's own column (not offset sideways --
    # with many window sizes, a sideways offset runs into the next box)
    top = max(np.max(block_sizes_kb[w]) for w in window_sizes)
    ax.set_ylim(top=top * 2.2)
    for i, w in zip(positions_x, window_sizes):
        _, pct_out, upper = outlier_stats(block_sizes_kb[w])
        ax.annotate(f"{pct_out:.0f}%", xy=(i, upper), xytext=(i, upper * 1.35),
                    fontsize=8, color=MUTED, ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "block_size_boxplot.png", dpi=150)
    plt.close(fig)

    # ---- SNP-density boxplot ----
    fig, ax = plt.subplots(figsize=(box_width, 6))
    bp = ax.boxplot([snp_density[w] for w in window_sizes], vert=True, whis=1.5,
                     positions=positions_x, widths=0.5, patch_artist=True,
                     medianprops=dict(color=INK, linewidth=1.6),
                     whiskerprops=dict(color=MUTED, linewidth=1.2),
                     capprops=dict(color=MUTED, linewidth=1.2),
                     flierprops=dict(marker="o", markersize=4, markerfacecolor="none",
                                      markeredgecolor=MUTED, alpha=0.6))
    for patch, w in zip(bp["boxes"], window_sizes):
        patch.set_facecolor(colors[w])
        patch.set_alpha(0.25)
        patch.set_edgecolor(colors[w])
        patch.set_linewidth(1.3)
    ax.set_yscale("log")
    ax.set_xticks(positions_x)
    ax.set_xticklabels([str(w) for w in window_sizes])
    ax.set_xlabel("window size (SNPs)")
    ax.set_ylabel("SNP density within block (SNPs/kb, log scale)")
    ax.set_title("Chr3L: SNP density within LD blocks by window size", fontweight="bold", color=INK)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "snp_density_boxplot.png", dpi=150)
    plt.close(fig)

    # ---- real bridging-score curves for one region ----
    if region_data is not None:
        region = region_data["region"]
        r2 = region_data["r2"]
        positions = region_data["positions"]
        max_score, min_sep = results["max_bridge_score"], results["min_separation"]

        scores = {w: bridging_score(r2, w) for w in window_sizes}
        boundaries = {w: find_block_boundaries(scores[w], max_score, min_sep) for w in window_sizes}

        fig, ax = plt.subplots(figsize=(11, 5))
        for w in window_sizes:
            ax.plot(positions / 1000, scores[w], label=f"window={w}", color=colors[w], linewidth=1.3)
        ax.axhline(max_score, color=MUTED, linestyle="--", linewidth=1, label="MAX_BRIDGE_SCORE")
        ax.set_xlabel("position (kb)")
        ax.set_ylabel("bridging score (mean r$^2$)")
        ax.set_title(f"Real bridging score across {region} -- all window sizes", fontweight="bold", color=INK)
        ax.legend(fontsize=9, frameon=False)
        _style_axis(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "bridging_score_overlay.png", dpi=150)
        plt.close(fig)

        fig_height = 2.6 * len(window_sizes)
        fig, axes = plt.subplots(len(window_sizes), 1, figsize=(11, fig_height),
                                  sharex=True, gridspec_kw=dict(hspace=0.55))
        for ax, w in zip(np.atleast_1d(axes), window_sizes):
            ax.plot(positions / 1000, scores[w], color=colors[w], linewidth=1.3)
            ax.axhline(max_score, color=MUTED, linestyle="--", linewidth=1)
            for bi in boundaries[w]:
                ax.axvline(positions[bi] / 1000, color=colors[w], alpha=0.3, linewidth=1.2)
            ax.set_ylabel("bridging score")
            ax.set_title(f"window={w}  --  {len(boundaries[w])} boundaries in this region",
                         fontsize=11, loc="left", color=INK, pad=8)
            _style_axis(ax)
        np.atleast_1d(axes)[-1].set_xlabel("position (kb)")
        fig.suptitle(f"Real bridging score across {region} -- own boundaries per window",
                     fontweight="bold", color=INK)
        # tight_layout warns it's unreliable for this sharex + custom-styled
        # axes combination (confirmed empirically -- it either overlaps the
        # suptitle with the first subplot's title or leaves a huge gap
        # depending on the rect passed). Fixed fractional margins instead.
        fig.subplots_adjust(top=0.945, bottom=0.035, left=0.07, right=0.98, hspace=0.55)
        fig.savefig(out_dir / "bridging_score_per_window.png", dpi=150)
        plt.close(fig)

        # ---- real SNP x SNP LD matrix, one panel per window -- zoomed to a
        # small SNP-index slice. The full 300kb region has ~30 real LD blocks
        # (median ~10kb each) packed into it, so showing the whole thing
        # squeezes each block down to a sub-pixel sliver -- nothing but noise
        # is visible at that zoom level. `zoom_snps` this small (a handful of
        # block-widths, not hundreds) is what actually shows block structure
        # -- at 800 SNPs, 30+ boundaries land on top of each other and the
        # dashed lines are indistinguishable from the r^2 speckle.
        #
        # The slice can't just be "the first zoom_snps SNPs": bridging_score
        # is undefined within `window` of either edge of the chunk, so the
        # earliest boundary for the biggest window size can easily land
        # hundreds of SNPs in -- a naive [:zoom_n] slice then shows zero
        # boundaries, not because there's no structure there but because we
        # never looked far enough in. Instead, pick the zoom_n-wide window
        # that contains the most detected boundaries (pooled across all
        # window sizes), centering candidate windows on each boundary so the
        # search actually considers boundary-dense regions.
        zoom_n = min(zoom_snps, r2.shape[0])
        all_bounds = np.array(sorted(set(np.concatenate([boundaries[w] for w in window_sizes]))))
        if len(all_bounds) > 0:
            candidates = np.clip(all_bounds - zoom_n // 2, 0, r2.shape[0] - zoom_n)
            counts = [np.sum((all_bounds >= c) & (all_bounds < c + zoom_n)) for c in candidates]
            zoom_start = int(candidates[int(np.argmax(counts))])
        else:
            zoom_start = 0
        r2_zoom = r2[zoom_start:zoom_start + zoom_n, zoom_start:zoom_start + zoom_n]
        mask = np.triu(np.ones_like(r2_zoom, dtype=bool), k=1)
        masked_r2 = np.where(mask, r2_zoom, np.nan)
        # r^2 is unsigned and mostly small here -- a fixed 0-1 scale crushes
        # real signal down near black. The 99th percentile is already
        # saturated at ~1.0 (many rare-variant coincidences with only 19
        # independent flies), so use the 95th instead for real contrast.
        vmax = np.nanpercentile(masked_r2, 95)

        ncols = min(4, len(window_sizes))
        nrows = int(np.ceil(len(window_sizes) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(12, 4.2 * ncols), 4.4 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad(color="white")

        im = None
        for ax, w in zip(axes_flat, window_sizes):
            im = ax.imshow(masked_r2, cmap=cmap, vmin=0, vmax=vmax, interpolation="none")
            zoom_boundaries = [bi - zoom_start for bi in boundaries[w]
                               if zoom_start <= bi < zoom_start + zoom_n]
            for bi in zoom_boundaries:
                ax.axvline(bi, color="white", linestyle="--", linewidth=0.9, alpha=0.9)
                ax.axhline(bi, color="white", linestyle="--", linewidth=0.9, alpha=0.9)
            ax.set_title(f"window={w}  ({len(zoom_boundaries)} boundaries shown)", fontsize=10, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes_flat[len(window_sizes):]:
            ax.axis("off")

        zoom_span_kb = (positions[zoom_start + zoom_n - 1] - positions[zoom_start]) / 1000
        fig.suptitle(f"Real SNP x SNP LD matrix, zoomed to SNPs {zoom_start}-{zoom_start + zoom_n} "
                     f"(~{zoom_span_kb:.0f}kb) of {region} -- boundaries by window size",
                     fontweight="bold", color=INK)
        fig.subplots_adjust(top=0.90, bottom=0.03, left=0.02, right=0.90, hspace=0.25, wspace=0.1)
        cax = fig.add_axes((0.92, 0.15, 0.02, 0.65))
        fig.colorbar(im, cax=cax, label="r$^2$")
        fig.savefig(out_dir / "ld_matrix_per_window.png", dpi=150)
        plt.close(fig)

    print(f"\nfigures saved to {out_dir}/")


def print_summary(results):
    for w in results["window_sizes"]:
        sizes = results["block_sizes_kb"][w]
        density = results["snp_density"][w]
        n_out, pct_out, upper = outlier_stats(sizes)
        print(f"window={w}: {len(sizes)} LD blocks, median={np.median(sizes):.1f}kb, "
              f"mean={np.mean(sizes):.1f}kb, p10={np.percentile(sizes, 10):.1f}kb, "
              f"p90={np.percentile(sizes, 90):.1f}kb")
        print(f"           outliers (>1.5xIQR past Q3, i.e. >{upper:.1f}kb): "
              f"{n_out} of {len(sizes)} blocks ({pct_out:.1f}%)")
        print(f"           SNP density (SNPs/kb) median={np.median(density):.2f}, "
              f"mean={np.mean(density):.2f}")


def print_block_snp_table(results):
    """Simple, direct summary: median (and mean) block size and SNP count
    per block, per window size -- with outliers included, then again with
    them excluded (Tukey's rule, same as the boxplots), since mean is
    sensitive to the outlier tail but median mostly isn't."""
    print("\n-- including outliers --")
    print(f"{'window':>8} {'n blocks':>9} {'median kb':>10} {'mean kb':>9} "
          f"{'median SNPs':>12} {'mean SNPs':>10}")
    for w in results["window_sizes"]:
        sizes = results["block_sizes_kb"][w]
        n_snps = results["n_snps_per_block"][w]
        print(f"{w:>8} {len(sizes):>9} {np.median(sizes):>10.1f} {np.mean(sizes):>9.1f} "
              f"{np.median(n_snps):>12.0f} {np.mean(n_snps):>10.1f}")

    print("\n-- excluding outliers (>1.5xIQR past Q3) --")
    print(f"{'window':>8} {'n blocks':>9} {'median kb':>10} {'mean kb':>9} "
          f"{'median SNPs':>12} {'mean SNPs':>10}")
    for w in results["window_sizes"]:
        sizes = results["block_sizes_kb"][w]
        n_snps = results["n_snps_per_block"][w]
        q1, q3 = np.percentile(sizes, [25, 75])
        iqr = q3 - q1
        keep = (sizes >= q1 - 1.5 * iqr) & (sizes <= q3 + 1.5 * iqr)
        print(f"{w:>8} {int(keep.sum()):>9} {np.median(sizes[keep]):>10.1f} {np.mean(sizes[keep]):>9.1f} "
              f"{np.median(n_snps[keep]):>12.0f} {np.mean(n_snps[keep]):>10.1f}")


def print_comparison_table(results):
    """Across-window comparison for picking a window size: not about finding
    a single 'optimal' value, just enough roughly-independent blocks to
    block-bootstrap with. Two things matter for that:
      - non-outlier block count: more = more bootstrap replicates
      - CV (coefficient of variation, std/mean) of the non-outlier block
        sizes: lower = more uniform/consistent block sizes ("reliable")
    """
    print(f"\n{'window':>8} {'total blocks':>13} {'outliers':>10} {'non-outlier':>12} "
          f"{'CV (non-out)':>13} {'median (non-out)':>17}")
    for w in results["window_sizes"]:
        sizes = results["block_sizes_kb"][w]
        n_out, _, _ = outlier_stats(sizes)
        is_outlier = np.zeros(len(sizes), dtype=bool)
        q1, q3 = np.percentile(sizes, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_outlier = (sizes < lower) | (sizes > upper)
        kept = sizes[~is_outlier]
        cv = np.std(kept) / np.mean(kept)
        print(f"{w:>8} {len(sizes):>13} {n_out:>10} {len(kept):>12} {cv:>13.3f} {np.median(kept):>17.2f}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcf", default=DEFAULT_VCF)
    p.add_argument("--popfile", default=DEFAULT_POPFILE)
    p.add_argument("--window-sizes", default="75,150,300",
                   help="comma-separated SNP half-widths for the bridging-score scan")
    p.add_argument("--chunk-size", type=int, default=300_000,
                   help="bp per GPU chunk (300kb keeps even dense regions well within GPU memory)")
    p.add_argument("--max-bridge-score", type=float, default=0.065,
                   help="boundary must dip to/below this mean r^2")
    p.add_argument("--min-separation", type=int, default=200,
                   help="min SNP-index gap between reported boundaries")
    p.add_argument("--test-length", type=int, default=None,
                   help="bp to process from the start of Chr3L, for a quick test (default: whole arm)")
    p.add_argument("--gpu-device", type=int, default=0,
                   help="check `nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv` first")
    p.add_argument("--plot-region", default="Chr3L:1866532-2166532",
                   help="one chunk-sized region for the real bridging-score curve plots")
    p.add_argument("--zoom-snps", type=int, default=500,
                   help="SNPs shown in the zoomed LD-matrix-vs-boundaries panel. Sized to comfortably "
                        "span a full median-sized block (~250-290 SNPs) plus its flanking boundaries "
                        "on both sides -- much smaller and you only see one edge with no block context; "
                        "much bigger (e.g. 800+) and 30+ boundaries pack in on top of each other")
    p.add_argument("--out-dir", default="/sietch_colab/akapoor/Infer_Demography/figures/ld_blocks_Chr3L")
    p.add_argument("--from-pickle", default=None,
                   help="skip the GPU pass entirely and re-plot from a saved results.pkl")
    args = p.parse_args()

    window_sizes = [int(w) for w in args.window_sizes.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_pickle:
        print(f"loading saved results from {args.from_pickle}")
        with open(args.from_pickle, "rb") as f:
            saved = pickle.load(f)
        results, region_data = saved["results"], saved["region_data"]
    else:
        results = run_gpu_scan(args.vcf, args.popfile, window_sizes, args.chunk_size,
                                args.max_bridge_score, args.min_separation,
                                args.test_length, args.gpu_device)
        region_data = run_region_scan(args.vcf, args.popfile, args.plot_region,
                                       window_sizes, args.gpu_device)

        pickle_path = out_dir / "results.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump({"results": results, "region_data": region_data}, f)
        print(f"raw results saved to {pickle_path}")

    results = compute_block_stats(results)
    print_summary(results)
    print_block_snp_table(results)
    print_comparison_table(results)
    make_plots(results, region_data, out_dir, zoom_snps=args.zoom_snps)


if __name__ == "__main__":
    main()

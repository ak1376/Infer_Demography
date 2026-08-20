#!/usr/bin/env python3
# godambe_correction_LRT/scripts/godambe_lrt_growth.py
#
# Per-arm generalization of bootstrapping_for_LRT_Chr3L.py: identical Godambe-
# adjusted LRT for CO growth, but driven entirely by CLI arguments instead of
# a hardcoded chromosome, so the Snakefile can request this for any arm.
# All bootstrap/LRT logic is unchanged from the Chr3L script.

"""
Godambe-adjusted likelihood-ratio test: does CO have exponential growth?

H0 (simple model): CO constant post-split, only FR grows.
H1 (complex/"both" model): both CO and FR grow.

Composite-likelihood LRT statistics are inflated by linkage, so we correct the
raw statistic with the Godambe factor adjust = H / J:
  * H is the observed information for the tested parameter (empirical SFS only).
  * J is the variance of the score under block-bootstrap resampling, which is
    what carries the linkage information.
D_adj = adjust * D ~ chi^2_1 under H0.

Single-chromosome block bootstrap
----------------------------------
The demographic fits are on the requested arm's OWN SFS (not pooled with any
other arm), so the bootstrap resamples blocks tiling that arm only. The arm is
tiled into non-overlapping blocks of a fixed physical size (--block-sizes-kb);
a bootstrap replicate draws (with replacement) as many blocks as there are and
sums their per-block SFS.

Block-size sensitivity: --block-sizes-kb sweeps several candidate block sizes
(kb). The default range's ~100 kb point comes from bootstrap_window_size.py,
which found that ~90-100 kb is the smallest block that is still ~independent
on the slowest-decaying arm/population (Chr2R-FR) in the pooled analysis --
treat it as a generic starting point, not an arm-specific guarantee. The
validated, per-population block size from `rule validated_blocks` (read from
--validated-blocks-bed) is reported alongside the sweep as the arm-specific,
LD-decay-justified reference point.
"""

import argparse
import gzip
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moments
from tqdm import tqdm
from src.demes_models import split_migration_growth_both_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="Chromosome arm, e.g. Chr3L")
    p.add_argument("--simple-fit", type=Path, required=True, help="sfs_fit_simple/best_fit.pkl")
    p.add_argument("--complex-fit", type=Path, required=True, help="sfs_fit_complex/best_fit.pkl")
    p.add_argument("--sfs", type=Path, required=True, help="This arm's own unfolded.sfs.pkl")
    p.add_argument("--vcf", type=Path, required=True, help="This arm's polarized (haploid+AA) VCF")
    p.add_argument("--popfile", type=Path, required=True)
    p.add_argument("--validated-blocks-bed", type=Path, required=True,
                    help="validated_blocks.bed from rule validated_blocks")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True,
                    help="Per-arm cache dir for parsed per-block SFS")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--num-boot-reps", type=int, default=10_000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--block-sizes-kb", type=str, default="50,75,100,150,200,300,500",
                    help="Comma-separated candidate block sizes (kb) for the sensitivity sweep")
    p.add_argument("--score-eps", type=float, default=0.01,
                    help="Finite-difference step for the growth_CO score (J) and Hessian (H)")
    return p.parse_args()


# ---- Complex model in the growth_CO parameterization -----------------------
# The null "no CO growth" (N_CO0 == N_CO1) isn't a single coordinate of the
# (N_CO0, N_CO1) vector, so we swap N_CO0 for growth_CO = log(N_CO0 / N_CO1):
# growth_CO == 0  <=>  N_CO0 == N_CO1, i.e. the null is exactly growth_CO = 0.
PARAM_NAMES = [
    "N_ANC",
    "growth_CO",  # log(N_CO0 / N_CO1); == 0 under H0
    "N_CO1",
    "N_FR0",
    "N_FR1",
    "T",
    "m_CO_FR",
    "m_FR_CO",
]
GROWTH_IDX = PARAM_NAMES.index("growth_CO")

# ProcessPoolExecutor pickles the mapped function by reference, so it must be a
# module-level function, not a closure -- these two globals are set once in
# main() (before the pool is created) and inherited by forked workers.
_POPFILE = None
_SAMPLE_SIZES = None


def _parse_chunk(job):
    """Parse one (vcf_path, interval) block into an unfolded SFS.

    `interval` in parse_vcf is 1-indexed and half-open, so consecutive
    (start, end) pairs tile an arm with no overlap.
    """
    vp, interval = job
    try:
        return moments.Parsing.parse_vcf(
            vp, pop_file=_POPFILE, use_AA=True, ploidy=1, interval=interval)
    except ValueError:
        # No SNPs in this interval -> all-zero SFS instead of crashing.
        return moments.Spectrum(np.zeros([n + 1 for n in _SAMPLE_SIZES]))


def split_migration_growth_both_sfs(p, ns):
    """Expected SFS for the complex model, converting growth_CO back to
    N_CO0 = N_CO1 * exp(growth_CO) for the demes graph."""
    sampled = dict(zip(PARAM_NAMES, p))
    growth_CO = sampled.pop("growth_CO")
    sampled["N_CO0"] = sampled["N_CO1"] * np.exp(growth_CO)
    graph = split_migration_growth_both_model(sampled)
    return moments.Spectrum.from_demes(graph, sampled_demes=["CO", "FR"], sample_sizes=ns)


def _get_vcf_bounds(vcf_path):
    first = int(subprocess.check_output(
        f"bcftools query -f '%POS\n' '{vcf_path}' | head -n 1", shell=True).strip())
    last = int(subprocess.check_output(
        f"bcftools query -f '%POS\n' '{vcf_path}' | tail -n 1", shell=True).strip())
    return first, last


def _parse_popfile(popfile_path):
    sample_to_pop, pop_order = {}, []
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


def _get_vcf_sample_indices(vcf_path, popfile_path):
    pop_names, sample_to_pop = _parse_popfile(popfile_path)
    sample_indices = {pop: [] for pop in pop_names}
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                for i, s in enumerate(line.rstrip("\n").split("\t")[9:]):
                    pop = sample_to_pop.get(s)
                    if pop is not None:
                        sample_indices[pop].append(i)
                break
    return pop_names, sample_indices


def main():
    args = parse_args()
    arm = args.arm
    popfile = str(args.popfile)
    vcf_path = str(args.vcf)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = args.out_dir / "block_sensitivity.csv"
    sensitivity_plot_path = args.out_dir / "block_sensitivity.png"
    j_plot_path = args.out_dir / "J_and_adjust_by_block_size.png"
    adjust_plot_path = args.out_dir / "adjust_by_block_size.png"
    hist_template = str(args.out_dir / "J_bootstrap_hist_{block_kb}kb.png")

    block_sizes_kb = [float(x) for x in args.block_sizes_kb.split(",") if x.strip()]
    num_boot_reps = args.num_boot_reps
    rng_seed = args.rng_seed
    score_eps = args.score_eps

    # ---- Null point p0 and fixed theta --------------------------------------
    with open(args.simple_fit, "rb") as f:
        simple_fit = pickle.load(f)
    simple_params = simple_fit["best_params"][0]

    # H0-consistent point in the 8-param growth_CO order (growth_CO = 0).
    p0 = [
        simple_params["N_ANC"],
        0.0,                     # growth_CO
        simple_params["N_CO"],   # N_CO1
        simple_params["N_FR0"],
        simple_params["N_FR1"],
        simple_params["T"],
        simple_params["m_CO_FR"],
        simple_params["m_FR_CO"],
    ]

    with open(args.sfs, "rb") as f:
        data_sfs = pickle.load(f)

    # multinom theta: fix theta at the optimal scaling for p0, then hold it fixed.
    model = split_migration_growth_both_sfs(p0, data_sfs.sample_sizes)
    theta_opt = moments.Inference.optimal_sfs_scaling(model, data_sfs)
    p0_theta = np.array(list(p0) + [theta_opt], dtype=float)

    # ---- H: observed information for growth_CO (empirical SFS only) --------
    def _ll_growth(growth_vec, data):
        full = p0_theta.copy()
        full[GROWTH_IDX] = growth_vec[0]
        fs = full[-1] * split_migration_growth_both_sfs(full[:-1], data.sample_sizes)
        return moments.Inference.ll(fs, data)

    H = -moments.Godambe.get_hess(_ll_growth, [p0_theta[GROWTH_IDX]], eps=score_eps, args=[data_sfs])
    H_growth = float(H[0, 0])
    print(f"[{arm}] H (growth_CO):", H_growth)

    # ---- Per-block SFS (block bootstrap units) ------------------------------
    pop_names, sample_indices = _get_vcf_sample_indices(vcf_path, popfile)
    sample_sizes = [len(sample_indices[pop]) for pop in pop_names]
    arm_lo, arm_hi = _get_vcf_bounds(vcf_path)
    print(f"  {arm}: {arm_lo:,}..{arm_hi:,}  ({(arm_hi - arm_lo + 1) / 1e6:.2f} Mb)")

    # _parse_chunk (module-level, for ProcessPoolExecutor picklability) reads
    # these instead of closing over local variables.
    global _POPFILE, _SAMPLE_SIZES
    _POPFILE = popfile
    _SAMPLE_SIZES = sample_sizes

    def _block_jobs(block_bp):
        """List of (vcf_path, (start, end)) blocks tiling the arm at ~block_bp.

        Gets floor(arm_len / block_bp) blocks, so the realised block size is
        >= block_bp (guarantees the LD-independence target is met, never undershot).
        """
        arm_len = arm_hi - arm_lo + 1
        n = max(1, int(arm_len // block_bp))
        bounds = np.linspace(arm_lo, arm_hi + 1, n + 1).astype(int)
        return [(vcf_path, (int(bounds[i]), int(bounds[i + 1]))) for i in range(n)]

    def _validated_jobs():
        """(vcf_path, (start, end)) blocks from the per-population-validated,
        independence-checked tiling in --validated-blocks-bed -- same job
        format as _block_jobs, but read from disk instead of computed
        arithmetically (and already restricted to the arm's validated usable
        range, unlike _block_jobs, which tiles the raw, unmasked arm bounds)."""
        jobs = []
        with open(args.validated_blocks_bed) as f:
            for line in f:
                chrom, start, end = line.split()
                jobs.append((vcf_path, (int(start), int(end))))
        return jobs

    def get_chunk_spectra(block_bp, jobs=None):
        """Per-block unfolded SFS, pooled into one list. Cached to disk (keyed
        by block size, or 'validated' when `jobs` is given explicitly) so
        re-runs skip parsing."""
        key = "validated" if jobs is not None else f"{int(block_bp)}bp"
        cache = args.cache_dir / f"chunk_spectra_{arm}_{key}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                return pickle.load(f)

        if jobs is None:
            jobs = _block_jobs(block_bp)
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            chunk_spectra = list(tqdm(pool.map(_parse_chunk, jobs), total=len(jobs),
                                      desc=f"parse {len(jobs)} blocks @ {key}"))
        with open(cache, "wb") as f:
            pickle.dump(chunk_spectra, f)
        return chunk_spectra

    # ---- J: score variance under block-bootstrap resampling ----------------
    # For a single nested parameter, bootstrap b contributes J_b = U_b**2, where
    # U_b is the score (gradient of the composite ll w.r.t. growth_CO at
    # growth_CO = 0) on bootstrap b's SFS. J = mean_b(J_b). theta is held at
    # theta_opt throughout, and the two null-point model spectra don't depend
    # on the bootstrap, so build them once. This reproduces moments' finite-
    # difference score, so J matches LRT_adjust's internal J.
    ns = list(data_sfs.sample_sizes)
    m0 = theta_opt * split_migration_growth_both_sfs(p0_theta[:-1], ns)   # growth_CO = 0
    p_eps = p0_theta[:-1].copy()
    p_eps[GROWTH_IDX] = score_eps
    mp = theta_opt * split_migration_growth_both_sfs(p_eps, ns)          # growth_CO = +eps

    def bootstrap_J(block_bp, rng, jobs=None):
        """Per-bootstrap J_b: resample the arm's blocks with replacement
        (n_blocks draws), sum to one bootstrap SFS, and square its growth_CO score."""
        chunk_spectra = get_chunk_spectra(block_bp, jobs=jobs)
        n_blocks = len(chunk_spectra)
        idx = np.arange(n_blocks)
        J_boot = np.empty(num_boot_reps)
        for b in range(num_boot_reps):
            sampled = rng.choice(idx, size=n_blocks, replace=True)
            boot = moments.Spectrum(sum(chunk_spectra[i] for i in sampled))
            score_b = (moments.Inference.ll(mp, boot) - moments.Inference.ll(m0, boot)) / score_eps
            J_boot[b] = score_b ** 2
        return J_boot

    def check_chunk_reconstruction(block_bp, jobs=None):
        """Sanity check: the non-overlapping blocks should sum to the empirical
        arm SFS used in the likelihood. If this fails badly, the bootstrap data
        are not matched to data_sfs (e.g. parse_vcf vs. how the arm SFS was
        built), and the correction is not trustworthy."""
        chunk_spectra = get_chunk_spectra(block_bp, jobs=jobs)
        chunk_sum = moments.Spectrum(sum(chunk_spectra))

        return {
            "chunk_sum_snps": float(chunk_sum.S()),
            "data_sfs_snps": float(data_sfs.S()),
            "snp_diff": float(chunk_sum.S() - data_sfs.S()),
            "max_abs_sfs_diff": float(np.max(np.abs(chunk_sum - data_sfs))),
            "sample_sizes_match": list(chunk_sum.sample_sizes) == list(data_sfs.sample_sizes),
        }

    def summarize_block_size(block_kb, seed, D, jobs=None):
        """Run the Godambe correction for one block size and return one row for
        the sensitivity table, plus the vector of per-bootstrap J values. `jobs`
        overrides the uniform tiling with an explicit block list (the validated,
        independence-checked tiling); `block_kb` is then just a label for plotting."""
        block_bp = int(block_kb * 1e3)
        rng = np.random.default_rng(seed)
        J_boot = bootstrap_J(block_bp, rng, jobs=jobs)
        mean_J = float(J_boot.mean())

        n_blocks = len(get_chunk_spectra(block_bp, jobs=jobs))

        adjust = H_growth / mean_J
        D_adj = adjust * D
        p_adj = float(moments.Godambe.sum_chi2_ppf(D_adj, weights=(0, 1)))

        recon = check_chunk_reconstruction(block_bp, jobs=jobs)

        row = {
            "block_kb": float(block_kb),
            "block_bp": float(block_bp),
            "n_blocks": int(n_blocks),
            "validated": jobs is not None,
            "H": float(H_growth),
            "mean_J": mean_J,
            "sd_J_boot": float(J_boot.std(ddof=1)),
            "adjust_H_over_J": float(adjust),
            "D_adj": float(D_adj),
            "p_adj": p_adj,
            **recon,
        }
        return row, J_boot

    # ---- Godambe-adjusted LRT across candidate block sizes ------------------
    # D = 2*(ll_complex - ll_simple), block-size independent. Only J, H/J,
    # D_adj, and p_adj change with block size.
    #
    # IMPORTANT -- log-likelihood convention. The stored best_ll from
    # moments_dadi_inference_real.py is a cross-entropy-style quantity that
    # differs from moments.Inference.ll (the full Poisson ll) by a constant
    # that depends only on the data. That constant cancels in the difference,
    # so D is identical either way -- but H and J are computed with
    # moments.Inference.ll, so we compute D with moments.Inference.ll too,
    # keeping D, H, and J in ONE consistent convention.
    with open(args.complex_fit, "rb") as f:
        complex_fit = pickle.load(f)

    def _ll_moments(fit_params, growth_CO):
        """moments.Inference.ll of the (theta-profiled) complex-model SFS at a
        fit's absolute params and a given growth_CO, evaluated on data_sfs."""
        full = [
            fit_params["N_ANC"], growth_CO, fit_params["N_CO1"],
            fit_params["N_FR0"], fit_params["N_FR1"], fit_params["T"],
            fit_params["m_CO_FR"], fit_params["m_FR_CO"],
        ]
        fsm = split_migration_growth_both_sfs(full, ns)
        return moments.Inference.ll(moments.Inference.optimal_sfs_scaling(fsm, data_sfs) * fsm, data_sfs)

    # Simple MLE embeds as growth_CO = 0 with N_CO1 = the simple model's single N_CO.
    _sp = simple_fit["best_params"][0]
    ll_simple = _ll_moments(
        {"N_ANC": _sp["N_ANC"], "N_CO1": _sp["N_CO"], "N_FR0": _sp["N_FR0"],
         "N_FR1": _sp["N_FR1"], "T": _sp["T"], "m_CO_FR": _sp["m_CO_FR"], "m_FR_CO": _sp["m_FR_CO"]},
        0.0,
    )
    # Complex MLE: growth_CO = log(N_CO0 / N_CO1).
    _cp = complex_fit["best_params"][0]
    ll_complex = _ll_moments(_cp, float(np.log(_cp["N_CO0"] / _cp["N_CO1"])))

    D = 2.0 * (ll_complex - ll_simple)
    p_raw = float(moments.Godambe.sum_chi2_ppf(D, weights=(0, 1)))

    # Consistency guard: D from the stored best_ll should match D computed here.
    # It will differ ONLY if the two fits were run on different data (their
    # constant offsets would then no longer cancel) -- the real "mismatched
    # SFS" signal.
    D_stored = 2.0 * (float(complex_fit["best_ll"][0]) - float(simple_fit["best_ll"][0]))
    print(f"\n[{arm}] Consistency check (fits on the same SFS?):")
    print(f"  D (moments.Inference.ll) = {D:.6g}")
    print(f"  D (stored best_ll)       = {D_stored:.6g}")
    if abs(D - D_stored) > 1e-3 * max(1.0, abs(D)):
        print(f"  *** WARNING: D differs between conventions for {arm} -- the two "
              f"fits were likely run on DIFFERENT SFS. Re-fit both models on the "
              f"current {arm} SFS. ***")

    results = []
    J_by_block = {}
    print(f"\n[{arm}] Raw LRT: D = {D:.6g}; unadjusted p = {p_raw:.6g}")
    print(f"H (growth_CO): {H_growth:.6g}\n")

    for block_kb in block_sizes_kb:
        # Different, reproducible seed per block size -- avoids accidentally
        # reusing the same bootstrap-index stream for all block sizes.
        seed = rng_seed + int(block_kb)
        row, J_boot = summarize_block_size(block_kb, seed, D)
        results.append(row)
        J_by_block[block_kb] = J_boot

        print(
            f"{block_kb:>5.0f} kb  "
            f"n_blocks={row['n_blocks']:>5d}  "
            f"mean_J={row['mean_J']:>12.3g}  "
            f"H/J={row['adjust_H_over_J']:>10.5g}  "
            f"D_adj={row['D_adj']:>10.5g}  "
            f"p_adj={row['p_adj']:>10.5g}  "
            f"SFS maxdiff={row['max_abs_sfs_diff']:.3g}"
        )

    # ---- Validated blocks: per-population, independence-checked tiling from
    # `rule validated_blocks`, reported alongside the uniform sweep (not
    # instead of it) -- see --validated-blocks-bed above.
    validated_jobs = _validated_jobs()
    _starts_ends = [job[1] for job in validated_jobs]
    validated_block_kb = (_starts_ends[0][1] - _starts_ends[0][0]) / 1e3  # actual bp size, in kb
    seed = rng_seed + 999_999  # distinct from every sweep seed (rng_seed + int(block_kb))
    row, J_boot = summarize_block_size(validated_block_kb, seed, D, jobs=validated_jobs)
    results.append(row)
    J_by_block[validated_block_kb] = J_boot

    print(
        f"{validated_block_kb:>5.1f} kb  "
        f"n_blocks={row['n_blocks']:>5d}  "
        f"mean_J={row['mean_J']:>12.3g}  "
        f"H/J={row['adjust_H_over_J']:>10.5g}  "
        f"D_adj={row['D_adj']:>10.5g}  "
        f"p_adj={row['p_adj']:>10.5g}  "
        f"SFS maxdiff={row['max_abs_sfs_diff']:.3g}  "
        f"[validated]"
    )

    # Write CSV summary without requiring pandas.
    fieldnames = list(results[0].keys())
    with open(summary_csv_path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in results:
            f.write(",".join(str(row[k]) for k in fieldnames) + "\n")
    print(f"\nwrote {summary_csv_path}")

    # ---- Figure 1: sensitivity of adjusted p-value across block sizes ------
    block_kb_arr = np.array([r["block_kb"] for r in results])
    n_blocks_arr = np.array([r["n_blocks"] for r in results])
    p_adj_vals = np.array([r["p_adj"] for r in results])
    adjust_vals = np.array([r["adjust_H_over_J"] for r in results])
    mean_J_vals = np.array([r["mean_J"] for r in results])
    is_validated = np.array([r["validated"] for r in results])

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(block_kb_arr[~is_validated], p_adj_vals[~is_validated], marker="o", label="uniform sweep")
    ax1.plot(block_kb_arr[is_validated], p_adj_vals[is_validated], marker="*", markersize=16,
             linestyle="none", color="#eb6834", label="validated (per-population, CO/FR)")
    ax1.axhline(0.05, ls="--", lw=1, label="p = 0.05")
    ax1.set_xlabel("Block size (kb)")
    ax1.set_ylabel("Godambe-adjusted p-value")
    ax1.set_title(f"Block-size sensitivity of Godambe-adjusted LRT ({arm})")
    ax1.set_xticks(block_kb_arr)
    for x, y, nb in zip(block_kb_arr, p_adj_vals, n_blocks_arr):
        ax1.annotate(f"{nb} blocks", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sensitivity_plot_path, dpi=150)
    print(f"wrote {sensitivity_plot_path}")

    # ---- Figure 2: mean J and H/J across block sizes ------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(block_kb_arr, mean_J_vals, marker="o", label="mean J")
    ax.set_xlabel("Block size (kb)")
    ax.set_ylabel("mean J")
    ax.set_title(f"Score-variance estimate J across block sizes ({arm})")
    ax.set_xticks(block_kb_arr)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(j_plot_path, dpi=150)
    print(f"wrote {j_plot_path}")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(block_kb_arr, adjust_vals, marker="o", label="H/J")
    ax.set_xlabel("Block size (kb)")
    ax.set_ylabel("Adjustment factor H/J")
    ax.set_title(f"Godambe adjustment factor across block sizes ({arm})")
    ax.set_xticks(block_kb_arr)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(adjust_plot_path, dpi=150)
    print(f"wrote {adjust_plot_path}")

    # ---- Per-block-size J histograms ----------------------------------------
    for block_kb in block_sizes_kb:
        J_boot = J_by_block[block_kb]
        row = next(r for r in results if r["block_kb"] == block_kb)
        hist_path = hist_template.format(block_kb=int(block_kb))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(J_boot, bins=30, alpha=0.85, edgecolor="white")
        ax.axvline(row["mean_J"], lw=1.8, ls="--", label=f"mean J = {row['mean_J']:,.0f}")
        ax.axvline(H_growth, lw=2.0, label=f"H = {H_growth:,.0f}")
        ax.set_xlabel(r"per-bootstrap $J_b = (\partial_{\mathrm{growth\_CO}}\,\ell)^2$")
        ax.set_ylabel("count")
        ax.set_title(
            f"{arm}: {block_kb:.0f} kb ({row['n_blocks']} blocks)   "
            f"H/J = {row['adjust_H_over_J']:.4g}   p = {row['p_adj']:.3g}"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)

    print(f"[{arm}] wrote per-block-size J histograms")


if __name__ == "__main__":
    main()

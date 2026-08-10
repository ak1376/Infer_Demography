#!/usr/bin/env python3
"""
snakemake_scripts/debug_ld_replicates_vs_windowing.py

Debugging tool (NOT wired into the Snakefile): for one fixed demography
(model_type + sampled_params taken from an existing sim_dir), compute
empirical LD statistics two ways and compare them.

  A) "replicates" — simulate N independent tree sequences, each of length
     config['genome_length'], and compute LD stats on each one
     (src.simulation.simulate_one_window_replicate, same as the production
     pipeline).

  B) "windowing" — simulate ONE tree sequence of length --big-genome-length
     (default: 10x config['genome_length']) and chop it into N overlapping
     windows of size config['genome_length'] (bcftools view -r), computing
     LD stats on each chunk — mirrors snakemake_scripts/split_vcf_windows.py.

Both methods use the same model, sampled_params, r_bins, and per-window
size, so any systematic difference between the two aggregated (means,
varcovs) is attributable to the estimation method itself rather than to the
underlying demography.

Output layout under --out-dir:
  replicates/windows/window_<i>.vcf.gz + samples.txt + flat_map.txt
  replicates/LD_stats/LD_stats_window_<i>.pkl
  replicates/means.varcovs.pkl, replicates/bootstrap_sets.pkl   (via aggregate_ld_statistics)
  windowing/full.vcf.gz
  windowing/windows/window_<i>.vcf.gz + samples.txt + flat_map.txt
  windowing/LD_stats/LD_stats_window_<i>.pkl
  windowing/means.varcovs.pkl, windowing/bootstrap_sets.pkl
  ld_comparison.pdf
  comparison_summary.txt
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import moments  # noqa: E402

from src.simulation import (  # noqa: E402
    simulate_one_window_replicate,
    simulation,
    write_samples_and_map,
)
from src.LD_stats import compute_ld_window, ld_names, het_names  # noqa: E402
from src.MomentsLD_inference import (  # noqa: E402
    DEFAULT_R_BINS,
    aggregate_ld_statistics,
    load_sampled_params,
)


def run_cmd(cmd: str) -> None:
    subprocess.check_call(cmd, shell=True)


# -----------------------------------------------------------------------------
# Method A: independent replicates
# -----------------------------------------------------------------------------


def build_replicates(
    *,
    sim_dir: Path,
    config_file: Path,
    ld_root: Path,
    num_replicates: int,
    r_bins: np.ndarray,
    seed_stride: int,
) -> None:
    windows_dir = ld_root / "windows"
    ld_stats_dir = ld_root / "LD_stats"
    ld_stats_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_file.read_text())

    for i in range(num_replicates):
        out_pkl = ld_stats_dir / f"LD_stats_window_{i}.pkl"
        if out_pkl.exists():
            print(f"[replicates] window {i}: already computed, skipping")
            continue

        simulate_one_window_replicate(
            sim_dir=sim_dir,
            rep_index=i,
            config_file=config_file,
            out_dir=windows_dir,
            meta_file=None,
            seed_stride=seed_stride,
        )
        stats = compute_ld_window(
            window_index=i,
            vcf_gz=windows_dir / f"window_{i}.vcf.gz",
            samples_file=windows_dir / "samples.txt",
            rec_map_file=windows_dir / "flat_map.txt",
            r_bins=r_bins,
            config=cfg,
        )
        with out_pkl.open("wb") as fh:
            pickle.dump(stats, fh)
        print(f"[replicates] window {i}: LD stats written")


# -----------------------------------------------------------------------------
# Method B: one big simulation, chopped into overlapping windows
# -----------------------------------------------------------------------------


def build_big_vcf(
    *,
    sim_dir: Path,
    config_file: Path,
    ld_root: Path,
    big_genome_length: float,
) -> Path:
    ld_root.mkdir(parents=True, exist_ok=True)
    full_vcf_gz = ld_root / "full.vcf.gz"
    if full_vcf_gz.exists():
        print("[windowing] full.vcf.gz already exists, skipping simulation")
        return full_vcf_gz

    cfg: Dict[str, Any] = json.loads(config_file.read_text())
    model_type = cfg.get("model_type") or cfg.get("demographic_model")
    sampled_params = pickle.load((sim_dir / "sampled_params.pkl").open("rb"))

    big_cfg = dict(cfg)
    big_cfg["genome_length"] = float(big_genome_length)

    print(
        f"[windowing] simulating one tree sequence of length {big_genome_length:.3g} bp..."
    )
    ts, _g = simulation(
        sampled_params=sampled_params,
        model_type=model_type,
        experiment_config=big_cfg,
    )

    raw_vcf = ld_root / "full.vcf"
    with raw_vcf.open("w") as fh:
        ts.write_vcf(fh, allow_position_zero=True)
    with raw_vcf.open("rb") as f_in, gzip.open(full_vcf_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    raw_vcf.unlink()
    run_cmd(f"bcftools index -t '{full_vcf_gz}'")

    # samples.txt + a flat_map.txt covering the *full* big span (shared by all windows)
    write_samples_and_map(
        L=int(big_genome_length),
        r=float(cfg["recombination_rate"]),
        samples={k: int(v) for k, v in cfg["num_samples"].items()},
        out_dir=ld_root,
    )
    return full_vcf_gz


def chop_and_compute(
    *,
    full_vcf_gz: Path,
    ld_root: Path,
    window_size: int,
    num_windows: int,
    config_file: Path,
    r_bins: np.ndarray,
) -> None:
    windows_dir = ld_root / "windows"
    ld_stats_dir = ld_root / "LD_stats"
    windows_dir.mkdir(parents=True, exist_ok=True)
    ld_stats_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_file.read_text())

    chrom = (
        subprocess.check_output(
            f"bcftools query -f '%CHROM\n' '{full_vcf_gz}' | head -n 1", shell=True
        )
        .decode()
        .strip()
    )
    first_pos = int(
        subprocess.check_output(
            f"bcftools query -f '%POS\n' '{full_vcf_gz}' | head -n 1", shell=True
        )
    )
    last_pos = int(
        subprocess.check_output(
            f"bcftools query -f '%POS\n' '{full_vcf_gz}' | tail -n 1", shell=True
        )
    )
    total_span = last_pos - first_pos
    if total_span < window_size:
        raise ValueError(
            f"Simulated span ({total_span} bp) is smaller than the requested "
            f"window size ({window_size} bp) — increase --big-genome-length."
        )

    step = (total_span - window_size) / (num_windows - 1) if num_windows > 1 else 0

    if not (windows_dir / "samples.txt").exists():
        shutil.copy(ld_root / "samples.txt", windows_dir / "samples.txt")
    if not (windows_dir / "flat_map.txt").exists():
        shutil.copy(ld_root / "flat_map.txt", windows_dir / "flat_map.txt")

    for i in range(num_windows):
        out_pkl = ld_stats_dir / f"LD_stats_window_{i}.pkl"
        if out_pkl.exists():
            print(f"[windowing] window {i}: already computed, skipping")
            continue

        w_start = int(first_pos + i * step)
        w_end = min(w_start + window_size - 1, last_pos)
        win_vcf = windows_dir / f"window_{i}.vcf.gz"
        if not win_vcf.exists():
            run_cmd(
                f"bcftools view -r {chrom}:{w_start}-{w_end} -O z -o '{win_vcf}' '{full_vcf_gz}'"
            )
            run_cmd(f"bcftools index -t '{win_vcf}'")

        stats = compute_ld_window(
            window_index=i,
            vcf_gz=win_vcf,
            samples_file=windows_dir / "samples.txt",
            rec_map_file=windows_dir / "flat_map.txt",
            r_bins=r_bins,
            config=cfg,
        )
        with out_pkl.open("wb") as fh:
            pickle.dump(stats, fh)
        print(f"[windowing] window {i}: {chrom}:{w_start}-{w_end} → LD stats written")


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------


def compare_and_plot(
    *,
    mv_replicates: Dict[str, List[np.ndarray]],
    mv_windowing: Dict[str, List[np.ndarray]],
    r_bins: np.ndarray,
    num_pops: int,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ld_stat_names, het_stat_names = ld_names(num_pops), het_names(num_pops)
    all_names = ld_stat_names + het_stat_names

    r_mid = np.sqrt(r_bins[:-1] * np.clip(r_bins[1:], 1e-12, None))
    r_mid[r_bins[:-1] == 0] = r_bins[1] / 2.0  # avoid log(0) for the first bin

    means_A = mv_replicates["means"]  # list length num_bins (last = heterozygosity)
    varcovs_A = mv_replicates["varcovs"]
    means_B = mv_windowing["means"]
    varcovs_B = mv_windowing["varcovs"]

    n_ld_bins = len(means_A) - 1  # last entry is heterozygosity, not r-binned
    n_stats = len(all_names)

    ncols = 4
    nrows = int(np.ceil(n_stats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    summary_lines = []
    for s_idx, stat_name in enumerate(all_names):
        ax = axes[s_idx // ncols][s_idx % ncols]

        if stat_name in ld_stat_names:
            j = ld_stat_names.index(stat_name)
            yA = np.array([means_A[b][j] for b in range(n_ld_bins)])
            yB = np.array([means_B[b][j] for b in range(n_ld_bins)])
            seA = np.array(
                [np.sqrt(np.array(varcovs_A[b])[j, j]) for b in range(n_ld_bins)]
            )
            seB = np.array(
                [np.sqrt(np.array(varcovs_B[b])[j, j]) for b in range(n_ld_bins)]
            )
            x = r_mid[: len(yA)]
            ax.errorbar(x, yA, yerr=seA, fmt="o-", label="replicates", color="tab:blue")
            ax.errorbar(x, yB, yerr=seB, fmt="s-", label="windowing", color="tab:orange")
            ax.set_xscale("log")
        else:
            j = het_stat_names.index(stat_name)
            yA = means_A[-1][j]
            yB = means_B[-1][j]
            seA = np.sqrt(np.array(varcovs_A[-1])[j, j])
            seB = np.sqrt(np.array(varcovs_B[-1])[j, j])
            ax.errorbar([0], [yA], yerr=[seA], fmt="o", label="replicates", color="tab:blue")
            ax.errorbar([0.3], [yB], yerr=[seB], fmt="s", label="windowing", color="tab:orange")
            ax.set_xlim(-1, 1)
            ax.set_xticks([])

        ax.set_title(stat_name, fontsize=9)
        if s_idx == 0:
            ax.legend(fontsize=7)

        rel_diff = np.abs(np.mean(yA) - np.mean(yB)) / (np.abs(np.mean(yB)) + 1e-300)
        summary_lines.append(f"{stat_name}: mean(replicates)={np.mean(yA):.4g}, "
                              f"mean(windowing)={np.mean(yB):.4g}, rel_diff={rel_diff:.3g}")

    for s_idx in range(n_stats, nrows * ncols):
        axes[s_idx // ncols][s_idx % ncols].axis("off")

    fig.suptitle("Empirical LD: replicates vs. windowing", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "ld_comparison.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "comparison_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\n✓ Wrote {out_dir / 'ld_comparison.pdf'}")
    print(f"✓ Wrote {summary_path}")
    print("\n".join(summary_lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare empirical LD stats: independent replicates vs. windowing a single long simulation."
    )
    p.add_argument("--config-file", required=True, type=Path)
    p.add_argument(
        "--sim-dir",
        required=True,
        type=Path,
        help="Existing sim dir containing sampled_params.pkl for the fixed demography",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--num-replicates", type=int, default=100)
    p.add_argument("--num-windows", type=int, default=100)
    p.add_argument(
        "--big-genome-length",
        type=float,
        default=None,
        help="Total length of the single simulation chopped for the windowing method "
        "(default: 10x config['genome_length'])",
    )
    p.add_argument("--seed-stride", type=int, default=10000)
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Override config['use_gpu_ld']=False for this run (portability)",
    )
    args = p.parse_args()

    cfg = json.loads(args.config_file.read_text())
    if args.force_cpu:
        cfg = dict(cfg)
        cfg["use_gpu_ld"] = False
        tmp_cfg_file = args.out_dir / "_config_cpu_override.json"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg_file.write_text(json.dumps(cfg))
        config_file = tmp_cfg_file
    else:
        config_file = args.config_file

    window_size = int(cfg["genome_length"])
    big_genome_length = args.big_genome_length or 10.0 * window_size
    num_pops = len(cfg["num_samples"])
    r_bins = DEFAULT_R_BINS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    replicates_root = args.out_dir / "replicates"
    windowing_root = args.out_dir / "windowing"

    print(f"=== Method A: {args.num_replicates} independent replicates, each {window_size:.3g} bp ===")
    build_replicates(
        sim_dir=args.sim_dir,
        config_file=config_file,
        ld_root=replicates_root,
        num_replicates=args.num_replicates,
        r_bins=r_bins,
        seed_stride=args.seed_stride,
    )
    mv_replicates = aggregate_ld_statistics(replicates_root)

    print(
        f"\n=== Method B: one {big_genome_length:.3g} bp simulation, "
        f"chopped into {args.num_windows} overlapping {window_size:.3g} bp windows ==="
    )
    full_vcf_gz = build_big_vcf(
        sim_dir=args.sim_dir,
        config_file=config_file,
        ld_root=windowing_root,
        big_genome_length=big_genome_length,
    )
    chop_and_compute(
        full_vcf_gz=full_vcf_gz,
        ld_root=windowing_root,
        window_size=window_size,
        num_windows=args.num_windows,
        config_file=config_file,
        r_bins=r_bins,
    )
    mv_windowing = aggregate_ld_statistics(windowing_root)

    print("\n=== Comparing ===")
    compare_and_plot(
        mv_replicates=mv_replicates,
        mv_windowing=mv_windowing,
        r_bins=r_bins,
        num_pops=num_pops,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# godambe_correction_LRT/scripts/compute_H_J.py
"""
Raw Godambe H and J for the CO-growth test (growth_CO = log(N_CO0/N_CO1) in
the complex model; H0 is growth_CO = 0), for one arm at one block size.

Uses moments.Godambe's own internal machinery (moments.Godambe._get_godambe,
the function LRT_adjust/score_stat/GIM_uncert all call internally) to compute
H and J, rather than a hand-rolled score/J loop -- this script only builds the
one input moments.Godambe can't build for you: `all_boot`, the list of
bootstrap-replicate spectra (resample blocks with replacement, sum).

H: observed information for growth_CO on the arm's own SFS (no bootstrap).
J: block-bootstrap score variance, from resampling the block SFS parsed out
   of --block-vcf-dir (produced by the tile_polarized_blocks Snakemake rule).
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import moments
import moments.Godambe as Godambe

from src.demes_models import split_migration_growth_both_model

MAX_WORKERS = 8

# growth_CO = log(N_CO0 / N_CO1); growth_CO == 0 <=> N_CO0 == N_CO1 (H0).
param_names = [
    "N_ANC",
    "growth_CO",
    "N_CO1",
    "N_FR0",
    "N_FR1",
    "T",
    "m_CO_FR",
    "m_FR_CO",
]
growth_idx = param_names.index("growth_CO")


def split_migration_growth_both_sfs(p, ns):
    sampled = dict(zip(param_names, p))
    growth_CO = sampled.pop("growth_CO")
    sampled["N_CO0"] = sampled["N_CO1"] * np.exp(growth_CO)
    graph = split_migration_growth_both_model(sampled)
    return moments.Spectrum.from_demes(graph, sampled_demes=["CO", "FR"], sample_sizes=ns)


def _parse_block(vcf_path, popfile, sample_sizes):
    try:
        return moments.Parsing.parse_vcf(vcf_path, pop_file=popfile, use_AA=True, ploidy=1)
    except ValueError:
        # No SNPs in this block -> all-zero SFS instead of crashing.
        return moments.Spectrum(np.zeros([n + 1 for n in sample_sizes]))


def _parse_block_job(job):
    vcf_path, popfile, sample_sizes = job
    return _parse_block(vcf_path, popfile, sample_sizes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--block-vcf-dir", required=True, type=Path)
    ap.add_argument("--popfile", required=True, type=Path)
    ap.add_argument("--sfs", required=True, type=Path,
                     help="arm's own unfolded SFS -- the data the fits were run on")
    ap.add_argument("--simple-fit", required=True, type=Path)
    ap.add_argument("--complex-fit", required=True, type=Path)
    ap.add_argument("--n-boot-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--pop-ids", default="CO,FR")
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    with open(args.sfs, "rb") as f:
        data_sfs = pickle.load(f)
    data_sfs = moments.Spectrum(data_sfs)
    data_sfs.pop_ids = [s.strip() for s in args.pop_ids.split(",")]
    ns = list(data_sfs.sample_sizes)

    with open(args.simple_fit, "rb") as f:
        simple_fit = pickle.load(f)
    with open(args.complex_fit, "rb") as f:
        complex_fit = pickle.load(f)

    sp = simple_fit["best_params"][0]
    # H0-consistent point: growth_CO = 0, N_CO1 = the simple model's single N_CO.
    p0 = [
        sp["N_ANC"], 0.0, sp["N_CO"], sp["N_FR0"], sp["N_FR1"], sp["T"],
        sp["m_CO_FR"], sp["m_FR_CO"],
    ]
    model = split_migration_growth_both_sfs(p0, ns)
    theta_opt = moments.Inference.optimal_sfs_scaling(model, data_sfs)
    p0_theta = np.array(list(p0) + [theta_opt], dtype=float)

    def func_ex(p, ns):
        return p[-1] * split_migration_growth_both_sfs(p[:-1], ns)

    def diff_func(diff_params, ns):
        # Mirrors moments.Godambe.LRT_adjust's internal diff_func: only the
        # nested (tested) parameter varies, everything else stays at p0_theta.
        full = p0_theta.copy()
        full[growth_idx] = diff_params[0]
        return func_ex(full, ns)

    # --- block SFS, parsed once from the pre-tiled block VCFs ---
    block_vcfs = sorted(glob.glob(str(args.block_vcf_dir / "window_*.vcf.gz")))
    if not block_vcfs:
        raise SystemExit(f"no block VCFs found under {args.block_vcf_dir}")
    jobs = [(v, str(args.popfile), list(ns)) for v in block_vcfs]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        block_spectra = list(pool.map(_parse_block_job, jobs))
    n_blocks = len(block_spectra)
    print(f"[{args.arm}] {n_blocks} blocks parsed from {args.block_vcf_dir}")

    # --- bootstrap replicates: resample blocks with replacement, sum ---
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n_blocks)
    all_boot = []
    for _ in range(args.n_boot_reps):
        sampled = rng.choice(idx, size=n_blocks, replace=True)
        all_boot.append(moments.Spectrum(sum(block_spectra[i] for i in sampled)))

    # --- H and J, straight from moments.Godambe's own internals ---
    p_nested = np.array([0.0])
    godambe, hess, J, cU = Godambe._get_godambe(
        diff_func, all_boot, p_nested, data_sfs, args.eps, log=False
    )
    H_growth = float(hess[0, 0])
    J_growth = float(J[0, 0])
    adjust = H_growth / J_growth

    # --- raw LRT D, same convention as the arm-specific bootstrap scripts ---
    def _ll_moments(fit_params, growth_CO_val):
        full = [
            fit_params["N_ANC"], growth_CO_val, fit_params["N_CO1"],
            fit_params["N_FR0"], fit_params["N_FR1"], fit_params["T"],
            fit_params["m_CO_FR"], fit_params["m_FR_CO"],
        ]
        fsm = split_migration_growth_both_sfs(full, ns)
        return moments.Inference.ll(moments.Inference.optimal_sfs_scaling(fsm, data_sfs) * fsm, data_sfs)

    ll_simple = _ll_moments(
        {"N_ANC": sp["N_ANC"], "N_CO1": sp["N_CO"], "N_FR0": sp["N_FR0"],
         "N_FR1": sp["N_FR1"], "T": sp["T"], "m_CO_FR": sp["m_CO_FR"], "m_FR_CO": sp["m_FR_CO"]},
        0.0,
    )
    cp = complex_fit["best_params"][0]
    ll_complex = _ll_moments(cp, float(np.log(cp["N_CO0"] / cp["N_CO1"])))
    D = 2.0 * (ll_complex - ll_simple)
    D_adj = adjust * D
    p_raw = float(moments.Godambe.sum_chi2_ppf(D, weights=(0, 1)))
    p_adj = float(moments.Godambe.sum_chi2_ppf(D_adj, weights=(0, 1)))

    summary = dict(
        arm=args.arm, n_blocks=n_blocks, n_boot_reps=args.n_boot_reps,
        H=H_growth, J=J_growth, adjust_H_over_J=adjust,
        raw_D=float(D), D_adj=float(D_adj), p_raw=p_raw, p_adj=p_adj,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{args.arm}] H={H_growth:.6g}  J={J_growth:.6g}  H/J={adjust:.6g}")
    print(f"[{args.arm}] D={D:.6g}  D_adj={D_adj:.6g}  p_raw={p_raw:.6g}  p_adj={p_adj:.6g}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()

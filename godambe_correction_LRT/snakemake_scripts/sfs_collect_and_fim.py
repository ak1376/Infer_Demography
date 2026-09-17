#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/sfs_collect_and_fim.py
"""
Thin wrapper called by Snakemake rule `sfs_collect_fim`.

Collects per-start moments SFS fits, picks the best, computes the slice
profiles + FIM/railing diagnostic, and writes best_fit.pkl + fim_summary.json.

Heavy lifting lives in:
  godambe_correction_LRT/src/sfs_collect_and_fim.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import moments

from sfs_collect_and_fim import compute_sfs_fim_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sfs", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--pop-ids", default="CO,FR")
    ap.add_argument("--starts", required=True, type=Path, nargs="+",
                     help="per-start pickles from sfs_fit_one_start.py")
    ap.add_argument("--rel-step", type=float, default=1e-4)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    arm, model_name = args.arm, args.model

    with open(args.config) as f:
        cfg = json.load(f)
    param_order = list(cfg["parameter_order"])
    model_func = getattr(
        importlib.import_module("src.demes_models"), f"{model_name}_model"
    )

    with open(args.sfs, "rb") as f:
        sfs = pickle.load(f)
    sfs = moments.Spectrum(sfs)
    sfs.pop_ids = [s.strip() for s in args.pop_ids.split(",")]

    starts = []
    for p in args.starts:
        with open(p, "rb") as f:
            starts.append(pickle.load(f))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_fit_blob, summary = compute_sfs_fim_summary(
        arm=arm, model_name=model_name, sfs=sfs, cfg=cfg, param_order=param_order,
        model_func=model_func, starts=starts, out_dir=out_dir, rel_step=args.rel_step,
    )

    with open(out_dir / "best_fit.pkl", "wb") as f:
        pickle.dump(best_fit_blob, f)
    with open(out_dir / "fim_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{arm}/{model_name}] best ll={summary['best_ll']:.6f} (start {summary['best_start_seed']})")
    print(f"[{arm}/{model_name}] cond(H)={summary['cond_H']:.3e}  "
          f"({'ILL-CONDITIONED' if summary['ill_conditioned'] else 'ok'})")
    railed = [b["param"] for b in summary["bounds_report"] if b["railed"]]
    print(f"[{arm}/{model_name}] railed params (scaled MLE vs prior bound): {railed if railed else 'none'}")
    edge_hit = [s["param"] for s in summary["slice_profile_report"] if s["at_window_edge"]]
    print(f"[{arm}/{model_name}] slice profiles peaking at window edge: {edge_hit if edge_hit else 'none'}")
    print(f"[{arm}/{model_name}] -> {out_dir}/best_fit.pkl , {out_dir}/fim_summary.json , "
          f"{summary['slice_profile_dir']}/profile_<param>.png")


if __name__ == "__main__":
    main()

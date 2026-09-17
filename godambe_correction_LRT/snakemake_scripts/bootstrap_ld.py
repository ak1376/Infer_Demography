#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/bootstrap_ld.py

"""
Thin wrapper called by Snakemake rule `aggregate_overlap`.

Builds ALL THREE Godambe LD inputs from the NON-overlapping window LD stats
(means, varcovs, bootstrap replicates) and writes them out. See module
docstring in src/bootstrap_ld.py for the statistics.

Heavy lifting lives in:
  godambe_correction_LRT/src/bootstrap_ld.py
"""

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bootstrap_ld import NORM_LD, build_bootstrap_ld_inputs

ROOT = Path(__file__).resolve().parents[2]
GC = ROOT / "godambe_correction_LRT"


def main():
    ap = argparse.ArgumentParser(description="Means/varcovs + bootstrap replicates from non-overlapping windows.")
    ap.add_argument("--ld-stats-dir", type=Path,
                    default=GC / "ld" / "nonoverlap" / "LD_stats")
    ap.add_argument("--out-dir", type=Path, default=GC / "ld" / "nonoverlap")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows, mv, all_boot = build_bootstrap_ld_inputs(
        args.ld_stats_dir, n_boot=args.n_boot, seed=args.seed)

    win_list = list(windows.values())
    ld_names, h_names = win_list[0]["stats"]
    n_windows = len(windows)
    print(f"Loaded {n_windows} non-overlapping windows "
          f"({len(ld_names)} LD stats x {len(win_list[0]['sums']) - 1} r-bins + "
          f"{len(h_names)} H stats).")

    means_file = args.out_dir / "means.varcovs.pkl"
    with means_file.open("wb") as f:
        pickle.dump(mv, f)

    boots_file = args.out_dir / "bootstrap_sets.pkl"
    with boots_file.open("wb") as f:
        pickle.dump(all_boot, f)

    # --- summary / sanity ----------------------------------------------------
    print(f"\nmeans/varcovs -> {means_file}")
    print(f"  means: {len(mv['means'])} arrays, shapes {[np.asarray(a).shape for a in mv['means']]}")
    print(f"  varcovs: {len(mv['varcovs'])} matrices, "
          f"shapes {[np.asarray(a).shape for a in mv['varcovs']]}")
    print(f"bootstrap replicates -> {boots_file}")
    print(f"  {len(all_boot)} replicates, each {len(all_boot[0])} arrays "
          f"(shapes {[a.shape for a in all_boot[0]]})")

    # normalizer should be ~1 in both means and each replicate (kept, not removed)
    pi2_idx = ld_names.index(NORM_LD)
    print(f"  normalizer {NORM_LD} (should be ~1): "
          f"means={np.asarray(mv['means'][0])[pi2_idx]:.4g}, "
          f"boot={all_boot[0][0][pi2_idx]:.4g}")


if __name__ == "__main__":
    main()

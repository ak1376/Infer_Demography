#!/usr/bin/env python3
# godambe_correction_LRT/scripts/bootstrap_ld.py

"""
Build ALL THREE Godambe LD inputs from the NON-overlapping window LD stats, from
one consistent set of windows:

  means, varcovs   (the observed LD curve + its covariance)  -> feed H
  all_boot         (bootstrap replicate LD curves)           -> feed J

means/varcovs come from moments.LD.Parsing.bootstrap_data (aggregate the windows
+ estimate their covariance -- no optimization, pure data summary).

all_boot is built explicitly here: each replicate samples the windows WITH
REPLACEMENT (as many draws as there are windows -> genome-sized), sums their RAW
LD sums, then sigmaD2-normalizes. Summing-then-normalizing is the right way to
"average" ratio statistics: (sum D^2)/(sum pi2) equals the mean ratio. This
reproduces moments' get_bootstrap_sets(..., remove_norm_stats=False).

All three KEEP the normalizing statistic (pi2_0_0_0_0 / H_0_0 -> value 1),
because moments.LD.Godambe.LRT_adjust removes it itself and errors if it's gone.

OUTPUT (under --out-dir, default ld/nonoverlap)
  means.varcovs.pkl   dict with "means" and "varcovs"   (LRT_adjust ms, vcs)
  bootstrap_sets.pkl  list[replicate]                   (LRT_adjust all_boot)
"""

import pickle
import argparse
from pathlib import Path

import numpy as np
import moments

ROOT = Path(__file__).resolve().parents[2]
GC = ROOT / "godambe_correction_LRT"

NORM_LD = "pi2_0_0_0_0"   # σD² normalizer for the LD-stat bins
NORM_H  = "H_0_0"          # normalizer for the heterozygosity array


def load_windows(ld_stats_dir: Path):
    """Load every window's LD stats into a dict {window_id: data}."""
    files = sorted(ld_stats_dir.glob("LD_stats_window_*.pkl"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"No LD_stats_window_*.pkl in {ld_stats_dir}")
    windows = {}
    for f in files:
        wid = int(f.stem.split("_")[-1])
        with f.open("rb") as fh:
            s = pickle.load(fh)
        if isinstance(s, dict) and s.get("empty"):
            continue                                  # skip empty-window sentinels
        windows[wid] = s
    return windows


def average_ld_structure(sampled_windows, ld_names, h_names):
    """Sum raw sums across sampled windows, then sigmaD2-normalize."""
    n_arrays = len(sampled_windows[0]["sums"])      # 5 LD bins + 1 H = 6
    summed = [np.sum([w["sums"][k] for w in sampled_windows], axis=0)
              for k in range(n_arrays)]

    pi2_idx = ld_names.index(NORM_LD)
    h_idx = h_names.index(NORM_H)
    normed = [arr.copy() for arr in summed]
    for k in range(n_arrays - 1):                   # the LD-stat r-bins
        normed[k] = summed[k] / summed[k][pi2_idx]
    normed[-1] = summed[-1] / summed[-1][h_idx]     # the H array
    return normed


def main():
    ap = argparse.ArgumentParser(description="Means/varcovs + bootstrap replicates from non-overlapping windows.")
    ap.add_argument("--ld-stats-dir", type=Path,
                    default=GC / "ld" / "nonoverlap" / "LD_stats")
    ap.add_argument("--out-dir", type=Path, default=GC / "ld" / "nonoverlap")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_windows(args.ld_stats_dir)          # dict {id: data}
    win_list = list(windows.values())
    ld_names, h_names = win_list[0]["stats"]
    n_windows = len(windows)
    print(f"Loaded {n_windows} non-overlapping windows "
          f"({len(ld_names)} LD stats x {len(win_list[0]['sums']) - 1} r-bins + "
          f"{len(h_names)} H stats).")

    # --- means + varcovs (observed curve + covariance) via moments -----------
    # bootstrap_data resamples the windows internally for the covariance, so seed
    # numpy's global RNG for reproducibility.
    np.random.seed(args.seed)
    mv = moments.LD.Parsing.bootstrap_data(windows)
    means_file = args.out_dir / "means.varcovs.pkl"
    with means_file.open("wb") as f:
        pickle.dump(mv, f)

    # --- all_boot (bootstrap replicate curves) -- explicit resampling --------
    rng = np.random.default_rng(args.seed)
    all_boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n_windows, size=n_windows)   # with replacement
        sampled = [win_list[i] for i in idx]
        all_boot.append(average_ld_structure(sampled, ld_names, h_names))
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

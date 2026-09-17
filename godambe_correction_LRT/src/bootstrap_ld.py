#!/usr/bin/env python3
# godambe_correction_LRT/src/bootstrap_ld.py

"""
Logic for building ALL THREE Godambe LD inputs from the NON-overlapping window
LD stats, from one consistent set of windows:

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

CLI wrapper: godambe_correction_LRT/snakemake_scripts/bootstrap_ld.py
  OUTPUT (under --out-dir, default ld/nonoverlap)
    means.varcovs.pkl   dict with "means" and "varcovs"   (LRT_adjust ms, vcs)
    bootstrap_sets.pkl  list[replicate]                   (LRT_adjust all_boot)
"""

import pickle
from pathlib import Path

import numpy as np

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


def build_bootstrap_ld_inputs(ld_stats_dir: Path, n_boot: int = 200, seed: int = 42):
    """Load the non-overlapping windows and build (mv, all_boot).

    mv       : dict with "means"/"varcovs" from moments.LD.Parsing.bootstrap_data
    all_boot : list of n_boot resampled-and-normalized replicate curves

    Returns (windows, mv, all_boot) -- `windows` (the raw per-window dict) is
    also returned since callers print diagnostics off it.
    """
    import moments  # local import: keeps this module importable without moments installed elsewhere

    windows = load_windows(ld_stats_dir)
    win_list = list(windows.values())
    ld_names, h_names = win_list[0]["stats"]
    n_windows = len(windows)

    # bootstrap_data resamples the windows internally for the covariance, so seed
    # numpy's global RNG for reproducibility.
    np.random.seed(seed)
    mv = moments.LD.Parsing.bootstrap_data(windows)

    rng = np.random.default_rng(seed)
    all_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_windows, size=n_windows)   # with replacement
        sampled = [win_list[i] for i in idx]
        all_boot.append(average_ld_structure(sampled, ld_names, h_names))

    return windows, mv, all_boot

#!/usr/bin/env python3
# godambe_correction_LRT/src/aggregate_and_J_ld.py

"""
Aggregate the per-tile LD stats (from the parallel per-window jobs) for one
(arm, block size), then compute J at p0 (the SIMPLE-model fit, embedded in the
complex model).

This is the tail of the old monolithic arm_J_at_blocksize.py, split out so the
tiling and per-window LD stats can be separate Snakemake jobs.

CLI wrapper: godambe_correction_LRT/snakemake_scripts/aggregate_and_J_ld.py
"""

import pickle
from pathlib import Path

import numpy as np
import moments

from bootstrap_ld import average_ld_structure
import compute_J_ld as cj


def aggregate_and_compute_J(ld_stats_dir: Path, null_params: dict, r_bins,
                            n_boot: int = 200, seed: int = 42):
    """Aggregate the non-empty LD_stats_window_*.pkl tiles under `ld_stats_dir`
    and compute J (score variance) at `null_params`, using `r_bins` (must match
    the r_bins the LD stats were computed with).

    Returns (n_windows, J, scores).
    """
    windows = {}
    for p in ld_stats_dir.glob("LD_stats_window_*.pkl"):
        s = pickle.load(p.open("rb"))
        if isinstance(s, dict) and s.get("empty"):
            continue                                  # skip empty-window sentinels
        windows[int(p.stem.split("_")[-1])] = s
    if not windows:
        raise RuntimeError(f"No (non-empty) LD stats in {ld_stats_dir}")
    win_list = list(windows.values())
    ld_names, h_names = win_list[0]["stats"]
    nn = len(win_list)

    np.random.seed(seed)
    mv = moments.LD.Parsing.bootstrap_data(windows)
    rng = np.random.default_rng(seed)
    all_boot = [
        average_ld_structure([win_list[i] for i in rng.integers(0, nn, size=nn)],
                             ld_names, h_names)
        for _ in range(n_boot)
    ]

    cj.R_BINS = np.asarray(r_bins, dtype=float)
    J, scores = cj.compute_J(all_boot, mv["varcovs"], null_params)
    return nn, J, scores

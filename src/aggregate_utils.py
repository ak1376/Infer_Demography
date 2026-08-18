"""Shared aggregation logic for the *_opts_* Snakemake rules.

Every ``aggregate_opts_*`` rule in the Snakefile (moments / dadi / MomentsLD,
for both simulated and real data) loads a set of per-optimization-restart
pickles, keeps the top-K entries by log-likelihood, and writes out a
``best_params`` / ``best_ll`` / ``opt_index`` dict. This module factors out
that shared loop so each rule's ``run:`` block only has to supply the paths
and any rule-specific bookkeeping.
"""
import glob
import pickle
import re

import numpy as np


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple, np.ndarray)) else [x]


def discover_opt_pkls(glob_pattern, regex):
    """Glob for per-opt pkl paths and recover each one's opt index via `regex`.

    `regex` must contain a single capturing group matching the opt index.
    Returns a sorted list of (path, opt_idx) tuples; paths that don't match
    `regex` are skipped.
    """
    records = []
    for pkl_path in sorted(glob.glob(glob_pattern)):
        m = re.search(regex, pkl_path)
        if not m:
            continue
        records.append((pkl_path, int(m.group(1))))
    return records


def aggregate_top_k(records, top_k, extra_fields=(), min_nonempty=None,
                     err_label="", err_engine="", err_context=""):
    """Load best_params/best_ll (+ extra_fields) from each pkl, keep the
    top_k entries by best_ll.

    records: list of (path, opt_idx) pairs, e.g. from discover_opt_pkls()
        or [(path, i) for i, path in enumerate(paths)].
    extra_fields: additional dict keys to carry through per-entry (e.g.
        "theta_hat"); missing values default to NaN.
    min_nonempty: if set, raise ValueError when fewer than this many records
        yielded a non-empty best_ll list.

    Returns (best, diagnostics):
        best: {"best_params": [...], "best_ll": [...], "opt_index": [...],
               <extra_fields>: [...]}  (top_k entries only)
        diagnostics: {"n_records", "n_readable", "n_nonempty", "n_entries"}
    """
    params, lls, opt_ids = [], [], []
    extra = {f: [] for f in extra_fields}
    n_readable = 0
    n_nonempty = 0

    for pkl_path, opt_idx in records:
        try:
            with open(pkl_path, "rb") as fh:
                d = pickle.load(fh)
            n_readable += 1
        except Exception as e:
            print(f"WARNING: could not load {pkl_path}: {e}")
            continue

        this_lls = as_list(d.get("best_ll"))
        if len(this_lls) == 0:
            continue

        this_params = as_list(d.get("best_params"))
        n_nonempty += 1
        params.extend(this_params)
        lls.extend(this_lls)
        opt_ids.extend([opt_idx] * len(this_lls))
        for f in extra_fields:
            extra[f].extend(as_list(d.get(f, np.nan)))

    if min_nonempty is not None and n_nonempty < min_nonempty:
        raise ValueError(
            f"[{err_label}] Need >= {min_nonempty} non-empty {err_engine} optimizations {err_context}, "
            f"but got nonempty={n_nonempty} (readable={n_readable}, paths_found={len(records)}). "
            f"Not aggregating."
        )

    # np.argsort ranks NaN as the largest value, so a single failed
    # (NaN/inf) restart would otherwise outrank every real optimization and
    # get selected as "best". Sort on a key that pushes non-finite lls to
    # the bottom instead, so they're only ever selected if there aren't
    # enough finite entries to fill top_k.
    lls_arr = np.asarray(lls, dtype=float)
    sort_key = np.where(np.isfinite(lls_arr), lls_arr, -np.inf)
    keep = np.argsort(sort_key)[::-1][:top_k]

    best = {
        "best_params": [params[i] for i in keep],
        "best_ll":     [lls[i] for i in keep],
        "opt_index":   [opt_ids[i] for i in keep],
    }
    for f in extra_fields:
        best[f] = [extra[f][i] for i in keep]

    diagnostics = {
        "n_records": len(records),
        "n_readable": n_readable,
        "n_nonempty": n_nonempty,
        "n_entries": len(lls),
        "n_nonfinite": int(np.size(lls_arr) - np.isfinite(lls_arr).sum()),
    }
    return best, diagnostics

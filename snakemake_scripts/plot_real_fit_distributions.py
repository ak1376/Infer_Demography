#!/usr/bin/env python3
# snakemake_scripts/plot_real_fit_distributions.py
#
# Diagnostic: distribution of the real-data per-restart optimizer fits
# (dadi/moments/MomentsLD), across ALL optimization restarts under
# real_data_analysis/runs/run_*/inferences/{engine}/best_fit.pkl -- i.e.
# before top-K aggregation. Useful for seeing how consistent/multimodal the
# real-data SFS optimization landscape actually is per parameter and engine.

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_pickle(path: Path):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (FileNotFoundError, EOFError):
        return None


def best_params_from_fit(blob) -> Optional[Dict[str, float]]:
    """Per-run best_fit.pkl is a single restart: {'best_params': dict, ...}
    (falls back to list[dict]/best_ll in case a top-K-shaped file shows up)."""
    if blob is None:
        return None
    bp = blob.get("best_params")
    if isinstance(bp, dict):
        return {k: float(v) for k, v in bp.items()}
    if isinstance(bp, list) and bp:
        ll = blob.get("best_ll")
        idx = 0
        if isinstance(ll, list) and len(ll) == len(bp):
            idx = max(range(len(ll)), key=lambda i: ll[i])
        return {k: float(v) for k, v in bp[idx].items()}
    return None


_RUN_RE = re.compile(r"^run_(\d+)$")


def load_engine_runs(runs_dir: Path, engine_subdir: str) -> List[Dict[str, float]]:
    """Load best_params from every runs_dir/run_*/inferences/{engine_subdir}/best_fit.pkl."""
    rows = []
    for d in sorted(runs_dir.iterdir(), key=lambda p: (
        int(m.group(1)) if (m := _RUN_RE.match(p.name)) else -1
    )):
        if not _RUN_RE.match(d.name):
            continue
        p = d / "inferences" / engine_subdir / "best_fit.pkl"
        params = best_params_from_fit(_load_pickle(p))
        if params is not None:
            rows.append(params)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--runs-dir", required=True, type=Path,
                     help="real_data_analysis/runs directory (contains run_0, run_1, ...).")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    param_order = list(cfg["priors"].keys())

    ENGINES = [("dadi", "dadi"), ("moments", "moments"), ("momentsLD", "MomentsLD")]
    runs_by_engine = {label: load_engine_runs(args.runs_dir, subdir) for label, subdir in ENGINES}
    for label, rows in runs_by_engine.items():
        print(f"{label}: {len(rows)} restarts loaded from {args.runs_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_params = len(param_order)
    ncols = 4
    nrows = -(-n_params // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    colors = {"dadi": "tab:blue", "moments": "tab:orange", "momentsLD": "tab:green"}

    for i, p in enumerate(param_order):
        ax = axes[i]
        data, labels, colors_used = [], [], []
        for label, _ in ENGINES:
            vals = [row[p] for row in runs_by_engine[label] if p in row and np.isfinite(row[p]) and row[p] > 0]
            if vals:
                data.append(vals)
                labels.append(label)
                colors_used.append(colors[label])

        if not data:
            ax.set_title(f"{p} (no data)")
            continue

        parts = ax.violinplot(data, showmedians=True)
        for body, c in zip(parts["bodies"], colors_used):
            body.set_facecolor(c)
            body.set_alpha(0.5)
        for j, vals in enumerate(data):
            jitter = np.random.default_rng(0).normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), j + 1) + jitter, vals, s=8, color=colors_used[j], alpha=0.5)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_yscale("log")
        ax.set_title(p)

    for i in range(n_params, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Real-data per-restart optimizer fits — {cfg['demographic_model']} "
                 f"({len(runs_by_engine['dadi'])} dadi / {len(runs_by_engine['moments'])} moments / "
                 f"{len(runs_by_engine['momentsLD'])} momentsLD restarts)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = args.out_dir / "real_fit_distributions.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    # Summary stats (median / IQR) per engine/param, for a quick numeric look.
    summary = {}
    for label, _ in ENGINES:
        summary[label] = {}
        for p in param_order:
            vals = [row[p] for row in runs_by_engine[label] if p in row and np.isfinite(row[p])]
            if vals:
                lo, med, hi = np.percentile(vals, [25, 50, 75])
                summary[label][p] = {"n": len(vals), "p25": lo, "median": med, "p75": hi}
    out_json = args.out_dir / "real_fit_distributions_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# snakemake_scripts/plot_real_ld_comparison.py
#
# Diagnostic: empirical vs. theoretical LD curves for the real Drosophila data,
# using the MomentsLD best_fit.pkl parameter estimates from specific
# real_data_analysis/runs/run_<id> restarts (rather than the "true" sampled
# params used for simulated data). Reuses the pipeline's own
# src.MomentsLD_inference.create_comparison_plot so the plot matches what the
# sim-side diagnostics look like.

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.MomentsLD_inference import DEFAULT_R_BINS, create_comparison_plot, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--empirical", required=True, type=Path, help="means.varcovs.pkl")
    ap.add_argument("--runs-dir", required=True, type=Path,
                     help="real_data_analysis/runs directory (contains run_0, run_1, ...).")
    ap.add_argument("--run-ids", required=True, type=str,
                     help="Comma-separated run indices, e.g. '36,6'.")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.empirical, "rb") as fh:
        empirical_data = pickle.load(fh)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for run_id in args.run_ids.split(","):
        run_id = run_id.strip()
        fit_path = args.runs_dir / f"run_{run_id}" / "inferences" / "MomentsLD" / "best_fit.pkl"
        with open(fit_path, "rb") as fh:
            fit = pickle.load(fh)
        best_params = fit["best_params"]

        out_pdf = args.out_dir / f"empirical_vs_theoretical_comparison_run{run_id}.pdf"
        out_pdf.unlink(missing_ok=True)  # create_comparison_plot skips if it already exists
        create_comparison_plot(cfg, best_params, empirical_data, DEFAULT_R_BINS, out_pdf)
        print(f"run_{run_id}: best_ll={fit['best_ll']:.2f}  N_ANC={best_params['N_ANC']:.1f} -> {out_pdf}")


if __name__ == "__main__":
    main()

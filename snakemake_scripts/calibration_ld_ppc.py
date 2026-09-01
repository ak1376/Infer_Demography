#!/usr/bin/env python3
# snakemake_scripts/calibration_ld_ppc.py
#
# Model calibration check (posterior-predictive), LD version: does the
# theoretical LD decay curve at the fitted params (e.g. predict_real_data.py's
# output, or a raw moments/momentsLD best_fit) reproduce the empirical LD
# decay measured directly from the real data?
#
# Thin wrapper around src.MomentsLD_inference.create_comparison_plot -- the
# same moments.LD.Plotting.plot_ld_curves_comp-based comparison the pipeline
# already produces (with sampled_params=None, i.e. no comparison curve) via
# aggregate_ld_windows_real/LD_inference.py --skip-optimize. This just feeds
# it real fitted params instead, no LD-window resimulation needed: the
# theoretical curve is computed analytically from the params, the same way
# the MomentsLD optimizer itself evaluates the objective.

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.MomentsLD_inference import create_comparison_plot, DEFAULT_R_BINS  # noqa: E402


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                     help="Experiment config JSON (demographic_model, priors, num_samples).")
    ap.add_argument("--real-ld-pkl", required=True, type=Path,
                     help="Real means.varcovs.pkl (empirical LD stats), e.g. "
                          "real_data_analysis/data/drosophila/MomentsLD/means.varcovs.pkl.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--params-json", type=Path,
                      help="JSON file with fitted params. Accepts either a flat "
                           "{param: value} object or a predict_real_data.py output "
                           "({'predictions': {param: value}, ...}).")
    src.add_argument("--params", type=str,
                      help="Fitted params as an inline JSON object string.")
    ap.add_argument("--out-path", required=True, type=Path,
                     help="Output PDF path.")
    return ap.parse_args()


def _load_params(args) -> dict:
    if args.params_json is not None:
        raw = json.loads(args.params_json.read_text())
    else:
        raw = json.loads(args.params)
    if isinstance(raw, dict) and isinstance(raw.get("predictions"), dict):
        raw = raw["predictions"]
    if not isinstance(raw, dict):
        raise SystemExit("Fitted params must be a JSON object of {param: value}.")
    return {k: float(v) for k, v in raw.items()}


def main() -> None:
    args = _parse_args()
    cfg = json.loads(args.config.read_text())

    params = _load_params(args)
    required = list(cfg["priors"].keys())
    missing = [p for p in required if p not in params]
    if missing:
        raise SystemExit(f"Fitted params missing required keys: {missing}")

    with open(args.real_ld_pkl, "rb") as fh:
        empirical_data = pickle.load(fh)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.unlink(missing_ok=True)  # create_comparison_plot no-ops if the path already exists

    create_comparison_plot(cfg, params, empirical_data, DEFAULT_R_BINS, args.out_path)

    if args.out_path.stat().st_size == 0:
        raise SystemExit(f"{args.out_path} is empty -- plot generation failed "
                          f"(see the warning logged by create_comparison_plot above).")
    print(f"✓ wrote LD calibration PPC → {args.out_path}")


if __name__ == "__main__":
    main()

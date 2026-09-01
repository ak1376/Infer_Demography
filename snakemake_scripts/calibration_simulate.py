#!/usr/bin/env python3
# snakemake_scripts/calibration_simulate.py
#
# Model calibration check: simulate under the demographic model at a fixed
# parameter point (e.g. the real-data fitted params from predict_real_data.py)
# and save the tree sequence + SFS per replicate, for later posterior-
# predictive comparison against the real observed data/summary stats.
#
# One invocation runs --n-replicates replicates starting at --start-replicate-index
# (sequentially, in-process). For SLURM job-array parallelism, call this with
# --n-replicates 1 --start-replicate-index "$SLURM_ARRAY_TASK_ID" -- one array
# task per replicate, run concurrently by SLURM (see bash_scripts/
# calibration_simulate.sh). For an ad hoc multi-replicate run in one process,
# just pass a larger --n-replicates.
#
# Reuses src.simulation.simulation()/create_SFS() -- the same functions the
# sim-generation pipeline (run_one_simulation_to_dir) uses -- just fed an
# explicit params dict instead of a prior draw. All physical simulation
# parameters (sequence_length, mutation_rate, recombination_rate, num_samples,
# engine) come from --config, same as every other pipeline stage.

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import simulation, create_SFS, sample_coverage_percent  # noqa: E402


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                     help="Experiment config JSON (priors, num_samples, engine, demographic_model, "
                          "sequence_length, mutation_rate, recombination_rate, ...).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--params-json", type=Path,
                      help="JSON file with fitted params. Accepts either a flat "
                           "{param: value} object or a predict_real_data.py output "
                           "({'predictions': {param: value}, ...}).")
    src.add_argument("--params", type=str,
                      help="Fitted params as an inline JSON object string.")
    ap.add_argument("--model-type", default=None,
                     help="Defaults to config['demographic_model'].")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Directory to hold replicate_0/, replicate_1/, ... plus fitted_params.json.")
    ap.add_argument("--n-replicates", type=int, default=1,
                     help="Independent stochastic replicates simulated at this same param point.")
    ap.add_argument("--start-replicate-index", type=int, default=0,
                     help="First replicate index to simulate; replicate i in "
                          "[start, start+n_replicates) is written to replicate_{i}/ "
                          "with seed base_seed+i. Set to $SLURM_ARRAY_TASK_ID with "
                          "--n-replicates 1 for one array task per replicate.")
    ap.add_argument("--seed", type=int, default=None,
                     help="Base seed; replicate i uses seed+i. Defaults to config['seed'] "
                          "offset by 1_000_000 (to avoid colliding with training-sim seeds), "
                          "or a random seed if config has none.")
    ap.add_argument("--coverage-percent", type=float, default=None,
                     help="Fixed BGS coverage percent; only used when engine=='slim'. If "
                          "omitted under slim, coverage is randomly sampled per replicate "
                          "from selection.coverage_percent (same as training).")
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


def _simulate_one_replicate(*, rep_dir, rep_index, params, model_type, cfg, engine,
                             base_seed, coverage_percent):
    rep_dir.mkdir(parents=True, exist_ok=True)

    if base_seed is not None:
        replicate_seed = base_seed + rep_index
        rng = np.random.default_rng(replicate_seed)
    else:
        replicate_seed = None
        rng = np.random.default_rng()

    sim_cfg = dict(cfg)
    if replicate_seed is not None:
        sim_cfg["seed"] = replicate_seed

    if engine == "slim":
        sel_cfg = cfg.get("selection") or {}
        coverage = (
            coverage_percent
            if coverage_percent is not None
            else sample_coverage_percent(sel_cfg, rng=rng)
        )
    else:
        coverage = None

    ts, g = simulation(params, model_type, sim_cfg, sampled_coverage=coverage)
    sfs = create_SFS(ts, pop_names=tuple(cfg["num_samples"].keys()))

    ts.dump(rep_dir / "tree_sequence.trees")
    (rep_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    (rep_dir / "meta.json").write_text(json.dumps({
        "model_type": model_type,
        "engine": engine,
        "replicate_index": rep_index,
        "seed": replicate_seed,
        "coverage_percent": coverage,
        "params": params,
    }, indent=2))

    print(f"[replicate {rep_index}] wrote {rep_dir}/tree_sequence.trees + SFS.pkl "
          f"(seed={replicate_seed}, sum(SFS)={float(np.asarray(sfs).sum()):.6g})")


def main() -> None:
    args = _parse_args()
    cfg = json.loads(args.config.read_text())
    model_type = args.model_type or cfg["demographic_model"]

    params = _load_params(args)
    required = list(cfg["priors"].keys())
    missing = [p for p in required if p not in params]
    if missing:
        raise SystemExit(f"Fitted params missing required keys: {missing}")
    params = {p: params[p] for p in required}  # drop extras, fix order

    engine = str(cfg["engine"]).lower()
    if engine not in ("slim", "msprime"):
        raise SystemExit("config['engine'] must be 'slim' or 'msprime'.")

    if args.seed is not None:
        base_seed = args.seed
    elif cfg.get("seed") is not None:
        base_seed = int(cfg["seed"]) + 1_000_000
    else:
        base_seed = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fitted_path = args.out_dir / "fitted_params.json"
    if not fitted_path.exists():
        fitted_path.write_text(json.dumps(params, indent=2))

    start = args.start_replicate_index
    for i in range(start, start + args.n_replicates):
        _simulate_one_replicate(
            rep_dir=args.out_dir / f"replicate_{i}",
            rep_index=i,
            params=params,
            model_type=model_type,
            cfg=cfg,
            engine=engine,
            base_seed=base_seed,
            coverage_percent=args.coverage_percent,
        )

    print(f"✓ calibration simulation done -> {args.out_dir} "
          f"(replicates {start}..{start + args.n_replicates - 1})")


if __name__ == "__main__":
    main()

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
# task per replicate, run concurrently by SLURM (see
# bash_scripts/simulation/calibration_simulate.sh). For an ad hoc multi-replicate run in one process,
# just pass a larger --n-replicates.
#
# Reuses src.simulation.simulation()/create_SFS() -- the same functions the
# sim-generation pipeline (run_one_simulation_to_dir) uses -- just fed an
# explicit params dict instead of a prior draw. All physical simulation
# parameters (sequence_length, mutation_rate, recombination_rate, num_samples,
# engine) come from --config, same as every other pipeline stage.
#
# Calibration-only overrides (never seen by the training simulations): any
# keys in the config's "calibration" block replace the top-level ones for
# this script only, e.g.
#   "calibration": {"recombination": {"type": "map", "file": ..., "region": [s, e]}}
# (see src/simulation.py::simulation_runner and src/bgs_intervals.py::_contig_from_cfg).
# --observed-sfs sets num_samples to the observed SFS's sample sizes (so the
# simulated sample always matches the data), and --sfs-meta sets
# simulation_kept_fraction to the fraction of sites the real data kept
# (sequence_length / region_length; replacing the config's value), so the
# simulated region's mutation rate is thinned to yield the same expected SNP
# count as the effective length the fit used.

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
                      help="JSON file with fitted params. Accepts a flat "
                           "{param: value} object, a predict_real_data.py output "
                           "({'predictions': {param: value}, ...}), or a real-data fit "
                           "summary like sfs_fit.json ({'best_params': {param: value}, ...}).")
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
    ap.add_argument("--observed-sfs", type=Path, default=None,
                     help="Observed SFS pickle: num_samples is set to its sample sizes "
                          "(pop order from config num_samples).")
    ap.add_argument("--sfs-meta", type=Path, default=None,
                     help="Observed SFS meta JSON: simulation_kept_fraction is set to "
                          "sequence_length / region_length (fraction of sites kept).")
    ap.add_argument("--no-trees", action="store_true",
                     help="Don't save tree_sequence.trees (SFS-only checks don't need it; "
                          "whole-chromosome tree sequences are large).")
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
    elif isinstance(raw, dict) and isinstance(raw.get("best_params"), dict):
        raw = raw["best_params"]
    if not isinstance(raw, dict):
        raise SystemExit("Fitted params must be a JSON object of {param: value}.")
    return {k: float(v) for k, v in raw.items()}


def _simulate_one_replicate(*, rep_dir, rep_index, params, model_type, cfg, engine,
                             base_seed, coverage_percent, save_trees=True):
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

    if save_trees:
        ts.dump(rep_dir / "tree_sequence.trees")
    (rep_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    (rep_dir / "meta.json").write_text(json.dumps({
        "model_type": model_type,
        "engine": engine,
        "replicate_index": rep_index,
        "seed": replicate_seed,
        "coverage_percent": coverage,
        "num_samples": cfg["num_samples"],
        "sample_ploidy": cfg.get("sample_ploidy", 2),
        "mutation_rate": cfg["mutation_rate"],
        "simulation_kept_fraction": cfg.get("simulation_kept_fraction", 1.0),
        "recombination": cfg.get("recombination") or {"type": "flat", "rate": cfg.get("recombination_rate")},
        "sequence_length": float(ts.sequence_length),
        "params": params,
    }, indent=2))

    print(f"[replicate {rep_index}] wrote {rep_dir}/{'tree_sequence.trees + ' if save_trees else ''}SFS.pkl "
          f"(seed={replicate_seed}, sum(SFS)={float(np.asarray(sfs).sum()):.6g})")


def main() -> None:
    args = _parse_args()
    cfg = json.loads(args.config.read_text())
    model_type = args.model_type or cfg["demographic_model"]

    calib = cfg.get("calibration") or {}
    cfg = {**cfg, **calib}
    if args.observed_sfs is not None:
        with open(args.observed_sfs, "rb") as fh:
            obs = pickle.load(fh)
        cfg["num_samples"] = {p: int(n) - 1 for p, n in zip(cfg["num_samples"], obs.shape)}
    if args.sfs_meta is not None:
        meta = json.loads(args.sfs_meta.read_text())
        cfg["simulation_kept_fraction"] = float(meta["sequence_length"]) / float(meta["region_length"])
        print(f"simulation_kept_fraction from {args.sfs_meta}: {cfg['simulation_kept_fraction']:.4f} "
              f"-> simulated mutation rate {float(cfg['mutation_rate']) * cfg['simulation_kept_fraction']:.4g}")
    print(f"calibration overrides: {sorted(calib)}; num_samples={cfg['num_samples']}, "
          f"sample_ploidy={cfg.get('sample_ploidy', 2)}")

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
            save_trees=not args.no_trees,
        )

    print(f"✓ calibration simulation done -> {args.out_dir} "
          f"(replicates {start}..{start + args.n_replicates - 1})")


if __name__ == "__main__":
    main()

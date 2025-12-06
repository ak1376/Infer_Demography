#!/usr/bin/env python3
"""
Simulate one windowed replicate and write:
  - window_<idx>.trees
  - window_<idx>.vcf.gz
  - samples.txt
  - flat_map.txt

This script mirrors simulation.py's engine switch:
  - engine == "msprime" → msprime_simulation(...)
  - engine == "slim"    → stdpopsim_slim_simulation(...)
and reuses those functions directly (no re-implementation here).
"""
from __future__ import annotations
import argparse, json, pickle, sys, gzip, shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import msprime
import demes

# ------------------------------------------------------------------ local imports
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import (
    bottleneck_model,
    split_isolation_model,
    split_migration_model,
    drosophila_three_epoch,
    split_migration_growth_model,
    msprime_simulation,          # reuse neutral path
    stdpopsim_slim_simulation,   # reuse SLiM/BGS path
)

# ────────────────────────────────────────────────────────────────────────────────
# helpers for small artifacts we keep next to each window
# ────────────────────────────────────────────────────────────────────────────────


def write_samples_and_map(
    *, L: int, r: float, samples: Dict[str, int], out_dir: Path
) -> None:
    # samples.txt
    lines = ["sample\tpop"]
    tsk_i = 0
    for pop, n in samples.items():
        for _ in range(int(n)):
            lines.append(f"tsk_{tsk_i}\t{pop}")
            tsk_i += 1
    (out_dir / "samples.txt").write_text("\n".join(lines) + "\n")

    # flat_map.txt (simple linear map for plotting/debug)
    cm_total = r * L * 100.0
    (out_dir / "flat_map.txt").write_text(f"pos\tMap(cM)\n0\t0\n{L}\t{cm_total}\n")


def demes_from_model(model_id: str, sampled: Dict[str, float]) -> demes.Graph:
    if model_id == "bottleneck":
        return bottleneck_model(sampled)
    if model_id == "split_isolation":
        return split_isolation_model(sampled)
    if model_id == "split_migration":
        return split_migration_model(sampled)
    if model_id == "drosophila_three_epoch":
        return drosophila_three_epoch(sampled)
    if model_id == "split_migration_growth":
        return split_migration_growth_model(sampled)
    raise ValueError(f"Unsupported demographic_model: {model_id}")


def simulate_window(
    *,
    graph: demes.Graph,
    cfg: Dict[str, Any],
    sampled_cov: Optional[float],
    sampled_params: Dict[str, float],
    rep_index: int,
) -> Tuple[msprime.TreeSequence, Dict]:
    """
    Delegate to the same engine-specific code paths as simulation.py:
      - engine == "msprime" → msprime_simulation(...)
      - engine == "slim"    → stdpopsim_slim_simulation(...)

    Returns:
        tuple: (TreeSequence, window_metadata_dict)
    """
    engine = cfg["engine"]
    model_type = cfg["demographic_model"]
    sel_cfg = cfg.get("selection") or {}

    # Create a modified config with unique seed for this window
    window_cfg = cfg.copy()
    base_seed = cfg.get("seed")
    window_seed = None
    if base_seed is not None:
        # Generate unique seed for this replicate: base_seed + rep_index * 10000
        # The * 10000 ensures seeds don't overlap between simulations and windows
        window_seed = int(base_seed) + rep_index * 10000
        window_cfg["seed"] = window_seed
        print(
            f"• Window {rep_index}: using seed {window_seed} (base: {base_seed} + {rep_index} * 10000)"
        )
    else:
        print(f"• Window {rep_index}: no seed specified, using random state")

    # Create window metadata for storage
    window_metadata = {
        "window_index": rep_index,
        "engine": engine,
        "model_type": model_type,
        "base_seed": base_seed,
        "window_seed": window_seed,
        "genome_length": float(cfg["genome_length"]),
        "mutation_rate": float(cfg["mutation_rate"]),
        "recombination_rate": float(cfg["recombination_rate"]),
        "num_samples": {k: int(v) for k, v in cfg["num_samples"].items()},
        "sampled_params": sampled_params,
        "sampled_coverage": sampled_cov,
        "selection_enabled": (
            bool(sel_cfg.get("enabled", False)) if engine == "slim" else False
        ),
    }

    if engine == "msprime":
        ts, _ = msprime_simulation(graph, window_cfg)
        window_metadata["sequence_length"] = float(ts.sequence_length)
        return ts, window_metadata

    if engine == "slim":
        if not bool(sel_cfg.get("enabled", False)):
            raise RuntimeError(
                "engine='slim' requires selection.enabled=true in config."
            )
        if sampled_cov is None:
            raise RuntimeError("engine='slim' requires a coverage value (from meta).")
        ts, _ = stdpopsim_slim_simulation(
            g=graph,
            experiment_config=window_cfg,
            sampled_coverage=sampled_cov,  # fraction (<=1) or percent (>1) are both supported by your code
            model_type=model_type,
            sampled_params=sampled_params,
        )

        # Add BGS-specific metadata
        sel_summary = getattr(ts, "_bgs_selection_summary", {}) or {}
        window_metadata.update(
            {
                "sequence_length": float(ts.sequence_length),
                "species": str(sel_cfg.get("species", "HomSap")),
                "dfe_id": str(sel_cfg.get("dfe_id", "Gamma_K17")),
                "selected_bp": int(sel_summary.get("selected_bp", 0)),
                "selected_frac": float(sel_summary.get("selected_frac", 0.0)),
                "slim_scaling": float(sel_cfg.get("slim_scaling", 10.0)),
                "slim_burn_in": float(sel_cfg.get("slim_burn_in", 5.0)),
            }
        )
        return ts, window_metadata

    raise ValueError("engine must be 'slim' or 'msprime'.")


def load_sampled_coverage_from_meta(meta_file: Optional[Path]) -> Optional[float]:
    """
    Pull the exact same coverage used in the base BGS sim, preferring:
      sampled_coverage_percent  >  target_coverage_frac / coverage_fraction  >  selected_frac
    Returns None if meta_file is not provided.
    """
    if not meta_file:
        return None
    if not meta_file.exists():
        raise FileNotFoundError(f"--meta-file not found: {meta_file}")

    meta = json.loads(meta_file.read_text())
    if (
        "sampled_coverage_percent" in meta
        and meta["sampled_coverage_percent"] is not None
    ):
        return float(meta["sampled_coverage_percent"])  # percent
    if "target_coverage_frac" in meta and meta["target_coverage_frac"] is not None:
        return float(meta["target_coverage_frac"])  # fraction
    if "coverage_fraction" in meta and meta["coverage_fraction"] is not None:
        return float(meta["coverage_fraction"])  # fraction
    if "selected_frac" in meta and meta["selected_frac"] is not None:
        return float(meta["selected_frac"])  # fraction (realized)
    raise RuntimeError(
        "Meta file is missing coverage fields. "
        "Expected one of: sampled_coverage_percent | target_coverage_frac | coverage_fraction | selected_frac"
    )


def write_outputs(
    *,
    ts: msprime.TreeSequence,
    out_dir: Path,
    rep_index: int,
    genome_length: int,
    recomb_rate: float,
    samples_cfg: Dict[str, int],
    window_metadata: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # .trees
    ts_file = out_dir / f"window_{rep_index}.trees"
    ts.dump(ts_file)

    # .vcf.gz
    raw_vcf = out_dir / f"window_{rep_index}.vcf"
    with raw_vcf.open("w") as fh:
        ts.write_vcf(fh, allow_position_zero=True)
    with raw_vcf.open("rb") as f_in, gzip.open(f"{raw_vcf}.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    raw_vcf.unlink()

    # samples + flat map
    write_samples_and_map(
        L=int(genome_length),
        r=float(recomb_rate),
        samples={k: int(v) for k, v in samples_cfg.items()},
        out_dir=out_dir,
    )

    # window metadata (includes seed information for reproducibility)
    window_meta_file = out_dir / f"window_{rep_index}.meta.json"
    window_meta_file.write_text(json.dumps(window_metadata, indent=2))

    rel = (
        ts_file.relative_to(out_dir.parent.parent)
        if out_dir.parent.parent in ts_file.parents
        else ts_file.name
    )
    print(f"✓ replicate {rep_index:04d} → {rel} + metadata")


def main() -> None:
    cli = argparse.ArgumentParser("simulate one windowed replicate (neutral or BGS)")
    cli.add_argument(
        "--sim-dir", required=True, type=Path, help="directory with sampled_params.pkl"
    )
    cli.add_argument("--rep-index", required=True, type=int)
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--out-dir", required=True, type=Path)
    cli.add_argument(
        "--meta-file",
        type=Path,
        required=False,
        help="bgs.meta.json from base simulation (to reuse exact coverage)",
    )
    args = cli.parse_args()

    cfg: Dict[str, Any] = json.loads(args.config_file.read_text())
    samp: Dict[str, float] = pickle.load(
        (args.sim_dir / "sampled_params.pkl").open("rb")
    )

    # Build Demes graph from model + sampled params
    graph = demes_from_model(cfg["demographic_model"], samp)

    # Reuse exact coverage for BGS (if engine=slim)
    sampled_cov = (
        load_sampled_coverage_from_meta(args.meta_file)
        if cfg["engine"] == "slim"
        else None
    )

    # Simulate via engine switch (reusing simulation.py functions)
    ts, window_metadata = simulate_window(
        graph=graph,
        cfg=cfg,
        sampled_cov=sampled_cov,
        sampled_params=samp,
        rep_index=args.rep_index,
    )

    # Persist outputs
    write_outputs(
        ts=ts,
        out_dir=args.out_dir,
        rep_index=args.rep_index,
        genome_length=int(cfg["genome_length"]),
        recomb_rate=float(cfg["recombination_rate"]),
        samples_cfg={k: int(v) for k, v in cfg["num_samples"].items()},
        window_metadata=window_metadata,
    )


if __name__ == "__main__":
    main()

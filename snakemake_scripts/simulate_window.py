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
from typing import Dict, Any, Optional

import msprime
import demes

# ------------------------------------------------------------------ local imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulation import (
    bottleneck_model,
    split_isolation_model,
    split_migration_model,
    drosophila_three_epoch,
    msprime_simulation,            # reuse neutral path
    stdpopsim_slim_simulation,     # reuse SLiM/BGS path
)

# ────────────────────────────────────────────────────────────────────────────────
# helpers for small artifacts we keep next to each window
# ────────────────────────────────────────────────────────────────────────────────

def write_samples_and_map(*, L: int, r: float, samples: Dict[str, int], out_dir: Path) -> None:
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
    raise ValueError(f"Unsupported demographic_model: {model_id}")


def simulate_window(
    *,
    graph: demes.Graph,
    cfg: Dict[str, Any],
    sampled_cov: Optional[float],
    sampled_params: Dict[str, float],
) -> msprime.TreeSequence:
    """
    Delegate to the same engine-specific code paths as simulation.py:
      - engine == "msprime" → msprime_simulation(...)
      - engine == "slim"    → stdpopsim_slim_simulation(...)
    """
    engine = cfg["engine"]
    model_type = cfg["demographic_model"]
    sel_cfg = cfg.get("selection") or {}

    if engine == "msprime":
        ts, _ = msprime_simulation(graph, cfg)
        return ts

    if engine == "slim":
        if not bool(sel_cfg.get("enabled", False)):
            raise RuntimeError("engine='slim' requires selection.enabled=true in config.")
        if sampled_cov is None:
            raise RuntimeError("engine='slim' requires a coverage value (from meta).")
        ts, _ = stdpopsim_slim_simulation(
            g=graph,
            experiment_config=cfg,
            sampled_coverage=sampled_cov,   # fraction (<=1) or percent (>1) are both supported by your code
            model_type=model_type,
            sampled_params=sampled_params,
        )
        return ts

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
    if "sampled_coverage_percent" in meta and meta["sampled_coverage_percent"] is not None:
        return float(meta["sampled_coverage_percent"])  # percent
    if "target_coverage_frac" in meta and meta["target_coverage_frac"] is not None:
        return float(meta["target_coverage_frac"])      # fraction
    if "coverage_fraction" in meta and meta["coverage_fraction"] is not None:
        return float(meta["coverage_fraction"])         # fraction
    if "selected_frac" in meta and meta["selected_frac"] is not None:
        return float(meta["selected_frac"])             # fraction (realized)
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

    rel = ts_file.relative_to(out_dir.parent.parent) if out_dir.parent.parent in ts_file.parents else ts_file.name
    print(f"✓ replicate {rep_index:04d} → {rel}")


def main() -> None:
    cli = argparse.ArgumentParser("simulate one windowed replicate (neutral or BGS)")
    cli.add_argument("--sim-dir", required=True, type=Path, help="directory with sampled_params.pkl")
    cli.add_argument("--rep-index", required=True, type=int)
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--out-dir", required=True, type=Path)
    cli.add_argument("--meta-file", type=Path, required=False,
                     help="bgs.meta.json from base simulation (to reuse exact coverage)")
    args = cli.parse_args()

    cfg: Dict[str, Any] = json.loads(args.config_file.read_text())
    samp: Dict[str, float] = pickle.load((args.sim_dir / "sampled_params.pkl").open("rb"))

    # Build Demes graph from model + sampled params
    graph = demes_from_model(cfg["demographic_model"], samp)

    # Reuse exact coverage for BGS (if engine=slim)
    sampled_cov = load_sampled_coverage_from_meta(args.meta_file) if cfg["engine"] == "slim" else None

    # Simulate via engine switch (reusing simulation.py functions)
    ts = simulate_window(
        graph=graph,
        cfg=cfg,
        sampled_cov=sampled_cov,
        sampled_params=samp,
    )

    # Persist outputs
    write_outputs(
        ts=ts,
        out_dir=args.out_dir,
        rep_index=args.rep_index,
        genome_length=int(cfg["genome_length"]),
        recomb_rate=float(cfg["recombination_rate"]),
        samples_cfg={k: int(v) for k, v in cfg["num_samples"].items()},
    )


if __name__ == "__main__":
    main()

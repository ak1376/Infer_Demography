#!/usr/bin/env python3
"""
Standalone simulator + cache (BGS only)

Generates one SLiM/stdpopsim simulation (tree-sequence + SFS) for the chosen model
and stores artefacts under <simulation-dir>/<simulation-number>/.

Requires experiment_config["selection"]["enabled"] = true.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import demesdraw
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulation import (  # noqa: E402
    simulation,  # runs BGS using your demes graph + SLiM engine
    create_SFS,  # builds a moments.Spectrum from the ts
)


# ------------------------------------------------------------------
# parameter sampling helper
# ------------------------------------------------------------------
def sample_params(
    priors: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    params = {k: float(rng.uniform(*bounds)) for k, bounds in priors.items()}
    # keep bottleneck start > end if both are present
    if {"t_bottleneck_start", "t_bottleneck_end"}.issubset(params) and params[
        "t_bottleneck_start"
    ] <= params["t_bottleneck_end"]:
        params["t_bottleneck_start"], params["t_bottleneck_end"] = (
            params["t_bottleneck_end"],
            params["t_bottleneck_start"],
        )
    return params


def sample_coverage(
    selection_cfg: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> float:
    """
    Sample a coverage percentage from the prior specified in selection_cfg.

    Parameters
    ----------
    selection_cfg : dict
        Should contain key "coverage_percent" with [low, high] bounds.
    rng : np.random.Generator, optional
        Random number generator to use. Defaults to np.random.default_rng().

    Returns
    -------
    float
        Sampled coverage percentage.
    """
    rng = rng or np.random.default_rng()
    low, high = selection_cfg["coverage_percent"]
    return float(rng.uniform(low, high))


# ------------------------------------------------------------------
# main workflow
# ------------------------------------------------------------------
def run_simulation(
    simulation_dir: Path,
    experiment_config: Path,
    model_type: str,
    simulation_number: Optional[str] = None,
):
    cfg: Dict[str, object] = json.loads(experiment_config.read_text())
    sel_cfg = cfg.get("selection") or {}
    if not sel_cfg.get("enabled", False):
        raise ValueError(
            "This runner is BGS-only. Set selection.enabled = true in your config."
        )

    # decide destination folder name
    if simulation_number is None:
        existing = {int(p.name) for p in simulation_dir.glob("[0-9]*") if p.is_dir()}
        simulation_number = f"{max(existing, default=0) + 1:04d}"
    out_dir = simulation_dir / simulation_number
    out_dir.mkdir(parents=True, exist_ok=True)

    # simulate (seeded sampling for reproducibility)
    sampled_params = sample_params(cfg["priors"])
    sampled_coverage = sample_coverage(cfg["selection"])  # percent (e.g. 37.4)
    print(f"• sampled coverage: {sampled_coverage:.2f}%")

    # Run SLiM/stdpopsim path through src/simulation.simulation(...)
    ts, g = simulation(sampled_params, model_type, cfg, sampled_coverage)
    sfs = create_SFS(ts)

    # save artefacts
    (out_dir / "sampled_params.pkl").write_bytes(pickle.dumps(sampled_params))
    (out_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts.dump(out_dir / "tree_sequence.trees")

    # demography plot (always your demes graph)
    fig_path = out_dir / "demes.png"
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("N")
    ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # metadata sidecar (now includes the ACTUAL sampled coverage,
    # plus realized selection tiling summary from ts._bgs_selection_summary)
    sel_summary = getattr(ts, "_bgs_selection_summary", {}) or {}

    # build a JSON-serializable dict (cast everything to primitives)
    meta = dict(
        selection=True,
        species=str(sel_cfg.get("species", "HomSap")),
        dfe_id=str(sel_cfg.get("dfe_id", "Gamma_K17")),
        # Real-window keys (None if unused)
        chromosome=sel_cfg.get("chromosome"),
        left=sel_cfg.get("left"),
        right=sel_cfg.get("right"),
        genetic_map=sel_cfg.get("genetic_map"),
        # Synthetic-contig parameters
        genome_length=float(cfg.get("genome_length")),
        mutation_rate=float(cfg.get("mutation_rate")),
        recombination_rate=float(cfg.get("recombination_rate")),
        # BGS tiling / coverage knobs from config (if present)
        coverage_fraction=(
            None
            if sel_cfg.get("coverage_fraction") is None
            else float(sel_cfg.get("coverage_fraction"))
        ),
        coverage_percent=(
            None
            if sel_cfg.get("coverage_percent") is None
            else [
                float(sel_cfg["coverage_percent"][0]),
                float(sel_cfg["coverage_percent"][1]),
            ]
        ),
        exon_bp=int(sel_cfg.get("exon_bp", 200)),
        jitter_bp=int(sel_cfg.get("jitter_bp", 0)),
        tile_bp=(None if sel_cfg.get("tile_bp") is None else int(sel_cfg["tile_bp"])),
        # Realized selection span (after interval building)
        selected_bp=int(sel_summary.get("selected_bp", 0)),
        selected_frac=float(sel_summary.get("selected_frac", 0.0)),
        # CRITICAL: persist the actual coverage you sampled for this sim
        sampled_coverage_percent=float(sampled_coverage),
        # also store a fraction version for downstream (window) scripts
        target_coverage_frac=(
            float(sampled_coverage) / 100.0
            if float(sampled_coverage) > 1.0
            else float(sampled_coverage)
        ),
        # SLiM options
        slim_scaling=float(sel_cfg.get("slim_scaling", 10.0)),
        slim_burn_in=float(sel_cfg.get("slim_burn_in", 5.0)),
        # misc
        num_samples={k: int(v) for k, v in (cfg.get("num_samples") or {}).items()},
        seed=(None if cfg.get("seed") is None else int(cfg.get("seed"))),
        sequence_length=float(ts.sequence_length),
        tree_sequence=str(out_dir / "tree_sequence.trees"),
        model_type=str(model_type),
        # sampled priors (floats only)
        sampled_params={k: float(v) for k, v in sampled_params.items()},
    )

    (out_dir / "bgs.meta.json").write_text(json.dumps(meta, indent=2))

    # friendly path for log message
    try:
        rel = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_dir
    print(f"✓ simulation written to {rel}")


# ------------------------------------------------------------------
# argparse entry-point
# ------------------------------------------------------------------
def main():
    cli = argparse.ArgumentParser(description="Generate one BGS simulation")
    cli.add_argument(
        "--simulation-dir",
        type=Path,
        required=True,
        help="Base directory that will hold <number>/ subfolders",
    )
    cli.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="JSON config with priors, genome length (or real window), etc.",
    )
    cli.add_argument(
        "--model-type",
        required=True,
        choices=[
            "bottleneck",
            "split_isolation",
            "split_migration",
            "drosophila_three_epoch",
        ],  # ← added drosophila
        help="Which demographic model to simulate",
    )
    cli.add_argument(
        "--simulation-number",
        type=str,
        help="Folder name to create (e.g. '0005').  If omitted, the next free index is used.",
    )
    args = cli.parse_args()
    run_simulation(
        args.simulation_dir,
        args.experiment_config,
        args.model_type,
        args.simulation_number,
    )


if __name__ == "__main__":
    main()

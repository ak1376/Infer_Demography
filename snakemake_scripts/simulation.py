#!/usr/bin/env python3
"""Standalone simulator + cache

Generates one simulation (tree-sequence + SFS) for the chosen model and
stores all artefacts under <simulation-dir>/<simulation-number>/.

Now supports Background Selection (BGS) via stdpopsim when
experiment_config["selection"]["enabled"] is True.
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
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulation import (  # noqa: E402
    bottleneck_model,
    split_isolation_model,
    split_migration_model,
    drosophila_three_epoch,
    simulation,
    create_SFS
)

# ---- local helper to reconstruct exon tiling for metadata/BED when BGS ----
def _build_tiling_intervals(L: int, exon_bp: int = 200, tile_bp: int = 1000) -> np.ndarray:
    starts = np.arange(0, max(0, int(L) - exon_bp + 1), tile_bp, dtype=int)
    ends   = np.minimum(starts + exon_bp, int(L)).astype(int)
    return np.column_stack([starts, ends])


# ------------------------------------------------------------------
# parameter sampling helper
# ------------------------------------------------------------------
def sample_params(priors: Dict[str, List[float]], *,
                  rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    params = {k: float(rng.uniform(*bounds)) for k, bounds in priors.items()}
    if {"t_bottleneck_start", "t_bottleneck_end"}.issubset(params) and \
       params["t_bottleneck_start"] <= params["t_bottleneck_end"]:
        params["t_bottleneck_start"], params["t_bottleneck_end"] = (
            params["t_bottleneck_end"], params["t_bottleneck_start"])
    return params


# ------------------------------------------------------------------
# main workflow
# ------------------------------------------------------------------
def run_simulation(simulation_dir: Path, experiment_config: Path, model_type: str,
                   simulation_number: Optional[str] = None):
    cfg: Dict[str, object] = json.loads(experiment_config.read_text())
    rng = np.random.default_rng(cfg.get("seed"))

    # decide destination folder name
    if simulation_number is None:
        existing = {int(p.name) for p in simulation_dir.glob("[0-9]*") if p.is_dir()}
        simulation_number = f"{max(existing, default=0) + 1:04d}"
    out_dir = simulation_dir / simulation_number
    out_dir.mkdir(parents=True, exist_ok=True)

    # simulate
    sampled_params = sample_params(cfg["priors"])
    ts, g = simulation(sampled_params, model_type, cfg)   # g may be None for BGS
    sfs   = create_SFS(ts)

    # save artefacts
    (out_dir / "sampled_params.pkl").write_bytes(pickle.dumps(sampled_params))
    (out_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts.dump(out_dir / "tree_sequence.trees")

    # Always produce a demography image to satisfy Snakemake
    # plot demography (neutral demes graph, or stdpopsim demography when BGS)
    fig_path = out_dir / "demes.png"
    sel_cfg = (cfg.get("selection") or {})
    if g is not None:
        # Neutral/msprime path: plot your demes graph
        ax = demesdraw.tubes(g)
        ax.set_xlabel("Time (generations)")
        ax.set_ylabel("N")
        ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
    elif sel_cfg.get("enabled", False):
        # BGS/stdpopsim path: plot the stdpopsim catalog demography you used
        import stdpopsim
        sp  = stdpopsim.get_species(sel_cfg.get("species", "HomSap"))
        dm  = sp.get_demographic_model(sel_cfg.get("demography_id", "OutOfAfrica_3G09"))
        # stdpopsim models expose an underlying object with a demes export; try both common APIs
        g_dem = None
        for attr in ("to_demes", "model"):
            obj = getattr(dm, attr, None)
            if obj is None:
                continue
            try:
                g_dem = obj.to_demes()
                break
            except Exception:
                pass
        if g_dem is None:
            # last-resort tiny placeholder (shouldn't happen on current stdpopsim)
            plt.figure(figsize=(3, 2)); plt.axis("off")
            plt.title(f"{dm.id} demography (plot error)")
            plt.tight_layout(); plt.savefig(fig_path, dpi=200); plt.close()
        else:
            ax = demesdraw.tubes(g_dem)
            ax.set_xlabel("Time (generations)")
            ax.set_ylabel("N")
            ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
    else:
        # Shouldn’t hit this branch, but keep a harmless placeholder
        plt.figure(figsize=(3, 2)); plt.axis("off")
        plt.title("demes plot not available")
        plt.tight_layout(); plt.savefig(fig_path, dpi=200); plt.close()

    # If selection is enabled, emit helpful sidecar files for diagnostics
    sel_cfg = (cfg.get("selection") or {})
    if sel_cfg.get("enabled", False):
        outroot   = (out_dir / "tree_sequence").with_suffix("")  # experiments/.../<sid>/tree_sequence
        meta_path = out_dir / "bgs.meta.json"
        bed_path  = out_dir / "bgs.exons.bed"

        # metadata summary
        meta = dict(
            selection=True,
            species=sel_cfg.get("species", "HomSap"),
            demography_id=sel_cfg.get("demography_id", "OutOfAfrica_3G09"),
            dfe_id=sel_cfg.get("dfe_id", "Gamma_K17"),
            exon_bp=int(sel_cfg.get("exon_bp", 200)),
            tile_bp=int(sel_cfg.get("tile_bp", 5000)),
            slim_scaling=float(sel_cfg.get("slim_scaling", 10.0)),
            slim_burn_in=float(sel_cfg.get("slim_burn_in", 5.0)),
            genome_length=float(cfg["genome_length"]),
            mutation_rate=float(cfg["mutation_rate"]),
            recombination_rate=float(cfg["recombination_rate"]),
            num_samples=cfg["num_samples"],
            seed=cfg.get("seed"),
            trees=str(out_dir / "tree_sequence.trees"),
        )
        meta_path.write_text(json.dumps(meta, indent=2))

        # write exon tiling used for BGS (so you can overlay later)
        intervals = _build_tiling_intervals(
            int(cfg["genome_length"]),
            exon_bp=meta["exon_bp"],
            tile_bp=meta["tile_bp"]
        )
        with open(bed_path, "w") as bed:
            for s, e in intervals:
                bed.write(f"chr1\t{s}\t{e}\n")

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
    cli = argparse.ArgumentParser(description="Generate one demographic simulation")
    cli.add_argument("--simulation-dir", type=Path, required=True,
                     help="Base directory that will hold <number>/ subfolders")
    cli.add_argument("--experiment-config", type=Path, required=True,
                     help="JSON config with priors, genome length, etc.")
    cli.add_argument("--model-type", required=True,
                     choices=["bottleneck", "split_isolation", "split_migration",
                              "drosophila_three_epoch"],
                     help="Which demographic model to simulate")
    cli.add_argument("--simulation-number", type=str,
                     help="Folder name to create (e.g. '0005').  If omitted, the next free index is used.")
    args = cli.parse_args()
    run_simulation(args.simulation_dir, args.experiment_config,
                   args.model_type, args.simulation_number)


if __name__ == "__main__":
    main()

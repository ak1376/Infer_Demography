#!/usr/bin/env python3
"""Simulate **one replicate** (formerly called “window”) and write it as
`bottleneck.<idx>.vcf.gz` in the final Windows directory.

CLI (all required)
------------------
--sim-dir        ld_experiments/bottleneck/simulations/<sid>
--rep-index      42                              (zero‑based)
--config-file    config_files/experiment_config_bottleneck.json
--out-dir        MomentsLD/LD_stats/sim_<sid>/windows

Effect
------
Reads `<sim-dir>/sampled_params.pkl`, builds the bottleneck demography,
then simulates **exactly one** ancestry + mutation replicate using
`msprime.sim_ancestry(..., num_replicates=1)` so the logic is identical
to your `run_msprime_reps` helper – just scoped to a single replicate.

The result is written to
    <out-dir>/bottleneck.<idx>.vcf.gz
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Any

import msprime

# ------------------------------------------------------------------ local imports
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulation import bottleneck_model  # noqa: E402

# ------------------------------------------------------------------ main

def main() -> None:
    cli = argparse.ArgumentParser("simulate one msprime replicate")
    cli.add_argument("--sim-dir",     required=True, type=Path,
                     help="ld_experiments/.../simulations/<sid>")
    cli.add_argument("--rep-index",   required=True, type=int,
                     help="0‑based replicate index")
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--out-dir",     required=True, type=Path,
                     help="MomentsLD/LD_stats/sim_<sid>/windows")
    args = cli.parse_args()

    # --- load cfg & sampled parameters -----------------------------
    cfg:  Dict[str, Any] = json.loads(args.config_file.read_text())
    samp: Dict[str, Any] = pickle.load((args.sim_dir / "sampled_params.pkl").open("rb"))

    # --- build demography -----------------------------------------
    graph = bottleneck_model(samp) #TODO: Change later to be generalizable to different demographic models
    demog = msprime.Demography.from_demes(graph)

    # --- simulate one replicate -----------------------------------
    ts = msprime.sim_ancestry(
        {"N0": cfg["num_samples"]["N0"]}, #TODO: Change later to be generalizable to different demographic models
        demography=demog,
        sequence_length=cfg["genome_length"],
        recombination_rate=cfg["recombination_rate"],
        random_seed=args.rep_index + 17,   # unique seed per replicate
    )
    ts = msprime.sim_mutations(ts, rate=cfg["mutation_rate"],
                               random_seed=args.rep_index + 197)

    # --- write compressed VCF -------------------------------------
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vcf = out_dir / f"window_{args.rep_index}.vcf"
    with vcf.open("w") as fh:
        ts.write_vcf(fh, allow_position_zero=True)
    os.system(f"gzip -f {vcf}")

    print(f"✓ replicate {args.rep_index:04d} → {vcf.with_suffix('.vcf.gz').relative_to(out_dir.parent.parent)}")

if __name__ == "__main__":
    main()

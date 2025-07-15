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

from simulation import bottleneck_model, split_isolation_model, split_migration_model, drosophila_three_epoch

def write_samples_and_map(*, L: int, r: float,
                          samples: dict[str, int],
                          out_dir: Path) -> None:
    """
    samples = {"N0": 10}  or  {"N1": 10, "N2": 10}  etc.

    Produces:
        samples.txt      (ID = tsk_<running‑index>, POP = dict key)
        flat_map.txt     (simple two‑point map)
    """
    # ------------ samples.txt ------------
    lines = ["sample\tpop"]
    tsk_i = 0
    for pop, n in samples.items():          # keep the dict order
        for _ in range(n):
            lines.append(f"tsk_{tsk_i}\t{pop}")
            tsk_i += 1
    (out_dir / "samples.txt").write_text("\n".join(lines) + "\n")

    # ------------ flat_map.txt -----------
    cm_total = r * L * 100                  # 1 Mbp * r  (in cM)
    (out_dir / "flat_map.txt").write_text(
        f"pos\tMap(cM)\n0\t0\n{L}\t{cm_total}\n"
    )

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

    model = cfg["demographic_model"]

    if model == "bottleneck":                       # ⇢ 3‑epoch, 1‑population
        # three_epoch params: (nu1, nu2, T1, T2, Ne)
        demo_func = bottleneck_model
 
    elif model == "split_isolation":                # ⇢ 2‑pop, split then no mig
        demo_func = split_isolation_model

    elif model == "split_migration":                # ⇢ 2‑pop, split + mig 
        demo_func = split_migration_model
        # (nu1, nu2, T_split, m12, m21, Ne)

    elif model == "drosophila_three_epoch":         # ⇢ your stdpopsim wrapper 
        demo_func = drosophila_three_epoch
        # map however the wrapped function expects; an illustrative example:
  
    else:
        raise ValueError(f"Need p_guess mapping for model '{model}'")

    graph = demo_func(samp) #TODO: Change later to be generalizable to different demographic models
    demog = msprime.Demography.from_demes(graph)

    # --- simulate one replicate -----------------------------------
    samples_dict = {pop: int(n) for pop, n in cfg["num_samples"].items()}

    ts = msprime.sim_ancestry(
        samples_dict,                 # <‑‑ works for *any* demographic model
        demography=demog,
        sequence_length=cfg["genome_length"],
        recombination_rate=cfg["recombination_rate"],
        random_seed=args.rep_index + 17,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=cfg["mutation_rate"],
        random_seed=args.rep_index + 197,
    )

    # --- write compressed VCF -------------------------------------
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vcf = out_dir / f"window_{args.rep_index}.vcf"
    with vcf.open("w") as fh:
        ts.write_vcf(fh, allow_position_zero=True)
    os.system(f"gzip -f {vcf}")

    write_samples_and_map(
        L=cfg["genome_length"],
        r=cfg["recombination_rate"],
        samples=samples_dict,
        out_dir=out_dir,           # or sim_dir
    )

    print(f"✓ replicate {args.rep_index:04d} → {vcf.with_suffix('.vcf.gz').relative_to(out_dir.parent.parent)}")

if __name__ == "__main__":
    main()

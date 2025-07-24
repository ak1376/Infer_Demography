#!/usr/bin/env python3
"""Simulate **one replicate** and write:
    - window_<idx>.vcf.gz (compressed VCF)
    - window_<idx>.trees (tree sequence)
    - samples.txt / flat_map.txt
"""

from __future__ import annotations
import argparse, json, os, pickle, sys, gzip, shutil
from pathlib import Path
from typing import Dict, Any
import msprime

# ------------------------------------------------------------------ local imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulation import (
    bottleneck_model,
    split_isolation_model,
    split_migration_model,
    drosophila_three_epoch
)

def write_samples_and_map(*, L: int, r: float,
                          samples: dict[str, int],
                          out_dir: Path) -> None:
    # ------------ samples.txt ------------
    lines = ["sample\tpop"]
    tsk_i = 0
    for pop, n in samples.items():
        for _ in range(n):
            lines.append(f"tsk_{tsk_i}\t{pop}")
            tsk_i += 1
    (out_dir / "samples.txt").write_text("\n".join(lines) + "\n")

    # ------------ flat_map.txt -----------
    cm_total = r * L * 100
    (out_dir / "flat_map.txt").write_text(
        f"pos\tMap(cM)\n0\t0\n{L}\t{cm_total}\n"
    )

def main() -> None:
    cli = argparse.ArgumentParser("simulate one msprime replicate")
    cli.add_argument("--sim-dir", required=True, type=Path)
    cli.add_argument("--rep-index", required=True, type=int)
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--out-dir", required=True, type=Path)
    args = cli.parse_args()

    cfg: Dict[str, Any] = json.loads(args.config_file.read_text())
    samp: Dict[str, Any] = pickle.load((args.sim_dir / "sampled_params.pkl").open("rb"))

    model = cfg["demographic_model"]
    if model == "bottleneck":
        demo_func = bottleneck_model
    elif model == "split_isolation":
        demo_func = split_isolation_model
    elif model == "split_migration":
        demo_func = split_migration_model
    elif model == "drosophila_three_epoch":
        demo_func = drosophila_three_epoch
    else:
        raise ValueError(f"Unsupported model: {model}")

    graph = demo_func(samp)
    demog = msprime.Demography.from_demes(graph)
    samples_dict = {pop: int(n) for pop, n in cfg["num_samples"].items()}

    ts = msprime.sim_ancestry(
        samples_dict,
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

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save tree sequence (.trees) ---
    ts_file = out_dir / f"window_{args.rep_index}.trees"
    ts.dump(ts_file)

    # --- Save compressed VCF (.vcf.gz) using gzip module ---
    raw_vcf = out_dir / f"window_{args.rep_index}.vcf"
    with raw_vcf.open("w") as fh:
        ts.write_vcf(fh, allow_position_zero=True)

    with raw_vcf.open("rb") as f_in, gzip.open(f"{raw_vcf}.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    raw_vcf.unlink()  # delete uncompressed VCF

    # --- Save sample and map files ---
    write_samples_and_map(
        L=cfg["genome_length"],
        r=cfg["recombination_rate"],
        samples=samples_dict,
        out_dir=out_dir,
    )

    print(f"✓ replicate {args.rep_index:04d} → {ts_file.relative_to(out_dir.parent.parent)}")

if __name__ == "__main__":
    main()

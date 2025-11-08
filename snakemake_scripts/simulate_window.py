#!/usr/bin/env python3
"""Simulate **one replicate** and write:
- window_<idx>.vcf.gz (compressed VCF)
- window_<idx>.trees (tree sequence)
- samples.txt / flat_map.txt
"""
from __future__ import annotations
import argparse, json, pickle, sys, gzip, shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import msprime
import numpy as np

# --- add these for BGS ---
import stdpopsim as sps
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
    _contig_from_cfg,  # ← you already have this
    _apply_dfe_intervals,  # ← and this
)


def write_samples_and_map(
    *, L: int, r: float, samples: dict[str, int], out_dir: Path
) -> None:
    lines = ["sample\tpop"]
    tsk_i = 0
    for pop, n in samples.items():
        for _ in range(n):
            lines.append(f"tsk_{tsk_i}\t{pop}")
            tsk_i += 1
    (out_dir / "samples.txt").write_text("\n".join(lines) + "\n")

    cm_total = r * L * 100
    (out_dir / "flat_map.txt").write_text(f"pos\tMap(cM)\n0\t0\n{L}\t{cm_total}\n")


def _demes_from_model(model: str, sampled: Dict[str, float]) -> demes.Graph:
    if model == "bottleneck":
        return bottleneck_model(sampled)
    if model == "split_isolation":
        return split_isolation_model(sampled)
    if model == "split_migration":
        print(f"Demes Graph: {split_migration_model(sampled)}")
        return split_migration_model(sampled)
    if model == "drosophila_three_epoch":
        return drosophila_three_epoch(sampled)
    raise ValueError(f"Unsupported model: {model}")


def _simulate_neutral_msprime(
    *, graph: demes.Graph, cfg: Dict[str, Any], seed_base: int
) -> msprime.TreeSequence:
    demog = msprime.Demography.from_demes(graph)
    samples_dict = {pop: int(n) for pop, n in cfg["num_samples"].items()}
    ts = msprime.sim_ancestry(
        samples_dict,
        demography=demog,
        sequence_length=cfg["genome_length"],
        recombination_rate=cfg["recombination_rate"],
        random_seed=seed_base + 17,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=cfg["mutation_rate"],
        random_seed=seed_base + 197,
    )
    return ts


def _simulate_bgs_stdpopsim(
    *,
    graph: demes.Graph,
    cfg: Dict[str, Any],
    sel_cfg: Dict[str, Any],
    seed_base: int,
    sampled_cov_percent_or_frac: float,
) -> msprime.TreeSequence:
    """
    Simulate background selection (BGS) using stdpopsim + SLiM.
    Handles both symmetric and asymmetric IM models and ensures leaf-first ordering
    to avoid 'p0 (ANC) is zero' SLiM errors.
    """
    model_id = cfg["demographic_model"]

    if model_id == "split_isolation":
        N0 = float(samp.get("N_anc", samp.get("N0")))
        N1 = float(samp.get("N_YRI", samp.get("N1")))
        N2 = float(samp.get("N_CEU", samp.get("N2")))
        T = float(samp.get("T_split", samp.get("t_split")))
        m = float(
            samp.get("m", samp.get("m_YRI_CEU", samp.get("m12", samp.get("m21", 0.0))))
        )

        class _IM_Symmetric(sps.DemographicModel):
            def __init__(self, N0, N1, N2, T, m):
                dem = msprime.Demography()
                dem.add_population(name="YRI", initial_size=float(N1))
                dem.add_population(name="CEU", initial_size=float(N2))
                dem.add_population(name="ANC", initial_size=float(N0))
                dem.set_migration_rate(source="YRI", dest="CEU", rate=float(m))
                dem.set_migration_rate(source="CEU", dest="YRI", rate=float(m))
                dem.add_population_split(
                    time=float(T), ancestral="ANC", derived=["YRI", "CEU"]
                )
                super().__init__(
                    id="IM_sym",
                    description="IM symmetric",
                    long_description="",
                    model=dem,
                    generation_time=1,
                )

        model = _IM_Symmetric(N0, N1, N2, T, m)

    elif model_id == "split_migration":
        # --- Build msprime demography directly from graph ---
        dem = msprime.Demography()
        leaf_names = [d.name for d in graph.demes if d.end_time == 0]  # YRI, CEU
        nonleaf_names = [d.name for d in graph.demes if d.end_time != 0]  # ANC
        ordered = leaf_names + nonleaf_names

        # Add populations
        for name in ordered:
            deme = graph[name]
            size = deme.epochs[-1].start_size
            dem.add_population(name=name, initial_size=size)

        # Add migrations (directly from graph)
        if hasattr(graph, "migrations"):
            for mig in graph.migrations:
                dem.set_migration_rate(source=mig.source, dest=mig.dest, rate=mig.rate)

        # Add split event
        anc_deme = nonleaf_names[0]
        derived = leaf_names
        split_time = max(d.end_time for d in graph.demes if d.end_time != 0)
        dem.add_population_split(time=split_time, ancestral=anc_deme, derived=derived)

        print("Population order for SLiM:", [p.name for p in dem.populations])

        model = sps.DemographicModel(
            id="IM_asym",
            description="Isolation-with-migration, asymmetric",
            long_description="ANC splits at T into YRI and CEU; asymmetric migration.",
            model=dem,
            generation_time=1,
        )

    else:
        model = sps.DemographicModel(
            id="from_demes",
            description="",
            long_description="",
            model=msprime.Demography.from_demes(graph),
            generation_time=1,
        )

    # --- Build contig + apply DFE intervals ---
    contig = _contig_from_cfg(cfg, sel_cfg)
    _ = _apply_dfe_intervals(
        contig, sel_cfg, sampled_coverage=sampled_cov_percent_or_frac
    )

    # --- Run SLiM via stdpopsim engine ---
    samples = {k: int(v) for k, v in (cfg.get("num_samples") or {}).items()}
    eng = sps.get_engine("slim")
    ts = eng.simulate(
        model,
        contig,
        samples,
        slim_scaling_factor=float(sel_cfg.get("slim_scaling", 10.0)),
        slim_burn_in=float(sel_cfg.get("slim_burn_in", 5.0)),
    )

    return ts


def main() -> None:
    cli = argparse.ArgumentParser("simulate one replicate (neutral or BGS)")
    cli.add_argument("--sim-dir", required=True, type=Path)
    cli.add_argument("--rep-index", required=True, type=int)
    cli.add_argument("--config-file", required=True, type=Path)
    cli.add_argument("--out-dir", required=True, type=Path)
    # NEW: use previously recorded metadata, so we don't resample coverage
    cli.add_argument(
        "--meta-file",
        type=Path,
        required=False,
        help="bgs.meta.json from the base simulation (per sid)",
    )
    args = cli.parse_args()

    cfg: Dict[str, Any] = json.loads(args.config_file.read_text())
    global samp  # used in _simulate_bgs_stdpopsim for a quick extraction
    samp = pickle.load((args.sim_dir / "sampled_params.pkl").open("rb"))

    graph = _demes_from_model(cfg["demographic_model"], samp)
    sel_cfg: Dict[str, Any] = cfg.get("selection") or {}
    seed_base = int(cfg.get("seed", 0)) + int(args.rep_index)

    # --- If BGS is enabled, get the SAME coverage as before from the meta file ---
    sampled_cov: Optional[float] = None  # may be percent (>1) or fraction (<=1)
    if bool(sel_cfg.get("enabled", False)):
        if args.meta_file and args.meta_file.exists():
            meta = json.loads(args.meta_file.read_text())
            # Preferred exact fields if you added them previously:
            if (
                "sampled_coverage_percent" in meta
                and meta["sampled_coverage_percent"] is not None
            ):
                sampled_cov = float(meta["sampled_coverage_percent"])  # percent
            elif (
                "target_coverage_frac" in meta
                and meta["target_coverage_frac"] is not None
            ):
                sampled_cov = float(meta["target_coverage_frac"])  # fraction
            elif "coverage_fraction" in meta and meta["coverage_fraction"] is not None:
                sampled_cov = float(meta["coverage_fraction"])  # fraction
            elif "selected_frac" in meta and meta["selected_frac"] is not None:
                # last-resort: reuse realized coverage; close enough for consistency
                sampled_cov = float(meta["selected_frac"])  # fraction
        # If meta missing, fall back to config bounds once (no resampling per window!)
        # i.e., do nothing here; better to error than silently resample:
        if sampled_cov is None:
            raise RuntimeError(
                "BGS enabled but --meta-file missing required coverage fields. "
                "Add 'sampled_coverage_percent' (or 'target_coverage_frac'/'selected_frac') to bgs.meta.json."
            )

    # --- choose path: BGS vs neutral ---
    do_bgs = bool(sel_cfg.get("enabled", False))
    if do_bgs:
        # Pass through *exact same* coverage as used originally
        ts = _simulate_bgs_stdpopsim(
            graph=graph,
            cfg=cfg,
            sel_cfg=sel_cfg,
            seed_base=seed_base,
            sampled_cov_percent_or_frac=sampled_cov,
        )
    else:
        ts = _simulate_neutral_msprime(graph=graph, cfg=cfg, seed_base=seed_base)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save tree sequence (.trees) ---
    ts_file = out_dir / f"window_{args.rep_index}.trees"
    ts.dump(ts_file)

    # --- Save compressed VCF (.vcf.gz) ---
    raw_vcf = out_dir / f"window_{args.rep_index}.vcf"
    with raw_vcf.open("w") as fh:
        ts.write_vcf(fh, allow_position_zero=True)
    with raw_vcf.open("rb") as f_in, gzip.open(f"{raw_vcf}.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    raw_vcf.unlink()

    # --- Save sample and map files ---
    write_samples_and_map(
        L=int(cfg["genome_length"]),
        r=float(cfg["recombination_rate"]),
        samples={k: int(v) for k, v in cfg["num_samples"].items()},
        out_dir=out_dir,
    )

    print(
        f"✓ replicate {args.rep_index:04d} → {ts_file.relative_to(out_dir.parent.parent)}"
    )


if __name__ == "__main__":
    main()

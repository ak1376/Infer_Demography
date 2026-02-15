from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any

import json
import pickle
from pathlib import Path

import demes
import numpy as np
import stdpopsim as sps
import tskit
import moments

import demesdraw
import matplotlib.pyplot as plt

import sys
import gzip
import shutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demes_models import (  # noqa: E402
    bottleneck_model,
    IM_symmetric_model,
    IM_asymmetric_model,
    drosophila_three_epoch,
    split_migration_growth_model,
    OOA_three_pop_Gutenkunst,
    OOA_three_pop_model_simplified,
)
from src.bgs_intervals import _contig_from_cfg, _apply_dfe_intervals  # noqa: E402
from src.stdpopsim_wrappers import define_sps_model  # noqa: E402


# ============================================================================
# Sampling helpers (moved from script)
# ============================================================================

def sample_params(
    priors: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    params: Dict[str, float] = {}

    for k, bounds in priors.items():
        params[k] = float(rng.uniform(*bounds))

    # # keep bottleneck start > end if both are present
    # if {"t_bottleneck_start", "t_bottleneck_end"}.issubset(params):
    #     if params["t_bottleneck_start"] <= params["t_bottleneck_end"]:
    #         params["t_bottleneck_start"], params["t_bottleneck_end"] = (
    #             params["t_bottleneck_end"],
    #             params["t_bottleneck_start"],
    #         )
    return params


def sample_coverage_percent(
    selection_cfg: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> float:
    """
    Sample a *percent* (e.g., 37.4) from selection_cfg["coverage_percent"] = [low, high].
    Only used when engine == "slim".
    """
    rng = rng or np.random.default_rng()
    low, high = selection_cfg["coverage_percent"]
    return float(rng.uniform(low, high))


def next_sim_number(simulation_dir: Path) -> str:
    existing = {int(p.name) for p in simulation_dir.glob("[0-9]*") if p.is_dir()}
    return f"{max(existing, default=0) + 1:04d}"


# ============================================================================
# Core simulators (your existing code)
# ============================================================================


# Make sampled_coverage optional 
def simulation_runner(
    g: demes.Graph, experiment_config: Dict[str, Any], sampled_coverage: Optional[float] = None
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    model = define_sps_model(g) 

    print(f'• Using engine: {experiment_config.get("engine")}')

    samples = {
        k: int(v) for k, v in (experiment_config.get("num_samples") or {}).items()
    }
    seed = experiment_config.get("seed", None)
    sel = experiment_config.get("selection") or {}
    contig = _contig_from_cfg(experiment_config, sel)
    print("contig.length:", contig.length)
    print("contig.mutation_rate:", contig.mutation_rate)
    print("contig.recombination_map.mean_rate:", contig.recombination_map.mean_rate)



    if experiment_config.get("engine") == "slim":

        sel_summary = _apply_dfe_intervals(contig, sel, sampled_coverage=sampled_coverage)

        eng = sps.get_engine("slim")
        ts = eng.simulate(
            model,
            contig,
            samples,
            slim_scaling_factor=float(sel.get("slim_scaling", 10.0)),
            slim_burn_in=float(sel.get("slim_burn_in", 5.0)),
            seed=seed

        )

        ts._bgs_selection_summary = sel_summary
    else: 

        eng = sps.get_engine("msprime")

        ts = eng.simulate(
            model,
            contig,
            samples,
            seed=seed
        )

    return ts, g


def simulation(
    sampled_params: Dict[str, float],
    model_type: str,
    experiment_config: Dict[str, Any],
    sampled_coverage: Optional[float] = None,
) -> Tuple[tskit.TreeSequence, demes.Graph]:
    # Build demes graph

    g = build_demes_graph(model_type, sampled_params, experiment_config)

    return simulation_runner(
        g, experiment_config, sampled_coverage=sampled_coverage
    )


# ============================================================================
# SFS utility (your existing code)
# ============================================================================

def pop_id_by_name(ts: tskit.TreeSequence, name: str) -> int:
    for pop in ts.populations():
        meta = pop.metadata if isinstance(pop.metadata, dict) else {}
        if meta.get("name") == name:
            return pop.id
    raise ValueError(
        f"Population with metadata name='{name}' not found. "
        "Available names: "
        + ", ".join(
            str((p.id, (p.metadata.get('name') if isinstance(p.metadata, dict) else None)))
            for p in ts.populations()
        )
    )

def create_SFS(ts: tskit.TreeSequence, pop_names: Sequence[str] = ("YRI", "CEU")) -> moments.Spectrum:
    """
    Create a 2D site-frequency spectrum for exactly the two populations in pop_names,
    using ts population metadata 'name' field (robust to population ID ordering).
    """
    sample_sets: List[np.ndarray] = []
    for name in pop_names:
        pid = pop_id_by_name(ts, name)
        samps = ts.samples(population=pid)
        if len(samps) == 0:
            raise ValueError(f"Population '{name}' (id={pid}) has zero samples in this TS.")
        sample_sets.append(samps)

    sfs = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False,
    )

    sfs = moments.Spectrum(sfs)
    sfs.pop_ids = list(pop_names)

    # Debug prints (optional)
    # print("create_SFS using:", list(pop_names))
    # print("ts.sequence_length:", ts.sequence_length)
    # print("ts.num_sites:", ts.num_sites)
    # print("sum(obs_sfs):", float(np.sum(np.asarray(sfs))))
    # print("sfs.shape:", sfs.shape)

    return sfs

# ============================================================================
# Plotting + metadata + run-to-disk (moved from script)
# ============================================================================

def save_demes_png(g: demes.Graph, fig_path: Path, model_type: str) -> None:
    if model_type == "OOA_three_pop_gutenkunst":
        fig, ax = plt.subplots(figsize=(8, 5))
        demesdraw.tubes(g, ax=ax)

        hide = {"OOA", "B"}
        for t in ax.texts:
            if t.get_text() in hide:
                t.set_visible(False)

        ax.set_title(f"{model_type}")
        ax.set_xlabel("time ago (generations)")
        ax.set_ylabel("")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("N")
    ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def write_bgs_meta_json(
    *,
    out_dir: Path,
    cfg: Dict[str, Any],
    model_type: str,
    engine: str,
    sel_cfg: Dict[str, Any],
    ts: tskit.TreeSequence,
    ts_path: Path,
    sampled_params: Dict[str, float],
    simulation_seed: Optional[int],
    sampled_coverage: Optional[float],
) -> None:
    sel_summary = getattr(ts, "_bgs_selection_summary", {}) or {}
    is_bgs = engine == "slim"

    meta = dict(
        engine=str(engine),
        model_type=str(model_type),
        selection=is_bgs,
        species=(str(sel_cfg.get("species", "HomSap")) if is_bgs else None),
        dfe_id=(str(sel_cfg.get("dfe_id", "Gamma_K17")) if is_bgs else None),
        chromosome=(sel_cfg.get("chromosome") if is_bgs else None),
        left=(sel_cfg.get("left") if is_bgs else None),
        right=(sel_cfg.get("right") if is_bgs else None),
        genetic_map=(sel_cfg.get("genetic_map") if is_bgs else None),
        genome_length=float(cfg.get("genome_length")),
        mutation_rate=float(cfg.get("mutation_rate")),
        recombination_rate=float(cfg.get("recombination_rate")),
        coverage_fraction=(
            None
            if not is_bgs
            else (None if sel_cfg.get("coverage_fraction") is None else float(sel_cfg["coverage_fraction"]))
        ),
        coverage_percent=(
            None
            if not is_bgs
            else (
                None
                if sel_cfg.get("coverage_percent") is None
                else [float(sel_cfg["coverage_percent"][0]), float(sel_cfg["coverage_percent"][1])]
            )
        ),
        exon_bp=(int(sel_cfg.get("exon_bp", 200)) if is_bgs else None),
        jitter_bp=(int(sel_cfg.get("jitter_bp", 0)) if is_bgs else None),
        tile_bp=(int(sel_cfg["tile_bp"]) if is_bgs and sel_cfg.get("tile_bp") is not None else None),
        selected_bp=(int(sel_summary.get("selected_bp", 0)) if is_bgs else 0),
        selected_frac=(float(sel_summary.get("selected_frac", 0.0)) if is_bgs else 0.0),
        sampled_coverage_percent=(float(sampled_coverage) if is_bgs and sampled_coverage is not None else None),
        target_coverage_frac=(
            (float(sampled_coverage) / 100.0)
            if is_bgs and sampled_coverage is not None and float(sampled_coverage) > 1.0
            else (float(sampled_coverage) if is_bgs and sampled_coverage is not None else None)
        ),
        slim_scaling=(float(sel_cfg.get("slim_scaling", 10.0)) if is_bgs else None),
        slim_burn_in=(float(sel_cfg.get("slim_burn_in", 5.0)) if is_bgs else None),
        num_samples={k: int(v) for k, v in (cfg.get("num_samples") or {}).items()},
        base_seed=(None if cfg.get("seed") is None else int(cfg.get("seed"))),
        simulation_seed=simulation_seed,
        sequence_length=float(ts.sequence_length),
        tree_sequence=str(ts_path),
        sampled_params={k: float(v) for k, v in sampled_params.items()},
    )

    (out_dir / "bgs.meta.json").write_text(json.dumps(meta, indent=2))


def run_one_simulation_to_dir(
    simulation_dir: Path,
    experiment_config_path: Path,
    model_type: str,
    simulation_number: Optional[str] = None,
) -> Path:
    """
    High-level runner used by Snakemake wrapper:
      - loads cfg
      - creates out_dir
      - seeds rng
      - samples params (+ coverage if SLiM)
      - runs simulation()
      - writes sampled_params.pkl, SFS.pkl, tree_sequence.trees, demes.png, bgs.meta.json
    Returns the out_dir.
    """
    cfg: Dict[str, Any] = json.loads(experiment_config_path.read_text())
    engine = str(cfg["engine"]).lower()
    sel_cfg = cfg.get("selection") or {}

    if simulation_number is None:
        simulation_number = next_sim_number(simulation_dir)

    out_dir = simulation_dir / simulation_number
    out_dir.mkdir(parents=True, exist_ok=True)

    # seed handling
    base_seed = cfg.get("seed")
    if base_seed is not None:
        simulation_seed = int(base_seed) + int(simulation_number)
        rng = np.random.default_rng(simulation_seed)
        print(f"• Using seed {simulation_seed} (base: {base_seed} + sim: {simulation_number})")
    else:
        simulation_seed = None
        rng = np.random.default_rng()
        print("• No seed specified, using random state")

    # sample params
    sampled_params = sample_params(cfg["priors"], rng=rng)

    # sample coverage if SLiM
    if engine == "slim":
        if "coverage_percent" not in sel_cfg:
            raise ValueError("engine='slim' requires selection.coverage_percent=[low, high].")
        sampled_coverage = sample_coverage_percent(sel_cfg, rng=rng)
        print(f"• engine=slim → sampled coverage: {sampled_coverage:.2f}%")
    elif engine == "msprime":
        sampled_coverage = None
        print("• engine=msprime → neutral (no BGS); skipping coverage sampling.")
    else:
        raise ValueError("engine must be 'slim' or 'msprime'.")

    # ensure we pass a per-simulation seed down
    sim_cfg = dict(cfg)
    if simulation_seed is not None:
        sim_cfg["seed"] = simulation_seed

    ts, g = simulation(sampled_params, model_type, sim_cfg, sampled_coverage)
    sfs = create_SFS(ts, pop_names = tuple(list(cfg['num_samples'].keys())))

    # --- DEBUG: site vs branch vs moments expectation ---

    # Reconstruct sample_sets and pop_ids EXACTLY like create_SFS
    sample_sets = []
    pop_ids = []
    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps):
            sample_sets.append(samps)
            meta = pop.metadata if isinstance(pop.metadata, dict) else {}
            pop_ids.append(meta.get("name", f"pop{pop.id}"))

    # Site-mode SFS (what you already save)
    arr_site = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False,
    )
    sfs_site = moments.Spectrum(arr_site)
    sfs_site.pop_ids = pop_ids

    # Branch-mode SFS (expected mutations on realized genealogy)
    arr_branch = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="branch",
        polarised=True,
        span_normalise=False,
    )

    mu = float(sim_cfg["mutation_rate"])
    arr_branch = arr_branch * mu
    sfs_branch = moments.Spectrum(arr_branch)
    sfs_branch.pop_ids = pop_ids

    print("DEBUG sum(site):", float(arr_site.sum()))
    print("DEBUG sum(branch * mu):", float(arr_branch.sum()))


    # save artifacts
    (out_dir / "sampled_params.pkl").write_bytes(pickle.dumps(sampled_params))
    (out_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts_path = out_dir / "tree_sequence.trees"
    ts.dump(ts_path)

    # plot + meta
    save_demes_png(g, out_dir / "demes.png", model_type=model_type)
    write_bgs_meta_json(
        out_dir=out_dir,
        cfg=cfg,
        model_type=model_type,
        engine=engine,
        sel_cfg=sel_cfg,
        ts=ts,
        ts_path=ts_path,
        sampled_params=sampled_params,
        simulation_seed=simulation_seed,
        sampled_coverage=sampled_coverage,
    )

    return out_dir

# ============================================================================
# Windowed replicate utilities (moved from simulate_one_window.py)
# ============================================================================

def build_demes_graph(
    model_type: str, sampled_params: Dict[str, float], cfg: Optional[Dict[str, Any]] = None
) -> demes.Graph:
    """
    Build a Demes graph for the given model_type + sampled_params.
    Mirrors the logic inside simulation(...), but returns only the graph.
    """
    cfg = cfg or {}
    if model_type == "bottleneck":
        return bottleneck_model(sampled_params, cfg)
    if model_type == "IM_symmetric":
        return IM_symmetric_model(sampled_params, cfg)
    if model_type == "IM_asymmetric":
        return IM_asymmetric_model(sampled_params, cfg)
    if model_type == "drosophila_three_epoch":
        return drosophila_three_epoch(sampled_params, cfg)
    if model_type == "split_migration_growth":
        return split_migration_growth_model(sampled_params, cfg)
    if model_type == "OOA_three_pop":
        return OOA_three_pop_model_simplified(sampled_params, cfg)
    if model_type == "OOA_three_pop_gutenkunst":
        return OOA_three_pop_Gutenkunst(sampled_params, cfg)
    raise ValueError(f"Unknown model_type: {model_type}")


def write_samples_and_map(*, L: int, r: float, samples: Dict[str, int], out_dir: Path) -> None:
    """
    Write two small helper files next to each window:
      - samples.txt  (tsk_i -> pop label)
      - flat_map.txt (two-point linear map in cM for quick plotting/debug)
    """
    # samples.txt
    lines = ["sample\tpop"]
    tsk_i = 0
    for pop, n in samples.items():
        for _ in range(int(n)):
            lines.append(f"tsk_{tsk_i}\t{pop}")
            tsk_i += 1
    (out_dir / "samples.txt").write_text("\n".join(lines) + "\n")

    # flat_map.txt
    cm_total = r * L * 100.0
    (out_dir / "flat_map.txt").write_text(f"pos\tMap(cM)\n0\t0\n{L}\t{cm_total}\n")


def load_sampled_coverage_from_meta(meta_file: Optional[Path]) -> Optional[float]:
    """
    Pull the exact same coverage used in the base BGS sim, preferring:
      sampled_coverage_percent  >  target_coverage_frac / coverage_fraction  >  selected_frac

    Returns None if meta_file is None.

    NOTE: This returns either:
      - a percent (>1), OR
      - a fraction (<=1)
    Your downstream BGS interval builder supports both.
    """
    if meta_file is None:
        return None
    if not meta_file.exists():
        raise FileNotFoundError(f"meta file not found: {meta_file}")

    meta = json.loads(meta_file.read_text())

    if meta.get("sampled_coverage_percent") is not None:
        return float(meta["sampled_coverage_percent"])  # percent
    if meta.get("target_coverage_frac") is not None:
        return float(meta["target_coverage_frac"])  # fraction
    if meta.get("coverage_fraction") is not None:
        return float(meta["coverage_fraction"])  # fraction
    if meta.get("selected_frac") is not None:
        return float(meta["selected_frac"])  # fraction (realized)

    raise RuntimeError(
        "Meta file is missing coverage fields. Expected one of: "
        "sampled_coverage_percent | target_coverage_frac | coverage_fraction | selected_frac"
    )


def window_seed_from_base(base_seed: Optional[int], rep_index: int, *, stride: int = 10000) -> Optional[int]:
    """
    Deterministic per-window seed from base seed.
    Keeps your existing behavior: base_seed + rep_index * 10000.
    """
    if base_seed is None:
        return None
    return int(base_seed) + int(rep_index) * int(stride)


def simulate_one_window_replicate(
    *,
    sim_dir: Path,
    rep_index: int,
    config_file: Path,
    out_dir: Path,
    meta_file: Optional[Path] = None,
    seed_stride: int = 10000,
) -> Path:
    """
    High-level window runner (engine-aware, reuses msprime_simulation/stdpopsim_slim_simulation).

    Inputs:
      - sim_dir: directory containing sampled_params.pkl from the *base* simulation
      - rep_index: which window replicate to generate
      - config_file: JSON config
      - out_dir: output directory (will contain window_<idx>.* + samples.txt + flat_map.txt + metadata)
      - meta_file: optional bgs.meta.json from base sim so we can reuse exact coverage
      - seed_stride: default 10000 matches your old script

    Writes:
      - window_<idx>.trees
      - window_<idx>.vcf.gz
      - samples.txt
      - flat_map.txt
      - window_<idx>.meta.json
    Returns:
      out_dir
    """
    cfg: Dict[str, Any] = json.loads(config_file.read_text())
    engine = str(cfg["engine"]).lower()

    sampled_params: Dict[str, float] = pickle.load((sim_dir / "sampled_params.pkl").open("rb"))

    # Determine model_type:
    # Prefer explicit "model_type" if present, else fall back to legacy "demographic_model".
    model_type = cfg.get("model_type", None) or cfg.get("demographic_model", None)
    if model_type is None:
        raise KeyError("Config must contain either 'model_type' or 'demographic_model'.")

    # Reuse exact coverage for BGS (if engine=slim)
    sampled_coverage = load_sampled_coverage_from_meta(meta_file) if engine == "slim" else None

    # Window seed
    base_seed = cfg.get("seed", None)
    w_seed = window_seed_from_base(base_seed, rep_index, stride=seed_stride)

    window_cfg = dict(cfg)
    if w_seed is not None:
        window_cfg["seed"] = w_seed
        print(f"• Window {rep_index}: using seed {w_seed} (base: {base_seed} + {rep_index} * {seed_stride})")
    else:
        print(f"• Window {rep_index}: no seed specified, using random state")

    # Metadata skeleton
    sel_cfg = window_cfg.get("selection") or {}
    window_metadata: Dict[str, Any] = {
        "window_index": int(rep_index),
        "engine": engine,
        "model_type": str(model_type),
        "base_seed": base_seed,
        "window_seed": w_seed,
        "genome_length": float(window_cfg["genome_length"]),
        "mutation_rate": float(window_cfg["mutation_rate"]),
        "recombination_rate": float(window_cfg["recombination_rate"]),
        "num_samples": {k: int(v) for k, v in window_cfg["num_samples"].items()},
        "sampled_params": {k: float(v) for k, v in sampled_params.items()},
        "sampled_coverage": sampled_coverage,
        "selection_enabled": (bool(sel_cfg.get("enabled", False)) if engine == "slim" else False),
    }

    # Simulate

    ts, g = simulation(sampled_params=sampled_params,
                      model_type=model_type,
                      experiment_config=window_cfg,
                      sampled_coverage=sampled_coverage)

    #     def simulation(
    #     sampled_params: Dict[str, float],
    #     model_type: str,
    #     experiment_config: Dict[str, Any],
    #     sampled_coverage: Optional[float] = None,
    # ) -> Tuple[tskit.TreeSequence, demes.Graph]:

    # if engine == "msprime":
    #     ts, _ = msprime_simulation(graph, window_cfg)
    # elif engine == "slim":
    #     if not bool(sel_cfg.get("enabled", False)):
    #         raise RuntimeError("engine='slim' requires selection.enabled=true in config.")
    #     if sampled_coverage is None:
    #         raise RuntimeError("engine='slim' requires coverage; pass --meta-file from base sim.")
    #     ts, _ = stdpopsim_slim_simulation(
    #         g=graph,
    #         experiment_config=window_cfg,
    #         sampled_coverage=sampled_coverage,
    #         model_type=str(model_type),
    #         sampled_params=sampled_params,
    #     )

    sel_summary = getattr(ts, "_bgs_selection_summary", {}) or {}
    window_metadata.update(
        {
            "species": str(sel_cfg.get("species", "HomSap")),
            "dfe_id": str(sel_cfg.get("dfe_id", "Gamma_K17")),
            "selected_bp": int(sel_summary.get("selected_bp", 0)),
            "selected_frac": float(sel_summary.get("selected_frac", 0.0)),
            "slim_scaling": float(sel_cfg.get("slim_scaling", 10.0)),
            "slim_burn_in": float(sel_cfg.get("slim_burn_in", 5.0)),
        }
    )

    window_metadata["sequence_length"] = float(ts.sequence_length)

    # Write outputs
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

    # samples.txt + flat_map.txt
    write_samples_and_map(
        L=int(window_cfg["genome_length"]),
        r=float(window_cfg["recombination_rate"]),
        samples={k: int(v) for k, v in window_cfg["num_samples"].items()},
        out_dir=out_dir,
    )

    # metadata
    (out_dir / f"window_{rep_index}.meta.json").write_text(json.dumps(window_metadata, indent=2))

    print(f"✓ replicate {rep_index:04d} → {ts_file.name} + .vcf.gz + metadata")
    return out_dir

#!/usr/bin/env python3
# Split isolation + tiled BGS via stdpopsim/SLiM.
# Coverage sweeps with Ray parallelism (via subprocess workers), Poisson moments fit (unchanged),
# skip-existing, range/grid coverages, cores/thread controls, and error plots.

from __future__ import annotations
import argparse, json, csv, warnings, os, sys, subprocess, tempfile
from pathlib import Path
from typing import List, Dict, Tuple, OrderedDict as _OD

import numpy as np
import msprime
import stdpopsim as sps
import tskit
import moments
import demes
import nlopt
import numdifftools as nd
import matplotlib.pyplot as plt
from moments import Inference as MInf

# Optional deps
try:
    import ray  # type: ignore
except Exception:
    ray = None
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", msprime.TimeUnitsMismatchWarning)

# ────────────────────────── SFS utility ──────────────────────────
def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    sample_sets: List[np.ndarray] = []
    pop_ids: List[str] = []
    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps) > 0:
            sample_sets.append(samps)
            meta = pop.metadata
            pop_ids.append(meta["name"] if isinstance(meta, dict) and "name" in meta else f"pop{pop.id}")
    if not sample_sets:
        raise ValueError("No sampled populations found in tree sequence.")
    arr = ts.allele_frequency_spectrum(sample_sets=sample_sets, mode="site",
                                       polarised=True, span_normalise=False)
    sfs = moments.Spectrum(arr)
    sfs.pop_ids = pop_ids
    return sfs

# ────────────────────────── split-isolation (local factory; Ray-safe) ───────
def _build_split_isolation_model(N0: float, N1: float, N2: float, T: float, m: float) -> sps.DemographicModel:
    dem = msprime.Demography()
    dem.add_population(name="YRI", initial_size=float(N1))
    dem.add_population(name="CEU", initial_size=float(N2))
    dem.add_population(name="ANC", initial_size=float(N0))
    m = float(m)
    dem.set_migration_rate("YRI", "CEU", m)
    dem.set_migration_rate("CEU", "YRI", m)
    dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])

    class _LocalSplitIsolationModel(sps.DemographicModel):
        def __init__(self):
            super().__init__(
                id="split_isolation",
                description="ANC → (YRI, CEU) at T; symmetric migration m.",
                long_description="Custom msprime demography: split isolation with symmetric migration.",
                model=dem,
                generation_time=1,
            )
    return _LocalSplitIsolationModel()

# ────────────────────────── demes graph (for moments) ────────────────────────
def split_isolation_graph(params: Dict[str, float]) -> demes.Graph:
    N0 = float(params.get("N_anc", params.get("N0")))
    N1 = float(params.get("N_YRI", params.get("N1")))
    N2 = float(params.get("N_CEU", params.get("N2")))
    T  = float(params.get("T_split", params.get("t_split")))
    m_candidates = [k for k in ("m_sym", "m", "m12", "m21") if k in params]
    m_vals = [float(params[k]) for k in m_candidates]
    m = float(np.mean(m_vals)) if m_vals else 0.0
    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme("YRI", ancestors=["ANC"], epochs=[dict(start_size=N1)])
    b.add_deme("CEU", ancestors=["ANC"], epochs=[dict(start_size=N2)])
    if m > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m)
        b.add_migration(source="CEU", dest="YRI", rate=m)
    return b.resolve()

# ────────────────────────── BGS tiling (coverage-aware, non-overlap) ────────
def _sanitize_nonoverlap(intervals: np.ndarray, L: int) -> np.ndarray:
    if intervals.size == 0:
        return intervals
    iv = intervals[np.argsort(intervals[:, 0])]
    out = []
    prev_end = -1
    for s, e in iv:
        s = int(max(0, min(s, L)))
        e = int(max(0, min(e, L)))
        if e <= s:
            continue
        if s < prev_end:
            continue
        out.append((s, e))
        prev_end = e
    return np.array(out, dtype=int) if out else np.empty((0, 2), dtype=int)

def build_tiling_intervals(L: int, exon_bp: int, tile_bp: int, jitter_bp: int = 0) -> np.ndarray:
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    if jitter_bp > 0 and len(starts) > 0:
        rng = np.random.default_rng()
        jitter = rng.integers(-jitter_bp, jitter_bp + 1, size=len(starts))
        starts = np.clip(starts + jitter, 0, max(0, L - exon_bp))
    ends = np.minimum(starts + exon_bp, L).astype(int)
    iv = np.column_stack([starts, ends])
    return _sanitize_nonoverlap(iv, L)

def intervals_from_coverage(L: int, exon_bp: int, coverage: float, jitter_bp: int = 0) -> np.ndarray:
    if coverage <= 0:
        return np.empty((0, 2), dtype=int)
    if coverage >= 1.0:
        return np.array([[0, int(L)]], dtype=int)
    tile_bp = max(exon_bp, int(round(exon_bp / float(max(coverage, 1e-12)))))
    return build_tiling_intervals(int(L), int(exon_bp), tile_bp, jitter_bp=jitter_bp)

def make_contig_and_apply_dfe(length: int, mu: float, r: float,
                              species: str, dfe_id: str, intervals: np.ndarray):
    sp = sps.get_species(species)
    try:
        contig = sp.get_contig(chromosome=None, length=int(length),
                               mutation_rate=float(mu), recombination_rate=float(r))
    except TypeError:
        print("[warn] stdpopsim.get_contig() does not accept 'recombination_rate'; using species default r.")
        contig = sp.get_contig(chromosome=None, length=int(length), mutation_rate=float(mu))
    if intervals.size > 0:
        dfe = sp.get_dfe(dfe_id)
        contig.add_dfe(intervals=intervals, DFE=dfe)
    return contig

# # ────────────────────────── moments helpers (Poisson; UNCHANGED) ────────────
# def _moments_expected_sfs(params_vec: np.ndarray, param_names: List[str],
#                           sample_sizes: "_OD[str, int]", config: Dict) -> moments.Spectrum:
#     p = {k: float(v) for k, v in zip(param_names, params_vec)}
#     g = split_isolation_graph(p)
#     hap = [2 * n for n in sample_sizes.values()]
#     demes_order = list(sample_sizes.keys())
#     theta = float(p[param_names[0]]) * 4.0 * float(config["mutation_rate"]) * float(config["genome_length"])
#     return moments.Spectrum.from_demes(g, sample_sizes=hap, sampled_demes=demes_order, theta=theta)

# def _geometric_mean(lo: float, hi: float) -> float:
#     return float(np.sqrt(float(lo) * float(hi)))

# def _prepare_sample_sizes_from_sfs(sfs: moments.Spectrum) -> "_OD[str, int]":
#     from collections import OrderedDict
#     if hasattr(sfs, "pop_ids") and sfs.pop_ids:
#         return OrderedDict((pid, (sfs.shape[i] - 1) // 2) for i, pid in enumerate(sfs.pop_ids))
#     return OrderedDict((f"pop{i}", (n - 1) // 2) for i, n in enumerate(sfs.shape))

# def fit_moments(sfs: moments.Spectrum, config: Dict,
#                 fixed_params: Dict[str, float] | None = None) -> Tuple[np.ndarray, float, List[str]]:
#     priors = config["priors"]
#     param_names = list(priors.keys())
#     lb = np.array([priors[p][0] for p in param_names], float)
#     ub = np.array([priors[p][1] for p in param_names], float)
#     start = np.array([_geometric_mean(*priors[p]) for p in param_names], float)

#     fixed_params = dict(fixed_params or {})
#     fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
#     free_idx  = [i for i, n in enumerate(param_names) if n not in fixed_params]
#     x0 = start.copy()
#     for i in fixed_idx:
#         x0[i] = float(fixed_params[param_names[i]])
#         if not (lb[i] <= x0[i] <= ub[i]):
#             raise ValueError(f"Fixed {param_names[i]}={x0[i]} outside bounds [{lb[i]},{ub[i]}]")

#     ns = _prepare_sample_sizes_from_sfs(sfs)

#     def pack_free(x_free: np.ndarray) -> np.ndarray:
#         x = x0.copy()
#         for j, i in enumerate(free_idx):
#             x[i] = float(x_free[j])
#         return x

#     # Poisson composite log-likelihood (UNCHANGED)
#     def obj_log10(xlog10_free: np.ndarray) -> float:
#         x_free = 10.0 ** np.asarray(xlog10_free, float)
#         x_full = pack_free(x_free)
#         try:
#             expected = _moments_expected_sfs(x_full, param_names, ns, config)
#             if getattr(sfs, "folded", False):
#                 expected = expected.fold()
#             expected = np.maximum(expected, 1e-300)
#             return float(np.sum(sfs * np.log(expected) - expected))
#         except Exception as e:
#             print(f"[moments obj] error: {e}")
#             return -np.inf

#     if len(free_idx) == 0:
#         return x0, obj_log10(np.array([], float)), param_names

#     lb_free = np.array([lb[i] for i in free_idx], float)
#     ub_free = np.array([ub[i] for i in free_idx], float)
#     x0_free = np.array([x0[i] for i in free_idx], float)
#     x0_free = moments.Misc.perturb_params(x0_free, fold=0.1)
#     x0_free = np.clip(x0_free, lb_free, ub_free)

#     grad_fn = nd.Gradient(obj_log10, step=1e-4)

#     def nlopt_objective(xlog10_free, grad):
#         ll = obj_log10(xlog10_free)
#         if grad.size > 0:
#             grad[:] = grad_fn(xlog10_free)
#         print(f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(xlog10_free), precision=4)}")
#         return ll

#     opt = nlopt.opt(nlopt.LD_LBFGS, len(free_idx))
#     opt.set_lower_bounds(np.log10(np.maximum(lb_free, 1e-300)))
#     opt.set_upper_bounds(np.log10(np.maximum(ub_free, 1e-300)))
#     opt.set_max_objective(nlopt_objective)
#     opt.set_ftol_rel(1e-8)
#     opt.set_maxeval(10000)

#     try:
#         x_free_hat_log10 = opt.optimize(np.log10(np.maximum(x0_free, 1e-300)))
#         ll_val = opt.last_optimum_value()
#     except Exception as e:
#         print(f"[moments] NLopt failed: {e}")
#         x_free_hat_log10 = np.log10(np.maximum(x0_free, 1e-300))
#         ll_val = obj_log10(x_free_hat_log10)

#     x_free_hat = 10.0 ** x_free_hat_log10
#     x_full_hat = pack_free(x_free_hat)
#     return x_full_hat, ll_val, param_names

# ────────────────────────── moments helpers (Multinomial; optimal scaling) ────────────
from moments import Inference as MInf

def _moments_expected_sfs_unscaled(
    params_vec: np.ndarray,
    param_names: List[str],
    sample_sizes: "_OD[str, int]",
) -> moments.Spectrum:
    """
    Expected SFS with theta=1 (unscaled). We scale to the data optimally in the objective.
    """
    p = {k: float(v) for k, v in zip(param_names, params_vec)}
    g = split_isolation_graph(p)
    hap = [2 * n for n in sample_sizes.values()]
    demes_order = list(sample_sizes.keys())
    return moments.Spectrum.from_demes(
        g, sample_sizes=hap, sampled_demes=demes_order, theta=1.0
    )

def _geometric_mean(lo: float, hi: float) -> float:
    return float(np.sqrt(float(lo) * float(hi)))

def _prepare_sample_sizes_from_sfs(sfs: moments.Spectrum) -> "_OD[str, int]":
    from collections import OrderedDict
    if hasattr(sfs, "pop_ids") and sfs.pop_ids:
        return OrderedDict((pid, (sfs.shape[i] - 1) // 2) for i, pid in enumerate(sfs.pop_ids))
    return OrderedDict((f"pop{i}", (n - 1) // 2) for i, n in enumerate(sfs.shape))

def fit_moments(
    sfs: moments.Spectrum,
    config: Dict,
    fixed_params: Dict[str, float] | None = None
) -> Tuple[np.ndarray, float, List[str]]:
    priors = config["priors"]
    param_names = list(priors.keys())
    lb = np.array([priors[p][0] for p in param_names], float)
    ub = np.array([priors[p][1] for p in param_names], float)
    start = np.array([_geometric_mean(*priors[p]) for p in param_names], float)

    fixed_params = dict(fixed_params or {})
    fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
    free_idx  = [i for i, n in enumerate(param_names) if n not in fixed_params]
    x0 = start.copy()
    for i in fixed_idx:
        x0[i] = float(fixed_params[param_names[i]])
        if not (lb[i] <= x0[i] <= ub[i]):
            raise ValueError(f"Fixed {param_names[i]}={x0[i]} outside bounds [{lb[i]},{ub[i]}]")

    ns = _prepare_sample_sizes_from_sfs(sfs)

    def pack_free(x_free: np.ndarray) -> np.ndarray:
        x = x0.copy()
        for j, i in enumerate(free_idx):
            x[i] = float(x_free[j])
        return x

    # Multinomial log-likelihood with optimal scaling (decouples θ from N0)
    def obj_log10(xlog10_free: np.ndarray) -> float:
        x_free = 10.0 ** np.asarray(xlog10_free, float)
        x_full = pack_free(x_free)
        try:
            model = _moments_expected_sfs_unscaled(x_full, param_names, ns)
            # Optimal overall scale to match total segregating sites
            scale = MInf.optimal_sfs_scaling(sfs, model)
            model *= scale
            # tiny floor to avoid zero cells, then renormalize to preserve total
            total = float(model.sum())
            if total <= 0:
                return -np.inf
            model = np.maximum(model, 1e-300)
            model *= total / float(model.sum())
            if getattr(sfs, "folded", False):
                model = model.fold()
            return float(MInf.ll_multinom(model, sfs))
        except Exception as e:
            print(f"[moments obj] error: {e}")
            return -np.inf

    if len(free_idx) == 0:
        return x0, obj_log10(np.array([], float)), param_names

    lb_free = np.array([lb[i] for i in free_idx], float)
    ub_free = np.array([ub[i] for i in free_idx], float)
    x0_free = np.array([x0[i] for i in free_idx], float)
    x0_free = moments.Misc.perturb_params(x0_free, fold=0.1)
    x0_free = np.clip(x0_free, lb_free, ub_free)

    grad_fn = nd.Gradient(obj_log10, step=1e-4)

    def nlopt_objective(xlog10_free, grad):
        ll = obj_log10(xlog10_free)
        if grad.size > 0:
            grad[:] = grad_fn(xlog10_free)
        print(f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(xlog10_free), precision=4)}")
        return ll

    opt = nlopt.opt(nlopt.LD_LBFGS, len(free_idx))
    opt.set_lower_bounds(np.log10(np.maximum(lb_free, 1e-300)))
    opt.set_upper_bounds(np.log10(np.maximum(ub_free, 1e-300)))
    opt.set_max_objective(nlopt_objective)
    opt.set_ftol_rel(1e-8)
    opt.set_maxeval(10000)

    try:
        x_free_hat_log10 = opt.optimize(np.log10(np.maximum(x0_free, 1e-300)))
        ll_val = opt.last_optimum_value()
    except Exception as e:
        print(f"[moments] NLopt failed: {e}")
        x_free_hat_log10 = np.log10(np.maximum(x0_free, 1e-300))
        ll_val = obj_log10(x_free_hat_log10)

    x_free_hat = 10.0 ** x_free_hat_log10
    x_full_hat = pack_free(x_free_hat)
    return x_full_hat, ll_val, param_names

# ────────────────────────── CLI ──────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Split isolation + tiled BGS via stdpopsim (SLiM). Poisson moments fit. Ray (subprocess) sweep + plots.")

    # Demography
    p.add_argument("--N-anc", type=float, required=True)
    p.add_argument("--N1", type=float, required=True)
    p.add_argument("--N2", type=float, required=True)
    p.add_argument("--t-split", type=float, required=True)
    p.add_argument("--m", type=float, default=0.0)

    # Samples
    p.add_argument("--samples", default="YRI:20,CEU:20")

    # Species/DFE for contig
    p.add_argument("--species", default="HomSap")
    p.add_argument("--dfe", default="Gamma_K17")

    # Genome/rates
    p.add_argument("--length", type=int, default=200_000)
    p.add_argument("--mu", type=float, default=1e-8)
    p.add_argument("--r", type=float, default=1e-8)

    # BGS tiling (classic)
    p.add_argument("--exon-bp", type=int, default=200)
    p.add_argument("--tile-bp", type=int, default=5000)

    # Coverage controls
    p.add_argument("--coverage", type=float, default=None,
                   help="Single coverage in [0,1], overrides tile-bp mode.")
    p.add_argument("--coverage-grid", type=str, default="",
                   help="Comma list (e.g. 0,0.01,0.02,...). Merged with any range-generated values.")
    p.add_argument("--coverage-min", type=float, default=None)
    p.add_argument("--coverage-max", type=float, default=None)
    p.add_argument("--coverage-n", type=int, default=None)
    p.add_argument("--coverage-space", choices=["lin","log"], default="lin")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--jitter-bp", type=int, default=0)

    # SLiM
    p.add_argument("--slim-scaling", type=float, default=10.0)
    p.add_argument("--slim-burn-in", type=float, default=5.0)

    # I/O
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--trees", default="sims/out.trees")
    p.add_argument("--vcf", default="")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")

    # Moments
    p.add_argument("--moments-config", type=Path,
                   help="JSON with {'priors':{...}, 'mutation_rate':..., 'genome_length':...}")
    p.add_argument("--fixed-json", type=str, default="")

    # Parallelism / cores
    p.add_argument("--ray", action="store_true", help="Use Ray to parallelize coverage×replicate sweep (via subprocess workers).")
    p.add_argument("--ray-num-cpus", type=int, default=None)
    p.add_argument("--ray-address", type=str, default=None)
    p.add_argument("--cores", type=int, default=None,
                   help="Convenience: sets --ray-num-cpus if not provided; also caps BLAS threads.")
    p.add_argument("--threads-per-task", type=int, default=1,
                   help="Set OMP/MKL/NUMEXPR threads per worker. Often 1 is best.")

    # Hidden: internal single-run for Ray subprocess workers
    p.add_argument("--_internal-single-run", action="store_true", default=False)
    p.add_argument("--_internal-coverage", type=float, default=None)
    p.add_argument("--_internal-label", type=str, default="")
    p.add_argument("--_internal-seed-offset", type=int, default=0)
    p.add_argument("--_internal-row-json", type=Path, default=None)
    return p.parse_args()

# ────────────────────────── run helpers ──────────────────────────
def _paths_from_suffix(base_trees: Path, suffix: str):
    base = base_trees.with_suffix("")
    outroot = base.parent / (f"{base.name}_{suffix}" if suffix else base.name)
    return (outroot.with_suffix(".trees"),
            outroot.with_suffix(".meta.json"),
            outroot.with_suffix(".exons.bed"),
            outroot.with_suffix(".sfs.npy"),
            outroot)

def run_single_sim_and_fit_serial(a, coverage_target: float | None, label_suffix: str,
                                  seed_offset: int = 0) -> Dict:
    # Build intervals (coverage mode) or tile-bp mode
    if coverage_target is None:
        intervals = build_tiling_intervals(a.length, a.exon_bp, a.tile_bp, jitter_bp=a.jitter_bp)
    else:
        intervals = intervals_from_coverage(a.length, a.exon_bp, coverage_target, jitter_bp=a.jitter_bp)

    trees_path, meta_path, bed_path, sfs_npy, outroot = _paths_from_suffix(Path(a.trees), label_suffix)
    vcf_path = (Path(a.vcf) if a.vcf else None)

    # ensure dirs
    for p in (trees_path, meta_path, bed_path, sfs_npy):
        p.parent.mkdir(parents=True, exist_ok=True)
    if vcf_path:
        vcf_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if exists
    if a.skip_existing and trees_path.exists():
        print(f"[skip-existing] Using existing {trees_path.name}")
        if not sfs_npy.exists():
            ts = tskit.load(str(trees_path))
            sfs = create_SFS(ts)
            np.save(sfs_npy, sfs.data)
        moments_pkl = outroot.with_suffix(".moments_best.pkl")
        if a.moments_config and not moments_pkl.exists():
            sfs = moments.Spectrum(np.load(sfs_npy))
            cfg = json.loads(Path(a.moments_config).read_text())
            fixed = json.loads(a.fixed_json) if a.fixed_json else {}
            best_vec, best_ll, names = fit_moments(sfs, cfg, fixed_params=fixed)
            best = {k: float(v) for k, v in zip(names, best_vec)}
            with open(moments_pkl, "wb") as f:
                import pickle; pickle.dump(dict(best_params=best, best_ll=float(best_ll),
                                                param_order=names, fixed_params=fixed), f)
        meta = {}
        if meta_path.exists():
            try: meta = json.loads(meta_path.read_text())
            except Exception: meta = {}
        row = dict(
            label=(label_suffix or "single"),
            coverage_target=(coverage_target if coverage_target is not None else float(a.exon_bp)/float(a.tile_bp)),
            selected_bp=int(meta.get("selected_bp", 0)),
            selected_frac=float(meta.get("selected_frac", 0.0)),
            seg_sites=int(meta.get("sites", 0)),
        )
        moments_pkl = outroot.with_suffix(".moments_best.pkl")
        if moments_pkl.exists():
            import pickle
            res = pickle.loads(moments_pkl.read_bytes())
            row["moments_best_ll"] = float(res.get("best_ll", np.nan))
            for k, v in (res.get("best_params") or {}).items():
                row[f"fit_{k}"] = float(v)
        return row

    # demography & contig
    model = _build_split_isolation_model(N0=a.N_anc, N1=a.N1, N2=a.N2, T=a.t_split, m=a.m)
    contig = make_contig_and_apply_dfe(a.length, a.mu, a.r, a.species, a.dfe, intervals)

    # samples
    samples = {k.strip(): int(v) for k, v in (x.split(":") for x in a.samples.split(","))}

    # simulate
    engine = sps.get_engine("slim")
    ts = engine.simulate(model, contig, samples,
                         seed=a.seed + seed_offset,
                         slim_scaling_factor=a.slim_scaling,
                         slim_burn_in=a.slim_burn_in)

    # write outputs
    ts.dump(str(trees_path))
    if vcf_path:
        with open(vcf_path, "w") as f:
            ts.write_vcf(f)
    with open(bed_path, "w") as bed:
        for s, e in intervals:
            bed.write(f"chr1\t{s}\t{e}\n")

    # SFS
    sfs = create_SFS(ts)
    np.save(sfs_npy, sfs.data)

    # meta
    selected_bp = int(np.sum((intervals[:, 1] - intervals[:, 0])) if intervals.size else 0)
    meta = dict(
        model="split_isolation",
        params=dict(N_anc=a.N_anc, N1=a.N1, N2=a.N2, t_split=a.t_split, m=a.m),
        species=a.species, samples=a.samples,
        length=a.length, mu=a.mu, r=a.r,
        dfe=a.dfe, exon_bp=a.exon_bp, tile_bp=a.tile_bp,
        coverage_target=coverage_target,
        selected_bp=selected_bp, selected_frac=(selected_bp / a.length),
        slim_scaling=a.slim_scaling, slim_burn_in=a.slim_burn_in,
        seed=a.seed + seed_offset,
        trees=str(trees_path), vcf=(str(vcf_path) if vcf_path else ""),
        sites=int(ts.num_sites), sequence_length=int(ts.sequence_length),
        sfs_npy=str(sfs_npy),
    )
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {trees_path}; sites={ts.num_sites}; length={ts.sequence_length:.0f} bp")
    if vcf_path: print(f"Wrote {vcf_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {bed_path}")
    print(f"Wrote {sfs_npy}")

    # moments (optional; Poisson unchanged)
    best = {}
    best_ll = None
    if a.moments_config:
        cfg = json.loads(Path(a.moments_config).read_text())
        if "priors" not in cfg:
            raise ValueError("--moments-config must include 'priors'")
        fixed = json.loads(a.fixed_json) if a.fixed_json else {}
        best_vec, best_ll, names = fit_moments(sfs, cfg, fixed_params=fixed)
        best = {k: float(v) for k, v in zip(names, best_vec)}
        out_pkl = Path(str(trees_path)).with_suffix("").with_suffix(".moments_best.pkl")
        with open(out_pkl, "wb") as f:
            import pickle; pickle.dump(
                dict(best_params=best, best_ll=float(best_ll),
                     param_order=names, fixed_params=fixed), f)
        print(f"[moments] best_ll={best_ll:.6g}  best_params={best}")
        print(f"[moments] wrote {out_pkl}")

    row = dict(
        label=(label_suffix or "single"),
        coverage_target=(coverage_target if coverage_target is not None else float(a.exon_bp)/float(a.tile_bp)),
        selected_bp=selected_bp, selected_frac=(selected_bp / a.length),
        seg_sites=int(ts.num_sites),
        moments_best_ll=(float(best_ll) if best_ll is not None else None),
    )
    for k, v in best.items():
        row[f"fit_{k}"] = v
    return row

# ────────────────────────── Ray worker (subprocess) ─────────────────────────
if ray is not None:
    @ray.remote
    def ray_worker_subprocess(script_path: str, a_dict: Dict, coverage: float, label: str, seed_offset: int,
                              threads_per_task: int) -> Dict:
        # Per-task thread caps
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads_per_task)
        env["MKL_NUM_THREADS"] = str(threads_per_task)
        env["NUMEXPR_NUM_THREADS"] = str(threads_per_task)

        # Build CLI args for the child
        with tempfile.TemporaryDirectory() as tmpd:
            row_json = Path(tmpd) / "row.json"
            # Reconstruct base CLI from dict
            args = [
                sys.executable, script_path,
                "--N-anc", str(a_dict["N_anc"]),
                "--N1", str(a_dict["N1"]),
                "--N2", str(a_dict["N2"]),
                "--t-split", str(a_dict["t_split"]),
                "--m", str(a_dict["m"]),
                "--samples", a_dict["samples"],
                "--species", a_dict["species"],
                "--dfe", a_dict["dfe"],
                "--length", str(a_dict["length"]),
                "--mu", str(a_dict["mu"]),
                "--r", str(a_dict["r"]),
                "--exon-bp", str(a_dict["exon_bp"]),
                "--tile-bp", str(a_dict["tile_bp"]),
                "--jitter-bp", str(a_dict["jitter_bp"]),
                "--slim-scaling", str(a_dict["slim_scaling"]),
                "--slim-burn-in", str(a_dict["slim_burn_in"]),
                "--seed", str(a_dict["seed"]),
                "--trees", a_dict["trees"],
                "--skip-existing" if a_dict["skip_existing"] else "--no-skip-existing",
                "--threads-per-task", str(threads_per_task),
                "--_internal-single-run",
                "--_internal-coverage", str(coverage),
                "--_internal-label", label,
                "--_internal-seed-offset", str(seed_offset),
                "--_internal-row-json", str(row_json),
            ]
            if a_dict.get("vcf"):
                args += ["--vcf", a_dict["vcf"]]
            if a_dict.get("moments_config"):
                args += ["--moments-config", a_dict["moments_config"]]
            if a_dict.get("fixed_json"):
                args += ["--fixed-json", a_dict["fixed_json"]]

            # Run child
            subprocess.run(args, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Read result row
            return json.loads(row_json.read_text())

# ────────────────────────── plotting: error vs coverage ─────────────────────
def _true_param_map_from_cli(a) -> Dict[str, float]:
    return {
        "N_anc": float(a.N_anc),
        "N_YRI": float(a.N1),
        "N_CEU": float(a.N2),
        "m_YRI_CEU": float(a.m),
        "T_split": float(a.t_split),
    }

def plot_errors_vs_coverage(rows: List[Dict], truths: Dict[str, float], out_png: Path):
    covs = sorted(set(float(r["coverage_target"]) for r in rows))
    fit_keys = sorted({k.replace("fit_","") for r in rows for k in r.keys() if k.startswith("fit_")})
    if not fit_keys:
        print("[plots] no fitted parameters found; skipping plot.")
        return
    n = len(fit_keys)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3.5), dpi=150, squeeze=False)
    for j, p in enumerate(fit_keys):
        ax = axes[0, j]
        xs, ys = [], []
        for r in rows:
            if f"fit_{p}" in r and p in truths and truths[p] > 0:
                xs.append(float(r["coverage_target"]))
                ys.append(100.0 * (float(r[f"fit_{p}"]) - truths[p]) / truths[p])
        if xs:
            ax.scatter(xs, ys, s=14, alpha=0.7)
        medx, medy = [], []
        for c in covs:
            vals = [100.0 * (float(r.get(f"fit_{p}", np.nan)) - truths[p]) / truths[p]
                    for r in rows if float(r["coverage_target"]) == c and p in truths and truths[p] > 0]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                medx.append(c); medy.append(float(np.median(vals)))
        if medx:
            ax.plot(medx, medy, "-", lw=2)
        ax.axhline(0, ls="--", lw=1, color="gray")
        ax.set_xlabel("Coverage (selected fraction)")
        ax.set_ylabel(f"% error: {p}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_png}")

# ────────────────────────── utils ──────────────────────────
def _build_coverage_list(a) -> List[float]:
    vals: List[float] = []
    if a.coverage_grid.strip():
        vals.extend(float(x) for x in a.coverage_grid.split(","))
    if a.coverage_min is not None and a.coverage_max is not None and a.coverage_n:
        lo, hi, n = float(a.coverage_min), float(a.coverage_max), int(a.coverage_n)
        if n < 1: raise ValueError("--coverage-n must be >=1")
        if a.coverage_space == "lin":
            vals.extend(np.linspace(lo, hi, n).tolist())
        else:
            if lo <= 0 or hi <= 0:
                raise ValueError("For log spacing, coverage min/max must be > 0.")
            vals.extend(np.exp(np.linspace(np.log(lo), np.log(hi), n)).tolist())
    if a.coverage is not None:
        vals.append(float(a.coverage))
    vals = sorted({max(0.0, min(1.0, float(v))) for v in vals})
    return vals

# ────────────────────────── main ──────────────────────────
def main():
    a = parse_args()

    # Hidden single-run branch (used by Ray subprocess worker)
    if a._internal_single_run:
        # per-task thread caps respected; build a tiny NS for reuse
        class NS: pass
        ns = NS(); ns.__dict__.update(vars(a))
        row = run_single_sim_and_fit_serial(ns, coverage_target=a._internal_coverage,
                                            label_suffix=a._internal_label,
                                            seed_offset=a._internal_seed_offset)
        if a._internal_row_json:
            a._internal_row_json.parent.mkdir(parents=True, exist_ok=True)
            a._internal_row_json.write_text(json.dumps(row))
        else:
            print(json.dumps(row))
        return

    # Thread hygiene (user can override)
    tp = int(a.threads_per_task)
    os.environ.setdefault("OMP_NUM_THREADS", str(tp))
    os.environ.setdefault("MKL_NUM_THREADS", str(tp))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(tp))

    # if --cores used and ray-num-cpus not set, apply it
    if a.cores is not None and a.ray_num_cpus is None:
        a.ray_num_cpus = int(a.cores)

    cov_list = _build_coverage_list(a)

    # If no coverage controls → single run (tile-based)
    if not cov_list:
        _ = run_single_sim_and_fit_serial(a, coverage_target=None, label_suffix="", seed_offset=0)
        return

    # Sweep coverages with replicates
    base = Path(a.trees).with_suffix("")
    summary_csv = base.with_suffix(".sweep_summary.csv")
    rows: List[Dict] = []
    seed_counter = 0
    total_jobs = len(cov_list) * max(1, int(a.replicates))

    if a.ray:
        if ray is None:
            raise RuntimeError("Ray is not installed. `pip install ray` and try again.")

        # Init Ray
        if a.ray_address:
            ray.init(address=a.ray_address, ignore_reinit_error=True)
        else:
            ray.init(num_cpus=a.ray_num_cpus, ignore_reinit_error=True)

        a_dict = {
            "N_anc": a.N_anc, "N1": a.N1, "N2": a.N2, "t_split": a.t_split, "m": a.m,
            "samples": a.samples, "species": a.species, "dfe": a.dfe,
            "length": a.length, "mu": a.mu, "r": a.r,
            "exon_bp": a.exon_bp, "tile_bp": a.tile_bp, "jitter_bp": a.jitter_bp,
            "slim_scaling": a.slim_scaling, "slim_burn_in": a.slim_burn_in,
            "seed": a.seed, "trees": a.trees, "vcf": a.vcf,
            "skip_existing": a.skip_existing,
            "moments_config": (str(a.moments_config) if a.moments_config else None),
            "fixed_json": a.fixed_json,
        }

        script_path = str(Path(sys.argv[0]).resolve())
        obj_ids = []
        labels = []
        for cov in cov_list:
            for rep in range(a.replicates):
                label = ("cov{:.6f}".format(max(0.0, cov))).replace(".", "p") + (f"_r{rep}" if a.replicates > 1 else "")
                labels.append((label, cov, rep))
                obj_ids.append(ray_worker_subprocess.remote(script_path, a_dict, cov, label, seed_counter, tp))
                seed_counter += 1

        # gather with simple progress
        pending = list(obj_ids)
        results = []
        bar = tqdm(total=total_jobs, desc="sims", smoothing=0) if tqdm else None
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            res = ray.get(done[0])
            results.append(res)
            if bar: bar.update(1)
        if bar: bar.close()
        ray.shutdown()

        # stitch labels back onto rows
        for (label, cov, rep), row in zip(labels, results):
            row["label"] = label
            row["coverage_target"] = cov
            row["replicate"] = rep
            rows.append(row)

    else:
        it = ((cov, rep) for cov in cov_list for rep in range(a.replicates))
        iterator = tqdm(list(it), desc="sims") if tqdm else ((cov, rep) for cov in cov_list for rep in range(a.replicates))
        for cov, rep in iterator:
            label = ("cov{:.6f}".format(max(0.0, cov))).replace(".", "p") + (f"_r{rep}" if a.replicates > 1 else "")
            row = run_single_sim_and_fit_serial(a, coverage_target=cov,
                                                label_suffix=label, seed_offset=seed_counter)
            row["label"] = label
            row["coverage_target"] = cov
            row["replicate"] = rep
            rows.append(row)
            seed_counter += 1

    # Write CSV summary
    with open(summary_csv, "w", newline="") as f:
        cols = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[sweep] wrote {summary_csv} with {len(rows)} rows")

    # Plot errors vs coverage if moments ran and we can map truths
    if rows and any(k.startswith("fit_") for k in rows[0].keys()):
        truths = _true_param_map_from_cli(a)
        out_png = base.with_suffix(".errors_vs_coverage.png")
        plot_errors_vs_coverage(rows, truths, out_png)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# stdpopsim_bgs_split_isolation.py
# Split isolation + tiled BGS via stdpopsim/SLiM.
# Coverage control/sweeps, Poisson moments fit (unchanged), skip-existing, and error plots.

from __future__ import annotations
import argparse, json, csv, warnings
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

# ────────────────────────── split-isolation (engine) ─────────────────────────
class SplitIsolationModel(sps.DemographicModel):
    def __init__(self, N0: float, N1: float, N2: float, T: float, m: float):
        dem = msprime.Demography()
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))
        m = float(m)
        dem.set_migration_rate("YRI", "CEU", m)
        dem.set_migration_rate("CEU", "YRI", m)
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])
        super().__init__(id="split_isolation",
            description="ANC → (YRI, CEU) at T; symmetric migration m.",
            long_description="Custom msprime demography: split isolation with symmetric migration.",
            model=dem, generation_time=1)

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
    """Sort and drop any overlapping or out-of-bounds intervals."""
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
            # overlap → drop this one
            continue
        out.append((s, e))
        prev_end = e
    return np.array(out, dtype=int) if out else np.empty((0, 2), dtype=int)

def build_tiling_intervals(L: int, exon_bp: int, tile_bp: int, jitter_bp: int = 0) -> np.ndarray:
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    if jitter_bp > 0 and len(starts) > 0:
        # keep jitter modest to avoid systematic overlaps
        jitter = np.random.randint(-jitter_bp, jitter_bp + 1, size=len(starts))
        starts = np.clip(starts + jitter, 0, max(0, L - exon_bp))
    ends = np.minimum(starts + exon_bp, L).astype(int)
    iv = np.column_stack([starts, ends])
    return _sanitize_nonoverlap(iv, L)

def intervals_from_coverage(L: int, exon_bp: int, coverage: float, jitter_bp: int = 0) -> np.ndarray:
    if coverage <= 0:
        return np.empty((0, 2), dtype=int)
    if coverage >= 1.0:
        return np.array([[0, int(L)]], dtype=int)
    # tile_bp ≥ exon_bp when 0<coverage≤1
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

# ────────────────────────── moments helpers (Poisson; UNCHANGED) ────────────
def _moments_expected_sfs(params_vec: np.ndarray, param_names: List[str],
                          sample_sizes: "_OD[str, int]", config: Dict) -> moments.Spectrum:
    p = {k: float(v) for k, v in zip(param_names, params_vec)}
    g = split_isolation_graph(p)
    hap = [2 * n for n in sample_sizes.values()]
    demes_order = list(sample_sizes.keys())
    theta = float(p[param_names[0]]) * 4.0 * float(config["mutation_rate"]) * float(config["genome_length"])
    return moments.Spectrum.from_demes(g, sample_sizes=hap, sampled_demes=demes_order, theta=theta)

def _geometric_mean(lo: float, hi: float) -> float:
    return float(np.sqrt(float(lo) * float(hi)))

def _prepare_sample_sizes_from_sfs(sfs: moments.Spectrum) -> "_OD[str, int]":
    from collections import OrderedDict
    if hasattr(sfs, "pop_ids") and sfs.pop_ids:
        return OrderedDict((pid, (sfs.shape[i] - 1) // 2) for i, pid in enumerate(sfs.pop_ids))
    return OrderedDict((f"pop{i}", (n - 1) // 2) for i, n in enumerate(sfs.shape))

def fit_moments(sfs: moments.Spectrum, config: Dict,
                fixed_params: Dict[str, float] | None = None) -> Tuple[np.ndarray, float, List[str]]:
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

    # Poisson composite log-likelihood (UNCHANGED)
    def obj_log10(xlog10_free: np.ndarray) -> float:
        x_free = 10.0 ** np.asarray(xlog10_free, float)
        x_full = pack_free(x_free)
        try:
            expected = _moments_expected_sfs(x_full, param_names, ns, config)
            if getattr(sfs, "folded", False):
                expected = expected.fold()
            expected = np.maximum(expected, 1e-300)
            return float(np.sum(sfs * np.log(expected) - expected))
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
    p = argparse.ArgumentParser(description="Split isolation + tiled BGS via stdpopsim (SLiM). Poisson moments fit. Coverage sweeps with skip-existing and plots.")
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
    p.add_argument("--coverage", type=float, default=None)
    p.add_argument("--coverage-grid", type=str, default="")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--jitter-bp", type=int, default=0)
    # SLiM
    p.add_argument("--slim-scaling", type=float, default=10.0)
    p.add_argument("--slim-burn-in", type=float, default=5.0)
    # I/O
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--trees", default="sims/out.trees")
    p.add_argument("--vcf", default="")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip simulation if output trees already exist (default true). Use --no-skip-existing to force.")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    # Moments
    p.add_argument("--moments-config", type=Path,
                   help="JSON with {'priors':{...}, 'mutation_rate':..., 'genome_length':...}")
    p.add_argument("--fixed-json", type=str, default="")
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

def run_single_sim_and_fit(a, intervals: np.ndarray, label_suffix: str,
                           seed_offset: int = 0, coverage_target: float | None = None):
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
        # ensure SFS exists; if not, build from trees
        if not sfs_npy.exists():
            ts = tskit.load(str(trees_path))
            sfs = create_SFS(ts)
            np.save(sfs_npy, sfs.data)
        # moments: only run if requested and best.pkl missing
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
        # build row from meta if present
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        row = dict(
            label=(label_suffix or "single"),
            coverage_target=(coverage_target if coverage_target is not None else float(a.exon_bp)/float(a.tile_bp)),
            selected_bp=int(meta.get("selected_bp", 0)),
            selected_frac=float(meta.get("selected_frac", 0.0)),
            seg_sites=int(meta.get("sites", 0)),
        )
        # add fitted if present
        moments_pkl = outroot.with_suffix(".moments_best.pkl")
        if moments_pkl.exists():
            import pickle
            res = pickle.loads(moments_pkl.read_bytes())
            row["moments_best_ll"] = float(res.get("best_ll", np.nan))
            for k, v in (res.get("best_params") or {}).items():
                row[f"fit_{k}"] = float(v)
        return row

    # demography & contig
    model = SplitIsolationModel(N0=a.N_anc, N1=a.N1, N2=a.N2, T=a.t_split, m=a.m)
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
        for k in ("priors", "mutation_rate", "genome_length"):
            if k not in cfg:
                raise ValueError(f"--moments-config missing key: {k}")
        fixed = json.loads(a.fixed_json) if a.fixed_json else {}
        best_vec, best_ll, names = fit_moments(sfs, cfg, fixed_params=fixed)
        best = {k: float(v) for k, v in zip(names, best_vec)}
        out_pkl = outroot.with_suffix(".moments_best.pkl")
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
    # collect unique coverages sorted
    covs = sorted(set(float(r["coverage_target"]) for r in rows))
    # which parameters are present in fits?
    fit_keys = sorted({k.replace("fit_","") for r in rows for k in r.keys() if k.startswith("fit_")})
    if not fit_keys:
        print("[plots] no fitted parameters found; skipping plot.")
        return
    n = len(fit_keys)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3.5), dpi=150, squeeze=False)
    for j, p in enumerate(fit_keys):
        ax = axes[0, j]
        # points per replicate
        xs, ys = [], []
        for r in rows:
            if f"fit_{p}" in r and p in truths and truths[p] > 0:
                xs.append(float(r["coverage_target"]))
                ys.append(100.0 * (float(r[f"fit_{p}"]) - truths[p]) / truths[p])
        if xs:
            ax.scatter(xs, ys, s=14, alpha=0.7)
        # median per coverage
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

# ────────────────────────── main ──────────────────────────
def main():
    a = parse_args()

    # Determine coverage list
    cov_list = None
    if a.coverage_grid.strip():
        cov_list = [float(x) for x in a.coverage_grid.split(",")]
    elif a.coverage is not None:
        cov_list = [float(a.coverage)]

    # If no coverage controls → single run (tile-based)
    if not cov_list:
        intervals = build_tiling_intervals(a.length, a.exon_bp, a.tile_bp, jitter_bp=a.jitter_bp)
        _ = run_single_sim_and_fit(a, intervals, label_suffix="", seed_offset=0, coverage_target=None)
        return

    # Sweep coverages with replicates
    base = Path(a.trees).with_suffix("")
    summary_csv = base.with_suffix(".sweep_summary.csv")
    rows = []
    seed_counter = 0
    for cov in cov_list:
        intervals = intervals_from_coverage(a.length, a.exon_bp, cov, jitter_bp=a.jitter_bp)
        for rep in range(a.replicates):
            label = ("cov{:.6f}".format(max(0.0, cov))).replace(".", "p") + (f"_r{rep}" if a.replicates > 1 else "")
            row = run_single_sim_and_fit(a, intervals, label_suffix=label,
                                         seed_offset=seed_counter, coverage_target=cov)
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
    if any(k.startswith("fit_") for k in rows[0].keys()):
        truths = _true_param_map_from_cli(a)
        out_png = base.with_suffix(".errors_vs_coverage.png")
        plot_errors_vs_coverage(rows, truths, out_png)

if __name__ == "__main__":
    main()

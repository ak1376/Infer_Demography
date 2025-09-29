#!/usr/bin/env python3
# stdpopsim_bgs_split_isolation.py
# Split isolation demography (custom msprime) + tiled BGS via stdpopsim + SLiM.
# Also: optional moments fitting of the generated SFS using a demes-based model.

import argparse
import json
from pathlib import Path
import warnings
from typing import List, Dict, Tuple, OrderedDict as _OD

import numpy as np
import msprime
import stdpopsim as sps
import tskit
import moments
import demes
import nlopt
import numdifftools as nd

# ────────────────────────── (optional) calm noisy warnings ──────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", msprime.TimeUnitsMismatchWarning)

# ────────────────────────── SFS utility ──────────────────────────
def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    """Build a moments.Spectrum from a tskit TreeSequence using pops with sampled individuals."""
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
    arr = ts.allele_frequency_spectrum(
        sample_sets=sample_sets, mode="site", polarised=True, span_normalise=False
    )
    sfs = moments.Spectrum(arr)
    sfs.pop_ids = pop_ids
    return sfs

# ────────────────────────── split-isolation (stdpopsim Engine model) ─────────────────────────
class SplitIsolationModel(sps.DemographicModel):
    """ANC splits at time T into YRI and CEU; symmetric migration m between them (SLiM-friendly)."""
    def __init__(self, N0: float, N1: float, N2: float, T: float, m: float):
        dem = msprime.Demography()
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))
        m = float(m)
        dem.set_migration_rate("YRI", "CEU", m)
        dem.set_migration_rate("CEU", "YRI", m)
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])
        super().__init__(
            id="split_isolation",
            description="ANC → (YRI, CEU) at T; symmetric migration m.",
            long_description="Custom msprime demography: split isolation with symmetric migration.",
            model=dem,
            generation_time=1,
        )

# ────────────────────────── split-isolation (demes Graph for moments) ─────────────────────────
def split_isolation_graph(params: Dict[str, float]) -> demes.Graph:
    """
    Demes graph:
      ANC (N0) → split at T → YRI (N1), CEU (N2) with symmetric migration m.
    Parameter keys accepted: N0|N_anc, N1|N_YRI, N2|N_CEU, T|t_split, m|m_sym|m12|m21
    """
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

# ────────────────────────── BGS tiling helpers ──────────────────────────
def build_tiling_intervals(L: int, exon_bp: int, tile_bp: int) -> np.ndarray:
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    ends = np.minimum(starts + exon_bp, L).astype(int)
    return np.column_stack([starts, ends])  # [N,2)

def make_contig_with_tiled_dfe(
    length: int, mu: float, r: float, species: str, dfe_id: str, exon_bp: int, tile_bp: int
):
    sp = sps.get_species(species)
    try:
        contig = sp.get_contig(
            chromosome=None, length=int(length), mutation_rate=float(mu), recombination_rate=float(r),
        )
    except TypeError:
        print("[warn] stdpopsim.get_contig() does not accept 'recombination_rate' "
              "in this version; using the species default r instead.")
        contig = sp.get_contig(chromosome=None, length=int(length), mutation_rate=float(mu))
    dfe = sp.get_dfe(dfe_id)
    intervals = build_tiling_intervals(int(length), exon_bp, tile_bp)
    contig.add_dfe(intervals=intervals, DFE=dfe)
    return contig, intervals

# ────────────────────────── moments helpers (informed by your moments_inference.py) ──────────────────────────
def _moments_expected_sfs(
    params_vec: np.ndarray,
    param_names: List[str],
    sample_sizes: "_OD[str, int]",
    config: Dict,
) -> moments.Spectrum:
    """
    Build expected SFS via moments.from_demes using our split-isolation demes graph.
    theta = 4 * N0 * mu * L, where N0 is first param in param_names (ancestral size).
    """
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

def fit_moments(
    sfs: moments.Spectrum,
    config: Dict,
    fixed_params: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Fit split-isolation moments model. Returns (best_params_vector in config['priors'] order, best_ll, param_names).
    """
    priors = config["priors"]
    param_names = list(priors.keys())
    lb = np.array([priors[p][0] for p in param_names], float)
    ub = np.array([priors[p][1] for p in param_names], float)
    start = np.array([_geometric_mean(*priors[p]) for p in param_names], float)

    # fixed handling
    fixed_params = dict(fixed_params or {})
    fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
    free_idx  = [i for i, n in enumerate(param_names) if n not in fixed_params]
    x0 = start.copy()
    for i in fixed_idx:
        x0[i] = float(fixed_params[param_names[i]])
        if not (lb[i] <= x0[i] <= ub[i]):
            raise ValueError(f"Fixed {param_names[i]}={x0[i]} outside bounds [{lb[i]},{ub[i]}]")

    # sample sizes from SFS
    ns = _prepare_sample_sizes_from_sfs(sfs)

    # objective (Poisson composite log-likelihood) on log10 free params
    def pack_free(x_free: np.ndarray) -> np.ndarray:
        x = x0.copy()
        for j, i in enumerate(free_idx):
            x[i] = float(x_free[j])
        return x

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
    # small perturbation for robustness
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
    p = argparse.ArgumentParser(description="Split isolation + tiled BGS with stdpopsim (SLiM engine) + optional moments fit.")
    # Demography (split-isolation only)
    p.add_argument("--N-anc", type=float, required=True, help="Ancestral size (ANC)")
    p.add_argument("--N1", type=float, required=True, help="Size of YRI")
    p.add_argument("--N2", type=float, required=True, help="Size of CEU")
    p.add_argument("--t-split", type=float, required=True, help="Split time (generations)")
    p.add_argument("--m", type=float, default=0.0, help="Symmetric migration rate (YRI↔CEU)")
    # Samples
    p.add_argument("--samples", default="YRI:20,CEU:20", help="Comma list pop:n, e.g. 'YRI:20,CEU:20'")
    # Species/DFE for contig
    p.add_argument("--species", default="HomSap", help="stdpopsim species for contig/DFE")
    p.add_argument("--dfe", default="Gamma_K17", help="stdpopsim DFE ID for tiled 'exons'")
    # Genome/rates
    p.add_argument("--length", type=int, default=200_000, help="Genome length (bp)")
    p.add_argument("--mu", type=float, default=1e-8, help="Mutation rate per bp/gen")
    p.add_argument("--r", type=float, default=1e-8, help="Recombination rate per bp/gen")
    # BGS tiling
    p.add_argument("--exon-bp", type=int, default=200, help="Exon tile size (bp)")
    p.add_argument("--tile-bp", type=int, default=5000, help="Tile period (bp): one exon every tile_bp")
    # SLiM knobs
    p.add_argument("--slim-scaling", type=float, default=10.0, help="SLiM rescaling factor Q")
    p.add_argument("--slim-burn-in", type=float, default=5.0, help="Burn-in in units of Ne")
    # I/O
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--trees", default="sims/out.trees")
    p.add_argument("--vcf", default="", help="Optional VCF path")
    # Moments fitting (optional)
    p.add_argument("--moments-config", type=Path, help="JSON with {'priors':{...}, 'mutation_rate':..., 'genome_length':...}")
    p.add_argument("--fixed-json", type=str, default="", help="JSON dict of fixed params, e.g. '{\"m\":1e-4}'")
    return p.parse_args()

# ────────────────────────── main ──────────────────────────
def main():
    a = parse_args()

    # 1) Demography (split-isolation for SLiM engine)
    model = SplitIsolationModel(N0=a.N_anc, N1=a.N1, N2=a.N2, T=a.t_split, m=a.m)

    # 2) Contig + tiled DFE (BGS)
    contig, intervals = make_contig_with_tiled_dfe(
        length=a.length, mu=a.mu, r=a.r, species=a.species, dfe_id=a.dfe,
        exon_bp=a.exon_bp, tile_bp=a.tile_bp,
    )

    # 3) Samples
    try:
        samples = {k.strip(): int(v) for k, v in (x.split(":") for x in a.samples.split(","))}
    except Exception as e:
        raise ValueError(f"Could not parse --samples '{a.samples}'. Use 'YRI:20,CEU:20'.") from e

    # 4) SLiM simulation via stdpopsim engine
    engine = sps.get_engine("slim")
    ts = engine.simulate(
        model, contig, samples, seed=a.seed,
        slim_scaling_factor=a.slim_scaling, slim_burn_in=a.slim_burn_in,
    )

    # 5) Ensure output dirs exist
    Path(a.trees).parent.mkdir(parents=True, exist_ok=True)
    if a.vcf:
        Path(a.vcf).parent.mkdir(parents=True, exist_ok=True)

    # 6) Write trees (+ optional VCF)
    ts.dump(a.trees)
    if a.vcf:
        with open(a.vcf, "w") as f:
            ts.write_vcf(f)

    # 7) Build & save SFS
    sfs = create_SFS(ts)
    sfs_npy = Path(a.trees).with_suffix(".sfs.npy")
    np.save(sfs_npy, sfs.data)
    print(f"Wrote {sfs_npy}")

    # 8) Sidecars (meta / exon BED)
    outroot = Path(a.trees).with_suffix("")   # e.g., sims/out
    meta_path = outroot.with_suffix(".meta.json")
    bed_path  = outroot.with_suffix(".exons.bed")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    bed_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(
        model="split_isolation",
        params=dict(N_anc=a.N_anc, N1=a.N1, N2=a.N2, t_split=a.t_split, m=a.m),
        species=a.species,
        samples=a.samples,
        length=a.length, mu=a.mu, r=a.r,
        dfe=a.dfe, exon_bp=a.exon_bp, tile_bp=a.tile_bp,
        slim_scaling=a.slim_scaling, slim_burn_in=a.slim_burn_in,
        seed=a.seed, trees=a.trees, vcf=a.vcf,
        sites=int(ts.num_sites), sequence_length=int(ts.sequence_length),
        sfs_npy=str(sfs_npy),
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    with open(bed_path, "w") as bed:
        for s, e in intervals:
            bed.write(f"chr1\t{s}\t{e}\n")

    print(f"Wrote {a.trees}; sites={ts.num_sites}; length={ts.sequence_length:.0f} bp")
    if a.vcf:
        print(f"Wrote {a.vcf}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {bed_path}")

    # 9) Optional: moments fitting (uses demes graph + config priors)
    if a.moments_config:
        cfg = json.loads(Path(a.moments_config).read_text())
        # sanity keys
        for k in ("priors", "mutation_rate", "genome_length"):
            if k not in cfg:
                raise ValueError(f"--moments-config missing key: {k}")

        fixed = json.loads(a.fixed_json) if a.fixed_json else {}
        best_vec, best_ll, names = fit_moments(sfs, cfg, fixed_params=fixed)
        best = {k: float(v) for k, v in zip(names, best_vec)}
        print(f"[moments] best_ll={best_ll:.6g}  best_params={best}")

        # save results + short log
        out_pkl = outroot.with_suffix(".moments_best.pkl")
        with open(out_pkl, "wb") as f:
            import pickle
            pickle.dump(
                dict(best_params=best, best_ll=float(best_ll), param_order=names, fixed_params=fixed),
                f
            )
        log_dir = outroot.parent / "logs" / "moments"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "optim_single.txt").write_text(
            "# moments single-run optimisation\n"
            f"best_ll: {best_ll}\n"
            f"best_params: {json.dumps(best)}\n"
        )
        print(f"[moments] wrote {out_pkl}")

if __name__ == "__main__":
    main()

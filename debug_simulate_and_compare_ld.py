#!/usr/bin/env python3
"""
debug_simulate_and_compare_ld.py

Simulate 2-pop or 3-pop tree sequences with msprime, split into windows,
write per-window .trees + .vcf.gz + shared samples.txt + flat_map.txt,
then compute LD stats on CPU (moments) vs GPU (pg_gpu) via your repo's
src/LD_stats.compute_ld_window.

This is meant to be a *minimal reproducible* harness to debug CPU/GPU mismatch.

Example:
  # 2 pops
  python debug_simulate_and_compare_ld.py --outdir /tmp/ld_dbg_2pop --k 2 --windows 3 --seed 1

  # 3 pops
  python debug_simulate_and_compare_ld.py --outdir /tmp/ld_dbg_3pop --k 3 --windows 3 --seed 1
"""

from __future__ import annotations

import sys
import json
import gzip
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# --- require msprime/tskit ---
import tskit
import msprime

# --- import your repo function ---
REPO_ROOT = Path(__file__).resolve().parent
if (REPO_ROOT / "src").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.LD_stats import compute_ld_window  # your file name
except ModuleNotFoundError:
    from src.ld_stats import compute_ld_window  # fallback


# ----------------------------
# I/O helpers
# ----------------------------

def write_flat_map(flat_map_path: Path, L: int, r_per_bp: float) -> None:
    """
    Write a flat recombination map in the format your LD_stats expects:
      pos   Map(cM)
    Map(cM) is cumulative, so Map(cM) = 100 * (pos_bp * r_per_bp).
    """
    flat_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flat_map_path, "w") as f:
        f.write("pos\tMap(cM)\n")
        # two-point map is enough for interpolation/average rate
        f.write(f"0\t0\n")
        f.write(f"{L}\t{(L * r_per_bp) * 100.0}\n")


def write_samples_file(samples_path: Path, pop_to_sample_names: Dict[str, List[str]]) -> None:
    """
    moments pop_file format expected by moments.LD.Parsing.compute_ld_statistics:

      sample<TAB>pop
      YRI_0<TAB>YRI
      ...

    IMPORTANT: must include a header with columns named exactly: 'sample' and 'pop'
    """
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(samples_path, "w") as f:
        f.write("sample\tpop\n")  # <-- REQUIRED for moments
        for pop, names in pop_to_sample_names.items():
            for s in names:
                f.write(f"{s}\t{pop}\n")


def gzip_file(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
        f_out.write(f_in.read())


def write_window_vcf_gz(ts_win: tskit.TreeSequence, vcf_gz_path: Path, sample_names: List[str]) -> None:
    """
    Write a VCF for this window and gzip it.
    """
    vcf_gz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_vcf = vcf_gz_path.with_suffix("")  # remove .gz -> .vcf
    # tskit writes VCF to an open text stream
    with open(tmp_vcf, "w") as f:
        ts_win.write_vcf(f, individual_names=sample_names)
    gzip_file(tmp_vcf, vcf_gz_path)
    tmp_vcf.unlink(missing_ok=True)


# ----------------------------
# Simulation helpers
# ----------------------------

def simulate_ts(
    *,
    k: int,
    L: int,
    n_dip_per_pop: int,
    r_per_bp: float,
    mu: float,
    seed: int,
) -> Tuple[tskit.TreeSequence, List[str], Dict[str, List[str]]]:
    """
    Simulate a simple demographic model that produces k populations at present.
    Returns:
      ts: mutated tree sequence
      sample_names (length = total haploid samples)
      pop_to_sample_names mapping for writing samples.txt
    """
    rng = np.random.default_rng(seed)

    # --- Demography ---
    dem = msprime.Demography()

    # We'll name pops like your config keys so it plugs into your LD code easily
    if k == 2:
        pop_names = ["YRI", "CEU"]
    elif k == 3:
        pop_names = ["YRI", "CEU", "CHB"]
    else:
        raise ValueError("k must be 2 or 3")

    # effective sizes (just reasonable defaults)
    for p in pop_names:
        dem.add_population(name=p, initial_size=10_000)

    # simple split history so pops share ancestry
    # times in generations
    if k == 2:
        dem.add_population(name="ANC", initial_size=10_000)
        dem.add_population_split(time=2000, derived=["YRI", "CEU"], ancestral="ANC")
    else:
        dem.add_population(name="ANC", initial_size=10_000)
        dem.add_population(name="EURASIA", initial_size=10_000)
        # CEU/CHB split more recent; both split from EURASIA; EURASIA + YRI split from ANC
        dem.add_population_split(time=1200, derived=["CEU", "CHB"], ancestral="EURASIA")
        dem.add_population_split(time=2500, derived=["YRI", "EURASIA"], ancestral="ANC")

    # --- Samples (diploid individuals) ---
    samples = []
    pop_to_sample_names: Dict[str, List[str]] = {p: [] for p in pop_names}
    sample_names_all: List[str] = []

    # In tskit VCF writing, `individual_names` length must equal number of individuals,
    # but `write_vcf(individual_names=...)` uses names for diploid individuals.
    # However, your moments pop_file is per-sample name. We will instead name *VCF samples*
    # by haploid sample nodes using `tskit` default? That’s awkward.
    #
    # Easiest: write VCF with individuals and let each individual be diploid;
    # moments pop_file expects the VCF sample IDs (individual IDs). So we name individuals.
    #
    # Therefore:
    #  - simulate with `ploidy=2` and `samples=[msprime.SampleSet(...)]` (diploids)
    #  - write VCF; sample IDs are individuals; pop_file uses those names.

    sample_sets = []
    for p in pop_names:
        sample_sets.append(msprime.SampleSet(num_samples=n_dip_per_pop, population=p, ploidy=2))

        # names of individuals for VCF/pop_file
        names = [f"{p}_{i}" for i in range(n_dip_per_pop)]
        pop_to_sample_names[p] = names
        sample_names_all.extend(names)

    # --- Simulate ancestry + mutations ---
    ts_anc = msprime.sim_ancestry(
        samples=sample_sets,
        demography=dem,
        sequence_length=L,
        recombination_rate=r_per_bp,
        random_seed=seed,
    )

    ts_mut = msprime.sim_mutations(ts_anc, rate=mu, random_seed=seed + 13)

    # Make sure we have variants; if not, bump mu slightly (rare at small L)
    if ts_mut.num_sites == 0:
        ts_mut = msprime.sim_mutations(ts_anc, rate=mu * 10, random_seed=seed + 17)

    return ts_mut, sample_names_all, pop_to_sample_names


def split_into_windows(ts: tskit.TreeSequence, L: int, windows: int) -> List[Tuple[int, int, tskit.TreeSequence]]:
    """
    Split genome [0, L) into `windows` equal bp chunks using keep_intervals.
    """
    edges = np.linspace(0, L, windows + 1).astype(int)
    out = []
    for i in range(windows):
        a, b = int(edges[i]), int(edges[i + 1])
        # keep_intervals expects list of [left, right) in *bp coordinates*
        ts_win = ts.keep_intervals([[a, b]], simplify=True)
        out.append((a, b, ts_win))
    return out


# ----------------------------
# Comparison helpers
# ----------------------------

def is_gpu_dict(x: Any) -> bool:
    return isinstance(x, dict) and ("sums" in x) and ("stats" in x) and ("bins" in x)


def extract_ld_dict(x: Any) -> Dict[str, np.ndarray]:
    """
    Extract per-bin LD arrays into dict[name] = array(nbins).
    Works for:
      - your GPU dict format
      - moments LDstats object (best-effort)
    """
    if is_gpu_dict(x):
        sums = x["sums"]
        ld_names, _h_names = x["stats"]
        nb = len(x["bins"])
        ld_mat = np.vstack([np.asarray(sums[i], dtype=np.float64) for i in range(nb)])
        return {nm: ld_mat[:, j].copy() for j, nm in enumerate(ld_names)}

    # moments object fallback
    names = None
    for meth in ("names", "stats", "keys"):
        if hasattr(x, meth):
            try:
                maybe = getattr(x, meth)()
                if isinstance(maybe, (list, tuple)) and maybe and isinstance(maybe[0], str):
                    names = list(maybe)
                    break
            except Exception:
                pass
    if names is None:
        raise RuntimeError("Could not extract stat names from moments object.")

    out: Dict[str, np.ndarray] = {}
    for nm in names:
        try:
            arr = np.asarray(x[nm], dtype=np.float64)
            if arr.ndim == 1:
                out[nm] = arr
        except Exception:
            pass
    return out


def first_mismatch(cpu: Dict[str, np.ndarray], gpu: Dict[str, np.ndarray], atol: float, rtol: float) -> Optional[str]:
    common = sorted(set(cpu) & set(gpu))
    for nm in common:
        a = cpu[nm]
        b = gpu[nm]
        if a.shape != b.shape:
            return f"{nm} shape mismatch {a.shape} vs {b.shape}"
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            diff = np.abs(a - b)
            worst = int(np.argmax(diff))
            return f"{nm} mismatch worst_bin={worst} CPU={a[worst]:.6e} GPU={b[worst]:.6e} abs={diff[worst]:.3e}"
    return None


# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--k", type=int, choices=[2, 3], required=True)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--L", type=int, default=2_000_000)
    p.add_argument("--n-dip", type=int, default=20)
    p.add_argument("--r-per-bp", type=float, default=1e-8)
    p.add_argument("--mu", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--r-bins", type=str, default="0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3")
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--use-gpu-ld", action="store_true", help="set config use_gpu_ld=true (still requires pg_gpu installed)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir.resolve()
    win_dir = outdir / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    r_bins = np.array([float(x) for x in args.r_bins.split(",")], dtype=float)

    # --- simulate ---
    ts, sample_names_all, pop_to_sample_names = simulate_ts(
        k=args.k,
        L=args.L,
        n_dip_per_pop=args.n_dip,
        r_per_bp=args.r_per_bp,
        mu=args.mu,
        seed=args.seed,
    )

    print(f"[sim] k={args.k} L={args.L} n_sites={ts.num_sites} n_trees={ts.num_trees}")

    # --- write shared files ---
    samples_file = win_dir / "samples.txt"
    flat_map_file = win_dir / "flat_map.txt"
    write_samples_file(samples_file, pop_to_sample_names)
    write_flat_map(flat_map_file, args.L, args.r_per_bp)

    # --- config to feed compute_ld_window ---
    if args.k == 2:
        pops = ["YRI", "CEU"]
    else:
        pops = ["YRI", "CEU", "CHB"]

    config = {
        "use_gpu_ld": bool(args.use_gpu_ld),
        "recombination_rate": float(args.r_per_bp),  # used if no map interpolation needed
        "num_samples": {p: int(args.n_dip) for p in pops},  # diploids per pop (matches pop_file)
    }

    # --- split + write per-window files ---
    windows = split_into_windows(ts, args.L, args.windows)

    # We write each window's VCF using the SAME individual ordering (sample_names_all)
    # because msprime created individuals in the order we provided SampleSet objects.
    # That matches pop_file names we wrote.
    for i, (a, b, ts_win) in enumerate(windows):
        ts_path = win_dir / f"window_{i}.trees"
        vcf_gz_path = win_dir / f"window_{i}.vcf.gz"

        ts_win.dump(str(ts_path))
        write_window_vcf_gz(ts_win, vcf_gz_path, sample_names=sample_names_all)

        print(f"[write] window {i}: [{a},{b}) sites={ts_win.num_sites} -> {ts_path.name}, {vcf_gz_path.name}")

    # --- compute CPU vs GPU ---
    print("\n" + "=" * 80)
    print("[run] CPU vs GPU compare via src/LD_stats.compute_ld_window")
    print("      config pops:", list(config["num_samples"].keys()))
    print("      use_gpu_ld:", config["use_gpu_ld"])
    print("=" * 80)

    for i in range(args.windows):
        vcf_gz = win_dir / f"window_{i}.vcf.gz"
        ts_file = win_dir / f"window_{i}.trees"

        cpu_out = compute_ld_window(
            window_index=i,
            vcf_gz=vcf_gz,
            samples_file=samples_file,
            rec_map_file=flat_map_file,
            ts_file=ts_file,
            r_bins=r_bins,
            config=config,
            request_gpu=False,
        )

        gpu_out = compute_ld_window(
            window_index=i,
            vcf_gz=vcf_gz,
            samples_file=samples_file,
            rec_map_file=flat_map_file,
            ts_file=ts_file,
            r_bins=r_bins,
            config=config,
            request_gpu=True,
        )

        cpu_ld = extract_ld_dict(cpu_out)
        gpu_ld = extract_ld_dict(gpu_out)

        mm = first_mismatch(cpu_ld, gpu_ld, atol=args.atol, rtol=args.rtol)
        if mm is None:
            print(f"[OK] window {i}: CPU == GPU (within atol/rtol)")
        else:
            print(f"[FAIL] window {i}: {mm}")
            # stop at first mismatch to keep output readable
            break


if __name__ == "__main__":
    main()

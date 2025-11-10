#!/usr/bin/env python3
"""Compute LD statistics for **one window** of one simulation.

Called by Snakemake rule ``ld_window``:

    python compute_ld_window.py \
        --sim-dir      MomentsLD/LD_stats/sim_0007          # same as --output-root/sim_<sid>
        --window-index 42
        --config-file  config_files/experiment_config_*DEMOGRAPHIC_MODEL*.json
        --r-bins       "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"
        --use-gpu      # optional flag to enable GPU acceleration

The script expects the following files already exist in *sim-dir*:
    windows/window_<idx>.vcf.gz       (compressed VCF of the replicate)
    samples.txt                       (two‑column sample/pop table)
    flat_map.txt                      (pos \t cM map)

It writes one pickle:
    LD_stats/LD_stats_window_<idx:04d>.pkl     (moments.LD.LDstats object)
"""
from __future__ import annotations

import argparse
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import moments
import tskit

# Optional GPU acceleration
try:
    from pg_gpu.haplotype_matrix import HaplotypeMatrix
    _HAVE_GPU = True
except ImportError:
    _HAVE_GPU = False

# ------------------------------------------------------------------ CLI


def parse_args():
    p = argparse.ArgumentParser("compute LD stats for one window")
    p.add_argument(
        "--sim-dir", required=True, type=Path, help="MomentsLD/LD_stats/sim_<sid>"
    )
    p.add_argument(
        "--window-index", required=True, type=int, help="zero‑based window index"
    )
    p.add_argument(
        "--config-file",
        required=True,
        type=Path,
        help="experiment_config_bottleneck.json",
    )
    p.add_argument(
        "--r-bins",
        required=True,
        help="comma‑separated list of recombination‑bin edges, e.g. '0,1e-6,1e-5,1e-4' ",
    )
    p.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration via pg_gpu"
    )
    return p.parse_args()


# ------------------------------------------------------------------ GPU helpers

def _hash_positions(positions: np.ndarray) -> str:
    """SHA1 of float64 positions, little-endian bytes."""
    buf = positions.astype(np.float64).tobytes(order="C")
    return hashlib.sha1(buf).hexdigest()


def build_sample_sets(ts: tskit.TreeSequence):
    """Build sample sets mapping from tree sequence populations."""
    # Use population names if present; else pick first two non-empty
    pop_names = {}
    for pid in range(ts.num_populations):
        pop = ts.population(pid)
        name = None
        if hasattr(pop, "name") and getattr(pop, "name", None):
            name = pop.name
        elif hasattr(pop, "metadata") and isinstance(pop.metadata, dict):
            name = pop.metadata.get("name")
        pop_names[pid] = name

    samples_by_pid = {pid: [int(x) for x in ts.samples(population=pid)]
                      for pid in range(ts.num_populations)}

    # Try to find deme0/deme1 by name, otherwise use first two non-empty
    pid_d0 = next((pid for pid, nm in pop_names.items() if nm == "deme0"), None)
    pid_d1 = next((pid for pid, nm in pop_names.items() if nm == "deme1"), None)
    
    if pid_d0 is None or pid_d1 is None:
        nonempty = [(pid, s) for pid, s in samples_by_pid.items() if len(s) > 0]
        nonempty.sort(key=lambda x: x[0])
        if len(nonempty) < 2:
            raise ValueError("Need two non-empty pops in the tree sequence.")
        pid_d0, pid_d1 = nonempty[0][0], nonempty[1][0]

    ss = {"deme0": samples_by_pid[pid_d0], "deme1": samples_by_pid[pid_d1]}
    if len(ss["deme0"]) == 0 or len(ss["deme1"]) == 0:
        raise ValueError("Empty sample set(s)")
    return ss


def gpu_ld_from_trees(ts_path: str, r_bins, r_per_bp: float, pop1: str = "deme0", pop2: str = "deme1") -> dict:
    """Compute LD stats using GPU acceleration from tree sequence file."""
    import cupy as cp
    
    # Find GPU with most available memory
    best_gpu = 0
    max_free_mem = 0
    for gpu_id in range(cp.cuda.runtime.getDeviceCount()):
        cp.cuda.Device(gpu_id).use()
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_gpu = gpu_id
    
    # Use the GPU with most free memory
    cp.cuda.Device(best_gpu).use()
    gpu_name = cp.cuda.runtime.getDeviceProperties(best_gpu)['name'].decode()
    print(f"📱 Using GPU {best_gpu} ({gpu_name}) with {max_free_mem/1e9:.1f}GB free memory")
    
    # Convert genetic distance bins to physical distance
    bp_bins = np.array(r_bins, dtype=float) / float(r_per_bp)
    
    ts = tskit.load(ts_path)
    sample_sets = build_sample_sets(ts)
    
    if pop1 not in sample_sets or pop2 not in sample_sets:
        raise KeyError(f"Requested pops ({pop1},{pop2}) not in sample_sets={list(sample_sets)}")

    h = HaplotypeMatrix.from_ts(ts)
    # Set sample sets
    normalized = {k: [int(x) for x in v] for k, v in sample_sets.items()}
    if hasattr(h, "set_sample_sets") and callable(getattr(h, "set_sample_sets")):
        h.set_sample_sets(normalized)
    else:
        h.sample_sets = normalized
    
    # Apply biallelic filter
    h_filt = h.apply_biallelic_filter()

    try:
        stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1=pop1,
            pop2=pop2,
            raw=True,
            ac_filter=True,
            fp64=True
        )
    except TypeError:
        stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1=pop1,
            pop2=pop2,
            raw=True,
            ac_filter=True
        )

    # Convert to moments-compatible format
    MOMENTS_ORDER = [
        'DD_0_0', 'DD_0_1', 'DD_1_1',
        'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1', 'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
        'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1', 'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1'
    ]
    H_STAT_NAMES = ['H_0_0', 'H_0_1', 'H_1_1']
    
    # Assemble per-bin vectors in moments order
    sums = []
    for (b0, b1) in zip(bp_bins[:-1], bp_bins[1:]):
        key = (float(b0), float(b1))
        od = stats_by_bin.get(key, None)
        if od is None:
            sums.append(np.zeros(len(MOMENTS_ORDER), dtype=float))
        else:
            sums.append(np.array([od[name] for name in MOMENTS_ORDER], dtype=float))

    # Compute H terms from tree sequence
    samples_vec = np.array(list(ts.samples()), dtype=np.int64)
    node_to_idx = {int(n): i for i, n in enumerate(samples_vec)}
    idx_A = np.array([node_to_idx[n] for n in sample_sets["deme0"] if n in node_to_idx], dtype=np.int64)
    idx_B = np.array([node_to_idx[n] for n in sample_sets["deme1"] if n in node_to_idx], dtype=np.int64)

    H00 = H01 = H11 = 0.0
    for var in ts.variants(samples=samples_vec, alleles=None, impute_missing_data=False):
        g = var.genotypes
        gA = g[idx_A]
        gB = g[idx_B]
        valA = gA[gA >= 0]
        valB = gB[gB >= 0]
        if valA.size == 0 or valB.size == 0:
            continue
        pA = float(valA.mean())
        pB = float(valB.mean())
        H00 += 2.0 * pA * (1.0 - pA)
        H11 += 2.0 * pB * (1.0 - pB)
        H01 += pA * (1.0 - pB) + (1.0 - pA) * pB

    sums.append(np.array([H00, H01, H11], dtype=float))

    bins_gen = [(np.float64(r_bins[i]), np.float64(r_bins[i + 1])) for i in range(len(r_bins) - 1)]

    result = {
        "bins": bins_gen,
        "sums": sums,
        "stats": (MOMENTS_ORDER, H_STAT_NAMES),
        "pops": [pop1, pop2],
    }
    
    # Aggressive GPU memory cleanup
    import cupy as cp
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass
    
    return result


# ------------------------------------------------------------------ main routine


def main():
    args = parse_args()

    sim_dir = args.sim_dir.resolve()
    idx = args.window_index
    r_bins = np.array([float(x) for x in args.r_bins.split(",")])

    # Load config to get recombination rate and GPU preference
    with open(args.config_file) as f:
        config = json.load(f)
    
    r_per_bp = float(config.get("recombination_rate", 1e-8))
    use_gpu = args.use_gpu and _HAVE_GPU and config.get("use_gpu_ld", False)

    vcf_gz = sim_dir / "windows" / f"window_{idx}.vcf.gz"
    samples_t = sim_dir / "windows" / "samples.txt"
    rec_map_t = sim_dir / "windows" / "flat_map.txt"
    out_dir = sim_dir / "LD_stats"
    out_pkl = out_dir / f"LD_stats_window_{idx}.pkl"

    # sanity checks -------------------------------------------------
    for path in (vcf_gz, samples_t, rec_map_t):
        if not path.exists():
            raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # skip if already done (idempotent rule) ------------------------
    if out_pkl.exists():
        print(f"✓ window {idx}: already computed → {out_pkl.relative_to(sim_dir)}")
        return

    # GPU path: use tree sequence if available
    if use_gpu:
        # Tree sequences are saved in the same windows directory
        ts_file = sim_dir / "windows" / f"window_{idx}.trees"
        
        if ts_file.exists():
            print(f"🚀 window {idx}: ATTEMPTING GPU acceleration from {ts_file}")
            try:
                import time
                gpu_start = time.perf_counter()
                stats = gpu_ld_from_trees(str(ts_file), r_bins, r_per_bp, pop1="deme0", pop2="deme1")
                gpu_time = time.perf_counter() - gpu_start
                
                # Clear GPU memory after computation
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except:
                    pass  # Ignore if cupy not available
                
                with out_pkl.open("wb") as fh:
                    pickle.dump(stats, fh)
                print(f"✅ window {idx:04d}: GPU LD stats completed in {gpu_time:.2f}s → {out_pkl.relative_to(sim_dir)}")
                print(f"🎯 window {idx}: GPU processed {stats.get('_sitecheck', {}).get('S_filt', 'unknown')} sites")
                return
            except Exception as e:
                # Clear GPU memory on error too
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except:
                    pass
                print(f"❌ window {idx}: GPU failed ({e}), falling back to traditional moments")
        else:
            print(f"⚠ window {idx}: tree sequence not found at {ts_file}, using traditional moments")
    else:
        gpu_reason = []
        if not args.use_gpu:
            gpu_reason.append("--use-gpu not specified")
        if not _HAVE_GPU:
            gpu_reason.append("pg_gpu not installed")
        if not config.get("use_gpu_ld", False):
            gpu_reason.append("use_gpu_ld=false in config")
        print(f"🐌 window {idx}: using traditional moments LD ({', '.join(gpu_reason)})")

    # Traditional moments path
    print(f"🐌 window {idx}: computing with traditional moments LD")
    # ----------------------------------------- grab every unique pop ID
    # read unique pop IDs from the samples file
    with samples_t.open() as fh:
        pops = sorted(
            {
                line.split()[1]
                for line in fh
                if line.strip() and not line.startswith("sample")
            }
        )

    # compute LD statistics ----------------------------------------
    import time
    traditional_start = time.perf_counter()
    stats = moments.LD.Parsing.compute_ld_statistics(
        str(vcf_gz),
        rec_map_file=str(rec_map_t),
        pop_file=str(samples_t),
        pops=pops,
        r_bins=r_bins,
        report=False,
    )
    traditional_time = time.perf_counter() - traditional_start

    # write pickle --------------------------------------------------
    with out_pkl.open("wb") as fh:
        pickle.dump(stats, fh)

    print(f"✓ window {idx:04d}: traditional LD stats completed in {traditional_time:.2f}s → {out_pkl.relative_to(sim_dir)}")
    print(f"📊 window {idx}: processed {len(stats.get('sums', []))-1 if stats.get('sums') else 'unknown'} r-bins")


if __name__ == "__main__":
    main()

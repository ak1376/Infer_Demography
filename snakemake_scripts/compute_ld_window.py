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

    samples_by_pid = {
        pid: [int(x) for x in ts.samples(population=pid)]
        for pid in range(ts.num_populations)
    }

    # Try to find deme0/deme1 by name, otherwise use first two non-empty
    pid_d0 = next((pid for pid, nm in pop_names.items() if nm == "deme0"), None)
    pid_d1 = next((pid for pid, nm in pop_names.items() if nm == "deme1"), None)

    # Check for ANC population (single population models like bottleneck)
    pid_anc = next((pid for pid, nm in pop_names.items() if nm == "ANC"), None)

    if pid_d0 is None or pid_d1 is None:
        # Check if we have ANC population - for single population models
        if pid_anc is not None and len(samples_by_pid[pid_anc]) > 0:
            # For single population, use all samples under the actual population name
            all_samples = samples_by_pid[pid_anc]
            ss = {"ANC": all_samples}  # Keep original population name
            print(f"Single population model detected - using all {len(all_samples)} samples as 'ANC'")
            return ss
        else:
            # Fall back to using first two non-empty populations with their actual names
            nonempty = [(pid, s, pop_names[pid]) for pid, s in samples_by_pid.items() if len(s) > 0]
            nonempty.sort(key=lambda x: x[0])
            if len(nonempty) < 1:
                raise ValueError("No non-empty populations in the tree sequence.")
            elif len(nonempty) == 1:
                # Single population - split samples but use actual name or fallback
                pid, all_samples, name = nonempty[0]
                pop_name = name if name else f"pop_{pid}"
                mid = len(all_samples) // 2
                ss = {pop_name + "_0": all_samples[:mid], pop_name + "_1": all_samples[mid:]}
            else:
                # Two or more populations - use actual population names
                pid_0, samples_0, name_0 = nonempty[0]
                pid_1, samples_1, name_1 = nonempty[1]
                pop_0 = name_0 if name_0 else f"pop_{pid_0}"
                pop_1 = name_1 if name_1 else f"pop_{pid_1}" 
                ss = {pop_0: samples_0, pop_1: samples_1}
    else:
        # Found deme0 and deme1 explicitly
        ss = {"deme0": samples_by_pid[pid_d0], "deme1": samples_by_pid[pid_d1]}

    # Validate that we have non-empty sample sets
    sample_counts = {k: len(v) for k, v in ss.items()}
    if any(count == 0 for count in sample_counts.values()):
        raise ValueError(f"Empty sample set(s): {sample_counts}")
    return ss


def gpu_ld_from_trees_single_pop(
    ts_path: str, r_bins, r_per_bp: float, pop: str = "ANC"
) -> dict:
    """Compute LD stats using GPU acceleration from tree sequence file for single population."""
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
    gpu_name = cp.cuda.runtime.getDeviceProperties(best_gpu)["name"].decode()
    print(
        f"📱 Using GPU {best_gpu} ({gpu_name}) with {max_free_mem/1e9:.1f}GB free memory"
    )

    # Convert genetic distance bins to physical distance
    bp_bins = np.array(r_bins, dtype=float) / float(r_per_bp)

    ts = tskit.load(ts_path)
    sample_sets = build_sample_sets(ts)

    if pop not in sample_sets:
        raise KeyError(
            f"Requested pop ({pop}) not in sample_sets={list(sample_sets)}"
        )

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
        stats_by_bin = h_filt.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins, raw=True, ac_filter=True
        )
        print(f"[DEBUG] stats_by_bin type: {type(stats_by_bin)}")
        print(f"[DEBUG] stats_by_bin keys/structure: {list(stats_by_bin.keys()) if hasattr(stats_by_bin, 'keys') else 'Not a dict'}")
        if hasattr(stats_by_bin, 'keys') and stats_by_bin:
            first_key = list(stats_by_bin.keys())[0]
            print(f"[DEBUG] First entry type: {type(stats_by_bin[first_key])}")
            print(f"[DEBUG] First entry: {stats_by_bin[first_key]}")
    except TypeError:
        stats_by_bin = h_filt.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins, raw=True
        )
        print(f"[DEBUG] (fallback) stats_by_bin type: {type(stats_by_bin)}")
        print(f"[DEBUG] (fallback) stats_by_bin structure: {stats_by_bin}")

    # For single population, pg_gpu returns tuples with 3 values in order: DD_0_0, Dz_0_0_0, pi2_0_0_0_0
    # Convert to moments-compatible format for single population
    MOMENTS_ORDER_SINGLE_POP = [
        "DD_0_0",
        "Dz_0_0_0", 
        "pi2_0_0_0_0"
    ]

    # Assemble per-bin vectors in moments order
    sums = []
    for b0, b1 in zip(bp_bins[:-1], bp_bins[1:]):
        key = (float(b0), float(b1))
        od = stats_by_bin.get(key, None)
        if od is None:
            sums.append(np.zeros(len(MOMENTS_ORDER_SINGLE_POP), dtype=float))
        else:
            # od is a tuple: (DD_0_0, Dz_0_0_0, pi2_0_0_0_0)
            sums.append(np.array(list(od), dtype=float))

    # Compute H terms from tree sequence for single population
    samples_vec = np.array(list(ts.samples()), dtype=np.int64)
    node_to_idx = {int(n): i for i, n in enumerate(samples_vec)}
    # Use all samples for single population
    idx_samples = np.array(
        [node_to_idx[n] for n in sample_sets[pop] if n in node_to_idx],
        dtype=np.int64,
    )

    H_stat = 0.0
    for var in ts.variants(
        samples=samples_vec, alleles=None, impute_missing_data=False
    ):
        g = var.genotypes
        g_samples = g[idx_samples]
        val_samples = g_samples[g_samples >= 0]
        if val_samples.size == 0:
            continue
        p_samples = float(val_samples.mean())
        H_stat += 2.0 * p_samples * (1.0 - p_samples)

    # Add H statistic as the final entry in sums
    sums.append(np.array([H_stat], dtype=float))

    # Return as LDstats compatible dict that matches the expected format from compute_ld_statistics
    # The 'stats' field should contain the NAMES of the statistics, not their values
    
    return {
        "bins": list(zip(bp_bins[:-1], bp_bins[1:])),  # List of (start, end) tuples
        "sums": sums,  # LD stats per bin + H stat array at the end
        "stats": (MOMENTS_ORDER_SINGLE_POP, ["H_0_0"]),  # Tuple of (LD_stat_names, H_stat_names)
        "pops": [pop],  # Single population list
    }


def gpu_ld_from_trees(
    ts_path: str, r_bins, r_per_bp: float, pop1: str = "deme0", pop2: str = "deme1"
) -> dict:
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
    gpu_name = cp.cuda.runtime.getDeviceProperties(best_gpu)["name"].decode()
    print(
        f"📱 Using GPU {best_gpu} ({gpu_name}) with {max_free_mem/1e9:.1f}GB free memory"
    )

    # Convert genetic distance bins to physical distance
    bp_bins = np.array(r_bins, dtype=float) / float(r_per_bp)

    ts = tskit.load(ts_path)
    sample_sets = build_sample_sets(ts)
    
    # Find the actual population names in sample_sets that correspond to pop1 and pop2
    available_pops = list(sample_sets.keys())
    
    # For two-population models, we need to map requested population names to available ones
    actual_pop1 = None
    actual_pop2 = None
    
    if pop1 in sample_sets and pop2 in sample_sets:
        # Direct match - populations exist with requested names
        actual_pop1, actual_pop2 = pop1, pop2
    elif len(available_pops) == 2:
        # Two populations available - assume they correspond to pop1, pop2 in order
        actual_pop1, actual_pop2 = available_pops[0], available_pops[1]
        print(f"Mapping requested ({pop1}, {pop2}) to available ({actual_pop1}, {actual_pop2})")
    else:
        raise KeyError(
            f"Cannot map requested pops ({pop1},{pop2}) to available sample_sets={available_pops}"
        )

    h = HaplotypeMatrix.from_ts(ts)
    # Set sample sets with actual population names
    normalized = {k: [int(x) for x in v] for k, v in sample_sets.items()}
    if hasattr(h, "set_sample_sets") and callable(getattr(h, "set_sample_sets")):
        h.set_sample_sets(normalized)
    else:
        h.sample_sets = normalized

    # Apply biallelic filter
    h_filt = h.apply_biallelic_filter()

    try:
        # Use actual population names for GPU computation
        stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins, pop1=actual_pop1, pop2=actual_pop2, raw=True, ac_filter=True, fp64=True
        )
    except TypeError:
        stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins, pop1=actual_pop1, pop2=actual_pop2, raw=True, ac_filter=True
        )

    # Convert to moments-compatible format
    MOMENTS_ORDER = [
        "DD_0_0",
        "DD_0_1",
        "DD_1_1",
        "Dz_0_0_0",
        "Dz_0_0_1",
        "Dz_0_1_1",
        "Dz_1_0_0",
        "Dz_1_0_1",
        "Dz_1_1_1",
        "pi2_0_0_0_0",
        "pi2_0_0_0_1",
        "pi2_0_0_1_1",
        "pi2_0_1_0_1",
        "pi2_0_1_1_1",
        "pi2_1_1_1_1",
    ]
    H_STAT_NAMES = ["H_0_0", "H_0_1", "H_1_1"]

    # Assemble per-bin vectors in moments order
    sums = []
    for b0, b1 in zip(bp_bins[:-1], bp_bins[1:]):
        key = (float(b0), float(b1))
        od = stats_by_bin.get(key, None)
        if od is None:
            sums.append(np.zeros(len(MOMENTS_ORDER), dtype=float))
        else:
            sums.append(np.array([od[name] for name in MOMENTS_ORDER], dtype=float))

    # Compute H terms from tree sequence
    samples_vec = np.array(list(ts.samples()), dtype=np.int64)
    node_to_idx = {int(n): i for i, n in enumerate(samples_vec)}
    idx_A = np.array(
        [node_to_idx[n] for n in sample_sets[actual_pop1] if n in node_to_idx],
        dtype=np.int64,
    )
    idx_B = np.array(
        [node_to_idx[n] for n in sample_sets[actual_pop2] if n in node_to_idx],
        dtype=np.int64,
    )

    H00 = H01 = H11 = 0.0
    for var in ts.variants(
        samples=samples_vec, alleles=None, impute_missing_data=False
    ):
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

    bins_gen = [
        (np.float64(r_bins[i]), np.float64(r_bins[i + 1]))
        for i in range(len(r_bins) - 1)
    ]

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

    print(
        f"[LD] window {idx}: args.use_gpu={args.use_gpu}, "
        f"_HAVE_GPU={_HAVE_GPU}, "
        f"use_gpu_ld_in_cfg={config.get('use_gpu_ld', False)}, "
        f"→ use_gpu={use_gpu}"
    )

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

    # Traditional moments path
    print(f"🐌 window {idx}: computing with traditional moments LD")
    # ----------------------------------------- grab every unique pop ID
    # read unique pop IDs from the samples file
    pops = list(config["num_samples"].keys())  # e.g. ["YRI", "CEU"] or ["ANC"]
    print(f"Pops found in samples.txt: {pops}")

    # compute LD statistics ----------------------------------------
    import time

    # Check if we should use GPU acceleration
    ts_file = sim_dir / "windows" / f"window_{idx}.trees"  # Tree sequences saved in windows directory
    
    if args.use_gpu and _HAVE_GPU and ts_file.exists():
        print(f"🚀 window {idx}: ATTEMPTING GPU acceleration from {ts_file}")
        try:
            gpu_start = time.perf_counter()
            
            # Choose correct GPU method based on number of populations
            if len(pops) == 1:
                print(f"📊 Using single-population GPU LD computation for {pops[0]}")
                stats = gpu_ld_from_trees_single_pop(
                    str(ts_file), r_bins, r_per_bp, pop=pops[0]
                )
            elif len(pops) == 2:
                print(f"📊 Using two-population GPU LD computation for {pops[0]}, {pops[1]}")
                stats = gpu_ld_from_trees(
                    str(ts_file), r_bins, r_per_bp, pop1=pops[0], pop2=pops[1]
                )
            else:
                raise ValueError(f"GPU LD computation not supported for {len(pops)} populations. Falling back to traditional method.")

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
            print(
                f"✅ window {idx:04d}: GPU LD stats completed in {gpu_time:.2f}s → {out_pkl.relative_to(sim_dir)}"
            )
            print(f"GPU acceleration successful; skipping traditional method.")
            return

        except Exception as e:
            print(f"❌ GPU acceleration failed: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise  # Re-raise to see the actual error instead of falling back

    # Traditional (CPU-based) LD computation as fallback
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

    print(
        f"✓ window {idx:04d}: traditional LD stats completed in {traditional_time:.2f}s → {out_pkl.relative_to(sim_dir)}"
    )
    print(
        f"📊 window {idx}: processed {len(stats.get('sums', []))-1 if stats.get('sums') else 'unknown'} r-bins"
    )


if __name__ == "__main__":
    main()

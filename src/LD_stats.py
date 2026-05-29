# src/LD_stats.py
"""
Core LD-stats computation utilities.

Design goals:
- Keep Snakemake wrapper thin.
- Provide one entrypoint: compute_ld_window(...)
- Always uses pg_gpu.moments_ld.compute_ld_statistics (GPU-accelerated, drop-in replacement
  for moments.LD.Parsing.compute_ld_statistics).
- Names/order for LD and H stats match moments' CPU naming functions for arbitrary num_pops.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import tskit

from pg_gpu.moments_ld import compute_ld_statistics as _pg_compute_ld_statistics


# -----------------------------------------------------------------------------
# moments-compatible stat naming (match moments.LD order)
# -----------------------------------------------------------------------------


def het_names(num_pops: int) -> List[str]:
    out: List[str] = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            out.append(f"H_{ii}_{jj}")
    return out


def ld_names(num_pops: int) -> List[str]:
    out: List[str] = []

    # DD
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            out.append(f"DD_{ii}_{jj}")

    # Dz
    for ii in range(num_pops):
        for jj in range(num_pops):
            for kk in range(jj, num_pops):
                out.append(f"Dz_{ii}_{jj}_{kk}")

    # pi2
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            for kk in range(ii, num_pops):
                for ll in range(kk, num_pops):
                    if kk == ii == ll and jj != ii:
                        continue
                    if ii == kk and ll < jj:
                        continue
                    out.append(f"pi2_{ii}_{jj}_{kk}_{ll}")

    return out


def moment_names(num_pops: int) -> Tuple[List[str], List[str]]:
    return (ld_names(num_pops), het_names(num_pops))


# -----------------------------------------------------------------------------
# recombination map helpers (flat map file: pos  Map(cM))
# -----------------------------------------------------------------------------


def _load_rec_map_cM(rec_map_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a map like:
      pos    Map(cM)
      0      0
      100000 0.15

    Returns:
      pos_bp (float64, sorted)
      map_M  (float64, cumulative Morgans, sorted)
    """
    arr = np.genfromtxt(str(rec_map_file), names=True, dtype=None, encoding=None)
    if arr.dtype.names is None or len(arr.dtype.names) < 2:
        raw = np.loadtxt(str(rec_map_file))
        pos_bp = np.asarray(raw[:, 0], dtype=np.float64)
        map_cM = np.asarray(raw[:, 1], dtype=np.float64)
    else:
        c0, c1 = arr.dtype.names[0], arr.dtype.names[1]
        pos_bp = np.asarray(arr[c0], dtype=np.float64)
        map_cM = np.asarray(arr[c1], dtype=np.float64)

    order = np.argsort(pos_bp)
    pos_bp = pos_bp[order]
    map_M = map_cM[order] / 100.0  # cM -> Morgans
    return pos_bp, map_M


def _interp_gen_pos_M(site_pos_bp: np.ndarray, rec_map_file: str | Path) -> np.ndarray:
    """Interpolate cumulative Morgans at SNP positions (bp -> Morgans)."""
    pos_bp_map, map_M = _load_rec_map_cM(rec_map_file)
    x = np.asarray(site_pos_bp, dtype=np.float64)
    return np.interp(x, pos_bp_map, map_M).astype(np.float64)


def _r_per_bp_from_flat_map(rec_map_file: str | Path) -> float:
    """
    Compute average recombination rate per bp from a flat map file:
      pos(bp)  map(cM)

    r_per_bp = (delta Morgans) / (delta bp)
    """
    pos_bp_map, map_M = _load_rec_map_cM(rec_map_file)
    L_bp = float(pos_bp_map[-1] - pos_bp_map[0])
    L_M = float(map_M[-1] - map_M[0])

    if L_bp <= 0 or L_M <= 0:
        raise ValueError(f"Bad recombination map: L_bp={L_bp}, L_M={L_M}")

    r_per_bp = L_M / L_bp
    print(f"[LD] r_per_bp from flat_map = {r_per_bp:.3e}")
    return r_per_bp


# -----------------------------------------------------------------------------
# population/sample utilities
# -----------------------------------------------------------------------------


def build_sample_sets(ts: tskit.TreeSequence) -> Dict[str, List[int]]:
    """
    Detect 1–3 populations from ts populations.

    Preference order:
      1) deme0/deme1/deme2 if present
      2) ANC if present (single-pop)
      3) first 1–3 non-empty populations in pid order

    IMPORTANT:
      Returns *row indices* into ts.samples() order (rows of ts.genotype_matrix().T).
      That is what pg_gpu's HaplotypeMatrix expects for sample_sets.
    """
    pop_names: Dict[int, Optional[str]] = {}
    for pid in range(ts.num_populations):
        pop = ts.population(pid)
        name = None
        if hasattr(pop, "name") and getattr(pop, "name", None):
            name = pop.name
        elif hasattr(pop, "metadata") and isinstance(pop.metadata, dict):
            name = pop.metadata.get("name")
        pop_names[pid] = name

    # node id -> row index in ts.samples() order
    samples_vec = [int(x) for x in ts.samples()]
    node_to_row = {node_id: row_idx for row_idx, node_id in enumerate(samples_vec)}

    def pid_nodes_to_rows(pid: int) -> List[int]:
        node_ids = [int(x) for x in ts.samples(population=pid)]  # node ids
        try:
            return [node_to_row[n] for n in node_ids]
        except KeyError as e:
            raise ValueError(
                f"Population {pid} has sample node {int(e.args[0])} not found in ts.samples()."
            )

    samples_by_pid = {pid: pid_nodes_to_rows(pid) for pid in range(ts.num_populations)}

    pid_d0 = next((pid for pid, nm in pop_names.items() if nm == "deme0"), None)
    pid_d1 = next((pid for pid, nm in pop_names.items() if nm == "deme1"), None)
    pid_d2 = next((pid for pid, nm in pop_names.items() if nm == "deme2"), None)
    pid_anc = next((pid for pid, nm in pop_names.items() if nm == "ANC"), None)

    # explicit demes
    if pid_d0 is not None and pid_d1 is not None:
        ss = {"deme0": samples_by_pid[pid_d0], "deme1": samples_by_pid[pid_d1]}
        if pid_d2 is not None and len(samples_by_pid[pid_d2]) > 0:
            ss["deme2"] = samples_by_pid[pid_d2]
        counts = {k: len(v) for k, v in ss.items()}
        if any(c == 0 for c in counts.values()):
            raise ValueError(f"Empty sample set(s) in explicit deme mapping: {counts}")
        return ss

    # explicit ANC single-pop
    if pid_anc is not None and len(samples_by_pid[pid_anc]) > 0:
        all_samples = samples_by_pid[pid_anc]
        print(
            f"Single population model detected - using all {len(all_samples)} samples as 'ANC'"
        )
        return {"ANC": all_samples}

    # fallback first 1–3 non-empty
    nonempty = [
        (pid, s, pop_names[pid]) for pid, s in samples_by_pid.items() if len(s) > 0
    ]
    nonempty.sort(key=lambda x: x[0])
    if len(nonempty) < 1:
        raise ValueError("No non-empty populations in the tree sequence.")

    chosen = nonempty[:3]
    ss: Dict[str, List[int]] = {}
    for pid, samples, name in chosen:
        pop_name = name if name else f"pop_{pid}"
        ss[pop_name] = samples

    counts = {k: len(v) for k, v in ss.items()}
    if any(c == 0 for c in counts.values()):
        raise ValueError(f"Empty sample set(s) in fallback mapping: {counts}")
    return ss


def _read_pop_order_from_samples_txt(samples_txt: str | Path) -> List[str]:
    """
    Parse pop order from moments pop_file (samples.txt).
    Typical format:
        POP  sample1 sample2 ...
    We just take the first token of each non-empty, non-comment line.
    """
    pops: List[str] = []
    with open(samples_txt, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            pops.append(ln.split()[0])
    return pops


def _select_best_gpu() -> None:
    import cupy as cp  # local import so CPU-only envs still work

    best_gpu = 0
    max_free_mem = 0
    for gpu_id in range(cp.cuda.runtime.getDeviceCount()):
        cp.cuda.Device(gpu_id).use()
        free_mem, _total = cp.cuda.runtime.memGetInfo()
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_gpu = gpu_id

    cp.cuda.Device(best_gpu).use()
    name = cp.cuda.runtime.getDeviceProperties(best_gpu)["name"].decode()
    print(f"📱 Using GPU {best_gpu} ({name}) with {max_free_mem/1e9:.1f}GB free memory")


def _gpu_cleanup() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Public entrypoint used by wrapper
# -----------------------------------------------------------------------------


def compute_ld_window(
    *,
    window_index: int,
    vcf_gz: Path,
    samples_file: Path,
    rec_map_file: Path,
    r_bins: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute LD stats for one window using pg_gpu.moments_ld.compute_ld_statistics.

    Always uses GPU-accelerated pg_gpu; no CPU fallback.

    Pop-order is taken from config["num_samples"].keys().
    """
    pops_cpu = list(config["num_samples"].keys())
    _select_best_gpu()
    t0 = time.perf_counter()
    stats = _pg_compute_ld_statistics(
        vcf_file=str(vcf_gz),
        rec_map_file=str(rec_map_file),
        pop_file=str(samples_file),
        pops=pops_cpu,
        r_bins=r_bins,
        report=False,
    )
    dt = time.perf_counter() - t0
    print(f"[LD] window {window_index:04d}: pg_gpu completed in {dt:.2f}s")
    return stats

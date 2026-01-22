# src/LD_stats.py
"""
Core LD-stats computation utilities.

Design goals:
- Keep Snakemake wrapper thin.
- Provide one entrypoint: compute_ld_window(...)
- GPU path uses pg_gpu + tree sequence if available.
- CPU fallback uses moments.LD.Parsing.compute_ld_statistics.
- Names/order for LD and H stats match moments' CPU naming functions for arbitrary num_pops.

CRITICAL UNITS:
- moments.LD bins are in *Morgans* (recombination fraction distance bins).
- tskit site positions are in *bp*.
- pg_gpu bins by differences in `hap.positions`.
Therefore, for GPU we MUST set `hap.positions` to *genetic positions in Morgans*
(interpolated from the rec map, or flat r_per_bp fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import tskit
import moments

# Optional GPU acceleration
try:
    from pg_gpu.haplotype_matrix import HaplotypeMatrix  # type: ignore

    _HAVE_GPU = True
except Exception:
    HaplotypeMatrix = None  # type: ignore
    _HAVE_GPU = False


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


def _compute_H_vals_from_ts(
    ts: tskit.TreeSequence,
    sample_sets: Dict[str, List[int]],
    pop_order: List[str],
) -> np.ndarray:
    """
    Return H values in het_names(k) order.

    IMPORTANT:
      sample_sets values are *row indices* into ts.samples() order
      (i.e., indices into var.genotypes when iterating with samples=ts.samples()).
    """
    samples_vec = np.array(list(ts.samples()), dtype=np.int64)

    idx_by_pop = {
        pop: np.asarray(sample_sets[pop], dtype=np.int64) for pop in pop_order
    }

    k = len(pop_order)
    H = {(i, j): 0.0 for i in range(k) for j in range(i, k)}

    for var in ts.variants(
        samples=samples_vec, alleles=None, impute_missing_data=False
    ):
        g = var.genotypes  # aligned with samples_vec order

        ps: List[float] = []
        ok = True
        for pop in pop_order:
            vv = g[idx_by_pop[pop]]
            vv = vv[vv >= 0]
            if vv.size == 0:
                ok = False
                break
            ps.append(float(vv.mean()))
        if not ok:
            continue

        for i in range(k):
            p_i = ps[i]
            H[(i, i)] += 2.0 * p_i * (1.0 - p_i)
            for j in range(i + 1, k):
                p_j = ps[j]
                H[(i, j)] += p_i * (1.0 - p_j) + (1.0 - p_i) * p_j

    vals: List[float] = []
    for i in range(k):
        for j in range(i, k):
            vals.append(H[(i, j)])
    return np.array(vals, dtype=float)


# -----------------------------------------------------------------------------
# GPU computation (auto-dispatch)
# -----------------------------------------------------------------------------


def gpu_ld_from_trees_auto(
    ts_path: str | Path,
    r_bins: np.ndarray,
    r_per_bp: float,
    *,
    cpu_fallback: bool,
    vcf_gz: Optional[str | Path],
    rec_map_file: Optional[str | Path],
    pop_file: Optional[str | Path],
    # Optional overrides so caller can force canonical subset + order
    sample_sets: Optional[Dict[str, List[int]]] = None,
    pop_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute LD stats from a tree sequence using pg_gpu when possible.

    Output format:
      {
        "bins": [(r0,r1), ...] in Morgans,
        "sums": [ld_vec_bin0, ..., ld_vec_lastbin, H_vec],
        "stats": (LD_names, H_names),
        "pops": pop_order
      }

    Notes:
    - `r_bins` are Morgans.
    - pg_gpu bins by differences in hap.positions, so we set hap.positions to Morgans.
    - If caller doesn't pass pop_order/sample_sets, we:
        (a) detect from ts, then
        (b) if pop_file is provided, we reorder to match pop_file order (CPU order).
    """
    if not _HAVE_GPU:
        raise RuntimeError("pg_gpu not available in this environment.")

    ts = tskit.load(str(ts_path))

    # sample_sets: row indices into ts.samples()
    if sample_sets is None:
        sample_sets = build_sample_sets(ts)

    # pop order: prefer pop_file order (CPU order), otherwise sample_sets key order
    if pop_order is None:
        if pop_file is not None:
            cpu_order = _read_pop_order_from_samples_txt(pop_file)
            cpu_order = [p for p in cpu_order if p in sample_sets]
            if len(cpu_order) > 0:
                pop_order = cpu_order
            else:
                pop_order = list(sample_sets.keys())
        else:
            pop_order = list(sample_sets.keys())

    # Ensure pop_order matches sample_sets keys
    missing = [p for p in pop_order if p not in sample_sets]
    if missing:
        raise ValueError(f"pop_order contains pops not in sample_sets: {missing}")

    # Keep only ordered pops
    sample_sets = {p: sample_sets[p] for p in pop_order}

    k = len(pop_order)
    LD_names, H_names = moment_names(k)

    _select_best_gpu()

    site_pos_bp = np.asarray(ts.tables.sites.position, dtype=np.float64)
    if rec_map_file is not None:
        site_pos_M = _interp_gen_pos_M(site_pos_bp, rec_map_file)
    else:
        site_pos_M = site_pos_bp * float(r_per_bp)

    dist_bins = np.asarray(r_bins, dtype=np.float64)

    # Build haplotype matrix (pg_gpu)
    h = HaplotypeMatrix.from_ts(ts)  # type: ignore[misc]
    normalized = {name: [int(x) for x in idxs] for name, idxs in sample_sets.items()}

    if hasattr(h, "set_sample_sets") and callable(getattr(h, "set_sample_sets")):
        h.set_sample_sets(normalized)
    else:
        h.sample_sets = normalized

    import cupy as cp

    h.positions = cp.asarray(site_pos_M)

    # filter to biallelic sites; keeps positions aligned
    h_filt = h.apply_biallelic_filter()

    sums: List[np.ndarray] = []

    def _bins_out() -> List[Tuple[np.float64, np.float64]]:
        return [
            (np.float64(r_bins[i]), np.float64(r_bins[i + 1]))
            for i in range(len(r_bins) - 1)
        ]

    def _finalize() -> Dict[str, Any]:
        _gpu_cleanup()
        return {
            "bins": _bins_out(),
            "sums": sums,
            "stats": (LD_names, H_names),
            "pops": pop_order,
        }

    def _append_ld_vectors(stats_by_bin: Dict[Tuple[float, float], Any]) -> None:
        for b0, b1 in zip(dist_bins[:-1], dist_bins[1:]):
            key = (float(b0), float(b1))
            od = stats_by_bin.get(key, None)
            if od is None:
                sums.append(np.zeros(len(LD_names), dtype=float))
                continue
            if isinstance(od, dict):
                sums.append(np.array([od[nm] for nm in LD_names], dtype=float))
            else:
                sums.append(np.array(list(od), dtype=float))

    # ---- k == 1 ----
    if k == 1:
        try:
            stats_by_bin = h_filt.compute_ld_statistics_gpu_single_pop(
                bp_bins=dist_bins, raw=True, ac_filter=True
            )
        except TypeError:
            stats_by_bin = h_filt.compute_ld_statistics_gpu_single_pop(
                bp_bins=dist_bins, raw=True
            )

        _append_ld_vectors(stats_by_bin)
        sums.append(_compute_H_vals_from_ts(ts, sample_sets, pop_order))
        return _finalize()

    # ---- k == 2 ----
    if k == 2:
        pop1, pop2 = pop_order[0], pop_order[1]
        try:
            stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
                bp_bins=dist_bins,
                pop1=pop1,
                pop2=pop2,
                raw=True,
                ac_filter=True,
                fp64=True,
            )
        except TypeError:
            stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
                bp_bins=dist_bins,
                pop1=pop1,
                pop2=pop2,
                raw=True,
                ac_filter=True,
            )

        _append_ld_vectors(stats_by_bin)
        sums.append(_compute_H_vals_from_ts(ts, sample_sets, pop_order))
        return _finalize()

    # ---- k >= 3: try multi-pop kernel ----
    multi_fn = None
    multi_name = None
    for cand in (
        "compute_ld_statistics_gpu_three_pops",
        "compute_ld_statistics_gpu_multi_pops",
        "compute_ld_statistics_gpu_k_pops",
    ):
        if hasattr(h_filt, cand) and callable(getattr(h_filt, cand)):
            multi_fn = getattr(h_filt, cand)
            multi_name = cand
            break

    if multi_fn is not None:
        print(f"🚀 Found pg_gpu multi-pop kernel: {multi_name} (k={k})")

        # --- Signature-aware call ---
        if multi_name == "compute_ld_statistics_gpu_three_pops":
            # Your 3-pop function expects pop1/pop2/pop3 as separate args
            pop1, pop2, pop3 = pop_order[0], pop_order[1], pop_order[2]
            stats_by_bin = multi_fn(
                bp_bins=dist_bins,
                pop1=pop1,
                pop2=pop2,
                pop3=pop3,
                raw=True,  # IMPORTANT: we want true sums for the "sums" field
                ac_filter=True,
            )
        else:
            # Other possible multi-pop APIs may accept pops=[...]
            try:
                stats_by_bin = multi_fn(
                    bp_bins=dist_bins,
                    pops=pop_order,
                    raw=True,  # IMPORTANT
                    ac_filter=True,
                )
            except TypeError:
                try:
                    stats_by_bin = multi_fn(
                        bp_bins=dist_bins,
                        pops=pop_order,
                        raw=True,  # IMPORTANT
                    )
                except TypeError:
                    # last resort (may return means) — but at least try
                    stats_by_bin = multi_fn(bp_bins=dist_bins)

        # Validate dict-of-dicts
        example = None
        for _key in stats_by_bin.keys():
            example = stats_by_bin[_key]
            if example is not None:
                break
        if not isinstance(example, dict):
            raise TypeError(
                f"Multi-pop kernel returned per-bin type={type(example)}; expected dict of stat_name->value."
            )

        missing = [nm for nm in LD_names if nm not in example]
        if missing:
            raise KeyError(
                f"GPU multi-pop output missing expected moments LD names (k={k}). "
                f"First few missing: {missing[:10]}"
            )

        _append_ld_vectors(stats_by_bin)
        sums.append(_compute_H_vals_from_ts(ts, sample_sets, pop_order))
        return _finalize()

    # ---- CPU fallback if no GPU multi-pop kernel ----
    if cpu_fallback:
        if vcf_gz is None or rec_map_file is None or pop_file is None:
            raise NotImplementedError(
                f"Detected k={k} pops but pg_gpu has no multi-pop kernel and CPU fallback inputs are missing."
            )

        print(
            f"🐌 Falling back to moments CPU compute_ld_statistics for k={k} pops: {pop_order}"
        )
        return moments.LD.Parsing.compute_ld_statistics(
            str(vcf_gz),
            rec_map_file=str(rec_map_file),
            pop_file=str(pop_file),
            pops=pop_order,
            r_bins=np.asarray([float(x) for x in r_bins], dtype=float),
            report=False,
        )

    raise NotImplementedError(
        f"Detected k={k} pops but pg_gpu has no multi-pop kernel and cpu_fallback=False."
    )


# -----------------------------------------------------------------------------
# Public entrypoint used by wrapper
# -----------------------------------------------------------------------------


def compute_ld_window(
    *,
    window_index: int,
    vcf_gz: Path,
    samples_file: Path,
    rec_map_file: Path,
    ts_file: Optional[Path],
    r_bins: np.ndarray,
    config: Dict[str, Any],
    request_gpu: bool,
) -> Dict[str, Any]:
    """
    Compute LD stats for one window. Uses GPU when requested + available + config allows.
    Falls back to CPU moments when GPU isn't possible or fails.

    Pop-order rule:
      CPU uses config["num_samples"].keys()
      GPU is forced to match CPU order by passing pop_order to gpu_ld_from_trees_auto.
    """
    r_per_bp = float(config.get("recombination_rate", 1e-8))
    use_gpu_cfg = bool(config.get("use_gpu_ld", False))
    use_gpu = bool(request_gpu and use_gpu_cfg and _HAVE_GPU and ts_file is not None)

    print(
        f"[LD] window {window_index}: request_gpu={request_gpu}, "
        f"use_gpu_ld_in_cfg={use_gpu_cfg}, have_gpu={_HAVE_GPU}, ts_exists={ts_file is not None} "
        f"→ use_gpu={use_gpu}"
    )

    pops_cpu = list(config["num_samples"].keys())
    print(f"[LD] CPU pop labels (from config num_samples keys): {pops_cpu}")

    if use_gpu:
        try:
            import time

            t0 = time.perf_counter()

            stats = gpu_ld_from_trees_auto(
                ts_path=ts_file,
                r_bins=r_bins,
                r_per_bp=r_per_bp,
                cpu_fallback=True,
                vcf_gz=vcf_gz,
                rec_map_file=rec_map_file,
                pop_file=samples_file,  # used to infer CPU pop order if pop_order not given
                pop_order=pops_cpu,  # force index order == CPU/config order
                sample_sets=None,  # if you later want canonical subsets, pass them here
            )

            dt = time.perf_counter() - t0
            print(f"✅ window {window_index:04d}: GPU/auto completed in {dt:.2f}s")
            return stats

        except Exception as e:
            print(
                f"❌ window {window_index:04d}: GPU path failed; falling back to CPU. Error: {e}"
            )

    import time

    t0 = time.perf_counter()
    stats = moments.LD.Parsing.compute_ld_statistics(
        str(vcf_gz),
        rec_map_file=str(rec_map_file),
        pop_file=str(samples_file),
        pops=pops_cpu,
        r_bins=r_bins,
        report=False,
    )
    dt = time.perf_counter() - t0
    print(f"✓ window {window_index:04d}: CPU completed in {dt:.2f}s")
    return stats

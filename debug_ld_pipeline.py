#!/usr/bin/env python3
"""
debug_ld_pipeline.py

Compare LD statistics computed on:
  - CPU: moments.LD.Parsing.compute_ld_statistics (via compute_ld_window with request_gpu=False)
  - GPU: pg_gpu path (via compute_ld_window with request_gpu=True, requires .trees + use_gpu_ld=true)

Usage:
  python debug_ld_pipeline.py \
    --sim-dir /path/to/.../MomentsLD \
    --config-file /path/to/experiment_config.json \
    --window-indices 0,1,2 \
    --r-bins "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3" \
    --atol 1e-10 --rtol 1e-6 \
    --compare-mode all_ld
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# -----------------------------------------------------------------------------
# Import compute_ld_window from your repo
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
if (REPO_ROOT / "src").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.LD_stats import compute_ld_window
except ModuleNotFoundError:
    from src.ld_stats import compute_ld_window


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("compare CPU vs GPU LD stats")
    p.add_argument("--sim-dir", required=True, type=Path)
    p.add_argument("--config-file", required=True, type=Path)
    p.add_argument("--window-indices", required=True, help="comma-separated indices, e.g. 0,1,2")
    p.add_argument("--r-bins", required=True, help="comma-separated recomb-bin edges")
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument(
        "--compare-mode",
        choices=["all_ld", "dd_only", "custom"],
        default="all_ld",
        help="which stat set to compare",
    )
    p.add_argument(
        "--stats",
        default="",
        help="(custom mode) comma-separated stat names to compare, e.g. DD_0_0,DD_0_1",
    )
    p.add_argument("--max-report", type=int, default=10)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Extraction utilities
# -----------------------------------------------------------------------------

def _is_gpu_dict(x: Any) -> bool:
    return isinstance(x, dict) and ("sums" in x) and ("stats" in x) and ("bins" in x)


def _extract_from_gpu_dict(
    gpu: Dict[str, Any],
) -> Tuple[List[Tuple[float, float]], Dict[str, np.ndarray], Dict[str, float], List[str]]:
    bins = [(float(a), float(b)) for (a, b) in gpu["bins"]]
    sums = gpu["sums"]
    ld_names, h_names = gpu["stats"]
    pop_order = list(gpu.get("pops", []))

    nb = len(bins)
    if len(sums) != nb + 1:
        raise ValueError(f"GPU dict sums length={len(sums)} but expected nbins+1={nb+1}")

    ld_mat = np.vstack([np.asarray(sums[i], dtype=np.float64) for i in range(nb)])  # (nb, n_ld)
    if ld_mat.shape[1] != len(ld_names):
        raise ValueError(f"GPU LD vector length mismatch: got {ld_mat.shape[1]} vs expected {len(ld_names)}")

    ld_dict = {name: ld_mat[:, j].copy() for j, name in enumerate(ld_names)}

    h_vec = np.asarray(sums[-1], dtype=np.float64)
    if h_vec.shape[0] != len(h_names):
        raise ValueError(f"GPU H vector length mismatch: got {h_vec.shape[0]} vs expected {len(h_names)}")
    h_dict = {name: float(h_vec[j]) for j, name in enumerate(h_names)}

    return bins, ld_dict, h_dict, pop_order


def _extract_from_moments_obj(
    obj: Any,
) -> Tuple[Optional[List[Tuple[float, float]]], Dict[str, np.ndarray], Optional[Dict[str, float]]]:
    names = None
    for meth in ("names", "stats", "keys"):
        if hasattr(obj, meth):
            try:
                maybe = getattr(obj, meth)()
                if isinstance(maybe, (list, tuple)) and maybe and isinstance(maybe[0], str):
                    names = list(maybe)
                    break
            except Exception:
                pass

    if names is None:
        raise RuntimeError(
            "Could not discover stat names from CPU moments object. "
            "Try --compare-mode custom --stats 'DD_0_0,DD_0_1,...'"
        )

    ld_dict: Dict[str, np.ndarray] = {}
    for name in names:
        try:
            arr = np.asarray(obj[name], dtype=np.float64)
            if arr.ndim == 1:
                ld_dict[name] = arr
        except Exception:
            continue

    return None, ld_dict, None


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------

def _choose_stats_to_compare(
    cpu_ld: Dict[str, np.ndarray],
    gpu_ld: Dict[str, np.ndarray],
    mode: str,
    custom: List[str],
) -> List[str]:
    common = sorted(set(cpu_ld.keys()) & set(gpu_ld.keys()))
    if mode == "all_ld":
        return common
    if mode == "dd_only":
        return [s for s in common if s.startswith("DD_")]
    if mode == "custom":
        want = [s for s in custom if s in cpu_ld and s in gpu_ld]
        missing = [s for s in custom if s not in cpu_ld or s not in gpu_ld]
        if missing:
            raise KeyError(f"Custom stats missing from one side: {missing}")
        return want
    raise ValueError(mode)


def compare_ld_dicts(
    cpu_ld: Dict[str, np.ndarray],
    gpu_ld: Dict[str, np.ndarray],
    stats: List[str],
    atol: float,
    rtol: float,
    *,
    max_report: int = 10,
    label: str = "",
) -> None:
    for s in stats:
        a = np.asarray(cpu_ld[s], dtype=np.float64)
        b = np.asarray(gpu_ld[s], dtype=np.float64)

        if a.shape != b.shape:
            raise AssertionError(f"{label}: shape mismatch for {s}: CPU {a.shape} vs GPU {b.shape}")

        if np.allclose(a, b, atol=atol, rtol=rtol):
            continue

        diff = np.abs(a - b)
        rel = diff / (np.abs(a) + 1e-300)
        worst = int(np.argmax(diff))

        top_idx = np.argsort(diff)[::-1][:max_report]
        lines = []
        for j in top_idx:
            lines.append(
                f"    bin={int(j):3d}  CPU={a[j]: .6e}  GPU={b[j]: .6e}  abs={diff[j]:.3e}  rel={rel[j]:.3e}"
            )

        msg = (
            f"{label}: MISMATCH in {s}\n"
            f"  worst bin={worst} CPU={a[worst]:.6e} GPU={b[worst]:.6e} abs={diff[worst]:.3e} rel={rel[worst]:.3e}\n"
            f"  top diffs:\n" + "\n".join(lines)
        )
        raise AssertionError(msg)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    sim_dir = args.sim_dir.resolve()

    with open(args.config_file) as f:
        config = json.load(f)

    window_indices = [int(x) for x in args.window_indices.split(",") if x.strip() != ""]
    r_bins = np.array([float(x) for x in args.r_bins.split(",")], dtype=float)

    samples_file = sim_dir / "windows" / "samples.txt"
    rec_map_file = sim_dir / "windows" / "flat_map.txt"
    if not samples_file.exists():
        raise FileNotFoundError(samples_file)
    if not rec_map_file.exists():
        raise FileNotFoundError(rec_map_file)

    pops_cpu = list(config["num_samples"].keys())
    custom_stats = [s for s in args.stats.split(",") if s.strip()] if args.compare_mode == "custom" else []

    print("============================================================")
    print("CPU vs GPU LD comparison")
    print(f"sim_dir:   {sim_dir}")
    print(f"windows:   {window_indices}")
    print(f"pops_cpu:  {pops_cpu}")
    print(f"atol/rtol: {args.atol} / {args.rtol}")
    print(f"mode:      {args.compare_mode}")
    if custom_stats:
        print(f"stats:     {custom_stats}")
    print("============================================================")

    for idx in window_indices:
        vcf_gz = sim_dir / "windows" / f"window_{idx}.vcf.gz"
        ts_file = sim_dir / "windows" / f"window_{idx}.trees"

        if not vcf_gz.exists():
            raise FileNotFoundError(vcf_gz)
        if not ts_file.exists():
            raise FileNotFoundError(ts_file)

        print("\n" + "=" * 80)
        print(f"WINDOW {idx}")

        # CPU
        cpu_out = compute_ld_window(
            window_index=idx,
            vcf_gz=vcf_gz,
            samples_file=samples_file,
            rec_map_file=rec_map_file,
            ts_file=ts_file,
            r_bins=r_bins,
            config=config,
            request_gpu=False,
        )

        # GPU
        gpu_out = compute_ld_window(
            window_index=idx,
            vcf_gz=vcf_gz,
            samples_file=samples_file,
            rec_map_file=rec_map_file,
            ts_file=ts_file,
            r_bins=r_bins,
            config=config,
            request_gpu=True,
        )

        # Extract CPU
        if _is_gpu_dict(cpu_out):
            _, cpu_ld, _, pops_cpu_detected = _extract_from_gpu_dict(cpu_out)
            print(f"[extract] CPU returned gpu-dict format; pops={pops_cpu_detected}")
        else:
            _, cpu_ld, _ = _extract_from_moments_obj(cpu_out)

        # Extract GPU
        if _is_gpu_dict(gpu_out):
            _, gpu_ld, _, pops_gpu = _extract_from_gpu_dict(gpu_out)
            print(f"[extract] GPU dict pops={pops_gpu}")
            # sanity: pop order should match config order (we forced this in src/LD_stats.py)
            if pops_gpu and pops_gpu != pops_cpu:
                raise AssertionError(
                    f"Pop-order mismatch!\n"
                    f"  CPU config order: {pops_cpu}\n"
                    f"  GPU dict order:   {pops_gpu}"
                )
        else:
            _, gpu_ld, _ = _extract_from_moments_obj(gpu_out)

        stats = _choose_stats_to_compare(cpu_ld, gpu_ld, args.compare_mode, custom_stats)
        if not stats:
            raise RuntimeError("No stats selected for comparison.")

        print(f"[compare] comparing {len(stats)} stats; example: {stats[:8]}{'...' if len(stats)>8 else ''}")

        compare_ld_dicts(
            cpu_ld, gpu_ld, stats,
            atol=args.atol, rtol=args.rtol,
            max_report=args.max_report,
            label=f"window={idx}",
        )

        print(f"PASS ✅ window {idx}: CPU vs GPU match for selected stats")

    print("\nALL WINDOWS PASSED ✅")


if __name__ == "__main__":
    main()

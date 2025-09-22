#!/usr/bin/env python3
# computing_residuals_from_sfs.py
# Build residuals = Expected SFS (ground-truth params) − Fitted SFS (best-LL params from saved inference),
# for dadi, moments, or both. No optimization is performed here.

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, OrderedDict, Any
import argparse
import importlib
import importlib.util
import json
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Expected-SFS helpers (engine-specific) ────────────────────────────────────
def expected_sfs_dadi(
    params_in_order: List[float],
    param_names: List[str],
    sample_sizes: OrderedDict[str, int],   # diploid counts per deme
    demo_model,                            # callable(dict)-> demes.Graph
    mutation_rate: float,
    sequence_length: int,
    pts: List[int],
):
    import dadi
    p_dict = dict(zip(param_names, params_in_order))
    graph = demo_model(p_dict)
    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())
    fs = dadi.Spectrum.from_demes(
        graph,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
        pts=pts,
    )
    # theta = 4 * N0 * mu * L  (N0 assumed to be first param)
    fs *= 4 * p_dict[param_names[0]] * mutation_rate * sequence_length
    return fs

def expected_sfs_moments(
    params_in_order: List[float],
    param_names: List[str],
    sample_sizes: OrderedDict[str, int],   # diploid counts per deme
    demo_model,                            # callable(dict)-> demes.Graph
    mutation_rate: float,
    genome_length: int,
):
    import moments
    p_dict = dict(zip(param_names, params_in_order))
    graph = demo_model(p_dict)
    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())
    theta = p_dict[param_names[0]] * 4 * mutation_rate * genome_length  # N0 reference
    return moments.Spectrum.from_demes(
        graph,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
        theta=theta,
    )

# ── Utility IO ────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def load_gt_params(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        obj = load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Ground-truth must be dict of param_name->value, got {type(obj)}")
    return {k: float(v) for k, v in obj.items()}

def save_np(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def save_json_obj(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

# ── Best-params loader (handles several saved formats) ────────────────────────
def _maybe_load_pickle(path: Path) -> Any | None:
    try:
        return pickle.load(open(path, "rb"))
    except Exception:
        return None

def _best_idx_from_ll_list(ll_list: List[float]) -> int:
    if not ll_list:
        raise ValueError("Empty best_ll list.")
    return int(np.argmax(np.array(ll_list, dtype=float)))

def load_best_params_from_inference(method_dir: Path, param_names: List[str]) -> Tuple[Dict[str, float], float | None]:
    """
    Try to read best parameters (highest likelihood) from common files:
      - fit_params.pkl   with keys like 'best_params' (list[dict]) and 'best_ll' (list[float])
      - best_fit.pkl     with keys 'best_params' (dict) and 'best_ll' (float)
    Returns: (best_params_dict, best_ll_or_None)
    """
    # Preferences: fit_params.pkl, then best_fit.pkl
    candidates = [method_dir / "fit_params.pkl", method_dir / "best_fit.pkl"]
    for path in candidates:
        obj = _maybe_load_pickle(path)
        if obj is None or not isinstance(obj, dict):
            continue

        # Case A: lists
        if "best_params" in obj and isinstance(obj["best_params"], list):
            best_params_list = obj["best_params"]
            # Optionally align with LL list if present
            if "best_ll" in obj and isinstance(obj["best_ll"], list) and len(obj["best_ll"]) == len(best_params_list):
                idx = _best_idx_from_ll_list(obj["best_ll"])
            else:
                idx = 0  # fallback to first
            best_params_dict = best_params_list[idx]
            if not isinstance(best_params_dict, dict):
                raise ValueError(f"{path} -> best_params[{idx}] is not a dict.")
            # ensure float
            best_params = {k: float(best_params_dict[k]) for k in best_params_dict}
            # order check: if a subset/superset, we’ll intersect with param_names
            best_params = {k: best_params[k] for k in param_names if k in best_params}
            best_ll = None
            if "best_ll" in obj and isinstance(obj["best_ll"], list) and len(obj["best_ll"]) > idx:
                try:
                    best_ll = float(obj["best_ll"][idx])
                except Exception:
                    best_ll = None
            return best_params, best_ll

        # Case B: single dict
        if "best_params" in obj and isinstance(obj["best_params"], dict):
            best_params_dict = obj["best_params"]
            best_params = {k: float(best_params_dict[k]) for k in best_params_dict}
            best_params = {k: best_params[k] for k in param_names if k in best_params}
            best_ll = None
            if "best_ll" in obj:
                try:
                    best_ll = float(obj["best_ll"])
                except Exception:
                    best_ll = None
            return best_params, best_ll

    raise FileNotFoundError(
        f"Could not find usable fit params in {method_dir} (looked for fit_params.pkl / best_fit.pkl)."
    )

# ── Residuals ────────────────────────────────────────────────────────────────
def compute_residuals(exp_sfs, fit_sfs) -> np.ndarray:
    E = np.asarray(exp_sfs)
    F = np.asarray(fit_sfs)
    if E.shape != F.shape:
        raise ValueError(f"SFS shape mismatch: expected {E.shape} vs fitted {F.shape}")
    return E - F

# ── Model loader ─────────────────────────────────────────────────────────────
def load_model_callable(model_spec: str):
    # model_spec: "module.sub:func"
    if ":" not in model_spec:
        raise ValueError("--model-py must be in the form 'module.sub:func'")
    mod_name, func_name = model_spec.split(":")
    if mod_name.startswith("src."):
        module_file = mod_name.replace("src.", "").replace(".", "/") + ".py"
        module_path = Path(__file__).parent.parent / "src" / module_file
        if not module_path.exists():
            module_path = Path(__file__).parent.parent / "src" / (mod_name.split(".")[-1] + ".py")
        spec = importlib.util.spec_from_file_location(mod_name.split(".")[-1], module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return getattr(module, func_name)
    return getattr(importlib.import_module(mod_name), func_name)

def get_sample_sizes(cfg: dict) -> dict:
    """
    Return diploid sample sizes per deme from config.
    Accepts either 'sample_sizes' or 'num_samples'.
    """
    if "sample_sizes" in cfg and isinstance(cfg["sample_sizes"], dict):
        return cfg["sample_sizes"]
    if "num_samples" in cfg and isinstance(cfg["num_samples"], dict):
        return cfg["num_samples"]
    raise KeyError("Config must provide 'sample_sizes' or 'num_samples' (diploid counts per deme).")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        "Residuals from Expected(GT) − Fitted(best-LL from saved inference) for dadi/moments/both"
    )
    ap.add_argument("--mode", choices=["dadi", "moments", "both"], required=True)
    ap.add_argument("--config", type=Path, required=True,
                    help="JSON config with 'priors', 'sample_sizes' (diploid), 'mutation_rate', 'genome_length' and for dadi also 'pts'.")
    ap.add_argument("--model-py", type=str, required=True,
                    help="module:function returning demes.Graph given a param dict")
    ap.add_argument("--ground-truth", type=Path, required=True,
                    help="Pickle/JSON dict of param_name->value")
    ap.add_argument("--inference-dir", type=Path, required=True,
                    help="Directory that contains subdirs 'dadi' and/or 'moments' with saved fit_params.pkl/best_fit.pkl")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Where to write residuals (will create outdir/{dadi,moments})")
    return ap.parse_args()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg: Dict[str, Any] = load_json(args.config)
    demo_func = load_model_callable(args.model_py)

    param_names: List[str] = list(cfg["priors"].keys())
    gt_dict = load_gt_params(args.ground_truth)
    missing = [p for p in param_names if p not in gt_dict]
    if missing:
        raise ValueError(f"Ground-truth file missing parameters: {missing}")
    gt_vec = [float(gt_dict[p]) for p in param_names]

    sample_sizes = get_sample_sizes(cfg)       # diploid counts
    mu = cfg["mutation_rate"]
    L = cfg["genome_length"]

    # ── Main (fragment) ───────────────────────────────────────────────────────
    modes = ["dadi", "moments"] if args.mode == "both" else [args.mode]

    for mode in modes:
        # If user asked for both → write under outdir/<mode>;
        # else write directly to args.outdir (which already includes the engine).
        mode_outdir = (Path(args.outdir) / mode) if len(modes) > 1 else Path(args.outdir)
        mode_outdir.mkdir(parents=True, exist_ok=True)

        method_dir = args.inference_dir / mode
        best_params_dict, best_ll = load_best_params_from_inference(method_dir, param_names)
        full_best = {p: best_params_dict.get(p, gt_dict[p]) for p in param_names}
        best_vec = [float(full_best[p]) for p in param_names]

        if mode == "dadi":
            pts = cfg.get("pts", None)
            if pts is None:
                raise KeyError("Config must provide 'pts' for dadi.")
            fitted  = expected_sfs_dadi(best_vec, param_names, sample_sizes, demo_func, mu, L, pts)
            expected = expected_sfs_dadi(gt_vec,   param_names, sample_sizes, demo_func, mu, L, pts)
        else:
            fitted  = expected_sfs_moments(best_vec, param_names, sample_sizes, demo_func, mu, L)
            expected = expected_sfs_moments(gt_vec,   param_names, sample_sizes, demo_func, mu, L)

        print(f"[{mode}] Expected SFS sum: {np.sum(expected)}, Fitted SFS sum: {np.sum(fitted)}")

        residuals = compute_residuals(expected, fitted)
        residuals_flat = residuals.ravel()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.hist(residuals_flat, bins=50, alpha=0.7)
        plt.title(f"Residuals Histogram ({mode})")
        plt.xlabel("Residual Value"); plt.ylabel("Frequency")
        plt.savefig(mode_outdir / "residuals_histogram.png")
        plt.close()

        save_np(mode_outdir / "residuals.npy", residuals)
        save_np(mode_outdir / "residuals_flat.npy", residuals_flat)
        save_json_obj(mode_outdir / "meta.json", {
            "mode": mode,
            "shape": list(residuals.shape),
            "param_order": param_names,
            "best_params": full_best,
            "best_ll": best_ll,
            "notes": "Residuals = Expected(GT) − Fitted(best-LL params).",
        })

        print(f"[{mode}] wrote → {mode_outdir}")

if __name__ == "__main__":
    main()

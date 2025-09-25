#!/usr/bin/env python3
# computing_residuals_from_sfs.py
# Build residuals = Observed − Fitted SFS at saved best params (dadi/moments).

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, OrderedDict, Any
import argparse
import importlib, importlib.util
import json, pickle, sys
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Expected SFS helpers ------------------

def expected_sfs_dadi(params_in_order: List[float],
                      param_names: List[str],
                      sample_sizes: "OrderedDict[str,int]",
                      demo_model,
                      mutation_rate: float,
                      sequence_length: int,
                      pts: List[int]):
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
    theta = 4 * p_dict[param_names[0]] * mutation_rate * sequence_length
    fs *= theta
    return fs

def expected_sfs_moments(params_in_order: List[float],
                         param_names: List[str],
                         sample_sizes: "OrderedDict[str,int]",
                         demo_model,
                         mutation_rate: float,
                         genome_length: int):
    import moments
    p_dict = dict(zip(param_names, params_in_order))
    graph = demo_model(p_dict)
    haploid_sizes = [2 * n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())
    theta = 4 * p_dict[param_names[0]] * mutation_rate * genome_length
    return moments.Spectrum.from_demes(
        graph,
        sample_sizes=haploid_sizes,
        sampled_demes=sampled_demes,
        theta=theta,
    )

# ------------------ IO utils ------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def save_np(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def save_json_obj(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

# ------------------ Fit loading (robust) ------------------

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
    Accept:
      - fit_params.pkl with {'best_params': list[dict], 'best_ll': list[float]}
      - best_fit.pkl    with {'best_params': dict,       'best_ll': float}
      - best_fit.pkl    with our new wrapper format (dict + 'param_order')
    """
    for name in ("fit_params.pkl", "best_fit.pkl"):
        path = method_dir / name
        obj = _maybe_load_pickle(path)
        if not isinstance(obj, dict):
            continue

        if isinstance(obj.get("best_params"), dict):
            pd = {k: float(v) for k, v in obj["best_params"].items()}
            # reduce / order to param_names
            pd = {p: pd[p] for p in param_names if p in pd}
            best_ll = None
            if "best_ll" in obj:
                try: best_ll = float(obj["best_ll"])
                except Exception: best_ll = None
            return pd, best_ll

        if isinstance(obj.get("best_params"), list):
            bp = obj["best_params"]
            idx = 0
            if "best_ll" in obj and isinstance(obj["best_ll"], list) and len(obj["best_ll"]) == len(bp):
                idx = _best_idx_from_ll_list(obj["best_ll"])
            pd = {k: float(v) for k, v in bp[idx].items()}
            pd = {p: pd[p] for p in param_names if p in pd}
            best_ll = None
            if "best_ll" in obj and isinstance(obj["best_ll"], list):
                try: best_ll = float(obj["best_ll"][idx])
                except Exception: best_ll = None
            return pd, best_ll

    raise FileNotFoundError(f"No fit params found under {method_dir}")

# ------------------ Misc helpers ------------------

def compute_residuals(obs_sfs, fit_sfs) -> np.ndarray:
    return np.asarray(obs_sfs) - np.asarray(fit_sfs)

def _load_model_callable(model_spec: str):
    if ":" not in model_spec:
        raise ValueError("--model-py must be 'module:func'")
    mod_name, func_name = model_spec.split(":")
    if mod_name.startswith("src."):
        base = Path(__file__).parent.parent / "src"
        module_file = mod_name.replace("src.", "").replace(".", "/") + ".py"
        module_path = base / module_file
        if not module_path.exists():
            module_path = base / (mod_name.split(".")[-1] + ".py")
        spec = importlib.util.spec_from_file_location(mod_name.split(".")[-1], module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return getattr(module, func_name)
    return getattr(importlib.import_module(mod_name), func_name)

def _diploid_sample_sizes_from_cfg(cfg: dict) -> "OrderedDict[str,int]":
    if "sample_sizes" in cfg and isinstance(cfg["sample_sizes"], dict):
        d = cfg["sample_sizes"]
    elif "num_samples" in cfg and isinstance(cfg["num_samples"], dict):
        d = cfg["num_samples"]
    else:
        raise KeyError("Config must provide 'sample_sizes' or 'num_samples'.")
    # preserve order
    return type(d)((str(k), int(v)) for k,v in d.items())

def _auto_pts_from_samples(sample_sizes: "OrderedDict[str,int]") -> List[int]:
    n_max_hap = max(2 * n for n in sample_sizes.values())
    return [n_max_hap + 20, n_max_hap + 40, n_max_hap + 60]

# ------------------ CLI ------------------

def parse_args():
    ap = argparse.ArgumentParser("Residuals (Observed − Fitted) for dadi/moments/both")
    ap.add_argument("--mode", choices=["dadi", "moments", "both"], required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model-py", type=str, required=True)
    ap.add_argument("--observed-sfs", type=Path, required=True)
    ap.add_argument("--inference-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    return ap.parse_args()

# ------------------ Main ------------------

def main():
    args = parse_args()
    cfg: Dict[str, Any] = load_json(args.config)
    orig_model = _load_model_callable(args.model_py)

    # wrap model to pass cfg if supported
    def demo_func(pdict):
        try:
            return orig_model(pdict, cfg)
        except TypeError:
            return orig_model(pdict)

    param_names: List[str] = list(cfg["priors"].keys())
    priors = cfg["priors"]

    sample_sizes = _diploid_sample_sizes_from_cfg(cfg)
    mu = float(cfg["mutation_rate"])
    L = int(cfg["genome_length"])

    with open(args.observed_sfs, "rb") as f:
        obs_obj = pickle.load(f)
    # Coerce to array (Spectrum is array-like)
    observed = np.asarray(obs_obj)

    modes = ["dadi", "moments"] if args.mode == "both" else [args.mode]

    for mode in modes:
        mode_outdir = (Path(args.outdir) / mode) if len(modes) > 1 else Path(args.outdir)
        mode_outdir.mkdir(parents=True, exist_ok=True)

        method_dir = args.inference_dir / mode
        best_params_dict, best_ll = load_best_params_from_inference(method_dir, param_names)

        # Fill any missing params with prior midpoint
        def prior_mid(p: str) -> float:
            lo, hi = priors[p]
            return (float(lo) + float(hi)) / 2.0

        full_best = {p: (float(best_params_dict[p]) if p in best_params_dict else prior_mid(p))
                     for p in param_names}
        best_vec = [full_best[p] for p in param_names]

        # Build fitted SFS
        if mode == "dadi":
            pts = cfg.get("pts", None)
            if pts is None:
                pts = _auto_pts_from_samples(sample_sizes)
            fitted = expected_sfs_dadi(best_vec, param_names, sample_sizes, demo_func, mu, L, pts)
        else:
            fitted = expected_sfs_moments(best_vec, param_names, sample_sizes, demo_func, mu, L)

        if np.asarray(observed).shape != np.asarray(fitted).shape:
            raise ValueError(
                f"SFS shape mismatch: observed {np.asarray(observed).shape} vs fitted {np.asarray(fitted).shape}. "
                "Check folding and sample sizes; for dadi also pts."
            )

        residuals = compute_residuals(observed, fitted)  # Observed − Fitted
        residuals_flat = residuals.ravel()

        print(f"[{mode}] sum(Fitted)={float(np.sum(fitted)):.6g}  "
              f"sum(Observed)={float(np.sum(observed)):.6g}  "
              f"sum(Residuals)={float(np.sum(residuals)):.6g}")

        # QC plot
        plt.figure(figsize=(8, 6))
        plt.hist(residuals_flat, bins=50, alpha=0.7)
        plt.title(f"Residuals Histogram ({mode})")
        plt.xlabel("Observed − Fitted"); plt.ylabel("Frequency")
        plt.savefig(mode_outdir / "residuals_histogram.png")
        plt.close()

        # Save
        save_np(mode_outdir / "residuals.npy", residuals)
        save_np(mode_outdir / "residuals_flat.npy", residuals_flat)
        save_json_obj(mode_outdir / "meta.json", {
            "mode": mode,
            "shape": list(residuals.shape),
            "param_order": param_names,
            "best_params": full_best,
            "best_ll": best_ll,
            "observed_source": str(args.observed_sfs),
            "notes": "Residuals = Observed − Fitted",
        })

        print(f"[{mode}] wrote → {mode_outdir}")


if __name__ == "__main__":
    main()

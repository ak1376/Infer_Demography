#!/usr/bin/env python3
# snakemake_scripts/computing_residuals_from_sfs.py
# Build residuals = Observed − Fitted SFS at saved best params (dadi/moments),
# with optional Gram–Schmidt projection (configured in the JSON config).

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, OrderedDict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Make project root importable (so "src" imports work from snakemake_scripts/)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gram_schmidt import project_vector_onto_gs_basis  # noqa: E402


# ------------------ Expected SFS helpers ------------------


def expected_sfs_dadi(
    params_in_order: List[float],
    param_names: List[str],
    sample_sizes: "OrderedDict[str,int]",
    demo_model,
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

    # Convention: first parameter in param_names is N0 (as in your pipeline)
    theta = 4 * p_dict[param_names[0]] * mutation_rate * sequence_length
    fs *= theta
    return fs


def expected_sfs_moments(
    params_in_order: List[float],
    param_names: List[str],
    sample_sizes: "OrderedDict[str,int]",
    demo_model,
    mutation_rate: float,
    genome_length: int,
):
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
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _best_idx_from_ll_list(ll_list: List[float]) -> int:
    if not ll_list:
        raise ValueError("Empty best_ll list.")
    return int(np.argmax(np.asarray(ll_list, dtype=float)))


def load_best_params_from_inference(
    method_dir: Path, param_names: List[str]
) -> Tuple[Dict[str, float], float | None]:
    """
    Accept:
      - fit_params.pkl with {'best_params': list[dict], 'best_ll': list[float]}
      - best_fit.pkl    with {'best_params': dict,       'best_ll': float}
      - best_fit.pkl    with wrapper format (dict + 'param_order') (ignored safely)
    """
    for fname in ("fit_params.pkl", "best_fit.pkl"):
        path = method_dir / fname
        obj = _maybe_load_pickle(path)
        if not isinstance(obj, dict):
            continue

        # Case A: best_params is a single dict
        if isinstance(obj.get("best_params"), dict):
            pd = {k: float(v) for k, v in obj["best_params"].items()}
            pd = {p: pd[p] for p in param_names if p in pd}

            best_ll = None
            if "best_ll" in obj:
                try:
                    best_ll = float(obj["best_ll"])
                except Exception:
                    best_ll = None
            return pd, best_ll

        # Case B: best_params is list of dicts
        if isinstance(obj.get("best_params"), list):
            bp = obj["best_params"]
            idx = 0
            if (
                "best_ll" in obj
                and isinstance(obj["best_ll"], list)
                and len(obj["best_ll"]) == len(bp)
            ):
                idx = _best_idx_from_ll_list(obj["best_ll"])

            pd = {k: float(v) for k, v in bp[idx].items()}
            pd = {p: pd[p] for p in param_names if p in pd}

            best_ll = None
            if "best_ll" in obj and isinstance(obj["best_ll"], list):
                try:
                    best_ll = float(obj["best_ll"][idx])
                except Exception:
                    best_ll = None
            return pd, best_ll

    raise FileNotFoundError(f"No fit params found under {method_dir}")


# ------------------ Misc helpers ------------------


def compute_residuals(obs_sfs, fit_sfs) -> np.ndarray:
    return np.asarray(obs_sfs) - np.asarray(fit_sfs)


def _load_model_callable(model_spec: str):
    """
    model_spec: 'module:func'. Supports src.* by loading from project/src/.
    """
    if ":" not in model_spec:
        raise ValueError("--model-py must be 'module:func'")
    mod_name, func_name = model_spec.split(":", 1)

    # If it is a src.* module, load file directly to avoid PYTHONPATH issues
    if mod_name.startswith("src."):
        base = ROOT / "src"
        rel = mod_name.replace("src.", "").replace(".", "/") + ".py"
        module_path = base / rel
        if not module_path.exists():
            # fallback: src/<last>.py
            module_path = base / (mod_name.split(".")[-1] + ".py")

        spec = importlib.util.spec_from_file_location(mod_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    module = importlib.import_module(mod_name)
    return getattr(module, func_name)


def _diploid_sample_sizes_from_cfg(cfg: dict) -> "OrderedDict[str,int]":
    if "sample_sizes" in cfg and isinstance(cfg["sample_sizes"], dict):
        d = cfg["sample_sizes"]
    elif "num_samples" in cfg and isinstance(cfg["num_samples"], dict):
        d = cfg["num_samples"]
    else:
        raise KeyError("Config must provide 'sample_sizes' or 'num_samples'.")
    return type(d)((str(k), int(v)) for k, v in d.items())


def _auto_pts_from_samples(sample_sizes: "OrderedDict[str,int]") -> List[int]:
    n_max_hap = max(2 * n for n in sample_sizes.values())
    return [n_max_hap + 20, n_max_hap + 40, n_max_hap + 60]


def collapse_sfs_bins(sfs: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Collapse SFS into fewer bins by summing adjacent entries along the flattened array.

    Note: this collapses the *flattened* SFS; shape becomes (n_bins,).
    """
    sfs_flat = np.asarray(sfs).ravel()
    original_bins = sfs_flat.size

    if n_bins >= original_bins:
        return np.asarray(sfs)

    edges = np.linspace(0, original_bins, n_bins + 1, dtype=int)
    collapsed = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        collapsed[i] = np.sum(sfs_flat[edges[i] : edges[i + 1]])
    return collapsed


def _safe_float(x: Any, *, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Expected {name} to be float-like, got {x!r}") from e


# ------------------ CLI ------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="Residuals (Observed − Fitted) for dadi/moments/both, optionally GS-projected."
    )
    ap.add_argument("--mode", choices=["dadi", "moments", "both"], required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model-py", type=str, required=True)
    ap.add_argument("--observed-sfs", type=Path, required=True)
    ap.add_argument("--inference-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Collapse SFS into this many bins (default: no collapsing).",
    )
    return ap.parse_args()


# ------------------ Main ------------------


def main():
    args = parse_args()
    cfg: Dict[str, Any] = load_json(args.config)

    # Load demographic model callable
    orig_model = _load_model_callable(args.model_py)

    # Wrap model to pass cfg if supported
    def demo_func(pdict):
        try:
            return orig_model(pdict, cfg)
        except TypeError:
            return orig_model(pdict)

    # Parameters / priors
    if "priors" not in cfg or not isinstance(cfg["priors"], dict):
        raise KeyError("Config missing 'priors' dict.")
    param_names: List[str] = list(cfg["priors"].keys())
    priors = cfg["priors"]

    # Samples and constants
    sample_sizes = _diploid_sample_sizes_from_cfg(cfg)
    mu = _safe_float(cfg.get("mutation_rate"), name="mutation_rate")
    L = int(float(cfg.get("genome_length")))

    # Optional Gram–Schmidt config
    use_gs = bool(cfg.get("gram_schmidt", False))
    gs_k = cfg.get("gram_schmidt_k", None)
    gs_basis_type = str(cfg.get("gram_schmidt_basis", "poly"))
    gs_eps = float(cfg.get("gram_schmidt_eps", 1e-12))

    if use_gs:
        if gs_k is None:
            raise KeyError("Config has gram_schmidt=true but missing gram_schmidt_k.")
        gs_k = int(gs_k)
        if gs_k < 1:
            raise ValueError("gram_schmidt_k must be >= 1.")

    # Load observed SFS object (Spectrum or array)
    with open(args.observed_sfs, "rb") as f:
        obs_obj = pickle.load(f)
    observed = np.asarray(obs_obj)

    modes = ["dadi", "moments"] if args.mode == "both" else [args.mode]

    for mode in modes:
        mode_outdir = (Path(args.outdir) / mode) if len(modes) > 1 else Path(args.outdir)
        mode_outdir.mkdir(parents=True, exist_ok=True)

        # Load best params
        method_dir = args.inference_dir / mode
        best_params_dict, best_ll = load_best_params_from_inference(method_dir, param_names)

        # Fill missing params with prior midpoint
        def prior_mid(p: str) -> float:
            lo, hi = priors[p]
            return (float(lo) + float(hi)) / 2.0

        full_best: Dict[str, float] = {
            p: (float(best_params_dict[p]) if p in best_params_dict else prior_mid(p))
            for p in param_names
        }
        best_vec = [full_best[p] for p in param_names]

        # Build fitted SFS
        if mode == "dadi":
            pts = cfg.get("pts", None)
            if pts is None:
                pts = _auto_pts_from_samples(sample_sizes)
            fitted = expected_sfs_dadi(best_vec, param_names, sample_sizes, demo_func, mu, L, pts)
        else:
            fitted = expected_sfs_moments(best_vec, param_names, sample_sizes, demo_func, mu, L)

        # Optional collapse (note: this changes shape to (n_bins,))
        obs_to_use = observed
        fit_to_use = fitted
        collapsed = False
        if args.n_bins is not None:
            obs_to_use = collapse_sfs_bins(observed, args.n_bins)
            fit_to_use = collapse_sfs_bins(fitted, args.n_bins)
            collapsed = True
        else:
            # Require identical shapes if not collapsing
            if np.asarray(observed).shape != np.asarray(fitted).shape:
                raise ValueError(
                    f"SFS shape mismatch: observed {np.asarray(observed).shape} vs fitted {np.asarray(fitted).shape}. "
                    "Check folding and sample sizes; for dadi also pts."
                )

        residuals = compute_residuals(obs_to_use, fit_to_use)  # Observed − Fitted
        residuals_flat = np.asarray(residuals).ravel()

        print(
            f"[{mode}] sum(Fitted)={float(np.sum(fitted)):.6g}  "
            f"sum(Observed)={float(np.sum(observed)):.6g}  "
            f"sum(Residuals)={float(np.sum(residuals)):.6g}"
        )
        if collapsed:
            print(
                f"[{mode}] Collapsed SFS from {np.asarray(observed).size} to {np.asarray(residuals_flat).size} bins"
            )

        # Optional Gram–Schmidt projection: produces k_eff coefficients
        gs_result = None
        residuals_gs_coeffs = None
        if use_gs:
            D = residuals_flat.size
            if gs_k > D:
                raise ValueError(f"gram_schmidt_k={gs_k} cannot exceed residual dimension D={D}.")

            gs_result = project_vector_onto_gs_basis(
                residuals_flat,
                k=gs_k,
                basis_type=gs_basis_type,
                eps=gs_eps,
            )
            residuals_gs_coeffs = np.asarray(gs_result.coeffs)

        # ---------------- QC plots ----------------
        # Histogram of full residuals
        plt.figure(figsize=(8, 6))
        plt.hist(residuals_flat, bins=50, alpha=0.7)
        plt.title(f"Residuals Histogram ({mode})")
        plt.xlabel("Observed − Fitted")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(mode_outdir / "residuals_histogram.png")
        plt.close()

        # Histogram of GS coefficients (if enabled)
        if use_gs and residuals_gs_coeffs is not None:
            plt.figure(figsize=(8, 6))
            plt.hist(residuals_gs_coeffs, bins=50, alpha=0.7)
            plt.title(f"GS Coeffs Histogram ({mode})")
            plt.xlabel("GS coefficient value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(mode_outdir / "residuals_gs_coeffs_histogram.png")
            plt.close()

        # ---------------- Saves ----------------
        save_np(mode_outdir / "residuals.npy", np.asarray(residuals))
        save_np(mode_outdir / "residuals_flat.npy", residuals_flat)

        if use_gs and gs_result is not None:
            save_np(mode_outdir / "residuals_gs_coeffs.npy", residuals_gs_coeffs)
            save_np(mode_outdir / "residuals_gs_basis.npy", np.asarray(gs_result.basis))
            save_np(mode_outdir / "residuals_gs_reconstruction.npy", np.asarray(gs_result.reconstruction))

        meta: Dict[str, Any] = {
            "mode": mode,
            "shape": list(np.asarray(residuals).shape),
            "flat_dim": int(residuals_flat.size),
            "param_order": param_names,
            "best_params": full_best,
            "best_ll": best_ll,
            "observed_source": str(args.observed_sfs),
            "n_bins": args.n_bins,
            "original_sfs_shape": list(np.asarray(observed).shape),
            "notes": "Residuals = Observed − Fitted",
            "gram_schmidt": use_gs,
        }
        if use_gs and gs_result is not None:
            meta.update(
                {
                    "gram_schmidt_k": int(gs_k),
                    "gram_schmidt_k_effective": int(np.asarray(gs_result.basis).shape[1]),
                    "gram_schmidt_basis": gs_basis_type,
                    "gram_schmidt_eps": float(gs_eps),
                }
            )

        save_json_obj(mode_outdir / "meta.json", meta)
        print(f"[{mode}] wrote → {mode_outdir}")


if __name__ == "__main__":
    main()

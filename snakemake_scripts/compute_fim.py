#!/usr/bin/env python3
"""
compute_fim.py — wrapper that selects the best params and calls fisher_info_sfs

Usage (from Snakefile rule compute_fim):
  python compute_fim.py \
      --engine {wildcards.engine} \
      --fit-pkl {input.fit} \
      --sfs {input.sfs} \
      --config {params.cfg} \
      --fim-npy {output.fim} \
      --summary-json {output.summ}
"""

from __future__ import annotations
import argparse, json, pickle, importlib, importlib.util, math
from pathlib import Path
from typing import Any, List, Dict
import numpy as np

from fisher_info_sfs import observed_fim_theta

# ------------------ Model import helpers ------------------


def _load_module_func(mod_name: str, func_name: str):
    """Import a function; support `src.*` path-to-file loading."""
    if mod_name.startswith("src."):
        base = Path(__file__).parent.parent / "src"
        module_file = mod_name.replace("src.", "").replace(".", "/") + ".py"
        module_path = base / module_file
        if not module_path.exists():
            module_path = base / (mod_name.split(".")[-1] + ".py")
        spec = importlib.util.spec_from_file_location(
            mod_name.split(".")[-1], module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader, f"Cannot load module: {module_path}"
        spec.loader.exec_module(module)
        return getattr(module, func_name)
    return getattr(importlib.import_module(mod_name), func_name)


def _model_import_from_cfg(cfg: dict):
    """Resolve model function from config['demographic_model'] to src.simulation:*."""
    model = cfg["demographic_model"]
    if model == "drosophila_three_epoch":
        mod_path, func_name = "src.simulation", "drosophila_three_epoch"
    else:
        mod_path, func_name = "src.simulation", f"{model}_model"
    return _load_module_func(mod_path, func_name)


# ------------------ Fit-parsing helpers ------------------


def _param_order_from_cfg(cfg: dict) -> list[str]:
    # JSON preserves insertion order ⇒ use priors key order
    return list(cfg["priors"].keys())


def _best_params_from_fit_any(fit_blob: dict, param_order: list[str]) -> np.ndarray:
    """
    Support both formats:
      A) {'best_params': [ {name: val}, ... ], 'best_ll': [ ... ]}
      B) {'best_params': {name: val, ...}, 'best_ll': float}
      C) (our latest wrapper) {'best_params': {name: val, ...}, 'param_order': [...], 'best_ll': float}
    Returns a vector ordered by `param_order`.
    """
    if not isinstance(fit_blob, dict):
        raise ValueError("fit-pkl must unpickle to a dict.")

    # Case C/B: single dict of params
    if isinstance(fit_blob.get("best_params"), dict):
        pd = {k: float(v) for k, v in fit_blob["best_params"].items()}
        return np.array([pd[p] for p in param_order], dtype=float)

    # Case A: list with (optional) best_ll list
    if isinstance(fit_blob.get("best_params"), list):
        params_list = fit_blob["best_params"]
        if (
            "best_ll" in fit_blob
            and isinstance(fit_blob["best_ll"], list)
            and len(fit_blob["best_ll"]) == len(params_list)
        ):
            idx = int(np.argmax(np.array(fit_blob["best_ll"], dtype=float)))
        else:
            idx = 0
        pd = {k: float(v) for k, v in params_list[idx].items()}
        return np.array([pd[p] for p in param_order], dtype=float)

    raise ValueError("Unrecognized fit format in best_fit/fit_params.pkl.")


# ------------------ SFS helpers ------------------


def _diploid_sample_sizes_from_cfg(cfg: dict) -> Dict[str, int]:
    if "sample_sizes" in cfg and isinstance(cfg["sample_sizes"], dict):
        return {str(k): int(v) for k, v in cfg["sample_sizes"].items()}
    if "num_samples" in cfg and isinstance(cfg["num_samples"], dict):
        return {str(k): int(v) for k, v in cfg["num_samples"].items()}
    raise KeyError(
        "Config must provide 'sample_sizes' or 'num_samples' (diploid per deme)."
    )


def _auto_pts_from_samples(sample_sizes: Dict[str, int]) -> List[int]:
    """
    For dadi: choose pts based on largest haploid sample size.
    """
    n_max_hap = max(2 * n for n in sample_sizes.values())
    return [n_max_hap + 20, n_max_hap + 40, n_max_hap + 60]


# ------------------ CLI ------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["dadi", "moments"])
    ap.add_argument("--fit-pkl", required=True, type=Path)
    ap.add_argument("--sfs", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fim-npy", required=True, type=Path)
    ap.add_argument("--summary-json", required=True, type=Path)
    ap.add_argument("--rel-step", type=float, default=1e-4)
    ap.add_argument(
        "--pts",
        type=str,
        default="auto",
        help='For dadi: "n1,n2,n3" or "auto" (default). Ignored for moments.',
    )
    args = ap.parse_args()

    # Load inputs
    fit_blob = pickle.load(args.fit_pkl.open("rb"))
    sfs = pickle.load(args.sfs.open("rb"))
    cfg = json.loads(args.config.read_text())
    mu = float(cfg["mutation_rate"])
    L = int(cfg["genome_length"])

    # Resolve model and wrap to pass config if supported
    orig_model = _model_import_from_cfg(cfg)

    def model_func(pdict):
        try:
            return orig_model(
                pdict, cfg
            )  # our src.simulation functions accept (params, config)
        except TypeError:
            return orig_model(pdict)

    param_order = _param_order_from_cfg(cfg)
    theta = _best_params_from_fit_any(fit_blob, param_order)

    # dadi pts
    pts = "auto"
    if args.engine == "dadi":
        if args.pts and args.pts.lower() != "auto":
            xs = [int(x) for x in args.pts.split(",")]
            if len(xs) != 3:
                raise SystemExit("--pts must be three integers or 'auto'")
            pts = xs
        else:
            pts = _auto_pts_from_samples(_diploid_sample_sizes_from_cfg(cfg))

    # Compute observed Fisher Information
    info, cov, se, free_idx = observed_fim_theta(
        sfs=sfs,
        param_names=param_order,
        theta_at=theta,
        model_func=model_func,
        mu=mu,
        L=L,
        engine=args.engine,
        pts=pts,
        fixed_params=None,
        rel_step=args.rel_step,
    )

    # Save outputs
    args.fim_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.fim_npy, info)

    summary = {}
    try:
        w, _ = np.linalg.eigh(info)
        w = np.asarray(w, float)
        w_clip = np.clip(w, 1e-300, None)
        summary["logdet"] = float(np.sum(np.log(w_clip)))
        summary["min_eigen"] = float(np.min(w))
        summary["max_eigen"] = float(np.max(w))
        summary["cond"] = float(np.max(w_clip) / np.min(w_clip))
    except Exception:
        pass

    if se is not None:
        summary["SE"] = se
    diag = np.diag(info)
    summary["diag"] = {param_order[i]: float(diag[k]) for k, i in enumerate(free_idx)}

    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanitize(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    (args.summary_json).write_text(json.dumps(_sanitize(summary), indent=2))
    print(f"✓ FIM → {args.fim_npy}")
    print(f"✓ summary → {args.summary_json}")


if __name__ == "__main__":
    main()

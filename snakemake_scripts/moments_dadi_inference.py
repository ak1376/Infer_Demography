#!/usr/bin/env python3
# snakemake_scripts/sfs_optimize_cli.py
# Unified custom Poisson SFS optimisation (dadi | moments | both) — NO --sampled-demes needed

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import importlib
import json
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt

import nlopt
import numdifftools as nd

# ────────────────────────────── CLI ──────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser("Custom Poisson SFS optimisation (dadi|moments|both) with NLopt")
    p.add_argument("--mode", choices=["dadi", "moments", "both"], required=True)
    p.add_argument("--sfs-file", type=Path, required=True,
                   help="Pickle of dadi.Spectrum | moments.Spectrum | numpy array (counts)")
    p.add_argument("--config", type=Path, required=True,
                   help="JSON with {'priors':{...}, 'mutation_rate':float, 'genome_length':float}")
    p.add_argument("--model-py", type=str, required=True,
                   help="module:function returning demes.Graph when called with a param dict")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Parent output directory. For --mode both, writes into outdir/{dadi,moments}")
    p.add_argument("--maxeval", type=int, default=800)
    p.add_argument("--rtol", type=float, default=1e-8, help="Relative tol for NLopt")
    p.add_argument("--use-bobyqa", action="store_true",
                   help="Use derivative-free LN_BOBYQA (default is LBFGS with finite-diff gradients)")
    p.add_argument("--seed-perturb", type=float, default=0.10,
                   help="Log-normal multiplicative perturbation around geom-mid start (0→off)")
    p.add_argument("--fix", action="append", default=[],
                   help="Fix params by name, e.g. --fix N0=10000 (repeatable)")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()

# ─────────────────────────── utilities ────────────────────────────────────

EPS_SFS = 1e-300  # guard for log(0) in Poisson LL

def ordered_param_names(priors: Dict[str, List[float]]) -> List[str]:
    return list(priors.keys())

def geom_mid(lo: float, hi: float) -> float:
    return float(np.sqrt(float(lo) * float(hi)))

def parse_fixed(items: List[str]) -> Dict[str, float]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--fix requires name=value (got {it!r})")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out

def haploid_ns_from_sfs(obj) -> List[int]:
    shape = getattr(obj, "shape", None)
    if shape is None:
        raise ValueError("Observed SFS object has no shape.")
    return [int(d - 1) for d in shape]

def default_pts(ns_haploid: List[int]) -> List[int]:
    nmax = int(max(ns_haploid))
    return [nmax + 10, nmax + 20, nmax + 30]

def load_sfs(path: Path, prefer: str):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, np.ndarray):
        if prefer == "dadi":
            import dadi; return dadi.Spectrum(obj)
        else:
            import moments; return moments.Spectrum(obj)
    return obj

# ───────────── demes → Spectrum (backend-agnostic, with θ scaling) ───────

def expected_sfs_log10(
    log10_params_full: np.ndarray,
    *,
    param_names: List[str],
    demo_func,
    sampled_demes: List[str],
    ns_haploid: List[int],
    mu_times_L: float,   # μ·L
    mode: str,           # "dadi" | "moments"
    pts: List[int] | None,
    sfs_folded: bool,
):
    vals = 10.0 ** np.asarray(log10_params_full, float)
    p_dict = {k: float(v) for k, v in zip(param_names, vals)}
    graph = demo_func(p_dict)

    N0 = float(p_dict[param_names[0]])
    theta = 4.0 * N0 * mu_times_L

    if mode == "dadi":
        import dadi
        if pts is None:
            pts = default_pts(ns_haploid)
        fs = dadi.Spectrum.from_demes(
            graph, sampled_demes=sampled_demes, sample_sizes=ns_haploid, pts=pts
        )
        fs = fs * theta
    elif mode == "moments":
        import moments
        fs = moments.Spectrum.from_demes(
            graph, sampled_demes=sampled_demes, sample_sizes=ns_haploid, theta=theta
        )
    else:
        raise ValueError("mode must be 'dadi' or 'moments'")

    if sfs_folded:
        fs = fs.fold()
    return fs

def poisson_ll_full(
    log10_params_full: np.ndarray,
    observed_sfs,
    *,
    param_names: List[str],
    demo_func,
    sampled_demes: List[str],
    ns_haploid: List[int],
    mu_times_L: float,
    mode: str,
    pts: List[int] | None,
) -> float:
    model = expected_sfs_log10(
        log10_params_full,
        param_names=param_names,
        demo_func=demo_func,
        sampled_demes=sampled_demes,
        ns_haploid=ns_haploid,
        mu_times_L=mu_times_L,
        mode=mode,
        pts=pts,
        sfs_folded=bool(getattr(observed_sfs, "folded", False)),
    )
    model = np.maximum(model, EPS_SFS)
    return float(np.sum(np.log(model) * observed_sfs - model))

# ───────────── fixed-parameter packing (remove from opt vector) ───────────

def build_packers(names, lb, ub, start, fixed_by_name):
    fixed_idx = [i for i, n in enumerate(names) if n in fixed_by_name]
    free_idx  = [i for i, n in enumerate(names) if n not in fixed_by_name]

    x_full0 = start.copy()
    for i in fixed_idx:
        x_full0[i] = float(fixed_by_name[names[i]])

    for i in fixed_idx:
        v = x_full0[i]
        if not (lb[i] <= v <= ub[i]):
            raise ValueError(f"Fixed value {names[i]}={v} outside bounds [{lb[i]}, {ub[i]}].")

    def pack(x_full):
        return np.asarray([x_full[i] for i in free_idx], float)

    def unpack(x_free):
        x_full = x_full0.copy()
        for j, i in enumerate(free_idx):
            x_full[i] = float(x_free[j])
        return x_full

    return free_idx, fixed_idx, x_full0, pack, unpack

# ─────────────── NLopt driver (maximization in log10 space) ───────────────

def optimize_poisson_with_fixed(
    *,
    names: List[str],
    lb_full: np.ndarray,
    ub_full: np.ndarray,
    start_full: np.ndarray,
    observed_sfs,
    demo_func,
    sampled_demes: List[str],
    ns_haploid: List[int],
    mu_times_L: float,
    mode: str,
    pts: List[int] | None,
    use_bobyqa: bool,
    rtol: float,
    maxeval: int,
    verbose: int,
    fixed_by_name: Dict[str, float],
) -> Tuple[np.ndarray, object, float, int]:
    free_idx, fixed_idx, x_full0, pack, unpack = build_packers(
        names, lb_full, ub_full, start_full, fixed_by_name
    )

    if len(free_idx) == 0:
        x_full_hat = x_full0
        best_ll = poisson_ll_full(
            np.log10(x_full_hat), observed_sfs,
            param_names=names, demo_func=demo_func,
            sampled_demes=sampled_demes, ns_haploid=ns_haploid,
            mu_times_L=mu_times_L, mode=mode, pts=pts,
        )
        fitted_sfs = expected_sfs_log10(
            np.log10(x_full_hat), param_names=names, demo_func=demo_func,
            sampled_demes=sampled_demes, ns_haploid=ns_haploid,
            mu_times_L=mu_times_L, mode=mode, pts=pts,
            sfs_folded=bool(getattr(observed_sfs, "folded", False)),
        )
        return x_full_hat, fitted_sfs, float(best_ll), 0

    lb_free = np.asarray([lb_full[i] for i in free_idx], float)
    ub_free = np.asarray([ub_full[i] for i in free_idx], float)
    x0_free = np.asarray([x_full0[i] for i in free_idx], float)

    def obj_free(xlog10_free):
        x_full = unpack(10.0 ** np.asarray(xlog10_free, float))
        return poisson_ll_full(
            np.log10(x_full), observed_sfs,
            param_names=names, demo_func=demo_func,
            sampled_demes=sampled_demes, ns_haploid=ns_haploid,
            mu_times_L=mu_times_L, mode=mode, pts=pts,
        )

    grad_fn = nd.Gradient(obj_free, step=1e-4)

    def objective(xlog10_free, grad):
        ll = obj_free(xlog10_free)
        if grad.size > 0 and not use_bobyqa:
            grad[:] = grad_fn(xlog10_free)
        if verbose:
            print(f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(xlog10_free), precision=4)}")
        return ll

    opt = nlopt.opt(nlopt.LN_BOBYQA if use_bobyqa else nlopt.LD_LBFGS, len(free_idx))
    opt.set_lower_bounds(np.log10(lb_free))
    opt.set_upper_bounds(np.log10(ub_free))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)
    opt.set_xtol_rel(rtol)
    opt.set_maxeval(maxeval)

    try:
        x_free_hat_log = opt.optimize(np.log10(x0_free))
        status = opt.last_optimize_result()
        best_ll = opt.last_optimum_value()
    except Exception as e:
        if verbose:
            print(f"[WARN] NLopt exception: {e}. Attempting to recover last best…")
        try:
            x_free_hat_log = opt.last_optimum_x()
            best_ll = opt.last_optimum_value()
            status = opt.last_optimize_result()
        except Exception:
            x_free_hat_log = np.log10(x0_free)
            best_ll = obj_free(x_free_hat_log)
            status = -1

    x_full_hat = unpack(10.0 ** x_free_hat_log)
    fitted_sfs = expected_sfs_log10(
        np.log10(x_full_hat), param_names=names, demo_func=demo_func,
        sampled_demes=sampled_demes, ns_haploid=ns_haploid,
        mu_times_L=mu_times_L, mode=mode, pts=pts,
        sfs_folded=bool(getattr(observed_sfs, "folded", False)),
    )
    return x_full_hat, fitted_sfs, float(best_ll), int(status)

# ─────────────────────────── plotting ─────────────────────────────────────

def save_profiles_grid(
    outdir: Path,
    param_names: List[str],
    fitted_params_full: np.ndarray,
    observed_sfs,
    mu_times_L: float,
    mode: str,
    sampled_demes: List[str],
    ns_haploid: List[int],
    demo_func,
    fixed_by_name: Dict[str, float],
):
    outpng = outdir / "profiles_grid.png"
    k = len(param_names)
    ncols = 3 if k >= 3 else k
    nrows = math.ceil(k / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), dpi=150, squeeze=False)
    axes = axes.ravel()
    pts = default_pts(ns_haploid) if mode == "dadi" else None

    for i, (ax, pname) in enumerate(zip(axes, param_names)):
        if pname in fixed_by_name:
            xstar = float(fixed_by_name[pname])
            ax.axvline(xstar, linestyle="--")
            ax.set_xscale("log")
            ax.set_xlabel(f"{pname} (fixed)")
            if i % ncols == 0:
                ax.set_ylabel("Poisson log-likelihood")
            ax.grid(alpha=0.25, which="both")
            continue

        center = float(fitted_params_full[i])
        glo = max(center / 50.0, 1e-12)
        ghi = center * 50.0
        if i >= k - 2:
            glo = max(center / 100.0, 1e-12)
            ghi = max(ghi, center * 100.0)
        if ghi <= glo * 1.0001:
            ghi = glo * 10.0

        grid = np.logspace(np.log10(glo), np.log10(ghi), 41)
        y = []
        for v in grid:
            test = fitted_params_full.copy()
            test[i] = float(v)
            ll = poisson_ll_full(
                np.log10(test), observed_sfs,
                param_names=param_names, demo_func=demo_func,
                sampled_demes=sampled_demes, ns_haploid=ns_haploid,
                mu_times_L=mu_times_L, mode=mode, pts=pts,
            )
            y.append(ll)

        ax.scatter(grid, y, s=10)
        ax.axvline(fitted_params_full[i], linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel(pname)
        if i % ncols == 0:
            ax.set_ylabel("Poisson log-likelihood")
        ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(outpng)
    plt.close(fig)
    print(f"[plot] saved {outpng}")

# ─────────────────────────── per-mode run ─────────────────────────────────

def _infer_sampled_demes(data_sfs, cfg, ns_len: int) -> List[str]:
    # 1) prefer SFS pop_ids if available
    pop_ids = getattr(data_sfs, "pop_ids", None)
    if pop_ids:
        return list(pop_ids)
    # 2) fall back to config order
    demes = list(cfg.get("num_samples", {}).keys())
    if demes and len(demes) == ns_len:
        return demes
    # 3) synthesize placeholder names
    return [f"pop{i}" for i in range(ns_len)]

def run_one_mode(mode: str, a, cfg, demo_func, data_sfs):
    outdir = a.outdir / ("dadi" if mode == "dadi" else "moments")
    outdir.mkdir(parents=True, exist_ok=True)

    pri = cfg["priors"]
    names = ordered_param_names(pri)
    mu = float(cfg["mutation_rate"])
    L = float(cfg["genome_length"])
    muL = mu * L

    ns = haploid_ns_from_sfs(data_sfs)
    sampled_demes = _infer_sampled_demes(data_sfs, cfg, len(ns))

    lb = np.array([pri[p][0] for p in names], float)
    ub = np.array([pri[p][1] for p in names], float)
    start = np.array([geom_mid(*pri[p]) for p in names], float)

    if a.seed_perturb and a.seed_perturb > 0:
        rng = np.random.default_rng(12345)
        start = np.clip(start * np.exp(rng.normal(0.0, a.seed_perturb, size=start.size)), lb, ub)

    fixed = parse_fixed(a.fix)
    pts = default_pts(ns) if mode == "dadi" else None

    fitted, fitted_sfs, best_ll, status = optimize_poisson_with_fixed(
        names=names, lb_full=lb, ub_full=ub, start_full=start,
        observed_sfs=data_sfs, demo_func=demo_func, sampled_demes=sampled_demes,
        ns_haploid=ns, mu_times_L=muL, mode=mode, pts=pts,
        use_bobyqa=a.use_bobyqa, rtol=a.rtol, maxeval=a.maxeval,
        verbose=a.verbose, fixed_by_name=fixed,
    )

    out_pkl = outdir / "best_fit.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(
            {
                "mode": mode,
                "best_params": {k: float(v) for k, v in zip(names, fitted)},
                "best_ll": float(best_ll),
                "status": int(status),
                "param_order": names,
                "fixed_params": fixed,
                "sampled_demes": sampled_demes,
            },
            f,
        )
    print(f"[{mode}] finished  LL={best_ll:.6g}, status={status}  → {out_pkl}")

    save_profiles_grid(
        outdir=outdir, param_names=names, fitted_params_full=fitted,
        observed_sfs=data_sfs, mu_times_L=muL, mode=mode,
        sampled_demes=sampled_demes, ns_haploid=ns, demo_func=demo_func,
        fixed_by_name=fixed,
    )

# ─────────────────────────────── main ─────────────────────────────────────

def main():
    a = _parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())

    mod_name, func_name = a.model_py.split(":")
    demo_func = getattr(importlib.import_module(mod_name), func_name)

    prefer = "dadi" if a.mode in {"dadi", "both"} else "moments"
    data_sfs = load_sfs(a.sfs_file, prefer=prefer)

    if a.mode == "both":
        for m in ["dadi", "moments"]:
            run_one_mode(m, a, cfg, demo_func, data_sfs)
    else:
        run_one_mode(a.mode, a, cfg, demo_func, data_sfs)

if __name__ == "__main__":
    main()

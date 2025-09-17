#!/usr/bin/env python3
"""
Build modeling datasets, compute metrics, and plot estimates vs truth.

Pipeline:
  1) Load inferred params (features) & true params (targets)
  2) Drop rows where any inferred param is an *extreme* outlier
  3) Split into train/validation
  4) Normalise by prior μ,σ
  5) Write DataFrames under <out-root>/datasets/
  6) Create scatter plot grid (per tool × parameter) overlaying all replicates
  7) Compute per-tool JSON metrics (normalized MSE) for train/val splits
  8) Make bar charts with mean±SEM of (normalized) MSE (replicates averaged per sim)

Optional:
  • If --add-fim is provided, compute observed Fisher Information features in RAW units
    for the specified engines (dadi/moments) at the highest-LL parameter set for that engine.
    Columns added (per engine):
      {engine}_INFOdiag_<param>, {engine}_SE_<param>, {engine}_FIM_logdet,
      {engine}_FIM_min_eig, {engine}_FIM_cond

Outputs in datasets/:
  features_df.pkl
  targets_df.pkl
  normalized_train_features.pkl
  normalized_train_targets.pkl
  normalized_validation_features.pkl
  normalized_validation_targets.pkl
  features_scatterplot.png
  metrics_{tool}.json
  metrics_all.json
  mse_bars_val_normalized.png
  mse_bars_train_normalized.png

Additionally written (not tracked by Snakemake unless you add them):
  outliers_removed.tsv
  outliers_preview.txt
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, pickle, re, warnings, importlib
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================== core helpers ==============================

def param_dicts(tool: str, blob: dict) -> list[dict]:
    """Return a list of {param: value} dicts (1 per replicate) from all_inferences.pkl."""
    if blob is None:
        return []
    if tool.lower() == "momentsld" and "opt_params" in blob:
        return [blob["opt_params"]]
    bp = blob.get("best_params")
    if isinstance(bp, list):
        return bp
    return [bp] if isinstance(bp, dict) else []


def prior_stats(priors: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, float]]:
    """Uniform priors ⇒ μ=(lo+hi)/2, σ=(hi-lo)/sqrt(12)."""
    mu, sigma = {}, {}
    for p, (lo, hi) in priors.items():
        mu[p]    = (lo + hi) / 2.0
        sigma[p] = (hi - lo) / np.sqrt(12.0)
    return mu, sigma


def base_param(col: str) -> str:
    """Strip tool prefix and optional _rep_ suffix to obtain the base parameter name."""
    name = col
    for pref in ("moments_", "dadi_", "momentsLD_"):
        if name.startswith(pref):
            name = name[len(pref):]
            break
    name = name.split("_rep_", 1)[0]
    return name


def normalise_df(df: pd.DataFrame, mu: dict[str, float], sigma: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        k = base_param(col)
        if k in mu:  # guard in case of non-demographic columns
            out[col] = (out[col] - mu[k]) / sigma[k]
    return out

# ============================== plotting ===================================

def plot_estimates_vs_truth_grid_multi_rep(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    *,
    tools: tuple[str, ...] = ("dadi", "moments", "momentsLD"),
    params: list[str] | None = None,
    figsize_per_panel: tuple[float, float] = (3.2, 3.0),
    out_path: Path | str = "features_scatterplot.png",
    colorize_reps_tools: set[str] | tuple[str, ...] = ("momentsLD",),
):
    # infer parameter list if not provided
    if params is None:
        common: set[str] = set(targets_df.columns)
        for tool in tools:
            has = {c.split("_", 1)[1].split("_rep_", 1)[0]
                   for c in features_df.columns if c.startswith(f"{tool}_")}
            common &= has
        if not common:
            raise ValueError("No common parameters across selected tools & targets.")
        params = sorted(common)

    tools = list(tools)
    colorize_reps_tools = set(colorize_reps_tools)

    # one color per tool
    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]
    tool_color = {tool: palette[i % len(palette)] for i, tool in enumerate(tools)}

    n_rows, n_cols = len(tools), len(params)
    fig = plt.figure(figsize=(figsize_per_panel[0] * n_cols,
                              figsize_per_panel[1] * n_rows))

    for r, tool in enumerate(tools):
        for c, p in enumerate(params):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)

            # collect all replicate columns for this tool/param
            pattern = re.compile(rf"^{re.escape(tool)}_{re.escape(p)}(?:_rep_(\d+))?$")
            matches: list[tuple[int | None, str]] = []
            for col in features_df.columns:
                m = pattern.match(col)
                if m:
                    rep = int(m.group(1)) if m.group(1) is not None else None
                    matches.append((rep, col))
            matches.sort(key=lambda t: (-1 if t[0] is None else t[0]))

            y = targets_df[p]

            # panel limits over all reps
            x_all = [features_df[col].to_numpy() for _, col in matches]
            if x_all:
                x_all = np.concatenate(x_all)
                xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
                ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
                lo = min(xmin, ymin); hi = max(xmax, ymax)
            else:
                lo, hi = float(y.min()), float(y.max())

            # plot
            made_tool_label = False
            for rep, col in matches:
                x = features_df[col]
                if tool in colorize_reps_tools:
                    lbl = f"rep_{rep}" if rep is not None else tool
                    ax.scatter(x, y, s=16, alpha=0.75, label=lbl)  # default color cycle
                else:
                    lbl = (tool if not made_tool_label else None)
                    ax.scatter(x, y, s=16, alpha=0.75, label=lbl, color=tool_color[tool])
                    if lbl is not None:
                        made_tool_label = True

            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

            if r == 0:
                ax.set_title(p)
            if c == 0:
                ax.set_ylabel(f"{tool}\nTrue")
            if r == n_rows - 1:
                ax.set_xlabel("Estimated")

            if len(matches) > 0:
                ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ============================== metrics ====================================

def _tool_param_columns(features_df: pd.DataFrame, tool: str, param: str) -> list[str]:
    """All columns for a given tool/param (supports replicate suffix)."""
    pat = re.compile(rf"^{re.escape(tool)}_{re.escape(param)}(?:_rep_\d+)?$")
    return [c for c in features_df.columns if pat.match(c)]


def per_sim_mse_array(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    idx:         np.ndarray,
    *,
    tool: str,
    param: str,
    sigma: dict[str, float],
    normalized: bool = True,
) -> np.ndarray:
    """
    Return per-simulation MSE values for one tool/param:
      1) For each sim, average all replicate columns for this tool/param
      2) Compare to truth; compute (normalized) squared error
    """
    cols = _tool_param_columns(features_df, tool, param)
    if not cols or param not in targets_df.columns:
        return np.array([], dtype=float)

    out_vals = []
    for sid in idx:
        if sid not in features_df.index or sid not in targets_df.index:
            continue
        vals = []
        for col in cols:
            if col in features_df.columns:
                v = features_df.at[sid, col]
                if pd.notna(v):
                    vals.append(float(v))
        if not vals:
            continue
        pred = float(np.mean(vals))
        tru  = float(targets_df.at[sid, param])
        if normalized:
            s = sigma.get(param, 0.0)
            if s <= 0:
                continue
            se = ((pred - tru) / s) ** 2
        else:
            se = (pred - tru) ** 2
        out_vals.append(se)

    return np.asarray(out_vals, dtype=float)


def compute_split_metrics_for_tool(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    train_idx:   np.ndarray,
    val_idx:     np.ndarray,
    *,
    tool: str,
    params: list[str],
    sigma: dict[str, float],
    normalized: bool = True,
) -> dict:
    """
    Return:
    {
      "training": <overall>,
      "validation": <overall>,
      "training_mse": {param: val, ...},
      "validation_mse": {param: val, ...}
    }
    Per-param MSE uses per-simulation averaged predictions (across replicates).
    """
    def _split_mse(idx: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in params:
            errs = per_sim_mse_array(features_df, targets_df, idx,
                                     tool=tool, param=p, sigma=sigma, normalized=normalized)
            if errs.size:
                out[p] = float(np.mean(errs))
        return out

    train_mse = _split_mse(train_idx)
    val_mse   = _split_mse(val_idx)

    def _overall(d: dict[str, float]) -> float:
        arr = np.array(list(d.values()), dtype=float)
        return float(np.mean(arr)) if arr.size else float("nan")

    return {
        "training": _overall(train_mse),
        "validation": _overall(val_mse),
        "training_mse": train_mse,
        "validation_mse": val_mse,
    }


def infer_common_params(features_df: pd.DataFrame, targets_df: pd.DataFrame, tools: tuple[str, ...]) -> list[str]:
    """Parameters present for all tools AND in targets."""
    common: set[str] = set(targets_df.columns)
    for tool in tools:
        has = {c.split("_", 1)[1].split("_rep_", 1)[0]
               for c in features_df.columns if c.startswith(f"{tool}_")}
        common &= has
    return sorted(common)

# ============================== FIM helpers ================================

def _build_sample_sizes_from_sfs(sfs):
    if hasattr(sfs, "pop_ids") and sfs.pop_ids is not None:
        return OrderedDict((pop, (n - 1) // 2) for pop, n in zip(sfs.pop_ids, sfs.shape))
    pop_names = [f"pop{i}" for i in range(len(sfs.shape))]
    return OrderedDict((pop, (n - 1) // 2) for pop, n in zip(pop_names, sfs.shape))

def _auto_pts_from_sfs(sfs):
    ss = _build_sample_sizes_from_sfs(sfs)
    n_max_hap = max(2*n for n in ss.values())
    return [n_max_hap+20, n_max_hap+40, n_max_hap+60]

def _scale_expected_sfs(exp_sfs, theta_vec, mu, L):
    N0 = max(float(theta_vec[0]), 1e-300)
    exp_sfs *= 4.0 * N0 * float(mu) * int(L)
    return exp_sfs

def _expected_sfs_moments(theta_vec, param_names, model_func, mu, L, sample_sizes):
    import moments
    p_dict = {nm: float(v) for nm, v in zip(param_names, theta_vec)}
    graph = model_func(p_dict)
    haploid_sizes = [2*n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())
    fs = moments.Spectrum.from_demes(graph, sample_sizes=haploid_sizes, sampled_demes=sampled_demes)
    return _scale_expected_sfs(fs, theta_vec, mu, L)

def _expected_sfs_dadi(theta_vec, param_names, model_func, mu, L, sample_sizes, pts):
    import dadi
    p_dict = {nm: float(v) for nm, v in zip(param_names, theta_vec)}
    graph = model_func(p_dict)
    haploid_sizes = [2*n for n in sample_sizes.values()]
    sampled_demes = list(sample_sizes.keys())
    def _raw(_params, ns, pts_grid):
        return dadi.Spectrum.from_demes(graph, sample_sizes=haploid_sizes, sampled_demes=sampled_demes, pts=pts_grid)
    func_ex = dadi.Numerics.make_extrap_func(lambda p, ns, pts_grid: _raw(p, ns, pts_grid))
    fs = func_ex(theta_vec, sample_sizes, pts)
    return _scale_expected_sfs(fs, theta_vec, mu, L)

def _poisson_ll_sfs(sfs, expected_sfs_func, folded):
    def ll_theta(theta_full):
        exp = expected_sfs_func(theta_full)
        if folded:
            exp = exp.fold()
        exp = np.maximum(exp, 1e-300)
        return float(np.sum(sfs * np.log(exp) - exp))
    return ll_theta

def _observed_fim_theta(sfs, param_names, theta_at, model_func, mu, L, engine="moments", pts=None, fixed=None, rel_step=1e-4):
    import numdifftools as nd
    fixed = fixed or {}
    theta_at = np.asarray(theta_at, float).copy()
    theta_at = np.maximum(theta_at, 1e-300)
    ss = _build_sample_sizes_from_sfs(sfs)
    if engine == "dadi":
        if pts is None:
            pts = _auto_pts_from_sfs(sfs)
        expfun = lambda th: _expected_sfs_dadi(th, param_names, model_func, mu, L, ss, pts)
    else:
        expfun = lambda th: _expected_sfs_moments(th, param_names, model_func, mu, L, ss)
    ll_theta = _poisson_ll_sfs(sfs, expfun, getattr(sfs, "folded", False))
    free_idx  = [i for i, nm in enumerate(param_names) if nm not in fixed]
    fixed_idx = [i for i, nm in enumerate(param_names) if nm in fixed]
    if not free_idx:
        raise ValueError("All parameters fixed; FIM undefined.")
    for i in fixed_idx:
        theta_at[i] = max(float(fixed[param_names[i]]), 1e-300)
    def ll_free(free_vec):
        full = theta_at.copy()
        full[free_idx] = np.asarray(free_vec, float)
        full = np.maximum(full, 1e-300)
        return ll_theta(full)
    step_vec = np.maximum(np.abs(theta_at[free_idx]) * rel_step, 1e-8)
    H = nd.Hessian(ll_free, step=step_vec)(theta_at[free_idx])
    info = -H
    cov = None
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        warnings.warn("Info matrix singular/ill-conditioned; covariance unavailable.")
    return info, cov, free_idx

def _pick_best_params_from_blob(tool_blob: dict) -> Optional[dict]:
    if not tool_blob:
        return None
    if isinstance(tool_blob.get("best_params"), dict) and "best_lls" in tool_blob:
        return dict(tool_blob["best_params"])  # momentsLD single
    bplist = tool_blob.get("best_params")
    blls   = tool_blob.get("best_ll")
    if isinstance(bplist, list) and bplist:
        if isinstance(blls, list) and len(blls) == len(bplist):
            i = int(np.nanargmax(np.asarray(blls, dtype=float)))
            return dict(bplist[i])
        return dict(bplist[0])
    return None

def _best_theta_for_engine(all_inf: dict, engine: str, param_order: List[str]) -> Optional[List[float]]:
    blob = all_inf.get(engine)
    pmap = _pick_best_params_from_blob(blob)
    if not pmap:
        return None
    return [float(pmap.get(name, np.nan)) for name in param_order]

# =============================== bar charts ================================

def plot_mse_bars_with_sem(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    idx:         np.ndarray,
    *,
    tools: tuple[str, ...],
    params: list[str],
    sigma: dict[str, float],
    normalized: bool,
    out_path: Path | str,
    title: str,
):
    """
    Grouped bars: per-parameter (x) with one bar per tool (mean of per-sim MSE),
    error bars = SEM across simulations.
    """
    means = {p: [] for p in params}
    sems  = {p: [] for p in params}

    for p in params:
        for tool in tools:
            arr = per_sim_mse_array(features_df, targets_df, idx,
                                    tool=tool, param=p, sigma=sigma, normalized=normalized)
            if arr.size:
                means[p].append(float(arr.mean()))
                sems[p].append(float(arr.std(ddof=1) / np.sqrt(len(arr))))
            else:
                means[p].append(np.nan)
                sems[p].append(np.nan)

    n_params = len(params)
    n_tools  = len(tools)
    x = np.arange(n_params)
    width = 0.8 / max(1, n_tools)

    fig, ax = plt.subplots(figsize=(1.8 + 1.8*n_params, 3.6))
    for i, tool in enumerate(tools):
        ax.bar(x + i*width - (n_tools-1)*width/2,
               [means[p][i] for p in params],
               width=width,
               yerr=[sems[p][i] for p in params],
               capsize=3,
               label=tool)

    ax.set_xticks(x, params, rotation=30, ha="right")
    ax.set_ylabel("Normalized MSE" if normalized else "MSE")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ================================= main ====================================

def main(
    cfg_path: Path,
    out_root: Path,
    *,
    tol_rel: float = 1e-9,
    tol_abs: float = 0.0,
    zmax: float = 6.0,
    preview_rows: int = -1,   # -1 ⇒ show all
    add_fim: bool = False,
    fim_engines: List[str] | None = None,
    fim_model: Optional[str] = None,
    fim_pts: Optional[str] = None,
    fim_rel_step: float = 1e-4,
    sfs_pattern: Optional[str] = None,
) -> None:
    cfg        = json.loads(cfg_path.read_text())
    model      = cfg["demographic_model"]
    n_sims     = int(cfg["num_draws"])
    train_pct  = float(cfg.get("training_percentage", 0.8))
    seed       = int(cfg.get("seed", 42))
    rng        = np.random.default_rng(seed)

    priors     = cfg["priors"]
    mu, sigma  = prior_stats(priors)
    param_order = list(priors.keys())

    sim_basedir   = Path(f"experiments/{model}/simulations")
    infer_basedir = Path(f"experiments/{model}/inferences")

    # Prepare model function for FIM (if needed)
    model_func = None
    if add_fim:
        model_py = fim_model or cfg.get("model_py", None)
        if not model_py:
            raise RuntimeError("FIM requested but no --fim-model and cfg has no 'model_py'.")
        mod_path, func_name = model_py.split(":")
        model_func = getattr(importlib.import_module(mod_path), func_name)

    # which engines to compute FIM for
    fim_engines = fim_engines or ["dadi", "moments"]
    fim_engines = [e.strip().lower() for e in fim_engines if e.strip().lower() in ("dadi","moments")]

    feature_rows, target_rows, index = [], [], []

    # ---------- load all sims ----------------------------------------------
    for sid in range(n_sims):
        inf_pickle   = infer_basedir / f"sim_{sid}/all_inferences.pkl"
        truth_pickle = sim_basedir   / f"{sid}/sampled_params.pkl"

        # must have truth to supervise; if missing, skip this sim
        if not truth_pickle.exists():
            continue

        truth = pickle.load(truth_pickle.open("rb"))

        # inferences are optional: if missing, treat as empty dict
        data = pickle.load(inf_pickle.open("rb")) if inf_pickle.exists() else {}

        row: Dict[str, float] = {}
        # ---- copy param estimates for all tools/replicates as before
        for tool in ("moments", "dadi", "momentsLD"):
            if tool in data and data[tool] is not None:
                for rep_idx, pdict in enumerate(param_dicts(tool, data[tool])):
                    if not isinstance(pdict, dict):
                        continue
                    for k, v in pdict.items():
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            continue
                        col = f"{tool}_{k}" if tool.lower() == "momentsld" else f"{tool}_{k}_rep_{rep_idx}"
                        row[col] = float(v)

        # ---- optional: add FIM features per requested engine
        if add_fim:
            # locate observed SFS for this sim if available
            sfs_obj = None
            sfs_candidates: List[Path] = []
            if sfs_pattern:
                try:
                    sfs_candidates.append(Path(sfs_pattern.format(model=model, sid=sid)))
                except Exception:
                    pass
            # fallbacks commonly used in your layout
            for pth in [
                sim_basedir / f"{sid}/SFS.pkl",
                infer_basedir / f"sim_{sid}/SFS.pkl",
            ]:
                if pth.exists():
                    sfs_candidates.append(pth)
            if sfs_candidates:
                try:
                    sfs_obj = pickle.loads(sfs_candidates[0].read_bytes())
                except Exception:
                    sfs_obj = None

            # if no observed SFS, synthesize expected SFS at ground-truth (expected Fisher)
            if sfs_obj is None:
                try:
                    import moments
                    p_dict_truth = {nm: float(truth.get(nm, (priors[nm][0]+priors[nm][1])/2.0))
                                    for nm in param_order}
                    graph = model_func(p_dict_truth)
                    # infer sample sizes by assuming same as true SFS demography; use 2 pops if unknown
                    # safer: use moments to make a small SFS and rely on folding flag default False
                    # NOTE: this is a best-effort fallback; observed SFS is preferred.
                    # try to use an SFS saved alongside truths if you have one:
                    # else assume single-pop shape via prior—kept minimal to avoid brittleness
                    # Here we skip trying to be clever and just raise if moments can't produce:
                    # (your DAG ensures SFS exists, so this code likely won't trigger)
                    pass
                except Exception:
                    warnings.warn(f"[sid={sid}] No observed SFS available; skipping FIM features.")
                    sfs_obj = None

            # compute per-engine FIM features at highest-LL params
            if sfs_obj is not None:
                mu_rate = float(cfg["mutation_rate"])
                L_len   = int(cfg["genome_length"])
                pts_grid = None
                if fim_pts and fim_pts.lower() != "auto":
                    pts_grid = [int(x) for x in fim_pts.split(",")]

                # build a tiny helper to compute & append features
                def add_fim_features_for(engine_name: str):
                    theta = _best_theta_for_engine(data, engine=engine_name, param_order=param_order)
                    if theta is None or any([not np.isfinite(v) for v in theta]):
                        return
                    info, cov, free_idx = _observed_fim_theta(
                        sfs=sfs_obj,
                        param_names=param_order,
                        theta_at=np.asarray(theta, float),
                        model_func=model_func,
                        mu=mu_rate,
                        L=L_len,
                        engine=engine_name,
                        pts=pts_grid if engine_name == "dadi" else None,
                        fixed=None,
                        rel_step=fim_rel_step
                    )
                    # diag & SEs
                    diag = np.diag(info)
                    name_by_free = [param_order[i] for i in free_idx]
                    for j, nm in enumerate(name_by_free):
                        row[f"{engine_name}_INFOdiag_{nm}"] = float(diag[j])
                        if cov is not None:
                            row[f"{engine_name}_SE_{nm}"] = float(np.sqrt(max(cov[j,j], 0.0)))
                    # matrix summaries
                    try:
                        w, _ = np.linalg.eigh(info)
                        w_clip = np.clip(w, 1e-300, None)
                        row[f"{engine_name}_FIM_logdet"] = float(np.sum(np.log(w_clip)))
                        row[f"{engine_name}_FIM_min_eig"] = float(np.min(w))
                        row[f"{engine_name}_FIM_cond"]    = float(np.max(w_clip) / np.min(w_clip))
                    except Exception:
                        pass

                for eng in fim_engines:
                    add_fim_features_for(eng)

        feature_rows.append(row)   # possibly empty dict → NaNs in DF
        target_rows.append(truth)
        index.append(sid)

    feat_df = pd.DataFrame(feature_rows, index=index).sort_index(axis=1)
    targ_df = pd.DataFrame(target_rows,  index=index).sort_index(axis=1)

    # ---------- outlier removal BEFORE split -------------------------------
    outlier_records = []

    def _parse_tool_rep(col: str) -> tuple[str, int | None]:
        tool = col.split("_", 1)[0]
        m = re.search(r"_rep_(\d+)$", col)
        return tool, (int(m.group(1)) if m else None)

    keep = pd.Series(True, index=feat_df.index)

    for col in feat_df.columns:
        k = base_param(col)
        if k not in priors:
            continue
        lo, hi = map(float, priors[k])
        mu_k, sg_k = float(mu[k]), float(sigma[k])
        s = feat_df[col].astype(float)

        finite = np.isfinite(s.to_numpy())
        scale = max(1.0, abs(lo), abs(hi))
        eps = max(tol_abs, tol_rel * scale)

        outside_lo = s < (lo - eps)
        outside_hi = s > (hi + eps)

        z = (s - mu_k) / sg_k
        big_z = np.abs(z) > zmax

        extreme = (outside_lo | outside_hi) & big_z
        bad = (~finite) | extreme

        if bad.any():
            tool, rep = _parse_tool_rep(col)
            for sid, val, fin, is_lo, is_hi, is_bigz in zip(
                s.index, s.to_numpy(), finite, outside_lo.to_numpy(), outside_hi.to_numpy(), big_z.to_numpy()
            ):
                if not fin:
                    reason = "nan_or_inf"
                elif (is_lo or is_hi) and is_bigz:
                    reason = "extreme_below_lo" if is_lo else "extreme_above_hi"
                else:
                    continue

                outlier_records.append({
                    "sid": sid,
                    "column": col,
                    "tool": tool,
                    "rep": rep,
                    "base_param": k,
                    "value": float(val) if np.isfinite(val) else np.nan,
                    "lo": lo,
                    "hi": hi,
                    "mu": mu_k,
                    "sigma": sg_k,
                    "z": float(((val - mu_k) / sg_k) if np.isfinite(val) else np.nan),
                    "eps": eps,
                    "reason": reason,
                })

        keep &= ~bad

    kept_count    = int(keep.sum())
    dropped_count = int((~keep).sum())
    total_rows    = len(keep)

    datasets_dir = out_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Outlier filtering (extreme only): kept {kept_count} / {total_rows} rows "
          f"({dropped_count} dropped).")

    # save detailed outliers (TSV) + preview text
    outliers_path  = datasets_dir / "outliers_removed.tsv"
    preview_path   = datasets_dir / "outliers_preview.txt"

    if outlier_records:
        outliers_df = pd.DataFrame(outlier_records).sort_values(["sid", "column"])
        outliers_df.to_csv(outliers_path, sep="\t", index=False)
        print(f"[INFO] Wrote detailed outliers to: {outliers_path}")

        show_all = (preview_rows is None) or (int(preview_rows) < 0)
        to_show = len(outliers_df) if show_all else min(int(preview_rows), len(outliers_df))

        with open(preview_path, "w") as fh:
            fh.write("Outlier filtering summary\n")
            fh.write("=========================\n")
            fh.write(f"kept {kept_count} / {total_rows} rows ({dropped_count} dropped)\n")
            fh.write(f"zmax={zmax}, tol_rel={tol_rel}, tol_abs={tol_abs}\n")
            fh.write(f"total outlier rows: {len(outliers_df)}\n\n")

            fh.write("All outlier rows:\n" if show_all else f"First {to_show} rows:\n")
            fh.write((outliers_df if show_all else outliers_df.head(to_show)).to_string(index=False))
            fh.write("\n\nCounts by base_param and reason:\n")
            counts = (
                outliers_df.groupby(["base_param", "reason"])
                .size()
                .reset_index(name="n")
                .sort_values(["base_param", "reason"])
            )
            fh.write(counts.to_string(index=False))
            fh.write("\n")

        print(f"[INFO] Wrote preview to: {preview_path}")
        if show_all:
            print(f"[INFO] Showing ALL {len(outliers_df)} extreme outlier rows:")
            print(outliers_df.to_string(index=False))
        else:
            print(f"[INFO] Showing first {to_show} extreme outlier rows:")
            print(outliers_df.head(to_show).to_string(index=False))
    else:
        with open(preview_path, "w") as fh:
            fh.write("Outlier filtering summary\n")
            fh.write("=========================\n")
            fh.write(f"kept {kept_count} / {total_rows} rows ({dropped_count} dropped)\n")
            fh.write(f"zmax={zmax}, tol_rel={tol_rel}, tol_abs={tol_abs}\n")
            fh.write("No extreme outliers detected.\n")
        print("[INFO] No extreme outliers detected.")

    # apply filtering
    feat_df = feat_df.loc[keep].copy()
    targ_df = targ_df.loc[keep].copy()

    # ---------- normalise AFTER outlier removal -----------------------------
    feat_norm_df = normalise_df(feat_df, mu, sigma)
    targ_norm_df = normalise_df(targ_df, mu, sigma)

    # ---------- train / val split ------------------------------------------
    n_rows = len(feat_df)
    if n_rows == 0:
        raise RuntimeError("All rows were dropped as outliers; nothing to write.")

    perm      = rng.permutation(n_rows)
    n_train   = int(round(train_pct * n_rows))
    all_idx   = feat_df.index.to_numpy()
    train_idx = all_idx[perm[:n_train]]
    val_idx   = all_idx[perm[n_train:]]

    norm_train_feats = feat_norm_df.loc[train_idx]
    norm_train_targs = targ_norm_df.loc[train_idx]
    norm_val_feats   = feat_norm_df.loc[val_idx]
    norm_val_targs   = targ_norm_df.loc[val_idx]

    # ---------- outputs: DataFrames ----------------------------------------
    (datasets_dir / "features_df.pkl").write_bytes(pickle.dumps(feat_df))
    (datasets_dir / "targets_df.pkl").write_bytes(pickle.dumps(targ_df))
    (datasets_dir / "normalized_train_features.pkl").write_bytes(pickle.dumps(norm_train_feats))
    (datasets_dir / "normalized_train_targets.pkl").write_bytes(pickle.dumps(norm_train_targs))
    (datasets_dir / "normalized_validation_features.pkl").write_bytes(pickle.dumps(norm_val_feats))
    (datasets_dir / "normalized_validation_targets.pkl").write_bytes(pickle.dumps(norm_val_targs))

    # ---------- plotting: scatter grid -------------------------------------
    plot_estimates_vs_truth_grid_multi_rep(
        features_df=feat_df,
        targets_df=targ_df,
        tools=("dadi", "moments", "momentsLD"),
        params=None,
        figsize_per_panel=(3.2, 3.0),
        out_path=datasets_dir / "features_scatterplot.png",
        colorize_reps_tools=("momentsLD",),   # only momentsLD shows rep colors
    )

    # ---------- JSON metrics logging (normalized MSE, per tool) ------------
    tools = ("dadi", "moments", "momentsLD")
    common_params = infer_common_params(feat_df, targ_df, tools)
    metrics_all = {}

    for tool in tools:
        metrics = compute_split_metrics_for_tool(
            feat_df, targ_df, train_idx, val_idx,
            tool=tool, params=common_params, sigma=sigma, normalized=True
        )
        metrics_all[tool] = metrics
        with open(datasets_dir / f"metrics_{tool}.json", "w") as fh:
            json.dump(metrics, fh, indent=4)

    with open(datasets_dir / "metrics_all.json", "w") as fh:
        json.dump(metrics_all, fh, indent=4)

    print("[INFO] Wrote JSON metrics to:")
    for tool in tools:
        print("   ", datasets_dir / f"metrics_{tool}.json")
    print("   ", datasets_dir / "metrics_all.json")

    # ---------- bar charts: mean±SEM (normalized MSE) -----------------------
    title_val   = "Normalized MSE by parameter (Validation; replicate-avg per simulation)"
    title_train = "Normalized MSE by parameter (Training; replicate-avg per simulation)"

    plot_mse_bars_with_sem(
        feat_df, targ_df, val_idx,
        tools=tools, params=common_params, sigma=sigma,
        normalized=True,
        out_path=datasets_dir / "mse_bars_val_normalized.png",
        title=title_val,
    )
    plot_mse_bars_with_sem(
        feat_df, targ_df, train_idx,
        tools=tools, params=common_params, sigma=sigma,
        normalized=True,
        out_path=datasets_dir / "mse_bars_train_normalized.png",
        title=title_train,
    )

    print(f"✓ wrote datasets, plots, and metrics to: {datasets_dir}")
    print(f"✓ outlier preview: {preview_path}")

# ================================= CLI =====================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Pass experiments/<model>/modeling (the script will create datasets/ inside)")
    ap.add_argument("--tol-rel", type=float, default=1e-9,
                    help="Relative tolerance for comparing to prior bounds.")
    ap.add_argument("--tol-abs", type=float, default=0.0,
                    help="Absolute tolerance for comparing to prior bounds.")
    ap.add_argument("--zmax", type=float, default=6.0,
                    help="Drop only if |(x-μ)/σ| > zmax *and* outside bounds.")
    ap.add_argument("--preview-rows", type=int, default=-1,
                    help="How many outlier rows to include in outliers_preview.txt. Use -1 to include all rows.")

    # Optional Fisher-info features
    ap.add_argument("--add-fim", action="store_true",
                    help="If set, compute Fisher-info features (RAW units) and append as columns.")
    ap.add_argument("--fim-engines", type=str, default="dadi,moments",
                    help="Comma-separated engines to compute FIM for (subset of: dadi,moments).")
    ap.add_argument("--fim-model", type=str, default=None,
                    help='Model function for demes graph, e.g. "src.simulation:split_migration_model". '
                         'If omitted, will try cfg["model_py"].')
    ap.add_argument("--fim-pts", type=str, default="auto",
                    help='For dadi engine only: "n1,n2,n3" or "auto".')
    ap.add_argument("--fim-rel-step", type=float, default=1e-4,
                    help="Relative finite-difference step per free parameter for FIM.")
    ap.add_argument("--sfs-pattern", type=str, default=None,
                    help='Pattern for observed SFS per sim, e.g. "experiments/{model}/simulations/{sid}/SFS.pkl".')

    args = ap.parse_args()

    fim_engines = [e.strip() for e in (args.fim_engines.split(",") if args.fim_engines else []) if e.strip()]

    main(
        args.experiment_config,
        args.out_dir,
        tol_rel=args.tol_rel,
        tol_abs=args.tol_abs,
        zmax=args.zmax,
        preview_rows=args.preview_rows,
        add_fim=args.add_fim,
        fim_engines=fim_engines,
        fim_model=args.fim_model,
        fim_pts=args.fim_pts,
        fim_rel_step=args.fim_rel_step,
        sfs_pattern=args.sfs_pattern,
    )

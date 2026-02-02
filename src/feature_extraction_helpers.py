# src/feature_extraction_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Core utilities
# =============================================================================

TOOLS_DEFAULT: Tuple[str, ...] = ("dadi", "moments", "momentsLD")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def dump_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent))


def dump_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(obj))


def load_pickle(path: Path) -> Any:
    return pickle.loads(path.read_bytes())


def prior_stats(priors: Dict[str, List[float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Uniform priors ⇒ μ=(lo+hi)/2, σ=(hi-lo)/sqrt(12)."""
    mu, sigma = {}, {}
    for p, (lo, hi) in priors.items():
        lo = float(lo); hi = float(hi)
        mu[p] = (lo + hi) / 2.0
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


def normalise_df(df: pd.DataFrame, mu: Dict[str, float], sigma: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        k = base_param(col)
        if k in mu:
            out[col] = (out[col] - mu[k]) / sigma[k]
    return out


def param_dicts(tool: str, blob: dict) -> List[Dict[str, float]]:
    """Return a list of {param: value} dicts (1 per replicate) from all_inferences.pkl."""
    if not blob:
        return []
    if tool.lower() == "momentsld" and isinstance(blob.get("opt_params"), dict):
        return [blob["opt_params"]]
    bp = blob.get("best_params")
    if isinstance(bp, list):
        return bp
    return [bp] if isinstance(bp, dict) else []


def infer_common_params(features_df: pd.DataFrame, targets_df: pd.DataFrame, tools: Sequence[str]) -> List[str]:
    """Parameters present for all tools AND in targets."""
    common: set[str] = set(targets_df.columns)
    for tool in tools:
        has = {
            c.split("_", 1)[1].split("_rep_", 1)[0]
            for c in features_df.columns
            if c.startswith(f"{tool}_")
        }
        common &= has
    return sorted(common)


def _tool_param_columns(features_df: pd.DataFrame, tool: str, param: str) -> List[str]:
    """All columns for a given tool/param (supports replicate suffix)."""
    pat = re.compile(rf"^{re.escape(tool)}_{re.escape(param)}(?:_rep_\d+)?$")
    return [c for c in features_df.columns if pat.match(c)]


def per_sim_mse_array(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    idx: np.ndarray,
    *,
    tool: str,
    param: str,
    sigma: Dict[str, float],
    normalized: bool = True,
) -> np.ndarray:
    """
    Per-simulation MSE for one tool/param:
      1) average all replicate columns for this tool/param
      2) squared error vs truth (optionally normalized by prior σ)
    """
    cols = _tool_param_columns(features_df, tool, param)
    if not cols or param not in targets_df.columns:
        return np.array([], dtype=float)

    out_vals: List[float] = []
    for sid in idx:
        if sid not in features_df.index or sid not in targets_df.index:
            continue
        vals = []
        for col in cols:
            v = features_df.at[sid, col]
            if pd.notna(v):
                vals.append(float(v))
        if not vals:
            continue
        pred = float(np.mean(vals))
        tru = float(targets_df.at[sid, param])
        if normalized:
            s = float(sigma.get(param, 0.0))
            if s <= 0:
                continue
            se = ((pred - tru) / s) ** 2
        else:
            se = (pred - tru) ** 2
        out_vals.append(se)

    return np.asarray(out_vals, dtype=float)


# =============================================================================
# Residual-engine selector
# =============================================================================

def norm_resid_engines(val: Any) -> List[str]:
    """Normalize residual engine selector from config."""
    if isinstance(val, str):
        v = val.lower()
        return ["moments", "dadi"] if v in ("both", "all") else [v]
    if isinstance(val, (list, tuple, set)):
        keep = [str(e).lower() for e in val if str(e).lower() in {"moments", "dadi"}]
        return keep or ["moments"]
    return ["moments"]


# =============================================================================
# Plotting helpers (moved verbatim-ish)
# =============================================================================

def plot_mse_bars_with_sem(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    idx: np.ndarray,
    *,
    tools: Sequence[str],
    params: Sequence[str],
    sigma: Dict[str, float],
    normalized: bool,
    out_path: Path,
    title: str,
) -> None:
    means = {p: [] for p in params}
    sems = {p: [] for p in params}

    for p in params:
        for tool in tools:
            arr = per_sim_mse_array(
                features_df,
                targets_df,
                idx,
                tool=tool,
                param=p,
                sigma=sigma,
                normalized=normalized,
            )
            if arr.size:
                means[p].append(float(arr.mean()))
                sems[p].append(float(arr.std(ddof=1) / np.sqrt(len(arr))))
            else:
                means[p].append(np.nan)
                sems[p].append(np.nan)

    n_params = len(list(params))
    n_tools = len(list(tools))
    x = np.arange(n_params)
    width = 0.8 / max(1, n_tools)

    fig, ax = plt.subplots(figsize=(1.8 + 1.8 * n_params, 3.6))
    for i, tool in enumerate(tools):
        ax.bar(
            x + i * width - (n_tools - 1) * width / 2,
            [means[p][i] for p in params],
            width=width,
            yerr=[sems[p][i] for p in params],
            capsize=3,
            label=tool,
        )

    ax.set_xticks(x, list(params), rotation=30, ha="right")
    ax.set_ylabel("Normalized MSE" if normalized else "MSE")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_estimates_vs_truth_grid_multi_rep(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    *,
    tools: Sequence[str] = TOOLS_DEFAULT,
    params: Optional[Sequence[str]] = None,
    figsize_per_panel: Tuple[float, float] = (3.2, 3.0),
    out_path: Path = Path("features_scatterplot.png"),
    colorize_reps_tools: Sequence[str] = ("momentsLD",),
) -> None:
    """Scatter panels: true vs estimated per tool × parameter."""
    if params is None:
        common: set[str] = set(targets_df.columns)
        for tool in tools:
            has = {
                c.split("_", 1)[1].split("_rep_", 1)[0]
                for c in features_df.columns
                if c.startswith(f"{tool}_")
            }
            common &= has
        if not common:
            raise ValueError("No common parameters across selected tools & targets.")
        params = sorted(common)

    tools = list(tools)
    params = list(params)
    colorize_reps_tools = set(colorize_reps_tools)

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0","C1","C2","C3","C4","C5"]
    tool_color = {tool: palette[i % len(palette)] for i, tool in enumerate(tools)}

    n_rows, n_cols = len(tools), len(params)
    fig = plt.figure(figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows))

    for r, tool in enumerate(tools):
        for c, p in enumerate(params):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1)

            pattern = re.compile(rf"^{re.escape(tool)}_{re.escape(p)}(?:_rep_(\d+))?$")
            matches: List[Tuple[Optional[int], str]] = []
            for col in features_df.columns:
                m = pattern.match(col)
                if m:
                    rep = int(m.group(1)) if m.group(1) is not None else None
                    matches.append((rep, col))
            matches.sort(key=lambda t: (-1 if t[0] is None else t[0]))

            y = targets_df[p]

            x_all = [features_df[col].to_numpy() for _, col in matches]
            if x_all:
                x_all = np.concatenate(x_all)
                xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
                ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
                lo = min(xmin, ymin)
                hi = max(xmax, ymax)
            else:
                lo, hi = float(y.min()), float(y.max())

            made_tool_label = False
            for rep, col in matches:
                x = features_df[col]
                if tool in colorize_reps_tools:
                    lbl = f"rep_{rep}" if rep is not None else tool
                    ax.scatter(x, y, s=16, alpha=0.75, label=lbl)
                else:
                    lbl = tool if not made_tool_label else None
                    ax.scatter(x, y, s=16, alpha=0.75, label=lbl, color=tool_color[tool])
                    if lbl is not None:
                        made_tool_label = True

            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
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


# =============================================================================
# Metrics
# =============================================================================

def compute_split_metrics_for_tool(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    tool: str,
    params: Sequence[str],
    sigma: Dict[str, float],
    normalized: bool = True,
) -> Dict[str, Any]:
    def _split_mse(idx: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in params:
            errs = per_sim_mse_array(
                features_df,
                targets_df,
                idx,
                tool=tool,
                param=p,
                sigma=sigma,
                normalized=normalized,
            )
            if errs.size:
                out[p] = float(np.mean(errs))
        return out

    train_mse = _split_mse(train_idx)
    val_mse = _split_mse(val_idx)

    def _overall(d: Dict[str, float]) -> float:
        arr = np.array(list(d.values()), dtype=float)
        return float(np.mean(arr)) if arr.size else float("nan")

    return {
        "training": _overall(train_mse),
        "validation": _overall(val_mse),
        "training_mse": train_mse,
        "validation_mse": val_mse,
    }


# =============================================================================
# Outlier filtering
# =============================================================================

@dataclass(frozen=True)
class OutlierFilterConfig:
    tol_rel: float = 1e-9
    tol_abs: float = 0.0
    zmax: float = 6.0
    preview_rows: int = -1  # -1 => all


def filter_extreme_outliers(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    priors: Dict[str, List[float]],
    mu: Dict[str, float],
    sigma: Dict[str, float],
    cfg: OutlierFilterConfig,
    *,
    save_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Drop a row if ANY demographic column is:
      - NaN/inf OR
      - outside prior bounds (with tol) AND |z|>zmax

    Returns:
      (features_kept, targets_kept, keep_mask, outliers_df)
    """
    def _parse_tool_rep(col: str) -> Tuple[str, Optional[int]]:
        tool = col.split("_", 1)[0]
        m = re.search(r"_rep_(\d+)$", col)
        return tool, (int(m.group(1)) if m else None)

    keep = pd.Series(True, index=features_df.index)
    outlier_records: List[Dict[str, Any]] = []

    for col in features_df.columns:
        k = base_param(col)
        if k not in priors:
            continue

        lo, hi = map(float, priors[k])
        mu_k, sg_k = float(mu[k]), float(sigma[k])
        s = features_df[col].astype(float)

        finite = np.isfinite(s.to_numpy())
        scale = max(1.0, abs(lo), abs(hi))
        eps = max(cfg.tol_abs, cfg.tol_rel * scale)

        outside_lo = s < (lo - eps)
        outside_hi = s > (hi + eps)

        z = (s - mu_k) / sg_k
        big_z = np.abs(z) > cfg.zmax

        extreme = (outside_lo | outside_hi) & big_z
        bad = (~finite) | extreme

        if bad.any():
            tool, rep = _parse_tool_rep(col)
            for sid_i, val, fin, is_lo, is_hi, is_bigz in zip(
                s.index,
                s.to_numpy(),
                finite,
                outside_lo.to_numpy(),
                outside_hi.to_numpy(),
                big_z.to_numpy(),
            ):
                if not fin:
                    reason = "nan_or_inf"
                elif (is_lo or is_hi) and is_bigz:
                    reason = "extreme_below_lo" if is_lo else "extreme_above_hi"
                else:
                    continue

                outlier_records.append(
                    {
                        "sid": sid_i,
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
                    }
                )

        keep &= ~bad

    outliers_df = pd.DataFrame(outlier_records).sort_values(["sid", "column"]) if outlier_records else pd.DataFrame()

    # optional write
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        preview_path = save_dir / "outliers_preview.txt"
        outliers_path = save_dir / "outliers_removed.tsv"

        kept_count = int(keep.sum())
        dropped_count = int((~keep).sum())
        total_rows = len(keep)

        if not outliers_df.empty:
            outliers_df.to_csv(outliers_path, sep="\t", index=False)

            show_all = (cfg.preview_rows is None) or (int(cfg.preview_rows) < 0)
            to_show = len(outliers_df) if show_all else min(int(cfg.preview_rows), len(outliers_df))

            with open(preview_path, "w") as fh:
                fh.write("Outlier filtering summary\n")
                fh.write("=========================\n")
                fh.write(f"kept {kept_count} / {total_rows} rows ({dropped_count} dropped)\n")
                fh.write(f"zmax={cfg.zmax}, tol_rel={cfg.tol_rel}, tol_abs={cfg.tol_abs}\n")
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
        else:
            with open(preview_path, "w") as fh:
                fh.write("Outlier filtering summary\n")
                fh.write("=========================\n")
                fh.write(f"kept {kept_count} / {total_rows} rows ({dropped_count} dropped)\n")
                fh.write(f"zmax={cfg.zmax}, tol_rel={cfg.tol_rel}, tol_abs={cfg.tol_abs}\n")
                fh.write("No extreme outliers detected.\n")

    return (
        features_df.loc[keep].copy(),
        targets_df.loc[keep].copy(),
        keep,
        outliers_df,
    )


# =============================================================================
# Dataset building: load → assemble features/targets
# =============================================================================

@dataclass(frozen=True)
class DatasetBuildConfig:
    model: str
    n_sims: int
    use_fim_features: bool
    use_residuals: bool
    residual_engines: List[str]
    priors: Dict[str, List[float]]
    seed: int = 42
    train_pct: float = 0.8
    tune_pct: float = 0.1


def build_feature_target_tables(
    cfg: DatasetBuildConfig,
    *,
    sim_basedir: Path,
    infer_basedir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: List[Dict[str, float]] = []
    target_rows: List[Dict[str, float]] = []
    index: List[int] = []

    for sid in range(cfg.n_sims):
        inf_pickle = infer_basedir / f"sim_{sid}/all_inferences.pkl"
        truth_pickle = sim_basedir / f"{sid}/sampled_params.pkl"
        if not truth_pickle.exists():
            continue

        truth = pickle.load(truth_pickle.open("rb"))
        data = pickle.load(inf_pickle.open("rb")) if inf_pickle.exists() else {}

        row: Dict[str, float] = {}

        # ---- inferred params
        for tool in TOOLS_DEFAULT:
            if tool in data and data[tool] is not None:
                for rep_idx, pdict in enumerate(param_dicts(tool, data[tool])):
                    if not isinstance(pdict, dict):
                        continue
                    for k, v in pdict.items():
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            continue
                        col = (
                            f"{tool}_{k}"
                            if tool.lower() == "momentsld"
                            else f"{tool}_{k}_rep_{rep_idx}"
                        )
                        row[col] = float(v)

        # ---- FIM
        if cfg.use_fim_features and isinstance(data.get("FIM"), dict) and data["FIM"]:
            engines = list(data["FIM"].keys())
            eng_pick = "moments" if "moments" in engines else engines[0]
            tri = data["FIM"].get(eng_pick, {}).get("tri_flat", None)
            if tri is not None:
                for k, v in enumerate(tri):
                    row[f"FIM_element_{k}"] = float(v)

        # ---- SFS residuals
        if cfg.use_residuals and isinstance(data.get("SFS_residuals"), dict):
            res_block = data["SFS_residuals"]
            for eng in cfg.residual_engines:
                payload = res_block.get(eng)
                if not isinstance(payload, dict):
                    continue
                flat = payload.get("flat")
                if flat is None:
                    continue
                for k, v in enumerate(flat):
                    row[f"SFSres_{eng}_{k}"] = float(v)

        feature_rows.append(row)
        target_rows.append(truth)
        index.append(sid)

    feat_df = pd.DataFrame(feature_rows, index=index).sort_index(axis=1)
    targ_df = pd.DataFrame(target_rows, index=index).sort_index(axis=1)
    return feat_df, targ_df


def split_indices(
    index: np.ndarray,
    *,
    seed: int,
    train_pct: float,
    tune_pct: float,
) -> Dict[str, np.ndarray]:
    if train_pct + tune_pct >= 1.0:
        raise ValueError("train_pct + tune_pct must be < 1.0")

    rng = np.random.default_rng(int(seed))
    n_rows = len(index)
    perm = rng.permutation(n_rows)

    n_train = int(round(train_pct * n_rows))
    n_tune = int(round(tune_pct * n_rows))
    n_train = min(n_train, n_rows)
    n_tune = min(n_tune, max(0, n_rows - n_train))

    train_idx = index[perm[:n_train]]
    tune_idx = index[perm[n_train:n_train + n_tune]]
    val_idx = index[perm[n_train + n_tune:]]

    return {"train_idx": train_idx, "tune_idx": tune_idx, "val_idx": val_idx}


def write_dataset_pickles(
    *,
    datasets_dir: Path,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    features_norm_df: pd.DataFrame,
    targets_norm_df: pd.DataFrame,
    split: Dict[str, np.ndarray],
) -> None:
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dump_pickle(features_df, datasets_dir / "features_df.pkl")
    dump_pickle(targets_df, datasets_dir / "targets_df.pkl")

    split_indices_json = {
        "train_idx": split["train_idx"].tolist(),
        "tune_idx": split["tune_idx"].tolist(),
        "val_idx": split["val_idx"].tolist(),
    }
    dump_json(split_indices_json, datasets_dir / "split_indices.json", indent=2)

    # raw splits
    for name, idx in split.items():
        dump_pickle(features_df.loc[idx], datasets_dir / f"{name.replace('_idx','')}_features.pkl")
        dump_pickle(targets_df.loc[idx], datasets_dir / f"{name.replace('_idx','')}_targets.pkl")

    # normalized splits
    for name, idx in split.items():
        dump_pickle(features_norm_df.loc[idx], datasets_dir / f"normalized_{name.replace('_idx','')}_features.pkl")
        dump_pickle(targets_norm_df.loc[idx], datasets_dir / f"normalized_{name.replace('_idx','')}_targets.pkl")


def write_metrics_and_plots(
    *,
    datasets_dir: Path,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    split: Dict[str, np.ndarray],
    sigma: Dict[str, float],
    tools: Sequence[str] = TOOLS_DEFAULT,
) -> None:
    # scatter grid
    plot_estimates_vs_truth_grid_multi_rep(
        features_df=features_df,
        targets_df=targets_df,
        tools=tools,
        params=None,
        figsize_per_panel=(3.2, 3.0),
        out_path=datasets_dir / "features_scatterplot.png",
        colorize_reps_tools=("momentsLD",),
    )

    # metrics
    common_params = infer_common_params(features_df, targets_df, tools)
    metrics_all: Dict[str, Dict[str, Any]] = {}

    for tool in tools:
        metrics = compute_split_metrics_for_tool(
            features_df,
            targets_df,
            split["train_idx"],
            split["val_idx"],
            tool=tool,
            params=common_params,
            sigma=sigma,
            normalized=True,
        )
        metrics_all[tool] = metrics
        dump_json(metrics, datasets_dir / f"metrics_{tool}.json", indent=4)

    dump_json(metrics_all, datasets_dir / "metrics_all.json", indent=4)

    # bars
    title_val = "Normalized MSE by parameter (Validation; replicate-avg per simulation)"
    title_train = "Normalized MSE by parameter (Training; replicate-avg per simulation)"

    plot_mse_bars_with_sem(
        features_df,
        targets_df,
        split["val_idx"],
        tools=tools,
        params=common_params,
        sigma=sigma,
        normalized=True,
        out_path=datasets_dir / "mse_bars_val_normalized.png",
        title=title_val,
    )
    plot_mse_bars_with_sem(
        features_df,
        targets_df,
        split["train_idx"],
        tools=tools,
        params=common_params,
        sigma=sigma,
        normalized=True,
        out_path=datasets_dir / "mse_bars_train_normalized.png",
        title=title_train,
    )


# =============================================================================
# One-call pipeline entrypoint (what your snakemake wrapper should call)
# =============================================================================

def build_modeling_datasets(
    *,
    experiment_config_path: Path,
    out_root: Path,
    tol_rel: float = 1e-9,
    tol_abs: float = 0.0,
    zmax: float = 6.0,
    preview_rows: int = -1,
) -> Path:
    """
    End-to-end dataset build:
      - load cfg
      - load feature/target rows from disk
      - outlier filter (once)
      - normalize
      - split train/tune/val
      - write pickles + indices + metrics + plots
    Returns datasets_dir.
    """
    cfg_raw = load_json(experiment_config_path)

    model = str(cfg_raw["demographic_model"])
    n_sims = int(cfg_raw["num_draws"])
    seed = int(cfg_raw.get("seed", 42))

    train_pct = float(cfg_raw.get("training_percentage", 0.8))
    tune_pct = float(cfg_raw.get("tune_percentage", 0.1))
    if train_pct + tune_pct >= 1.0:
        raise ValueError("training_percentage + tune_percentage must be < 1.0")

    use_fim_features = bool(cfg_raw.get("use_fim_features", False))
    use_residuals = bool(cfg_raw.get("use_residuals", False))
    residual_engines = norm_resid_engines(cfg_raw.get("residual_engines", "moments"))

    priors = cfg_raw["priors"]
    mu, sigma = prior_stats(priors)

    sim_basedir = Path(f"experiments/{model}/simulations")
    infer_basedir = Path(f"experiments/{model}/inferences")

    cfg = DatasetBuildConfig(
        model=model,
        n_sims=n_sims,
        use_fim_features=use_fim_features,
        use_residuals=use_residuals,
        residual_engines=residual_engines,
        priors=priors,
        seed=seed,
        train_pct=train_pct,
        tune_pct=tune_pct,
    )

    # build tables
    feat_df, targ_df = build_feature_target_tables(
        cfg,
        sim_basedir=sim_basedir,
        infer_basedir=infer_basedir,
    )

    if len(feat_df) == 0:
        raise RuntimeError("No rows loaded (did you generate inferences + sampled_params.pkl?).")

    datasets_dir = out_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # outlier filter ONCE
    of_cfg = OutlierFilterConfig(tol_rel=tol_rel, tol_abs=tol_abs, zmax=zmax, preview_rows=preview_rows)
    feat_df, targ_df, keep_mask, outliers_df = filter_extreme_outliers(
        feat_df, targ_df, priors, mu, sigma, of_cfg, save_dir=datasets_dir
    )

    if len(feat_df) == 0:
        raise RuntimeError("All rows were dropped as outliers; nothing to write.")

    # normalize AFTER filter
    feat_norm_df = normalise_df(feat_df, mu, sigma)
    targ_norm_df = normalise_df(targ_df, mu, sigma)

    # split
    split = split_indices(
        feat_df.index.to_numpy(),
        seed=seed,
        train_pct=train_pct,
        tune_pct=tune_pct,
    )

    # write pickles
    write_dataset_pickles(
        datasets_dir=datasets_dir,
        features_df=feat_df,
        targets_df=targ_df,
        features_norm_df=feat_norm_df,
        targets_norm_df=targ_norm_df,
        split=split,
    )

    # metrics + plots
    write_metrics_and_plots(
        datasets_dir=datasets_dir,
        features_df=feat_df,
        targets_df=targ_df,
        split=split,
        sigma=sigma,
        tools=TOOLS_DEFAULT,
    )

    return datasets_dir

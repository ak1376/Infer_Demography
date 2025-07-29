#!/usr/bin/env python3
"""
Build modeling datasets and plot estimates vs. truth.

Pipeline:
  1) load inferred params (features) & true params (targets)
  2) drop rows where any inferred param is an *extreme* outlier
  3) split into train/validation
  4) normalise by prior μ,σ
  5) write DataFrames under <out-root>/datasets/
  6) create scatter plot grid (per tool × parameter) overlaying all replicates

Outputs in datasets/:
  features_df.pkl
  targets_df.pkl
  normalized_train_features.pkl
  normalized_train_targets.pkl
  normalized_validation_features.pkl
  normalized_validation_targets.pkl
  features_scatterplot.png

Additionally written (not tracked by Snakemake unless you add them):
  outliers_removed.tsv         # all extreme outlier rows (tab-separated)
  outliers_preview.txt         # readable summary + rows + counts

Notes:
- Pass --preview-rows -1 to include *all* outlier rows in the preview file
  and stdout (this is the default here).
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, pickle, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# ------------------------------- helpers ---------------------------------- #
def param_dicts(tool: str, blob: dict) -> list[dict]:
    """Return a list of {param: value} dicts (1 per replicate)."""
    if tool.lower() == "momentsld" and "opt_params" in blob:
        return [blob["opt_params"]]
    bp = blob.get("best_params")
    if isinstance(bp, list):
        return bp
    return [bp]


def prior_stats(priors: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, float]]:
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
        if k in mu:  # guard in case of non‑demographic columns
            out[col] = (out[col] - mu[k]) / sigma[k]
    return out


# ----------------------------- plotting ----------------------------------- #
def plot_estimates_vs_truth_grid_multi_rep(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    *,
    tools: tuple[str, ...] = ("dadi", "moments", "momentsLD"),
    params: list[str] | None = None,
    figsize_per_panel: tuple[float, float] = (3.2, 3.0),
    out_path: Path | str = "features_scatterplot.png",
    # Only these tools will colorize by replicate. Leave default to just momentsLD.
    colorize_reps_tools: set[str] | tuple[str, ...] = ("momentsLD",),
):
    colorize_reps_tools = set()

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
                    # per‑rep labeling/coloring (only for the requested tools)
                    lbl = f"rep_{rep}" if rep is not None else tool
                    ax.scatter(x, y, s=16, alpha=0.75, label=lbl)  # default color cycle
                else:
                    # single color & legend entry for the whole tool
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

            # show legend only if there’s more than one plotted series
            if len(matches) > 0:
                ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------------- main ------------------------------------ #
def main(
    cfg_path: Path,
    out_root: Path,
    *,
    tol_rel: float = 1e-9,
    tol_abs: float = 0.0,
    zmax: float = 6.0,
    preview_rows: int = -1,   # -1 ⇒ show all rows
) -> None:
    cfg        = json.loads(cfg_path.read_text())
    model      = cfg["demographic_model"]
    n_sims     = int(cfg["num_draws"])
    train_pct  = float(cfg.get("training_percentage", 0.8))
    seed       = int(cfg.get("seed", 42))
    rng        = np.random.default_rng(seed)

    priors     = cfg["priors"]
    mu, sigma  = prior_stats(priors)

    sim_basedir   = Path(f"experiments/{model}/simulations")
    infer_basedir = Path(f"experiments/{model}/inferences")

    feature_rows, target_rows, index = [], [], []

    for sid in range(n_sims):
        inf_pickle   = infer_basedir / f"sim_{sid}/all_inferences.pkl"
        truth_pickle = sim_basedir   / f"{sid}/sampled_params.pkl"
        if not inf_pickle.exists() or not truth_pickle.exists():
            continue

        data  = pickle.load(inf_pickle.open("rb"))
        truth = pickle.load(truth_pickle.open("rb"))

        row = {}
        for tool in ("moments", "dadi", "momentsLD"):
            if tool in data and data[tool] is not None:
                for rep_idx, pdict in enumerate(param_dicts(tool, data[tool])):
                    if not isinstance(pdict, dict):
                        continue
                    for k, v in pdict.items():
                        if v is None or np.isnan(v):
                            continue
                        col = f"{tool}_{k}" if tool.lower() == "momentsld" else f"{tool}_{k}_rep_{rep_idx}"
                        row[col] = float(v)

        feature_rows.append(row)
        target_rows.append(truth)
        index.append(sid)

    feat_df = pd.DataFrame(feature_rows, index=index).sort_index(axis=1)
    targ_df = pd.DataFrame(target_rows,  index=index).sort_index(axis=1)

    # ---------- outlier removal BEFORE split --------------------------------
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

        # decide how many rows to show in preview/stdout
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

        # also echo to stdout (can be large!)
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
    train_idx = feat_df.index.to_numpy()[perm[:n_train]]
    val_idx   = feat_df.index.to_numpy()[perm[n_train:]]

    norm_train_feats = feat_norm_df.loc[train_idx]
    norm_train_targs = targ_norm_df.loc[train_idx]
    norm_val_feats   = feat_norm_df.loc[val_idx]
    norm_val_targs   = targ_norm_df.loc[val_idx]

    # ---------- outputs -----------------------------------------------------
    # full, unnormalised
    (datasets_dir / "features_df.pkl").write_bytes(pickle.dumps(feat_df))
    (datasets_dir / "targets_df.pkl").write_bytes(pickle.dumps(targ_df))

    # normalised split DataFrames
    (datasets_dir / "normalized_train_features.pkl").write_bytes(pickle.dumps(norm_train_feats))
    (datasets_dir / "normalized_train_targets.pkl").write_bytes(pickle.dumps(norm_train_targs))
    (datasets_dir / "normalized_validation_features.pkl").write_bytes(pickle.dumps(norm_val_feats))
    (datasets_dir / "normalized_validation_targets.pkl").write_bytes(pickle.dumps(norm_val_targs))

    # ---------- plotting ----------------------------------------------------
    plot_estimates_vs_truth_grid_multi_rep(
        features_df=feat_df,
        targets_df=targ_df,
        tools=("dadi", "moments", "momentsLD"),
        params=None,
        figsize_per_panel=(3.2, 3.0),
        out_path=datasets_dir / "features_scatterplot.png",
    )

    print(f"✓ wrote datasets & plot to: {datasets_dir}")
    print(f"✓ outlier preview: {preview_path}")  # not tracked by Snakemake unless added


# --------------------------------- CLI ------------------------------------ #
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
    args = ap.parse_args()

    main(
        args.experiment_config,
        args.out_dir,
        tol_rel=args.tol_rel,
        tol_abs=args.tol_abs,
        zmax=args.zmax,
        preview_rows=args.preview_rows,
    )

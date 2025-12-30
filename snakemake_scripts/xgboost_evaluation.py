#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, pickle, joblib, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as mse_sklearn
from xgboost import XGBRegressor

# ----- allow project root imports (for visualizing_results) -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting_helpers import visualizing_results  # your existing function


# ---------- helpers ----------
def mean_squared_error(y_true, y_pred) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean((yt - yp) ** 2))


def load_df_pickle(path: str):
    """
    Load a pkl that contains a DataFrame/Series, or ndarray/dict.
    Return (df, colnames). If we don't know names, we synthesize them.
    Mirrors your RF loader.
    """
    obj = pickle.load(open(path, "rb"))

    if isinstance(obj, pd.DataFrame):
        return obj, obj.columns.tolist()

    if isinstance(obj, pd.Series):
        return obj.to_frame(), [obj.name]

    if isinstance(obj, dict) and "features" in obj:
        arr = np.asarray(obj["features"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = obj.get("feature_names", None)
        if cols is None:
            cols = [f"feature_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=list(cols))
        return df, list(cols)

    if isinstance(obj, dict) and "targets" in obj:
        arr = np.asarray(obj["targets"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = obj.get("target_names", None)
        if cols is None:
            cols = [f"target_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=list(cols))
        return df, list(cols)

    arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = [f"col_{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    return df, cols


def _align_df_columns(df: pd.DataFrame, desired_order: list[str], name: str) -> pd.DataFrame:
    """
    Reorder df columns to desired_order if it has the same set of columns.
    Otherwise raise (so you catch silent mismatch early).
    """
    if df is None:
        return df

    have = list(df.columns)
    if have == desired_order:
        return df

    if set(have) != set(desired_order):
        missing = [c for c in desired_order if c not in have]
        extra = [c for c in have if c not in desired_order]
        raise ValueError(
            f"[COLUMN ORDER ERROR] {name} columns don't match TRAIN columns.\n"
            f"Missing: {missing}\n"
            f"Extra:   {extra}\n"
            f"Train:   {desired_order}\n"
            f"{name}: {have}\n"
        )

    return df.loc[:, desired_order]


def _default_model_dir(exp_cfg):
    return Path(f"experiments/{exp_cfg['demographic_model']}/modeling/xgboost")


def _get_param_names(exp_cfg):
    if "parameters_to_estimate" in exp_cfg:
        return list(exp_cfg["parameters_to_estimate"])
    if "priors" in exp_cfg:
        return list(exp_cfg["priors"].keys())
    return []


def _detect_outer_jobs(user, default=1):
    if user is not None:
        return int(user)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or default))


def _plot_feature_importances_grid(model, feature_names, target_names, save_path, top_k=None):
    estimators = model.estimators_
    n_out = len(estimators)
    n_cols = 3
    n_rows = (n_out + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), constrained_layout=True)
    axes = axes.flatten()

    for out_idx, est in enumerate(estimators):
        importances = est.feature_importances_
        idx_sorted = np.argsort(importances)[::-1]
        if top_k is not None:
            idx_sorted = idx_sorted[:top_k]

        imp_sorted = importances[idx_sorted]
        names_sorted = [feature_names[i] for i in idx_sorted]

        ax = axes[out_idx]
        ax.bar(range(len(imp_sorted)), imp_sorted, align="center")
        ax.set_xticks(range(len(imp_sorted)))
        ax.set_xticklabels(names_sorted, rotation=45, ha="right")
        ax.set_ylabel("Importance")
        ax.set_xlabel("Features")
        targ_label = (
            target_names[out_idx]
            if target_names and out_idx < len(target_names)
            else f"target_{out_idx}"
        )
        ax.set_title(f"Feature Importances – {targ_label}")

    for ax in axes[n_out:]:
        ax.axis("off")

    fig.suptitle("XGBoost Feature Importances Across Targets", fontsize=16)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def tune_xgb_on_tune_multioutput(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    n_iter: int = 20,
    random_state: int = 42,
    outer_jobs: int = 1,
    xgb_n_jobs: int = 1,
    tree_method: str = "hist",
):
    """
    Random search over XGB hyperparams using the TUNE set, scoring on ALL targets.

    IMPORTANT PARALLELISM RULE (same idea as your RF fix):
      - XGBRegressor n_jobs = xgb_n_jobs (keep small, typically 1)
      - MultiOutputRegressor n_jobs = outer_jobs (parallelize across targets)
    """
    rng = np.random.RandomState(random_state)

    # (Keep your discrete grids; expand if you want)
    n_estimators_grid = [150, 250]
    max_depth_grid = [3, 5]
    learning_rate_grid = [0.05, 0.10]
    subsample_grid = [0.8, 1.0]
    colsample_bytree_grid = [0.8, 1.0]
    min_child_weight_grid = [1, 3]
    reg_lambda_grid = [1.0, 5.0]
    reg_alpha_grid = [0.0, 0.1]

    best_mse = np.inf
    best_params = None

    print(
        f"[TUNE] Random search on TUNE (ALL targets): n_iter={n_iter}, "
        f"X_train={X_train.shape}, X_tune={X_tune.shape}, y_train={y_train.shape}"
    )

    for i in range(n_iter):
        params = dict(
            objective="reg:squarederror",
            n_estimators=int(rng.choice(n_estimators_grid)),
            max_depth=int(rng.choice(max_depth_grid)),
            learning_rate=float(rng.choice(learning_rate_grid)),
            subsample=float(rng.choice(subsample_grid)),
            colsample_bytree=float(rng.choice(colsample_bytree_grid)),
            min_child_weight=float(rng.choice(min_child_weight_grid)),
            reg_lambda=float(rng.choice(reg_lambda_grid)),
            reg_alpha=float(rng.choice(reg_alpha_grid)),
            tree_method=tree_method,
            n_jobs=int(xgb_n_jobs),
            random_state=int(random_state),
            verbosity=0,
        )

        model = MultiOutputRegressor(XGBRegressor(**params), n_jobs=int(outer_jobs))

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        preds = model.predict(X_tune)
        mse = mse_sklearn(y_tune, preds)
        dt = time.perf_counter() - t0

        print(
            f"[TUNE] iter={i+1}/{n_iter} "
            f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
            f"learning_rate={params['learning_rate']}, subsample={params['subsample']}, "
            f"colsample_bytree={params['colsample_bytree']}, min_child_weight={params['min_child_weight']}, "
            f"reg_lambda={params['reg_lambda']}, reg_alpha={params['reg_alpha']} "
            f"-> tune MSE(all targets)={mse:.6f} (time={dt:.1f}s)"
        )

        if mse < best_mse:
            best_mse = mse
            best_params = params

    print(f"[TUNE] Best params from TUNE: {best_params}, tune MSE={best_mse:.6f}")
    return best_params


# ---------- main ----------
def xgboost_evaluation(
    X_train_path=None,
    y_train_path=None,
    X_tune_path=None,
    y_tune_path=None,
    X_val_path=None,
    y_val_path=None,
    experiment_config_path=None,
    model_config_path=None,
    color_shades_path=None,
    main_colors_path=None,
    model_directory=None,
    # hyperparams
    n_estimators=None,
    max_depth=None,
    learning_rate=None,
    subsample=None,
    colsample_bytree=None,
    min_child_weight=None,
    reg_lambda=None,
    reg_alpha=None,
    # random search
    do_random_search=False,
    n_iter=20,
    random_state=42,
    top_k_features_plot=None,
    # perf controls
    outer_n_jobs=None,
    xgb_n_jobs=1,
    tree_method="hist",
    early_stopping_rounds=None,  # IMPORTANT: ES uses TUNE, not VAL
):

    if experiment_config_path is None:
        raise ValueError("--experiment_config_path is required")
    with open(experiment_config_path, "r") as f:
        exp_cfg = json.load(f)
    if model_config_path:
        with open(model_config_path) as f:
            _ = yaml.safe_load(f)

    # output dir
    if model_directory is None:
        model_directory = _default_model_dir(exp_cfg)
    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Outputs → {model_directory}")

    # colors
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors = pickle.load(open(main_colors_path, "rb"))

    # ---------- LOAD AS DFS (for alignment discipline) ----------
    X_train_df, feat_names_train = load_df_pickle(X_train_path)
    y_train_df, targ_names_train = load_df_pickle(y_train_path)
    X_tune_df, _ = load_df_pickle(X_tune_path) if X_tune_path else (None, None)
    y_tune_df, _ = load_df_pickle(y_tune_path) if y_tune_path else (None, None)
    X_val_df, _ = load_df_pickle(X_val_path) if X_val_path else (None, None)
    y_val_df, _ = load_df_pickle(y_val_path) if y_val_path else (None, None)

    # SOURCE OF TRUTH: TRAIN order
    feature_order = list(feat_names_train)
    target_order = list(targ_names_train) if targ_names_train else _get_param_names(exp_cfg)

    # Align tune/val to TRAIN order
    if X_tune_df is not None:
        X_tune_df = _align_df_columns(X_tune_df, feature_order, "X_tune")
    if X_val_df is not None:
        X_val_df = _align_df_columns(X_val_df, feature_order, "X_val")
    if y_tune_df is not None and len(target_order) > 0:
        y_tune_df = _align_df_columns(y_tune_df, target_order, "y_tune")
    if y_val_df is not None and len(target_order) > 0:
        y_val_df = _align_df_columns(y_val_df, target_order, "y_val")

    print("[INFO] Feature order used:")
    print("  " + ", ".join(feature_order[:10]) + (" ..." if len(feature_order) > 10 else ""))
    print("[INFO] Target order used:")
    print("  " + ", ".join(target_order))

    # Convert to arrays
    X_train = X_train_df.values
    y_train = y_train_df.values
    X_tune = None if X_tune_df is None else X_tune_df.values
    y_tune = None if y_tune_df is None else y_tune_df.values
    X_val = None if X_val_df is None else X_val_df.values
    y_val = None if y_val_df is None else y_val_df.values

    # sanity
    if X_train is None or y_train is None:
        raise ValueError("Need X_train_path and y_train_path.")
    if (X_val is None) ^ (y_val is None):
        raise ValueError("Need both X_val & y_val (or neither).")
    if (X_tune is None) ^ (y_tune is None):
        raise ValueError("Need both X_tune & y_tune (or neither).")

    # Environment hints to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    outer_jobs = _detect_outer_jobs(outer_n_jobs)
    print(f"[INFO] Parallelism: outer n_jobs={outer_jobs} (targets), inner xgb n_jobs={xgb_n_jobs}")

    # decide hyperparams
    user_provided = any(
        v is not None
        for v in [
            n_estimators,
            max_depth,
            learning_rate,
            subsample,
            colsample_bytree,
            min_child_weight,
            reg_lambda,
            reg_alpha,
        ]
    )

    if (do_random_search or not user_provided) and X_tune is not None:
        print("[INFO] Running random search on TUNE set for XGBoost (ALL targets) …")
        best = tune_xgb_on_tune_multioutput(
            X_train,
            y_train,
            X_tune,
            y_tune,
            n_iter=n_iter,
            random_state=random_state,
            outer_jobs=outer_jobs,
            xgb_n_jobs=xgb_n_jobs,
            tree_method=tree_method,
        )
        # apply
        n_estimators = best.get("n_estimators", n_estimators)
        max_depth = best.get("max_depth", max_depth)
        learning_rate = best.get("learning_rate", learning_rate)
        subsample = best.get("subsample", subsample)
        colsample_bytree = best.get("colsample_bytree", colsample_bytree)
        min_child_weight = best.get("min_child_weight", min_child_weight)
        reg_lambda = best.get("reg_lambda", reg_lambda)
        reg_alpha = best.get("reg_alpha", reg_alpha)
        tree_method = best.get("tree_method", tree_method)
        print("[INFO] Best params applied from TUNE search.")

    # Base estimator
    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators if n_estimators is not None else 200,
        max_depth=max_depth if max_depth is not None else 5,
        learning_rate=learning_rate if learning_rate is not None else 0.05,
        subsample=subsample if subsample is not None else 1.0,
        colsample_bytree=colsample_bytree if colsample_bytree is not None else 1.0,
        min_child_weight=min_child_weight if min_child_weight is not None else 1,
        reg_lambda=reg_lambda if reg_lambda is not None else 1.0,
        reg_alpha=reg_alpha if reg_alpha is not None else 0.0,
        tree_method=tree_method,
        n_jobs=int(xgb_n_jobs),
        random_state=random_state,
        verbosity=2,
    )

    print(
        f"[INFO] Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_tune={None if X_tune is None else X_tune.shape}, "
        f"y_tune={None if y_tune is None else y_tune.shape}, "
        f"X_val={None if X_val is None else X_val.shape}, y_val={None if y_val is None else y_val.shape}"
    )

    # ---------- FIT ----------
    # If early stopping is set, use TUNE as eval_set. VAL stays untouched.
    def _fit_multioutput_with_es_using_tune(X_tr, Y_tr, X_tu, Y_tu):
        est_list = []
        n_out = Y_tr.shape[1]
        for i in range(n_out):
            est = xgb_base.__class__(**xgb_base.get_params())
            lab = target_order[i] if target_order and i < len(target_order) else f"target_{i}"
            print(f"[FIT+ES] Target {i+1}/{n_out}: {lab}")
            t0 = time.perf_counter()
            est.fit(
                X_tr,
                Y_tr[:, i],
                eval_set=[(X_tu, Y_tu[:, i])],
                early_stopping_rounds=int(early_stopping_rounds),
                verbose=True,
            )
            best_iter = getattr(est, "best_iteration", None)
            print(f"[FIT+ES]   elapsed={time.perf_counter()-t0:.1f}s best_iteration={best_iter}")
            est_list.append(est)
        return est_list

    if early_stopping_rounds is not None:
        if X_tune is None or y_tune is None:
            raise ValueError("early_stopping_rounds requires TUNE data (X_tune/y_tune).")
        print(f"[INFO] Early stopping enabled: {early_stopping_rounds} rounds (eval_set=TUNE).")
        est_list = _fit_multioutput_with_es_using_tune(X_train, y_train, X_tune, y_tune)
        model = MultiOutputRegressor(xgb_base, n_jobs=outer_jobs)
        model.estimators_ = est_list
    else:
        model = MultiOutputRegressor(xgb_base, n_jobs=outer_jobs)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        print(f"[INFO] MultiOutputRegressor.fit elapsed: {time.perf_counter()-t0:.1f}s")

    # predictions
    train_preds = model.predict(X_train)
    if train_preds.ndim == 1:
        train_preds = train_preds.reshape(-1, 1)

    val_preds = None
    if X_val is not None:
        val_preds = model.predict(X_val)
        if val_preds.ndim == 1:
            val_preds = val_preds.reshape(-1, 1)

    # package container
    xgb_obj = {
        "model": model,
        "training": {"predictions": train_preds, "targets": y_train},
        "validation": {"predictions": val_preds, "targets": y_val} if X_val is not None else None,
        "param_names": list(target_order),
        "target_order": list(target_order),
        "feature_order": list(feature_order),
        "xgb_random_state": int(random_state),
        "outer_n_jobs": int(outer_jobs),
        "xgb_n_jobs": int(xgb_n_jobs),
        "tree_method": str(tree_method),
        "early_stopping_rounds": None if early_stopping_rounds is None else int(early_stopping_rounds),
    }

    # errors
    err = {
        "training": float(np.mean((y_train - train_preds) ** 2)),
        "validation": None if (X_val is None) else float(np.mean((y_val - val_preds) ** 2)),
        "training_mse": {},
        "validation_mse": {},
        # "target_order": list(target_order),
        # "feature_order": list(feature_order),
    }

    for i, pname in enumerate(target_order):
        err["training_mse"][pname] = float(np.mean((y_train[:, i] - train_preds[:, i]) ** 2))
        if X_val is not None:
            err["validation_mse"][pname] = float(np.mean((y_val[:, i] - val_preds[:, i]) ** 2))

    # save artifacts
    with open(model_directory / "xgb_mdl_obj.pkl", "wb") as f:
        pickle.dump(xgb_obj, f)
    with open(model_directory / "xgb_model_error.json", "w") as f:
        json.dump(err, f, indent=4)
    joblib.dump(model, model_directory / "xgb_model.pkl")

    # plots
    visualizing_results(
        xgb_obj,
        analysis="xgb_results",
        save_loc=model_directory,
        stages=["training", "validation"] if X_val is not None else ["training"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    _plot_feature_importances_grid(
        model,
        feature_names=list(feature_order),
        target_names=list(target_order),
        save_path=model_directory / "xgb_feature_importances.png",
        top_k=top_k_features_plot,
    )

    print("[INFO] XGBoost run complete.")


# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XGBoost multi-output evaluation script")

    # Data
    p.add_argument("--X_train_path", type=str, required=True)
    p.add_argument("--y_train_path", type=str, required=True)
    p.add_argument("--X_tune_path", type=str, required=True)
    p.add_argument("--y_tune_path", type=str, required=True)
    p.add_argument("--X_val_path", type=str, required=True)
    p.add_argument("--y_val_path", type=str, required=True)

    # Configs
    p.add_argument("--experiment_config_path", type=str, required=True)
    p.add_argument("--model_config_path", type=str, default=None)
    p.add_argument("--color_shades_file", type=str, required=True)
    p.add_argument("--main_colors_file", type=str, required=True)

    # Output dir
    p.add_argument("--model_directory", type=str, default=None)

    # Hyperparams
    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--colsample_bytree", type=float, default=None)
    p.add_argument("--min_child_weight", type=float, default=None)
    p.add_argument("--reg_lambda", type=float, default=None)
    p.add_argument("--reg_alpha", type=float, default=None)

    # Random search
    p.add_argument("--do_random_search", action="store_true")
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--random_state", type=int, default=42)

    # Plot options
    p.add_argument("--top_k_features_plot", type=int, default=None)

    # Perf controls
    p.add_argument("--outer-n-jobs", type=int, default=None)
    p.add_argument("--xgb-n-jobs", type=int, default=1)
    p.add_argument("--tree-method", type=str, default="hist")

    # Early stopping: uses TUNE (not VAL)
    p.add_argument("--early-stopping-rounds", type=int, default=None)

    args = p.parse_args()

    xgboost_evaluation(
        X_train_path=args.X_train_path,
        y_train_path=args.y_train_path,
        X_tune_path=args.X_tune_path,
        y_tune_path=args.y_tune_path,
        X_val_path=args.X_val_path,
        y_val_path=args.y_val_path,
        experiment_config_path=args.experiment_config_path,
        model_config_path=args.model_config_path,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file,
        model_directory=args.model_directory,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        do_random_search=args.do_random_search,
        n_iter=args.n_iter,
        random_state=args.random_state,
        top_k_features_plot=args.top_k_features_plot,
        outer_n_jobs=args.outer_n_jobs,
        xgb_n_jobs=args.xgb_n_jobs,
        tree_method=args.tree_method,
        early_stopping_rounds=args.early_stopping_rounds,
    )

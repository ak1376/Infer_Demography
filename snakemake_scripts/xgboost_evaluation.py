#!/usr/bin/env python3
import argparse, json, os, sys, pickle, joblib, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")  # headless
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
def mean_squared_error(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean((yt - yp) ** 2))


def _load_df_and_array(path):
    """
    Load .pkl or .npy and return (numpy_array, column_names_if_df_else_None).
    Assumes .pkl files are DataFrames or dict-like from your pipeline.
    """
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
        return arr, None
    elif p.suffix == ".pkl":
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy(), list(obj.columns)
        if isinstance(obj, pd.Series):
            return obj.to_numpy().reshape(-1, 1), [obj.name]
        if isinstance(obj, dict):
            if "features" in obj:
                arr = np.asarray(obj["features"])
                colnames = obj.get("feature_names", None)
                return arr, colnames
            if "targets" in obj:
                arr = np.asarray(obj["targets"])
                colnames = obj.get("target_names", None)
                return arr, colnames
        # fallback
        arr = np.asarray(obj)
        return arr, None
    else:
        raise ValueError(f"Unsupported extension for {path} (use .npy or .pkl).")


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


def _plot_feature_importances_grid(
    model, feature_names, target_names, save_path, top_k=None
):
    """Single PNG grid of per-target importances."""
    estimators = model.estimators_
    n_out = len(estimators)
    n_cols = 3
    n_rows = (n_out + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16, 5 * n_rows), constrained_layout=True
    )
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


def tune_xgb_on_tune(
    X_train,
    y_train,
    X_tune,
    y_tune,
    n_iter=20,
    random_state=42,
):
    """
    Manual random search over XGB hyperparameters using TUNE set.

    For each sampled hyperparameter set:
      - fit single-output XGB on TRAIN (first target)
      - evaluate MSE on TUNE (same target)
    Returns dict of best params.
    """
    rng = np.random.RandomState(random_state)

    # Discrete grids (same spirit as your old RandomizedSearchCV)
    n_estimators_grid = [150, 250]
    max_depth_grid = [3, 5]
    learning_rate_grid = [0.05, 0.10]
    subsample_grid = [0.8, 1.0]
    colsample_bytree_grid = [0.8, 1.0]
    min_child_weight_grid = [1, 3]
    reg_lambda_grid = [1.0, 5.0]
    reg_alpha_grid = [0.0, 0.1]
    tree_method = "hist"

    # single-output target (first column if multi-output)
    y_tr = y_train[:, 0] if y_train.ndim > 1 else y_train
    y_tu = y_tune[:, 0] if y_tune.ndim > 1 else y_tune

    best_mse = np.inf
    best_params = None

    print(
        f"[TUNE] Random search on TUNE set: n_iter={n_iter}, "
        f"X_train={X_train.shape}, X_tune={X_tune.shape}"
    )

    for i in range(n_iter):
        n_estimators = rng.choice(n_estimators_grid)
        max_depth = rng.choice(max_depth_grid)
        learning_rate = rng.choice(learning_rate_grid)
        subsample = rng.choice(subsample_grid)
        colsample_bytree = rng.choice(colsample_bytree_grid)
        min_child_weight = rng.choice(min_child_weight_grid)
        reg_lambda = rng.choice(reg_lambda_grid)
        reg_alpha = rng.choice(reg_alpha_grid)

        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            tree_method=tree_method,
            n_jobs=1,  # avoid oversubscription
            random_state=random_state,
            verbosity=0,
        )

        t0 = time.perf_counter()
        xgb.fit(X_train, y_tr)
        preds = xgb.predict(X_tune)
        mse = mse_sklearn(y_tu, preds)
        dt = time.perf_counter() - t0

        print(
            f"[TUNE] iter={i+1}/{n_iter} "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"learning_rate={learning_rate}, subsample={subsample}, "
            f"colsample_bytree={colsample_bytree}, min_child_weight={min_child_weight}, "
            f"reg_lambda={reg_lambda}, reg_alpha={reg_alpha} "
            f"-> tune MSE={mse:.6f} (time={dt:.1f}s)"
        )

        if mse < best_mse:
            best_mse = mse
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_weight": min_child_weight,
                "reg_lambda": reg_lambda,
                "reg_alpha": reg_alpha,
                "tree_method": tree_method,
            }

    print(f"[TUNE] Best params from TUNE set: {best_params}, tune MSE={best_mse:.6f}")
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
    early_stopping_rounds=None,
):

    if experiment_config_path is None:
        raise ValueError("--experiment_config_path is required")
    with open(experiment_config_path, "r") as f:
        exp_cfg = json.load(f)
    if model_config_path:
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)
    else:
        model_cfg = {}

    # output dir
    if model_directory is None:
        model_directory = _default_model_dir(exp_cfg)
    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Outputs → {model_directory}")

    # colors
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors = pickle.load(open(main_colors_path, "rb"))

    # data
    X_train, feat_names_train = (
        _load_df_and_array(X_train_path) if X_train_path else (None, None)
    )
    y_train, targ_names_train = (
        _load_df_and_array(y_train_path) if y_train_path else (None, None)
    )
    X_tune, _ = _load_df_and_array(X_tune_path) if X_tune_path else (None, None)
    y_tune, _ = _load_df_and_array(y_tune_path) if y_tune_path else (None, None)
    X_val, _ = _load_df_and_array(X_val_path) if X_val_path else (None, None)
    y_val, _ = _load_df_and_array(y_val_path) if y_val_path else (None, None)

    # fallbacks for names
    feature_names = (
        feat_names_train
        if feat_names_train is not None
        else (
            [f"feat_{i}" for i in range(X_train.shape[1])]
            if X_train is not None
            else None
        )
    )
    target_names = (
        targ_names_train if targ_names_train is not None else _get_param_names(exp_cfg)
    )

    # sanity checks
    if X_train is None:
        raise ValueError("X_train_path / y_train_path are required.")
    if (X_train is None) ^ (y_train is None):
        raise ValueError("Need both X_train & y_train (or neither).")
    if (X_val is None) ^ (y_val is None):
        raise ValueError("Need both X_val & y_val (or neither).")
    if (X_tune is None) ^ (y_tune is None):
        raise ValueError("Need both X_tune & y_tune (or neither).")

    # decide hyperparams:
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
        print("[INFO] Running random search on TUNE set for XGBoost …")
        best = tune_xgb_on_tune(
            X_train,
            y_train,
            X_tune,
            y_tune,
            n_iter=n_iter,
            random_state=random_state,
        )
        n_estimators = best.get("n_estimators", n_estimators)
        max_depth = best.get("max_depth", max_depth)
        learning_rate = best.get("learning_rate", learning_rate)
        subsample = best.get("subsample", subsample)
        colsample_bytree = best.get("colsample_bytree", colsample_bytree)
        min_child_weight = best.get("min_child_weight", min_child_weight)
        reg_lambda = best.get("reg_lambda", reg_lambda)
        reg_alpha = best.get("reg_alpha", reg_alpha)
        tree_method = best.get("tree_method", tree_method)
        print(f"[INFO] Best params applied from TUNE search.")

    # Environment hints to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    print(
        "[INFO] Thread caps: "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}"
    )

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
        verbosity=2,  # info-level
    )

    outer_jobs = _detect_outer_jobs(outer_n_jobs)
    print(
        f"[INFO] Parallelism: outer n_jobs={outer_jobs} (targets), inner xgb n_jobs={xgb_n_jobs}"
    )
    print(
        f"[INFO] Data shapes: "
        f"X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_tune={None if X_tune is None else X_tune.shape}, "
        f"y_tune={None if y_tune is None else y_tune.shape}, "
        f"X_val={X_val.shape}, y_val={y_val.shape}"
    )

    # fit final multi-output model on TRAIN only
    def _fit_multioutput_with_es(X_tr, Y_tr, X_va, Y_va):
        est_list = []
        n_out = Y_tr.shape[1]
        for i in range(n_out):
            est = xgb_base.__class__(**xgb_base.get_params())
            lab = (
                target_names[i]
                if target_names and i < len(target_names)
                else f"target_{i}"
            )
            print(f"[FIT] Target {i+1}/{n_out}: {lab}")
            t0 = time.perf_counter()
            fit_kwargs = {}
            if (
                early_stopping_rounds is not None
                and X_va is not None
                and Y_va is not None
            ):
                fit_kwargs.update(
                    dict(
                        eval_set=[(X_va, Y_va[:, i])],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=True,
                    )
                )
            est.fit(X_tr, Y_tr[:, i], **fit_kwargs)
            best_iter = getattr(est, "best_iteration", None)
            print(
                f"[FIT]   elapsed={time.perf_counter()-t0:.1f}s best_iteration={best_iter}"
            )
            est_list.append(est)
        return est_list

    if early_stopping_rounds is not None:
        print(f"[INFO] Using early stopping: {early_stopping_rounds} rounds.")
        est_list = _fit_multioutput_with_es(X_train, y_train, X_val, y_val)
        model = MultiOutputRegressor(xgb_base, n_jobs=outer_jobs)
        model.estimators_ = est_list
    else:
        model = MultiOutputRegressor(xgb_base, n_jobs=outer_jobs)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        print(f"[INFO] MultiOutputRegressor.fit elapsed: {time.perf_counter()-t0:.1f}s")

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    if train_preds.ndim == 1:
        train_preds = train_preds.reshape(-1, 1)
    if val_preds.ndim == 1:
        val_preds = val_preds.reshape(-1, 1)

    # package container
    features_and_targets = {
        "training": {"features": X_train, "targets": y_train},
        "validation": {"features": X_val, "targets": y_val},
    }

    xgb_obj = {
        "model": model,
        "training": {
            "predictions": train_preds,
            "targets": y_train,
        },
        "validation": {
            "predictions": val_preds,
            "targets": y_val,
        },
        "param_names": target_names,
    }

    # errors
    rrmse = {
        "training": None,
        "validation": None,
        "training_mse": {},
        "validation_mse": {},
    }

    def _fill_err(y_true, y_pred, split):
        if y_true is None or y_pred is None:
            return
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        rrmse[split] = float(np.mean((yt - yp) ** 2))
        for i, pname in enumerate(target_names):
            rrmse[f"{split}_mse"][pname] = float(np.mean((yt[:, i] - yp[:, i]) ** 2))

    _fill_err(y_train, train_preds, "training")
    _fill_err(y_val, val_preds, "validation")

    # save artifacts
    with open(model_directory / "xgb_mdl_obj.pkl", "wb") as f:
        pickle.dump(xgb_obj, f)
    with open(model_directory / "xgb_model_error.json", "w") as f:
        json.dump(rrmse, f, indent=4)
    joblib.dump(model, model_directory / "xgb_model.pkl")

    # plot predictions/results with your helper
    visualizing_results(
        xgb_obj,
        analysis="xgb_results",
        save_loc=model_directory,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    # Feature importances
    if feature_names is None:
        feature_names = [
            f"feat_{i}"
            for i in range((X_train if X_train is not None else X_val).shape[1])
        ]
    _plot_feature_importances_grid(
        model,
        feature_names=feature_names,
        target_names=target_names,
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
    p.add_argument(
        "--top_k_features_plot",
        type=int,
        default=None,
        help="Limit number of top features shown per target (optional).",
    )

    # Verbosity / performance controls
    p.add_argument(
        "--outer-n-jobs",
        type=int,
        default=None,
        help="Parallel jobs across targets for MultiOutputRegressor. "
        "Default: SLURM_CPUS_PER_TASK or os.cpu_count().",
    )
    p.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=1,
        help="Threads per XGBRegressor (keep small to avoid oversubscription).",
    )
    p.add_argument(
        "--tree-method",
        type=str,
        default="hist",
        help="XGBoost tree_method; 'hist' is fast on CPU.",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help="If set and validation data provided, use early stopping with per-iteration logs.",
    )

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

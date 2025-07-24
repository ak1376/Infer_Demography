#!/usr/bin/env python3
import argparse, json, os, sys, pickle, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error as mse_sklearn
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
    Assumes .pkl files are DataFrames or dicts from your pipeline.
    """
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
        return arr, None
    elif p.suffix == ".pkl":
        obj = pickle.load(open(p, "rb"))
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy(), list(obj.columns)
        if isinstance(obj, pd.Series):
            return obj.to_numpy().reshape(-1, 1), [obj.name]
        if isinstance(obj, dict):
            # try common keys
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
    # prefer the same scheme you used above
    if "parameters_to_estimate" in exp_cfg:
        return list(exp_cfg["parameters_to_estimate"])
    if "priors" in exp_cfg:
        return list(exp_cfg["priors"].keys())
    return []

def _random_search_xgb(X_train, y_train, n_iter=20, random_state=42):
    """
    RandomizedSearchCV over XGBRegressor. Uses first target column only
    for speed/compatibility, then applies best params to all outputs.
    """
    base = XGBRegressor(objective="reg:squarederror", n_jobs=-1)
    param_dist = {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [2, 3, 4, 5, 8, 10],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.6, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5],
        "reg_lambda":        [1, 2, 5, 10],
        "reg_alpha":         [0, 0.1, 0.5, 1.0],
    }
    scorer = make_scorer(mse_sklearn, greater_is_better=False)
    # if multi-output, just search on first target to get best hyperparams
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_search = y_train[:, 0]
    else:
        y_search = y_train

    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring=scorer,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    rs.fit(X_train, y_search)
    return rs.best_params_


def _plot_feature_importances_grid(model, feature_names, target_names, save_path, top_k=None):
    """
    model: MultiOutputRegressor(XGBRegressor)
    feature_names: list[str]
    target_names: list[str] (len = n_outputs), fallback to index
    top_k: limit number of features (optional)
    save_path: single PNG grid
    """
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
        targ_label = target_names[out_idx] if target_names and out_idx < len(target_names) else f"target_{out_idx}"
        ax.set_title(f"Feature Importances – {targ_label}")

    # Hide any extra axes
    for ax in axes[n_out:]:
        ax.axis("off")

    fig.suptitle("XGBoost Feature Importances Across Targets", fontsize=16)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------- main ----------
def xgboost_evaluation(
    X_train_path=None, y_train_path=None,
    X_val_path=None,   y_val_path=None,
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
    top_k_features_plot=None
):

    if experiment_config_path is None:
        raise ValueError("--experiment_config_path is required")
    exp_cfg = json.load(open(experiment_config_path, "r"))
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
    main_colors  = pickle.load(open(main_colors_path, "rb"))

    # data
    X_train, feat_names_train = _load_df_and_array(X_train_path) if X_train_path else (None, None)
    y_train, targ_names_train = _load_df_and_array(y_train_path) if y_train_path else (None, None)
    X_val,   _                = _load_df_and_array(X_val_path)   if X_val_path   else (None, None)
    y_val,   _                = _load_df_and_array(y_val_path)   if y_val_path   else (None, None)

    # fallbacks for names
    feature_names = feat_names_train if feat_names_train is not None else [f"feat_{i}" for i in range(X_train.shape[1])]
    target_names  = targ_names_train if targ_names_train is not None else _get_param_names(exp_cfg)

    if X_train is None and X_val is None:
        raise ValueError("Provide at least a training or validation split.")
    if (X_train is None) ^ (y_train is None):
        raise ValueError("Need both X_train & y_train (or neither).")
    if (X_val is None) ^ (y_val is None):
        raise ValueError("Need both X_val & y_val (or neither).")

    # decide hyperparams:
    user_provided = any(v is not None for v in [
        n_estimators, max_depth, learning_rate, subsample,
        colsample_bytree, min_child_weight, reg_lambda, reg_alpha
    ])

    if (not user_provided and do_random_search and X_train is not None):
        print("[INFO] Running RandomizedSearchCV for XGBoost …")
        best = _random_search_xgb(X_train, y_train, n_iter=n_iter, random_state=random_state)
        n_estimators     = best.get("n_estimators", n_estimators)
        max_depth        = best.get("max_depth", max_depth)
        learning_rate    = best.get("learning_rate", learning_rate)
        subsample        = best.get("subsample", subsample)
        colsample_bytree = best.get("colsample_bytree", colsample_bytree)
        min_child_weight = best.get("min_child_weight", min_child_weight)
        reg_lambda       = best.get("reg_lambda", reg_lambda)
        reg_alpha        = best.get("reg_alpha", reg_alpha)
        print(f"[INFO] Best params: {best}")

    # Build base estimator
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
        n_jobs=-1,
        random_state=random_state
    )
    model = MultiOutputRegressor(xgb_base)

    # fit
    if X_train is not None:
        model.fit(X_train, y_train)
    else:
        # rarely fit on val if only val is given
        model.fit(X_val, y_val)

    train_preds = model.predict(X_train) if X_train is not None else None
    val_preds   = model.predict(X_val)   if X_val is not None else None
    # ensure 2D
    if train_preds is not None and train_preds.ndim == 1:
        train_preds = train_preds.reshape(-1, 1)
    if val_preds is not None and val_preds.ndim == 1:
        val_preds = val_preds.reshape(-1, 1)

    # package container
    features_and_targets = {}
    if X_train is not None:
        features_and_targets["training"] = {"features": X_train, "targets": y_train}
    if X_val is not None:
        features_and_targets["validation"] = {"features": X_val, "targets": y_val}

    xgb_obj = {
        "model": model,
        "training": {},
        "validation": {},
        "param_names": target_names  # store param names
    }
    if "training" in features_and_targets:
        xgb_obj["training"]["predictions"] = train_preds
        xgb_obj["training"]["targets"] = np.asarray(features_and_targets["training"]["targets"])
    if "validation" in features_and_targets:
        xgb_obj["validation"]["predictions"] = val_preds
        xgb_obj["validation"]["targets"] = np.asarray(features_and_targets["validation"]["targets"])

    # errors
    rrmse = {"training": None, "validation": None, "training_mse": {}, "validation_mse": {}}

    def _fill_err(y_true, y_pred, split):
        if y_true is None or y_pred is None:
            return
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        rrmse[split] = float(np.mean((yt - yp) ** 2))
        for i, pname in enumerate(target_names):
            rrmse[f"{split}_mse"][pname] = float(np.mean((yt[:, i] - yp[:, i]) ** 2))

    _fill_err(xgb_obj["training"].get("targets"),     xgb_obj["training"].get("predictions"),     "training")
    _fill_err(xgb_obj["validation"].get("targets"),   xgb_obj["validation"].get("predictions"),   "validation")

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
        stages=[s for s in ["training", "validation"] if s in features_and_targets],
        color_shades=color_shades,
        main_colors=main_colors
    )

    # Feature importances
    _plot_feature_importances_grid(
        model,
        feature_names=feature_names,
        target_names=target_names,
        save_path=model_directory / "xgb_feature_importances.png",
        top_k=top_k_features_plot
    )

    print("[INFO] XGBoost run complete.")


# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XGBoost multi-output evaluation script")

    # Data
    p.add_argument("--X_train_path", type=str, default=None)
    p.add_argument("--y_train_path", type=str, default=None)
    p.add_argument("--X_val_path",   type=str, default=None)
    p.add_argument("--y_val_path",   type=str, default=None)

    # Configs
    p.add_argument("--experiment_config_path", type=str, required=True)
    p.add_argument("--model_config_path",      type=str, default=None)
    p.add_argument("--color_shades_file",      type=str, required=True)
    p.add_argument("--main_colors_file",       type=str, required=True)

    # Output dir
    p.add_argument("--model_directory", type=str, default=None)

    # Hyperparams
    p.add_argument("--n_estimators",      type=int,   default=None)
    p.add_argument("--max_depth",         type=int,   default=None)
    p.add_argument("--learning_rate",     type=float, default=None)
    p.add_argument("--subsample",         type=float, default=None)
    p.add_argument("--colsample_bytree",  type=float, default=None)
    p.add_argument("--min_child_weight",  type=float, default=None)
    p.add_argument("--reg_lambda",        type=float, default=None)
    p.add_argument("--reg_alpha",         type=float, default=None)

    # Random search
    p.add_argument("--do_random_search", action="store_true")
    p.add_argument("--n_iter",           type=int, default=20)
    p.add_argument("--random_state",     type=int, default=42)

    # Plot options
    p.add_argument("--top_k_features_plot", type=int, default=None,
                   help="Limit number of top features shown per target (optional).")

    args = p.parse_args()

    xgboost_evaluation(
        X_train_path=args.X_train_path,
        y_train_path=args.y_train_path,
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
        top_k_features_plot=args.top_k_features_plot
    )

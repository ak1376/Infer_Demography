#!/usr/bin/env python3
import argparse, json, os, sys, pickle, joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import yaml
import pandas as pd

# make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting_helpers import visualizing_results   # keep using your plotting util


# ---------- minimal helpers ----------

def mean_squared_error(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean((yt - yp) ** 2))

def _load_array(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix == ".pkl":
        obj = pickle.load(open(p, "rb"))
        # Common cases:
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy()
        if isinstance(obj, pd.Series):
            return obj.to_numpy().reshape(-1, 1)
        if isinstance(obj, dict) and "features" in obj:
            return np.asarray(obj["features"])
        if isinstance(obj, dict) and "targets" in obj:
            return np.asarray(obj["targets"])
        return np.asarray(obj)
    else:
        raise ValueError(f"Unsupported extension for {path} (use .npy or .pkl).")

def _build_default_model_dir(exp_cfg, regression_type):
    base = Path(f"experiments/{exp_cfg['demographic_model']}/modeling")
    return base / f"linear_{regression_type}"

def _pick_model(regression_type, alpha, l1_ratio):
    if regression_type == "ridge":
        return Ridge(alpha=alpha)
    if regression_type == "lasso":
        return Lasso(alpha=alpha, max_iter=10000)
    if regression_type == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    if regression_type == "standard":
        return LinearRegression()
    raise ValueError("regression_type must be one of: standard, ridge, lasso, elasticnet")

def _fit_and_predict(model, X_train, y_train, X_val, y_val):
    # Fit on train if available, else fit on val (rare case)
    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train)
    elif X_val is not None and y_val is not None:
        model.fit(X_val, y_val)
    else:
        raise ValueError("No data provided to fit the model.")

    train_preds = (model.predict(X_train).reshape(X_train.shape[0], -1)
                   if X_train is not None else None)
    val_preds   = (model.predict(X_val).reshape(X_val.shape[0], -1)
                   if X_val is not None else None)
    return train_preds, val_preds

def _organize_results(data_dict, train_preds, val_preds, model):
    out = {"model": model, "training": {}, "validation": {}}
    if "training" in data_dict:
        out["training"]["predictions"] = train_preds
        out["training"]["targets"]     = np.asarray(data_dict["training"]["targets"])
    if "validation" in data_dict:
        out["validation"]["predictions"] = val_preds
        out["validation"]["targets"]     = np.asarray(data_dict["validation"]["targets"])
    return out


# ---------- main ----------

def linear_evaluation(
    X_train_path=None,
    y_train_path=None,
    X_val_path=None,
    y_val_path=None,
    experiment_config_path=None,
    model_config_path=None,
    color_shades_path=None,
    main_colors_path=None,
    model_directory=None,
    regression_type="standard",
    alpha=0.0,
    l1_ratio=0.5,
    do_grid_search=False
):
    # configs
    if experiment_config_path is None:
        raise ValueError("experiment_config_path is required.")
    exp_cfg = json.load(open(experiment_config_path))
    if model_config_path:
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)
    else:
        model_cfg = {}

    # model dir
    if model_directory is None:
        model_directory = _build_default_model_dir(exp_cfg, regression_type)
    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving outputs to: {model_directory}")

    # colors
    color_shades = pickle.load(open(color_shades_path, "rb"))
    main_colors  = pickle.load(open(main_colors_path, "rb"))

    # data
    X_train = _load_array(X_train_path) if X_train_path else None
    y_train = _load_array(y_train_path) if y_train_path else None
    X_val   = _load_array(X_val_path)   if X_val_path   else None
    y_val   = _load_array(y_val_path)   if y_val_path   else None

    if X_train is None and X_val is None:
        raise ValueError("Provide at least train or val split.")
    if (X_train is None) ^ (y_train is None):
        raise ValueError("If you give X_train_path, also give y_train_path (and vice versa).")
    if (X_val is None) ^ (y_val is None):
        raise ValueError("If you give X_val_path, also give y_val_path (and vice versa).")

    # optional grid search
    if regression_type in ["ridge", "lasso", "elasticnet"] and do_grid_search and X_train is not None:
        if regression_type == "ridge":
            base = Ridge()
            grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
        elif regression_type == "lasso":
            base = Lasso(max_iter=10000)
            grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
        else:  # elasticnet
            base = ElasticNet(max_iter=10000)
            grid = {"alpha": [0.1, 1.0, 10.0, 100.0], "l1_ratio": [0.1, 0.5, 0.9]}
        gs = GridSearchCV(base, grid, scoring="neg_mean_squared_error", cv=5)
        gs.fit(X_train, y_train)
        print(f"[INFO] Best params: {gs.best_params_}")
        alpha    = gs.best_params_.get("alpha", alpha)
        l1_ratio = gs.best_params_.get("l1_ratio", l1_ratio)

    # fit + predict
    model = _pick_model(regression_type, alpha, l1_ratio)
    train_preds, val_preds = _fit_and_predict(model, X_train, y_train, X_val, y_val)

    # package data for downstream
    features_and_targets = {}
    if X_train is not None:
        features_and_targets["training"] = {"features": X_train, "targets": y_train}
    if X_val is not None:
        features_and_targets["validation"] = {"features": X_val, "targets": y_val}

    linear_obj = _organize_results(features_and_targets, train_preds, val_preds, model)
    linear_obj["param_names"] = list(exp_cfg["priors"].keys()) if "priors" in exp_cfg else []

    # errors
    # --- MSEs (overall + per-parameter) ----------------------------------------
    param_names = linear_obj["param_names"]
    rrmse = {"training": None, "validation": None,
            "training_mse": {}, "validation_mse": {}}

    def _fill_rrmse(y_true, y_pred, split_key):
        if y_true is None or y_pred is None:
            return
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        rrmse[split_key] = float(np.mean((yt - yp) ** 2))
        for i, name in enumerate(param_names):
            rrmse[f"{split_key}_mse"][name] = float(np.mean((yt[:, i] - yp[:, i]) ** 2))

    _fill_rrmse(y_train, train_preds, "training")
    _fill_rrmse(y_val,   val_preds,   "validation")

    # save artifacts
    with open(model_directory / f"linear_mdl_obj_{regression_type}.pkl", "wb") as f:
        pickle.dump(linear_obj, f)
    with open(model_directory / f"linear_model_error_{regression_type}.json", "w") as f:
        json.dump(rrmse, f, indent=4)
    joblib.dump(model, model_directory / f"linear_regression_model_{regression_type}.pkl")

    # plots
    visualizing_results(
        linear_obj,
        f"linear_results_{regression_type}",
        save_loc=model_directory,
        stages=[s for s in ["training", "validation"] if s in features_and_targets],
        color_shades=color_shades,
        main_colors=main_colors
    )
    print("[INFO] Linear model complete.")


# ---------- CLI ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser()

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

    # Model
    p.add_argument("--model_directory", type=str, default=None)
    p.add_argument("--regression_type", type=str, default="standard",
                   choices=["standard", "ridge", "lasso", "elasticnet"])
    p.add_argument("--alpha",     type=float, default=0.0)
    p.add_argument("--l1_ratio",  type=float, default=0.5)
    p.add_argument("--do_grid_search", action="store_true")

    args = p.parse_args()

    linear_evaluation(
        X_train_path=args.X_train_path,
        y_train_path=args.y_train_path,
        X_val_path=args.X_val_path,
        y_val_path=args.y_val_path,
        experiment_config_path=args.experiment_config_path,
        model_config_path=args.model_config_path,
        color_shades_path=args.color_shades_file,
        main_colors_path=args.main_colors_file,
        model_directory=args.model_directory,
        regression_type=args.regression_type,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        do_grid_search=args.do_grid_search
    )

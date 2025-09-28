#!/usr/bin/env python3
import argparse, json, os, sys, pickle, joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
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
            return obj  # KEEP DataFrame to preserve cols
        if isinstance(obj, pd.Series):
            return obj.to_frame()  # 2D
        if isinstance(obj, dict) and "features" in obj:
            return pd.DataFrame(np.asarray(obj["features"]))
        if isinstance(obj, dict) and "targets" in obj:
            return np.asarray(obj["targets"])
        return obj
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


# ---------- feature sanitation (Option A) ----------

def _to_dataframe(X):
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, pd.Series):
        return X.to_frame()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # anonymous columns if none
    return pd.DataFrame(arr)

def _sanitize_and_align(X_train_df: pd.DataFrame | None,
                        X_val_df:   pd.DataFrame | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """Replace ±Inf→NaN, align columns, drop columns that are all-NaN in both splits."""
    if X_train_df is not None:
        X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
    if X_val_df is not None:
        X_val_df = X_val_df.replace([np.inf, -np.inf], np.nan)

    # Align on intersection of columns if both provided
    if X_train_df is not None and X_val_df is not None:
        common_cols = X_train_df.columns.intersection(X_val_df.columns)
        X_train_df = X_train_df[common_cols]
        X_val_df   = X_val_df[common_cols]

        all_nan_both = X_train_df.isna().all(0) & X_val_df.isna().all(0)
        dropped_cols = list(common_cols[all_nan_both])
        keep_cols    = list(common_cols[~all_nan_both])
        X_train_df   = X_train_df[keep_cols]
        X_val_df     = X_val_df[keep_cols]
        return X_train_df, X_val_df, dropped_cols

    # Only one split present
    one = X_train_df if X_train_df is not None else X_val_df
    all_nan = one.isna().all(0)
    dropped_cols = list(one.columns[all_nan])
    keep_cols    = list(one.columns[~all_nan])
    if X_train_df is not None:
        X_train_df = X_train_df[keep_cols]
    if X_val_df is not None:
        X_val_df = X_val_df[keep_cols]
    return X_train_df, X_val_df, dropped_cols


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

    # ---- Option A: sanitize, align, drop all-NaN cols, then impute in-pipeline
    Xtr_df = _to_dataframe(X_train)
    Xva_df = _to_dataframe(X_val)
    Xtr_df, Xva_df, dropped = _sanitize_and_align(Xtr_df, Xva_df)

    if dropped:
        drop_log = model_directory / "dropped_feature_columns.txt"
        drop_log.write_text("\n".join(map(str, dropped)) + "\n")
        print(f"[INFO] Dropped {len(dropped)} all-NaN feature columns. Logged to {drop_log}")

    # Convert features to numpy; remaining sporadic NaNs will be imputed by the pipeline.
    X_train_np = None if Xtr_df is None else Xtr_df.to_numpy()
    X_val_np   = None if Xva_df is None else Xva_df.to_numpy()

    # Targets → numpy (ensure 2D for multi-output)
    def _to_targets(y):
        if y is None: return None
        if isinstance(y, (pd.Series, pd.DataFrame)):
            arr = y.to_numpy()
        else:
            arr = np.asarray(y)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
    y_train_np = _to_targets(y_train)
    y_val_np   = _to_targets(y_val)

    # optional grid search (tune the PIPELINE)
    model = None
    if regression_type in ["ridge", "lasso", "elasticnet"] and do_grid_search and X_train_np is not None:
        if regression_type == "ridge":
            pipe = make_pipeline(SimpleImputer(strategy="median"), Ridge())
            grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}
        elif regression_type == "lasso":
            pipe = make_pipeline(SimpleImputer(strategy="median"), Lasso(max_iter=10000))
            grid = {"lasso__alpha": [0.1, 1.0, 10.0, 100.0]}
        else:  # elasticnet
            pipe = make_pipeline(SimpleImputer(strategy="median"), ElasticNet(max_iter=10000))
            grid = {
                "elasticnet__alpha":   [0.1, 1.0, 10.0, 100.0],
                "elasticnet__l1_ratio": [0.1, 0.5, 0.9],
            }
        gs = GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", cv=5)
        gs.fit(X_train_np, y_train_np)
        print(f"[INFO] Best params: {gs.best_params_}")
        model = gs.best_estimator_

    # fit + predict (pipeline with imputer if no grid search or for 'standard')
    if model is None:
        base_model = _pick_model(regression_type, alpha, l1_ratio)
        model = make_pipeline(SimpleImputer(strategy="median"), base_model)

    train_preds, val_preds = _fit_and_predict(model, X_train_np, y_train_np, X_val_np, y_val_np)

    # package data for downstream
    features_and_targets = {}
    if X_train_np is not None:
        features_and_targets["training"] = {"features": X_train_np, "targets": y_train_np}
    if X_val_np is not None:
        features_and_targets["validation"] = {"features": X_val_np, "targets": y_val_np}

    linear_obj = _organize_results(features_and_targets, train_preds, val_preds, model)
    linear_obj["param_names"] = list(exp_cfg["priors"].keys()) if "priors" in exp_cfg else []

    # errors
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

    _fill_rrmse(y_train_np, train_preds, "training")
    _fill_rrmse(y_val_np,   val_preds,   "validation")

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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import joblib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as sk_mse

# ----------------------------
# constants
# ----------------------------
N_JOBS = 8                 # parallelize across targets
FIXED_RANDOM_STATE = 295   # RF randomness is fixed and NOT tuned

# Make project root importable (so "src" works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting_helpers import visualizing_results


# ------------------------ IO helpers ------------------------
def load_df_pickle(path: str) -> tuple[pd.DataFrame, list[str]]:
    obj = pickle.load(open(path, "rb"))

    if isinstance(obj, pd.DataFrame):
        return obj, obj.columns.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_frame(), [obj.name]

    if isinstance(obj, dict) and "features" in obj:
        arr = np.asarray(obj["features"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = obj.get("feature_names") or [f"feature_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=list(cols)), list(cols)

    if isinstance(obj, dict) and "targets" in obj:
        arr = np.asarray(obj["targets"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = obj.get("target_names") or [f"target_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=list(cols)), list(cols)

    arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = [f"col_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols), cols


def align_df_columns(df: pd.DataFrame, desired_order: list[str], name: str) -> pd.DataFrame:
    have = list(df.columns)
    if have == desired_order:
        return df
    if set(have) != set(desired_order):
        missing = [c for c in desired_order if c not in have]
        extra = [c for c in have if c not in desired_order]
        raise ValueError(
            f"[COLUMN ORDER ERROR] {name} columns don't match TRAIN columns.\n"
            f"Missing: {missing}\nExtra: {extra}\n"
            f"Train: {desired_order}\n{name}: {have}\n"
        )
    return df.loc[:, desired_order]


# ------------------------ plots ------------------------
def plot_feature_importances_grid(model, feature_names, target_names, out_path, top_k=20):
    n_outputs = len(model.estimators_)
    n_cols = 3
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
    axes = axes.flatten()

    for j, est in enumerate(model.estimators_):
        importances = est.feature_importances_
        order = np.argsort(importances)[::-1][:top_k]
        imp = importances[order]
        names = [feature_names[i] for i in order]

        ax = axes[j]
        ax.bar(range(len(imp)), imp)
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Importance")
        lab = target_names[j] if j < len(target_names) else f"Output {j}"
        ax.set_title(f"Feature Importance: {lab}")

    for ax in axes[n_outputs:]:
        ax.axis("off")

    fig.suptitle("Random Forest Feature Importances Across Outputs", fontsize=16)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# ------------------------ Optuna tuning ------------------------
def optuna_tune_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    n_trials: int,
    timeout_sec: int | None,
    seed: int,
) -> dict:
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError(
            "Optuna is not installed. Install with: pip install optuna"
        ) from e

    def objective(trial: "optuna.Trial") -> float:
        # Keep ranges reasonable for speed.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 400, step=20),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.7, 1.0]),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "bootstrap": True,  # needed if using max_samples
        }

        rf_single = RandomForestRegressor(
            **params,
            random_state=FIXED_RANDOM_STATE,
            n_jobs=1,  # IMPORTANT: avoid oversubscription
        )
        rf = MultiOutputRegressor(rf_single, n_jobs=N_JOBS)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_tune)
        return float(sk_mse(y_tune, preds))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    best = dict(study.best_params)
    best["bootstrap"] = True

    print(f"[OPTUNA] best_value (tune MSE) = {study.best_value:.6f}")
    print(f"[OPTUNA] best_params = {best}")
    return best


# ------------------------ main ------------------------
def main(args):
    # Load dataframes to preserve column names/order
    X_train_df, feat_names_train = load_df_pickle(args.X_train_path)
    y_train_df, targ_names_train = load_df_pickle(args.y_train_path)
    X_tune_df, _ = load_df_pickle(args.X_tune_path)
    y_tune_df, _ = load_df_pickle(args.y_tune_path)
    X_val_df, _ = load_df_pickle(args.X_val_path)
    y_val_df, _ = load_df_pickle(args.y_val_path)

    feature_order = list(feat_names_train)
    target_order = list(targ_names_train)

    X_tune_df = align_df_columns(X_tune_df, feature_order, "X_tune")
    X_val_df  = align_df_columns(X_val_df,  feature_order, "X_val")
    y_tune_df = align_df_columns(y_tune_df, target_order, "y_tune")
    y_val_df  = align_df_columns(y_val_df,  target_order, "y_val")

    X_train = X_train_df.values
    y_train = y_train_df.values
    X_tune  = X_tune_df.values
    y_tune  = y_tune_df.values
    X_val   = X_val_df.values
    y_val   = y_val_df.values

    color_shades = pickle.load(open(args.color_shades_file, "rb"))
    main_colors  = pickle.load(open(args.main_colors_file, "rb"))

    # Decide hyperparams
    user_specified = any(
        v is not None for v in [args.n_estimators, args.max_depth, args.min_samples_split]
    )

    if args.use_optuna and not user_specified:
        rf_params = optuna_tune_rf(
            X_train, y_train, X_tune, y_tune,
            n_trials=args.n_trials,
            timeout_sec=args.optuna_timeout,
            seed=args.optuna_seed,
        )
    else:
        # Manual / default params (bootstrap hardcoded True)
        rf_params = dict(
            n_estimators=args.n_estimators if args.n_estimators is not None else 200,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split if args.min_samples_split is not None else 2,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            max_samples=args.max_samples,
            bootstrap=True,
        )

    # Final fit data choice
    if args.final_fit == "train_plus_tune":
        X_fit = np.vstack([X_train, X_tune])
        y_fit = np.vstack([y_train, y_tune])
        fit_label = "TRAIN+TUNE"
    else:
        X_fit, y_fit = X_train, y_train
        fit_label = "TRAIN"

    print(f"[INFO] Final fit on: {fit_label} X={X_fit.shape} y={y_fit.shape}")
    print(f"[INFO] RF params: {rf_params}")
    print(f"[INFO] Parallelism: MultiOutput n_jobs={N_JOBS}, RF n_jobs=1, rf_random_state={FIXED_RANDOM_STATE}")

    rf_single = RandomForestRegressor(
        **rf_params,
        random_state=FIXED_RANDOM_STATE,
        n_jobs=1,
    )
    rf = MultiOutputRegressor(rf_single, n_jobs=N_JOBS)
    rf.fit(X_fit, y_fit)

    # Predictions for reporting
    tr_pred = rf.predict(X_train)
    va_pred = rf.predict(X_val)

    mse_dict = {
        "training": float(np.mean((y_train - tr_pred) ** 2)),
        "validation": float(np.mean((y_val - va_pred) ** 2)),
        "training_mse": {},
        "validation_mse": {}
    }
    for i, p in enumerate(target_order):
        mse_dict["training_mse"][p] = float(np.mean((y_train[:, i] - tr_pred[:, i]) ** 2))
        mse_dict["validation_mse"][p] = float(np.mean((y_val[:, i] - va_pred[:, i]) ** 2))

    rf_obj = {
        "model": rf,
        "training": {"predictions": tr_pred, "targets": y_train},
        "validation": {"predictions": va_pred, "targets": y_val},
        "param_names": target_order,
        "target_order": target_order,
        "feature_names": feature_order,
        "rf_params": rf_params,
        "final_fit": fit_label,
    }

    out_dir = Path(args.model_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "random_forest_mdl_obj.pkl", "wb") as f:
        pickle.dump(rf_obj, f)
    with open(out_dir / "random_forest_model_error.json", "w") as f:
        json.dump(mse_dict, f, indent=4)
    joblib.dump(rf, out_dir / "random_forest_model.pkl")

    # plots
    visualizing_results(
        rf_obj,
        "random_forest_results",
        save_loc=out_dir,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )
    plot_feature_importances_grid(
        model=rf,
        feature_names=feature_order,
        target_names=target_order,
        out_path=out_dir / "random_forest_feature_importances.png",
        top_k=args.top_k_importances,
    )

    with open(out_dir / "random_forest_best_params.json", "w") as f:
        json.dump(rf_params, f, indent=2)

    print("[INFO] Random Forest complete. Artifacts saved to:", out_dir)


# ------------------------ CLI ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--X_train_path", type=str, required=True)
    p.add_argument("--y_train_path", type=str, required=True)
    p.add_argument("--X_tune_path", type=str, required=True)
    p.add_argument("--y_tune_path", type=str, required=True)
    p.add_argument("--X_val_path", type=str, required=True)
    p.add_argument("--y_val_path", type=str, required=True)

    p.add_argument("--color_shades_file", type=str, required=True)
    p.add_argument("--main_colors_file", type=str, required=True)

    p.add_argument("--model_directory", type=str, required=True)

    p.add_argument(
        "--final_fit",
        choices=["train_only", "train_plus_tune"],
        default="train_plus_tune",
    )

    # If any of these are provided, Optuna is bypassed.
    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=None)

    # Extra knobs
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument(
        "--max_features",
        type=str,
        default="0.7",
        help='Can be "sqrt" or a float like 0.7',
    )
    p.add_argument("--max_samples", type=float, default=0.9)

    # Optuna
    p.add_argument("--use_optuna", action="store_true")
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--optuna_timeout", type=int, default=None)
    p.add_argument("--optuna_seed", type=int, default=295)

    # Plot
    p.add_argument("--top_k_importances", type=int, default=20)

    # (Optional) config paths passed by Snakemake for provenance/logging
    p.add_argument("--experiment_config_path", type=str, default=None)
    p.add_argument("--model_config_path", type=str, default=None)


    args = p.parse_args()

    # parse max_features string
    if args.max_features == "sqrt":
        pass
    else:
        args.max_features = float(args.max_features)

    main(args)

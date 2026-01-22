#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import joblib
import time
import inspect
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as sk_mse

import xgboost as xgb
from xgboost import XGBRegressor

# ----- allow project root imports (for visualizing_results) -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting_helpers import visualizing_results  # your existing function


# ------------------------ IO helpers ------------------------
def load_df_pickle(path: str) -> Tuple[pd.DataFrame, List[str]]:
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


def align_df_columns(
    df: pd.DataFrame, desired_order: List[str], name: str
) -> pd.DataFrame:
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


def _default_model_dir(exp_cfg: Dict[str, Any]) -> Path:
    return Path(f"experiments/{exp_cfg['demographic_model']}/modeling/xgboost")


def _detect_outer_jobs(user: Optional[int], fallback: int = 1) -> int:
    if user is not None:
        return int(user)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or fallback))


def _set_thread_env_sane_defaults() -> None:
    # avoid oversubscription from BLAS / OpenMP
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ------------------------ plotting ------------------------
def plot_feature_importances_grid(
    model: MultiOutputRegressor,
    feature_names: List[str],
    target_names: List[str],
    out_path: Path,
    top_k: int = 20,
) -> None:
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise ValueError("MultiOutputRegressor has no estimators_. Did fit() succeed?")

    n_outputs = len(estimators)
    if n_outputs == 0:
        raise ValueError("No estimators found in model.estimators_.")

    n_cols = 3
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16, 5 * n_rows), constrained_layout=True
    )
    axes = np.array(axes).reshape(-1)

    for j, est in enumerate(estimators):
        importances = getattr(est, "feature_importances_", None)
        ax = axes[j]

        if importances is None:
            ax.set_title(
                f"{target_names[j] if j < len(target_names) else f'Output {j}'}\n(no feature_importances_)"
            )
            ax.axis("off")
            continue

        order = np.argsort(importances)[::-1]
        if top_k is not None:
            order = order[:top_k]
        imp = importances[order]
        names = [feature_names[i] for i in order]

        ax.bar(range(len(imp)), imp)
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Importance")
        lab = target_names[j] if j < len(target_names) else f"Output {j}"
        ax.set_title(f"Feature Importances: {lab}")

    for ax in axes[n_outputs:]:
        ax.axis("off")

    fig.suptitle("XGBoost Feature Importances Across Outputs", fontsize=16)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# ------------------------ XGB version-adaptive early stopping ------------------------
def _fit_supports_kwarg(est: XGBRegressor, kw: str) -> bool:
    try:
        sig = inspect.signature(est.fit)
        return kw in sig.parameters
    except Exception:
        # extremely defensive: if we can't introspect, assume unsupported
        return False


def _fit_one_target(
    *,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train_1d: np.ndarray,
    X_tune: Optional[np.ndarray],
    y_tune_1d: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> XGBRegressor:
    """
    Fit a single XGBRegressor. If early_stopping_rounds is set, uses whichever
    API is supported by the installed xgboost:
      - callbacks=... (newer)
      - early_stopping_rounds=... (older)
    """
    est = XGBRegressor(**params)

    if early_stopping_rounds is None:
        est.fit(X_train, y_train_1d)
        return est

    if X_tune is None or y_tune_1d is None:
        raise ValueError("early_stopping_rounds requires TUNE data (X_tune/y_tune).")

    eval_set = [(X_tune, y_tune_1d)]

    # Prefer callbacks if supported
    if _fit_supports_kwarg(est, "callbacks"):
        # callbacks path (works on some versions)
        callbacks = [
            xgb.callback.EarlyStopping(
                rounds=int(early_stopping_rounds), save_best=True
            )
        ]
        est.fit(
            X_train,
            y_train_1d,
            eval_set=eval_set,
            verbose=False,
            callbacks=callbacks,
        )
        return est

    # Fallback: early_stopping_rounds kw
    fit_kwargs: Dict[str, Any] = {
        "eval_set": eval_set,
        "verbose": False,
    }
    if _fit_supports_kwarg(est, "early_stopping_rounds"):
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    # Some versions want eval_metric passed to fit (even if also in params).
    if _fit_supports_kwarg(est, "eval_metric") and ("eval_metric" in params):
        fit_kwargs["eval_metric"] = params["eval_metric"]

    est.fit(X_train, y_train_1d, **fit_kwargs)
    return est


def _fit_multioutput(
    *,
    base_params: Dict[str, Any],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    outer_jobs: int,
    X_tune: Optional[np.ndarray],
    y_tune: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
    target_names: List[str],
) -> MultiOutputRegressor:
    """
    - Without early stopping: use sklearn MultiOutputRegressor parallelism.
    - With early stopping: fit one estimator per target (early stopping needs per-target eval_set).
    """
    if early_stopping_rounds is None:
        base = XGBRegressor(**base_params)
        model = MultiOutputRegressor(base, n_jobs=int(outer_jobs))
        model.fit(X_fit, y_fit)
        return model

    est_list: List[XGBRegressor] = []
    n_out = y_fit.shape[1]

    for i in range(n_out):
        lab = target_names[i] if i < len(target_names) else f"target_{i}"
        print(f"[FIT+ES] target {i+1}/{n_out}: {lab}")
        t0 = time.perf_counter()

        est = _fit_one_target(
            params=base_params,
            X_train=X_fit,
            y_train_1d=y_fit[:, i],
            X_tune=X_tune,
            y_tune_1d=None if y_tune is None else y_tune[:, i],
            early_stopping_rounds=early_stopping_rounds,
        )

        dt = time.perf_counter() - t0
        best_iter = getattr(est, "best_iteration", None)
        best_ntree = getattr(est, "best_ntree_limit", None)
        print(
            f"[FIT+ES]   elapsed={dt:.1f}s best_iteration={best_iter} best_ntree_limit={best_ntree}"
        )
        est_list.append(est)

    # Wrap for consistent downstream API (.predict)
    wrapper = MultiOutputRegressor(XGBRegressor(**base_params), n_jobs=int(outer_jobs))
    wrapper.estimators_ = est_list
    return wrapper


# ------------------------ Optuna tuning ------------------------
def optuna_tune_xgb_multioutput(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    n_trials: int,
    timeout_sec: Optional[int],
    seed: int,
    outer_jobs: int,
    xgb_n_jobs: int,
    tree_method: str,
    early_stopping_rounds: Optional[int],
) -> Dict[str, Any]:
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError(
            "Optuna is not installed. Install with: pip install optuna"
        ) from e

    # minimize tune MSE over all outputs
    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": tree_method,
            "n_jobs": int(xgb_n_jobs),
            "random_state": int(seed),
            "verbosity": 0,
            # search space
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=200),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        }

        model = _fit_multioutput(
            base_params=params,
            X_fit=X_train,
            y_fit=y_train,
            outer_jobs=outer_jobs,
            X_tune=X_tune,
            y_tune=y_tune,
            early_stopping_rounds=early_stopping_rounds,
            target_names=[f"target_{i}" for i in range(y_train.shape[1])],
        )
        preds = model.predict(X_tune)
        return float(sk_mse(y_tune, preds))

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), timeout=timeout_sec)

    best = dict(study.best_params)
    best.update(
        {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": tree_method,
            "n_jobs": int(xgb_n_jobs),
            "random_state": int(seed),
            "verbosity": 2,
        }
    )

    print(f"[OPTUNA] best_value (tune MSE) = {study.best_value:.6f}")
    print(f"[OPTUNA] best_params = {best}")
    return best


# ------------------------ main ------------------------
def main(args: argparse.Namespace) -> None:
    _set_thread_env_sane_defaults()

    with open(args.experiment_config_path, "r") as f:
        exp_cfg = json.load(f)

    if args.model_config_path:
        with open(args.model_config_path, "r") as f:
            _ = yaml.safe_load(f)

    out_dir = (
        Path(args.model_directory)
        if args.model_directory
        else _default_model_dir(exp_cfg)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Outputs → {out_dir}")
    print(f"[INFO] xgboost version = {xgb.__version__}")

    color_shades = pickle.load(open(args.color_shades_file, "rb"))
    main_colors = pickle.load(open(args.main_colors_file, "rb"))

    # load data
    X_train_df, feat_order = load_df_pickle(args.X_train_path)
    y_train_df, targ_order = load_df_pickle(args.y_train_path)
    X_tune_df, _ = load_df_pickle(args.X_tune_path)
    y_tune_df, _ = load_df_pickle(args.y_tune_path)
    X_val_df, _ = load_df_pickle(args.X_val_path)
    y_val_df, _ = load_df_pickle(args.y_val_path)

    # align columns
    X_tune_df = align_df_columns(X_tune_df, feat_order, "X_tune")
    X_val_df = align_df_columns(X_val_df, feat_order, "X_val")
    y_tune_df = align_df_columns(y_tune_df, targ_order, "y_tune")
    y_val_df = align_df_columns(y_val_df, targ_order, "y_val")

    X_train = X_train_df.values
    y_train = y_train_df.values
    X_tune = X_tune_df.values
    y_tune = y_tune_df.values
    X_val = X_val_df.values
    y_val = y_val_df.values

    outer_jobs = _detect_outer_jobs(args.outer_n_jobs)
    xgb_n_jobs = int(args.xgb_n_jobs)
    print(
        f"[INFO] Parallelism: MultiOutput n_jobs={outer_jobs}, XGB n_jobs={xgb_n_jobs}"
    )

    # If any manual hyperparam is set -> bypass optuna
    user_specified = any(
        v is not None
        for v in [
            args.n_estimators,
            args.max_depth,
            args.learning_rate,
            args.subsample,
            args.colsample_bytree,
            args.min_child_weight,
            args.reg_lambda,
            args.reg_alpha,
        ]
    )

    if args.use_optuna and (not user_specified):
        base_params = optuna_tune_xgb_multioutput(
            X_train,
            y_train,
            X_tune,
            y_tune,
            n_trials=args.n_trials,
            timeout_sec=args.optuna_timeout,
            seed=args.optuna_seed,
            outer_jobs=outer_jobs,
            xgb_n_jobs=xgb_n_jobs,
            tree_method=args.tree_method,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        optuna_used = True
    else:
        base_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": args.tree_method,
            "n_jobs": int(xgb_n_jobs),
            "random_state": int(args.optuna_seed),
            "verbosity": 2,
            "n_estimators": int(
                args.n_estimators if args.n_estimators is not None else 600
            ),
            "max_depth": int(args.max_depth if args.max_depth is not None else 5),
            "learning_rate": float(
                args.learning_rate if args.learning_rate is not None else 0.05
            ),
            "subsample": float(args.subsample if args.subsample is not None else 0.8),
            "colsample_bytree": float(
                args.colsample_bytree if args.colsample_bytree is not None else 0.8
            ),
            "min_child_weight": float(
                args.min_child_weight if args.min_child_weight is not None else 3.0
            ),
            "reg_lambda": float(
                args.reg_lambda if args.reg_lambda is not None else 5.0
            ),
            "reg_alpha": float(args.reg_alpha if args.reg_alpha is not None else 0.0),
        }
        optuna_used = False

    # final fit dataset choice
    if args.final_fit == "train_plus_tune":
        X_fit = np.vstack([X_train, X_tune])
        y_fit = np.vstack([y_train, y_tune])
        fit_label = "TRAIN+TUNE"
    else:
        X_fit, y_fit = X_train, y_train
        fit_label = "TRAIN"

    # With early stopping, do NOT train on TRAIN+TUNE
    if args.early_stopping_rounds is not None and fit_label != "TRAIN":
        print(
            "[WARN] early_stopping_rounds set: forcing final fit on TRAIN (not TRAIN+TUNE)."
        )
        X_fit, y_fit = X_train, y_train
        fit_label = "TRAIN"

    print(f"[INFO] Final fit on: {fit_label} X={X_fit.shape} y={y_fit.shape}")
    print(f"[INFO] early_stopping_rounds={args.early_stopping_rounds}")
    print(f"[INFO] base_params={base_params}")

    # fit model
    t0 = time.perf_counter()
    model = _fit_multioutput(
        base_params=base_params,
        X_fit=X_fit,
        y_fit=y_fit,
        outer_jobs=outer_jobs,
        X_tune=X_tune,
        y_tune=y_tune,
        early_stopping_rounds=args.early_stopping_rounds,
        target_names=list(targ_order),
    )
    print(f"[INFO] Fit elapsed: {time.perf_counter() - t0:.1f}s")

    # predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    xgb_obj = {
        "model": model,
        "training": {"predictions": train_preds, "targets": y_train},
        "validation": {"predictions": val_preds, "targets": y_val},
        "param_names": list(targ_order),
        "target_order": list(targ_order),
        "feature_names": list(feat_order),
        "feature_order": list(feat_order),
        "xgb_params": dict(base_params),
        "final_fit": fit_label,
        "optuna_used": bool(optuna_used),
        "optuna_seed": int(args.optuna_seed),
        "outer_n_jobs": int(outer_jobs),
        "xgb_n_jobs": int(xgb_n_jobs),
        "tree_method": str(args.tree_method),
        "early_stopping_rounds": (
            None
            if args.early_stopping_rounds is None
            else int(args.early_stopping_rounds)
        ),
        "experiment_config_path": args.experiment_config_path,
        "model_config_path": args.model_config_path,
        "xgboost_version": xgb.__version__,
    }

    err: Dict[str, Any] = {
        "training": float(np.mean((y_train - train_preds) ** 2)),
        "validation": float(np.mean((y_val - val_preds) ** 2)),
        "training_mse": {},
        "validation_mse": {},
        "target_order": list(targ_order),
        "feature_order": list(feat_order),
        "xgb_params": dict(base_params),
        "final_fit": fit_label,
        "optuna_used": bool(optuna_used),
        "xgboost_version": xgb.__version__,
    }
    for i, pname in enumerate(targ_order):
        err["training_mse"][pname] = float(
            np.mean((y_train[:, i] - train_preds[:, i]) ** 2)
        )
        err["validation_mse"][pname] = float(
            np.mean((y_val[:, i] - val_preds[:, i]) ** 2)
        )

    # save
    with open(out_dir / "xgb_mdl_obj.pkl", "wb") as f:
        pickle.dump(xgb_obj, f)
    with open(out_dir / "xgb_model_error.json", "w") as f:
        json.dump(err, f, indent=4)
    joblib.dump(model, out_dir / "xgb_model.pkl")
    with open(out_dir / "xgb_best_params.json", "w") as f:
        json.dump(dict(base_params), f, indent=2)

    # plots
    visualizing_results(
        xgb_obj,
        analysis="xgb_results",
        save_loc=out_dir,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )
    plot_feature_importances_grid(
        model=model,
        feature_names=list(feat_order),
        target_names=list(targ_order),
        out_path=out_dir / "xgb_feature_importances.png",
        top_k=int(args.top_k_features_plot),
    )

    print("[INFO] XGBoost complete. Artifacts saved to:", out_dir)


# ------------------------ CLI ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="XGBoost multi-output evaluation script (version-adaptive early stopping)"
    )

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

    # Final fit
    p.add_argument(
        "--final_fit",
        choices=["train_only", "train_plus_tune"],
        default="train_plus_tune",
    )

    # Manual hyperparams (if any set, Optuna is bypassed)
    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--colsample_bytree", type=float, default=None)
    p.add_argument("--min_child_weight", type=float, default=None)
    p.add_argument("--reg_lambda", type=float, default=None)
    p.add_argument("--reg_alpha", type=float, default=None)

    # Optuna
    p.add_argument("--use_optuna", action="store_true")
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--optuna_timeout", type=int, default=None)
    p.add_argument("--optuna_seed", type=int, default=295)

    # Perf controls
    p.add_argument("--outer_n_jobs", type=int, default=None)
    p.add_argument("--xgb_n_jobs", type=int, default=1)
    p.add_argument("--tree_method", type=str, default="hist")

    # Early stopping (handled adaptively across xgboost versions)
    p.add_argument("--early_stopping_rounds", type=int, default=None)

    # Plot
    p.add_argument("--top_k_features_plot", type=int, default=20)

    args = p.parse_args()
    main(args)

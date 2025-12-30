#!/usr/bin/env python3
import argparse, json, pickle, joblib, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# constants (hardcoded)
# ----------------------------
N_JOBS = 8
FIXED_RANDOM_STATE = 295

# Make project root importable (so "src" works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as sk_mse

from src.plotting_helpers import visualizing_results  # keep using your plotting util


# ------------------------ helpers ------------------------


def load_df_pickle(path):
    """
    Load a pkl that contains a DataFrame/Series, or ndarray/dict.
    Return (df, colnames). If we don't know names, we synthesize them.
    """
    obj = pickle.load(open(path, "rb"))

    if isinstance(obj, pd.DataFrame):
        return obj, obj.columns.tolist()

    if isinstance(obj, pd.Series):
        return obj.to_frame(), [obj.name]

    # fallback: ndarray or dict
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

    # generic ndarray
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


def plot_feature_importances_grid(
    model, feature_names, target_names, out_path, max_num_features=None
):
    """
    model is a MultiOutputRegressor with a list of single-output estimators.
    Creates a grid of feature-importance plots, one per target.
    """
    n_outputs = len(model.estimators_)
    n_cols = 3
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True
    )
    axes = axes.flatten()

    for j, est in enumerate(model.estimators_):
        importances = est.feature_importances_
        order = np.argsort(importances)[::-1]
        if max_num_features is not None:
            order = order[:max_num_features]

        imp = importances[order]
        names = [feature_names[i] for i in order]

        ax = axes[j]
        ax.bar(range(len(imp)), imp)
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        title = (
            f"Feature Importance: {target_names[j]}"
            if j < len(target_names)
            else f"Output {j}"
        )
        ax.set_title(title)

    # hide unused axes
    for ax in axes[n_outputs:]:
        ax.axis("off")

    fig.suptitle("Random Forest Feature Importances Across Outputs", fontsize=16)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def tune_rf_on_tune(
    X_train,
    y_train,
    X_tune,
    y_tune,
    n_iter=10,
    base_random_state=295,
):
    """
    Manual random search over RF hyperparameters.

    IMPORTANT SPEED/CPU RULE:
      - RF n_jobs=1
      - MultiOutputRegressor n_jobs=N_JOBS
    so we parallelize ACROSS outputs, not inside each forest.

    random_state of the RF is FIXED (FIXED_RANDOM_STATE).
    base_random_state only controls the sampling of hyperparameters.
    """
    rng = np.random.RandomState(base_random_state)

    n_estimators_grid = [50, 100, 200, 300, 500]
    max_depth_grid = [None, 10, 20, 30, 40]
    min_samples_split_grid = [2, 5, 10, 15, 20]

    best_mse = np.inf
    best_params = None

    for i in range(n_iter):
        n_estimators = int(rng.choice(n_estimators_grid))
        max_depth = rng.choice(max_depth_grid)
        min_samples_split = int(rng.choice(min_samples_split_grid))

        rf_single = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,

            min_samples_leaf=1,
            max_features=0.7,
            max_samples=0.9,
            bootstrap=True,
            random_state=FIXED_RANDOM_STATE,
            n_jobs=1,
        )
        rf = MultiOutputRegressor(
            rf_single,
            n_jobs=N_JOBS
        )
        rf.fit(X_train, y_train)

        preds_tune = rf.predict(X_tune)
        mse_tune = sk_mse(y_tune, preds_tune)

        print(
            f"[RANDOM SEARCH] iter={i+1}/{n_iter} "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, rf_random_state={FIXED_RANDOM_STATE} "
            f"-> tune MSE={mse_tune:.6f}"
        )

        if mse_tune < best_mse:
            best_mse = mse_tune
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            }

    print(f"[INFO] Best params from TUNE set: {best_params}, tune MSE={best_mse:.6f}")
    return best_params


# ------------------------ main functional pipeline ------------------------


def main(args):
    # Load experiment config (only as fallback)
    exp_cfg = json.load(open(args.experiment_config_path))
    cfg_param_names = exp_cfg.get(
        "parameters_to_estimate", list(exp_cfg.get("priors", {}).keys())
    )

    # Optional model cfg (not critical here)
    if args.model_config_path:
        with open(args.model_config_path) as f:
            _ = yaml.safe_load(f)

    # Load colors
    color_shades = pickle.load(open(args.color_shades_file, "rb"))
    main_colors = pickle.load(open(args.main_colors_file, "rb"))

    # Load data as dataframes to keep names
    X_train_df, feat_names_train = load_df_pickle(args.X_train_path)
    y_train_df, targ_names_train = load_df_pickle(args.y_train_path)

    X_tune_df, _ = load_df_pickle(args.X_tune_path)
    y_tune_df, _ = load_df_pickle(args.y_tune_path)

    X_val_df, _ = load_df_pickle(args.X_val_path)
    y_val_df, _ = load_df_pickle(args.y_val_path)

    # ---------------------------
    # SOURCE OF TRUTH: TRAIN order
    # ---------------------------
    feature_order = list(feat_names_train)
    target_order = list(targ_names_train)

    # Align tune/val to TRAIN order (both X and y)
    X_tune_df = _align_df_columns(X_tune_df, feature_order, "X_tune")
    X_val_df  = _align_df_columns(X_val_df,  feature_order, "X_val")

    y_tune_df = _align_df_columns(y_tune_df, target_order, "y_tune")
    y_val_df  = _align_df_columns(y_val_df,  target_order, "y_val")

    # What we will report everywhere (like XGBoost)
    param_names = target_order if target_order else (cfg_param_names or targ_names_train)

    print("[INFO] Feature order used:")
    print("  " + ", ".join(feature_order[:10]) + (" ..." if len(feature_order) > 10 else ""))
    print("[INFO] Target order used:")
    print("  " + ", ".join(param_names))

    # Convert to arrays
    X_train = X_train_df.values
    y_train = y_train_df.values
    X_tune = X_tune_df.values
    y_tune = y_tune_df.values
    X_val = X_val_df.values
    y_val = y_val_df.values

    # Basic sanity checks for tune
    if X_tune is None or y_tune is None:
        raise ValueError(
            "X_tune and y_tune must be provided (via --X_tune_path / --y_tune_path) "
            "for hyperparameter tuning."
        )

    # Choose or search hyperparams
    user_specified = any(
        v is not None
        for v in [
            args.n_estimators,
            args.max_depth,
            args.min_samples_split,
        ]
    )

    if args.do_random_search or not user_specified:
        # base_random_state here only affects which hyperparams you try, not the RF randomness
        base_rs = FIXED_RANDOM_STATE
        best = tune_rf_on_tune(
            X_train,
            y_train,
            X_tune,
            y_tune,
            n_iter=args.n_iter,
            base_random_state=base_rs,
        )
        n_estimators = best.get("n_estimators", 200)
        max_depth = best.get("max_depth", None)
        min_samples_split = best.get("min_samples_split", 2)
    else:
        n_estimators = args.n_estimators or 200
        max_depth = args.max_depth
        min_samples_split = args.min_samples_split or 2

    print(
        f"[INFO] Final RF hyperparams: n_estimators={n_estimators}, "
        f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
        f"rf_random_state={FIXED_RANDOM_STATE}, N_JOBS={N_JOBS}"
    )

    # ---------------------------
    # FINAL FIT (speed-correct)
    # ---------------------------
    rf_single = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,

        min_samples_leaf=1,
        max_features=0.7,
        max_samples=0.9,
        bootstrap=True,
        random_state=FIXED_RANDOM_STATE,
        n_jobs=1,
    )
    rf = MultiOutputRegressor(
        rf_single,
        n_jobs=N_JOBS
    )
    rf.fit(X_train, y_train)

    tr_pred = rf.predict(X_train)
    va_pred = rf.predict(X_val)

    # MSE dict
    mse_dict = {
        "training": float(np.mean((y_train - tr_pred) ** 2)),
        "validation": float(np.mean((y_val - va_pred) ** 2)),
        "training_mse": {},
        "validation_mse": {},
        # "target_order": list(param_names),
        # "feature_order": list(feature_order),
        # "rf_random_state": FIXED_RANDOM_STATE,
        # "n_jobs_multioutput": N_JOBS,
        # "n_jobs_rf_single": 1,
    }

    for i, p in enumerate(param_names):
        mse_dict["training_mse"][p] = float(np.mean((y_train[:, i] - tr_pred[:, i]) ** 2))
        mse_dict["validation_mse"][p] = float(np.mean((y_val[:, i] - va_pred[:, i]) ** 2))

    # Prepare object like your old wrapper
    rf_obj = {
        "model": rf,
        "training": {"predictions": tr_pred, "targets": y_train},
        "validation": {"predictions": va_pred, "targets": y_val},
        "param_names": list(param_names),
        "target_order": list(param_names),
        "feature_names": list(feature_order),
    }

    out_dir = Path(args.model_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save rf_obj, mse, model
    with open(out_dir / "random_forest_mdl_obj.pkl", "wb") as f:
        pickle.dump(rf_obj, f)

    with open(out_dir / "random_forest_model_error.json", "w") as f:
        json.dump(mse_dict, f, indent=4)

    joblib.dump(rf, out_dir / "random_forest_model.pkl")

    # Plot predictions (use your plotting util)
    visualizing_results(
        rf_obj,
        "random_forest_results",
        save_loc=out_dir,
        stages=["training", "validation"],
        color_shades=color_shades,
        main_colors=main_colors,
    )

    # Feature importance per target (grid)
    plot_feature_importances_grid(
        model=rf,
        feature_names=list(feature_order),
        target_names=list(param_names),
        out_path=out_dir / "random_forest_feature_importances.png",
        max_num_features=20,
    )

    print("[INFO] Random Forest complete. Artifacts saved to:", out_dir)


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()

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
    p.add_argument("--model_directory", type=str, required=True)

    # RF hyperparams (all optional)
    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=None)

    # NOTE: we keep this arg for backward compatibility, but we DO NOT use it.
    # (You asked to fix random_state to 295 and not optimize it.)
    p.add_argument("--random_state", type=int, default=None)

    p.add_argument("--do_random_search", action="store_true")
    p.add_argument("--n_iter", type=int, default=10)

    args = p.parse_args()
    main(args)

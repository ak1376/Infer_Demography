#!/usr/bin/env python3
import argparse, json, pickle, joblib, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make project root importable (so "src" works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as sk_mse


# ------------------------ helpers ------------------------


def load_df_pickle(path):
    """Load a pkl that contains a DataFrame (or array). Return df and columns."""
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, pd.DataFrame):
        return obj, obj.columns.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_frame(), [obj.name]
    # fallback: ndarray or dict
    if isinstance(obj, dict) and "features" in obj:
        arr = np.asarray(obj["features"])
        cols = [f"feature_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        return df, cols
    if isinstance(obj, dict) and "targets" in obj:
        arr = np.asarray(obj["targets"])
        cols = [f"target_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        return df, cols
    # generic ndarray
    arr = np.asarray(obj)
    cols = [f"col_{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    return df, cols


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
    n_iter=20,
    base_random_state=42,
):
    """
    Manual random search over RF hyperparameters.

    For each sampled hyperparameter set:
      - fit RF on TRAIN
      - evaluate MSE on TUNE
    Return dict of best params.
    """
    rng = np.random.RandomState(base_random_state)

    # Discrete grids (same as before)
    n_estimators_grid = [50, 100, 200, 300, 500]
    max_depth_grid = [None, 10, 20, 30, 40]
    min_samples_split_grid = [2, 5, 10, 15, 20]
    random_state_grid = [42, 123, 2023, 295]

    best_mse = np.inf
    best_params = None

    for i in range(n_iter):
        n_estimators = rng.choice(n_estimators_grid)
        max_depth = rng.choice(max_depth_grid)
        min_samples_split = rng.choice(min_samples_split_grid)
        random_state = rng.choice(random_state_grid)

        rf_single = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        rf = MultiOutputRegressor(rf_single)
        rf.fit(X_train, y_train)

        preds_tune = rf.predict(X_tune)
        mse_tune = sk_mse(y_tune, preds_tune)

        print(
            f"[RANDOM SEARCH] iter={i+1}/{n_iter} "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, random_state={random_state} "
            f"-> tune MSE={mse_tune:.6f}"
        )

        if mse_tune < best_mse:
            best_mse = mse_tune
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "random_state": random_state,
            }

    print(f"[INFO] Best params from TUNE set: {best_params}, tune MSE={best_mse:.6f}")
    return best_params


# ------------------------ main functional pipeline ------------------------


def main(args):
    # Load experiment config (for param names)
    exp_cfg = json.load(open(args.experiment_config_path))
    param_names = exp_cfg.get(
        "parameters_to_estimate", list(exp_cfg.get("priors", {}).keys())
    )

    # Model cfg (not critical here, but optional)
    model_cfg = {}
    if args.model_config_path:
        with open(args.model_config_path) as f:
            model_cfg = yaml.safe_load(f)

    # Load colors
    color_shades = pickle.load(open(args.color_shades_file, "rb"))
    main_colors = pickle.load(open(args.main_colors_file, "rb"))

    # Load data as dataframes to keep names
    X_train_df, feat_names = load_df_pickle(args.X_train_path)
    y_train_df, targ_names = load_df_pickle(args.y_train_path)
    X_tune_df, _ = load_df_pickle(args.X_tune_path)
    y_tune_df, _ = load_df_pickle(args.y_tune_path)
    X_val_df, _ = load_df_pickle(args.X_val_path)
    y_val_df, _ = load_df_pickle(args.y_val_path)

    # Fall back if config names differ
    if not param_names:
        param_names = targ_names

    # Convert to arrays
    X_train = X_train_df.values
    y_train = y_train_df.values
    X_tune = X_tune_df.values
    y_tune = y_tune_df.values
    X_val = X_val_df.values
    y_val = y_val_df.values

    # Basic sanity checks for tune
    if (X_tune is None) or (y_tune is None):
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
            args.random_state,
        ]
    )

    if args.do_random_search or not user_specified:
        base_rs = args.random_state if args.random_state is not None else 42
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
        random_state = best.get("random_state", 42)
    else:
        n_estimators = args.n_estimators or 200
        max_depth = args.max_depth
        min_samples_split = args.min_samples_split or 2
        random_state = args.random_state or 42

    print(
        f"[INFO] Final RF hyperparams: n_estimators={n_estimators}, "
        f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
        f"random_state={random_state}"
    )

    # Build and fit multi-output RF on TRAIN (only)
    rf_single = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    rf = MultiOutputRegressor(rf_single)
    rf.fit(X_train, y_train)

    tr_pred = rf.predict(X_train)
    va_pred = rf.predict(X_val)

    # MSE dict
    mse_dict = {
        "training": None,
        "validation": None,
        "training_mse": {},
        "validation_mse": {},
    }

    mse_dict["training"] = float(np.mean((y_train - tr_pred) ** 2))
    for i, p in enumerate(param_names):
        mse_dict["training_mse"][p] = float(
            np.mean((y_train[:, i] - tr_pred[:, i]) ** 2)
        )

    mse_dict["validation"] = float(np.mean((y_val - va_pred) ** 2))
    for i, p in enumerate(param_names):
        mse_dict["validation_mse"][p] = float(
            np.mean((y_val[:, i] - va_pred[:, i]) ** 2)
        )

    # Prepare object like your old wrapper
    rf_obj = {
        "model": rf,
        "training": {
            "predictions": tr_pred,
            "targets": y_train,
        },
        "validation": {
            "predictions": va_pred,
            "targets": y_val,
        },
        "param_names": param_names,
        "feature_names": feat_names,
        "target_names": targ_names,
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
    from src.plotting_helpers import visualizing_results

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
        feature_names=feat_names,
        target_names=targ_names,
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
    p.add_argument("--random_state", type=int, default=None)
    p.add_argument("--do_random_search", action="store_true")
    p.add_argument("--n_iter", type=int, default=20)

    args = p.parse_args()
    main(args)

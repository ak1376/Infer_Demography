#!/usr/bin/env python3
import argparse, json, pickle, joblib, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import make_scorer


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


def overall_and_per_param_mse(y_true, y_pred, param_names):
    """Return dict with overall MSE + per-parameter MSE."""
    out = {"training": None, "validation": None,
           "training_mse": {}, "validation_mse": {}}
    if y_true is not None and y_pred is not None:
        out["training"] = float(np.mean((y_true - y_pred) ** 2))
        for i, p in enumerate(param_names):
            out["training_mse"][p] = float(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
    return out


def update_validation_mses(rrmse_dict, y_true_val, y_pred_val, param_names):
    if y_true_val is not None and y_pred_val is not None:
        rrmse_dict["validation"] = float(np.mean((y_true_val - y_pred_val) ** 2))
        for i, p in enumerate(param_names):
            rrmse_dict["validation_mse"][p] = float(np.mean((y_true_val[:, i] - y_pred_val[:, i]) ** 2))


def plot_feature_importances_grid(model, feature_names, target_names, out_path, max_num_features=None):
    """
    model is a MultiOutputRegressor with a list of single-output estimators.
    Creates a grid of feature-importance plots, one per target.
    """
    n_outputs = len(model.estimators_)
    n_cols = 3
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
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
        title = f"Feature Importance: {target_names[j]}" if j < len(target_names) else f"Output {j}"
        ax.set_title(title)

    # hide unused axes
    for ax in axes[n_outputs:]:
        ax.axis("off")

    fig.suptitle("Random Forest Feature Importances Across Outputs", fontsize=16)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def randomized_search_rf(X, y, n_iter=20, random_state=42, n_jobs=-1):
    """Run RandomizedSearchCV for RF hyperparams."""
    param_dist = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10, 15, 20],
        "random_state": [42, 123, 2023, 295],
    }
    scorer = make_scorer(sk_mse, greater_is_better=False)
    base = RandomForestRegressor()
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring=scorer,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )
    rs.fit(X, y)
    return rs.best_params_


# ------------------------ main functional pipeline ------------------------

def main(args):
    # Load experiment config (for param names)
    exp_cfg = json.load(open(args.experiment_config_path))
    param_names = exp_cfg.get("parameters_to_estimate", list(exp_cfg.get("priors", {}).keys()))

    # Model cfg (not critical here, but optional)
    model_cfg = {}
    if args.model_config_path:
        with open(args.model_config_path) as f:
            model_cfg = yaml.safe_load(f)

    # Load colors
    color_shades = pickle.load(open(args.color_shades_file, "rb"))
    main_colors  = pickle.load(open(args.main_colors_file, "rb"))

    # Load data as dataframes to keep names
    X_train_df, feat_names = load_df_pickle(args.X_train_path)
    y_train_df, targ_names = load_df_pickle(args.y_train_path)
    X_val_df,   _          = load_df_pickle(args.X_val_path)
    y_val_df,   _          = load_df_pickle(args.y_val_path)

    # Fall back if config names differ
    if not param_names:
        param_names = targ_names

    # Convert to arrays
    X_train = X_train_df.values
    y_train = y_train_df.values
    X_val   = X_val_df.values
    y_val   = y_val_df.values

    # Choose or search hyperparams
    user_specified = any(v is not None for v in [args.n_estimators,
                                                 args.max_depth,
                                                 args.min_samples_split,
                                                 args.random_state])
    if args.do_random_search or not user_specified:
        best = randomized_search_rf(
            X_train, y_train,
            n_iter=args.n_iter,
            random_state=args.random_state if args.random_state is not None else 42
        )
        print(f"[INFO] RandomizedSearchCV best params: {best}")
        n_estimators     = best.get("n_estimators", 200)
        max_depth        = best.get("max_depth", None)
        min_samples_split= best.get("min_samples_split", 2)
        random_state     = best.get("random_state", 42)
    else:
        n_estimators      = args.n_estimators or 200
        max_depth         = args.max_depth
        min_samples_split = args.min_samples_split or 2
        random_state      = args.random_state or 42

    # Build and fit multi-output RF
    rf_single = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    rf = MultiOutputRegressor(rf_single)
    rf.fit(X_train, y_train)

    tr_pred = rf.predict(X_train)
    va_pred = rf.predict(X_val)

    # MSE dict
    mse_dict = {"training": None, "validation": None,
                "training_mse": {}, "validation_mse": {}}

    mse_dict["training"] = float(np.mean((y_train - tr_pred) ** 2))
    for i, p in enumerate(param_names):
        mse_dict["training_mse"][p] = float(np.mean((y_train[:, i] - tr_pred[:, i]) ** 2))

    mse_dict["validation"] = float(np.mean((y_val - va_pred) ** 2))
    for i, p in enumerate(param_names):
        mse_dict["validation_mse"][p] = float(np.mean((y_val[:, i] - va_pred[:, i]) ** 2))

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
        "target_names": targ_names
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
        main_colors=main_colors
    )

    # Feature importance per target (grid)
    plot_feature_importances_grid(
        model=rf,
        feature_names=feat_names,
        target_names=targ_names,
        out_path=out_dir / "random_forest_feature_importances.png",
        max_num_features=20
    )

    print("[INFO] Random Forest complete. Artifacts saved to:", out_dir)


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--X_train_path", type=str, required=True)
    p.add_argument("--y_train_path", type=str, required=True)
    p.add_argument("--X_val_path",   type=str, required=True)
    p.add_argument("--y_val_path",   type=str, required=True)

    # Configs
    p.add_argument("--experiment_config_path", type=str, required=True)
    p.add_argument("--model_config_path",      type=str, default=None)
    p.add_argument("--color_shades_file",      type=str, required=True)
    p.add_argument("--main_colors_file",       type=str, required=True)

    # Output dir
    p.add_argument("--model_directory", type=str, required=True)

    # RF hyperparams (all optional)
    p.add_argument("--n_estimators",     type=int, default=None)
    p.add_argument("--max_depth",        type=int, default=None)
    p.add_argument("--min_samples_split",type=int, default=None)
    p.add_argument("--random_state",     type=int, default=None)
    p.add_argument("--do_random_search", action="store_true")
    p.add_argument("--n_iter",           type=int, default=20)

    args = p.parse_args()
    main(args)

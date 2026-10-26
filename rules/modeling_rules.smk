##############################################################################
# ML modeling: combine_features -> train_model (linear / random forest / xgboost)
##############################################################################

##############################################################################
# Modeling-only settings
##############################################################################
# CLI-flag builders for random_forest/xgboost — same optuna/manual-override
# knobs regardless of which modeling_{variant} the rule instantiates for.
def _rf_opt_flags():
    return " ".join([
        "--use_optuna" if config.get("rf", {}).get("use_optuna", False) else "",
        f"--n_trials {config['rf']['n_trials']}" if config.get("rf", {}).get("n_trials") is not None else "",
        f"--optuna_timeout {config['rf']['optuna_timeout']}" if config.get("rf", {}).get("optuna_timeout") is not None else "",
        f"--optuna_seed {config['rf']['optuna_seed']}" if config.get("rf", {}).get("optuna_seed") is not None else "",
        f"--final_fit {config['rf']['final_fit']}" if config.get("rf", {}).get("final_fit") is not None else "",
        f"--n_estimators {config['rf']['n_estimators']}" if config.get("rf", {}).get("n_estimators") is not None else "",
        f"--max_depth {config['rf']['max_depth']}" if config.get("rf", {}).get("max_depth") is not None else "",
        f"--min_samples_split {config['rf']['min_samples_split']}" if config.get("rf", {}).get("min_samples_split") is not None else "",
        f"--min_samples_leaf {config['rf']['min_samples_leaf']}" if config.get("rf", {}).get("min_samples_leaf") is not None else "",
        f"--max_features {config['rf']['max_features']}" if config.get("rf", {}).get("max_features") is not None else "",
        f"--max_samples {config['rf']['max_samples']}" if config.get("rf", {}).get("max_samples") is not None else "",
    ]).strip()

def _xgb_opt_flags():
    return " ".join([
        "--use_optuna" if config.get("xgb", {}).get("use_optuna", False) else "",
        f"--n_trials {config['xgb']['n_trials']}" if config.get("xgb", {}).get("n_trials") is not None else "",
        f"--optuna_timeout {config['xgb']['optuna_timeout']}" if config.get("xgb", {}).get("optuna_timeout") is not None else "",
        f"--optuna_seed {config['xgb']['optuna_seed']}" if config.get("xgb", {}).get("optuna_seed") is not None else "",
        f"--final_fit {config['xgb']['final_fit']}" if config.get("xgb", {}).get("final_fit") is not None else "",
        f"--early_stopping_rounds {config['xgb']['early_stopping_rounds']}" if config.get("xgb", {}).get("early_stopping_rounds") is not None else "",
        f"--n_estimators {config['xgb']['n_estimators']}" if config.get("xgb", {}).get("n_estimators") is not None else "",
        f"--max_depth {config['xgb']['max_depth']}" if config.get("xgb", {}).get("max_depth") is not None else "",
        f"--learning_rate {config['xgb']['learning_rate']}" if config.get("xgb", {}).get("learning_rate") is not None else "",
        f"--subsample {config['xgb']['subsample']}" if config.get("xgb", {}).get("subsample") is not None else "",
        f"--colsample_bytree {config['xgb']['colsample_bytree']}" if config.get("xgb", {}).get("colsample_bytree") is not None else "",
        f"--min_child_weight {config['xgb']['min_child_weight']}" if config.get("xgb", {}).get("min_child_weight") is not None else "",
        f"--reg_lambda {config['xgb']['reg_lambda']}" if config.get("xgb", {}).get("reg_lambda") is not None else "",
        f"--reg_alpha {config['xgb']['reg_alpha']}" if config.get("xgb", {}).get("reg_alpha") is not None else "",
        f"--top_k_features_plot {config['xgb']['top_k_plot']}" if config.get("xgb", {}).get("top_k_plot") is not None else "",
    ]).strip()

# Regressors


##############################################################################
# RULE combine_features – one modeling dataset per {variant}                 #
# (filter, split, normalize). Discovers sims at runtime and skips missing    #
# ones, so it has no per-sim inputs. Feature set per variant:                #
# VARIANT_FEATURE_FLAGS in rules/common.smk.                                 #
# Also writes (undeclared, inference variants only) estimate-vs-truth        #
# metrics_*.json / *.png and outliers_* reports next to these files.         #
##############################################################################
DATASETS_DIR = f"experiments/{MODEL}/modeling_{{variant}}/datasets"

rule combine_features:
    input:
        cfg  = EXP_CFG
    output:
        features_df = f"{DATASETS_DIR}/features_df.pkl",
        targets_df  = f"{DATASETS_DIR}/targets_df.pkl",
        split_idx   = f"{DATASETS_DIR}/split_indices.json",
        norm_stats  = f"{DATASETS_DIR}/empirical_norm_stats.json",
        ntrain_X    = f"{DATASETS_DIR}/normalized_train_features.pkl",
        ntrain_y    = f"{DATASETS_DIR}/normalized_train_targets.pkl",
        ntune_X     = f"{DATASETS_DIR}/normalized_tune_features.pkl",
        ntune_y     = f"{DATASETS_DIR}/normalized_tune_targets.pkl",
        nval_X      = f"{DATASETS_DIR}/normalized_val_features.pkl",
        nval_y      = f"{DATASETS_DIR}/normalized_val_targets.pkl",
    params:
        outdir = f"experiments/{MODEL}/modeling_{{variant}}",
        flags  = lambda w: VARIANT_FEATURE_FLAGS[w.variant],
    threads: 1
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/feature_extraction.py \
            --experiment-config "{input.cfg}" \
            --out-dir "{params.outdir}" \
            {params.flags}
        """

##############################################################################
# RULE train_model – fit one model on one variant's dataset                  #
# {model_key}: linear_<reg> (REG_TYPES), random_forest or xgboost. Script,  #
# threads and model-specific flags come from the tables below.              #
##############################################################################
DATASET = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_{{split}}_{{xy}}.pkl"

TRAIN_SCRIPTS = {
    "linear":        "snakemake_scripts/linear_evaluation.py",
    "random_forest": "snakemake_scripts/random_forest.py",
    "xgboost":       "snakemake_scripts/xgboost_evaluation.py",
}
TRAIN_THREADS = {"linear": 2, "random_forest": 8, "xgboost": 4}

def _model_family(model_key):
    return "linear" if model_key.startswith("linear_") else model_key

def _model_flags(w):
    if w.model_key.startswith("linear_"):
        reg = w.model_key[len("linear_"):]
        opts = config["linear"].get(reg, {})
        return " ".join([
            f"--regression_type {reg}",
            f"--alpha {opts.get('alpha', 0.0)}",
            f"--l1_ratio {opts.get('l1_ratio', 0.5)}",
            "--do_grid_search" if opts.get("grid_search", False) else "",
        ]).strip()
    return _rf_opt_flags() if w.model_key == "random_forest" else _xgb_opt_flags()

wildcard_constraints:
    model_key = "|".join(MODEL_KEYS),

rule train_model:
    input:
        X_train = DATASET.replace("{split}", "train").replace("{xy}", "features"),
        y_train = DATASET.replace("{split}", "train").replace("{xy}", "targets"),
        X_tune  = DATASET.replace("{split}", "tune").replace("{xy}", "features"),
        y_tune  = DATASET.replace("{split}", "tune").replace("{xy}", "targets"),
        X_val   = DATASET.replace("{split}", "val").replace("{xy}", "features"),
        y_val   = DATASET.replace("{split}", "val").replace("{xy}", "targets"),
        expcfg  = EXP_CFG,
        mdlcfg  = "config_files/model_config.yaml",
    output:
        obj   = f"{TRAINED_MODEL_DIR}/mdl_obj.pkl",
        errjs = f"{TRAINED_MODEL_DIR}/model_error.json",
        mdl   = f"{TRAINED_MODEL_DIR}/model.pkl",
        plot  = f"{TRAINED_MODEL_DIR}/results.png",
    params:
        script    = lambda w: TRAIN_SCRIPTS[_model_family(w.model_key)],
        model_dir = lambda w: TRAINED_MODEL_DIR.format(variant=w.variant, model_key=w.model_key),
        flags     = _model_flags,
    threads: lambda w: TRAIN_THREADS[_model_family(w.model_key)]
    benchmark:
        "benchmarks/train_model_{variant}_{model_key}.tsv"
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --X_train_path "{input.X_train}" \
            --y_train_path "{input.y_train}" \
            --X_tune_path  "{input.X_tune}" \
            --y_tune_path  "{input.y_tune}" \
            --X_val_path   "{input.X_val}" \
            --y_val_path   "{input.y_val}" \
            --experiment_config_path "{input.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --model_directory "{params.model_dir}" \
            {params.flags}
        """

##############################################################################
# RULE modeling_all – every model for every MODELING_VARIANTS entry          #
# Usage:  snakemake -s modeling.smk -j <N> modeling_all                     #
##############################################################################
rule modeling_all:
    input:
        expand(f"{TRAINED_MODEL_DIR}/mdl_obj.pkl", variant=MODELING_VARIANTS, model_key=MODEL_KEYS),

##############################################################################
# ML modeling: combine_features -> linear / random forest / xgboost
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
REG_TYPES = config["linear"]["types"]  # e.g., ["standard","ridge","lasso","elasticnet"]


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
# RULE make_color_scheme – build color_shades.pkl & main_colors.pkl
##############################################################################
rule make_color_scheme:
    output:
        shades = f"experiments/{MODEL}/modeling/color_shades.pkl",
        mains  = f"experiments/{MODEL}/modeling/main_colors.pkl"
    params:
        script = "snakemake_scripts/setup_colors.py",
        cfg    = EXP_CFG
    threads: 1
    benchmark:
        "benchmarks/make_color_scheme.tsv"
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --config "{params.cfg}" \
            --out-dir "$(dirname {output.shades})"
        """

##############################################################################
# RULE linear_regression                                                     #
##############################################################################
rule linear_regression:
    input:
        X_train = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_features.pkl",
        y_train = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_targets.pkl",

        # ✅ add tune
        X_tune  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_features.pkl",
        y_tune  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_targets.pkl",

        X_val   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_features.pkl",
        y_val   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_targets.pkl",
        shades  = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors  = f"experiments/{MODEL}/modeling/main_colors.pkl",
        mdlcfg  = "config_files/model_config.yaml"
    output:
        obj   = f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl",
        errjs = f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_model_error_{{reg}}.json",
        mdl   = f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_regression_model_{{reg}}.pkl",
        plot  = f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_results_{{reg}}.png"
    params:
        script    = "snakemake_scripts/linear_evaluation.py",
        expcfg    = EXP_CFG,
        model_dir = f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}",
        alpha    = lambda w: config["linear"].get(w.reg, {}).get("alpha", 0.0),
        l1_ratio = lambda w: config["linear"].get(w.reg, {}).get("l1_ratio", 0.5),
        gridflag = lambda w: "--do_grid_search" if config["linear"].get(w.reg, {}).get("grid_search", False) else ""
    threads: 2
    benchmark:
        f"benchmarks/linear_regression_{{variant}}_{{reg}}.tsv"
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
            --experiment_config_path "{params.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            --regression_type "{wildcards.reg}" \
            --model_directory "{params.model_dir}" \
            {params.gridflag} \
            --alpha {params.alpha} \
            --l1_ratio {params.l1_ratio}
        """

##############################################################################
# RULE random_forest                                                         #
##############################################################################
rule random_forest:
    input:
        X_train = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_features.pkl",
        y_train = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_targets.pkl",
        X_tune  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_features.pkl",
        y_tune  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_targets.pkl",
        X_val   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_features.pkl",
        y_val   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_targets.pkl",
        shades  = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors  = f"experiments/{MODEL}/modeling/main_colors.pkl",
        expcfg  = EXP_CFG,
        mdlcfg  = "config_files/model_config.yaml"
    output:
        obj   = f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_mdl_obj.pkl",
        errjs = f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_model_error.json",
        mdl   = f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_model.pkl",
        plot  = f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_results.png",
        fi    = f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_feature_importances.png"
    params:
        script    = "snakemake_scripts/random_forest.py",
        model_dir = f"experiments/{MODEL}/modeling_{{variant}}/random_forest",
        opt_flags = lambda w: _rf_opt_flags(),
    threads: 8
    benchmark:
        f"benchmarks/random_forest_{{variant}}.tsv"
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
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            {params.opt_flags} \
            --model_directory "{params.model_dir}"
        """

##############################################################################
# RULE xgboost                                                               #
##############################################################################
rule xgboost:
    input:
        X_train = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_features.pkl",
        y_train = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_targets.pkl",
        X_tune  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_features.pkl",
        y_tune  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_targets.pkl",
        X_val   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_features.pkl",
        y_val   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_targets.pkl",
        shades  = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors  = f"experiments/{MODEL}/modeling/main_colors.pkl",
        expcfg  = EXP_CFG,
        mdlcfg  = "config_files/model_config.yaml"
    output:
        obj   = f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_mdl_obj.pkl",
        errjs = f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_model_error.json",
        mdl   = f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_model.pkl",
        plot  = f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_results.png",
        fi    = f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_feature_importances.png"
    params:
        script    = "snakemake_scripts/xgboost_evaluation.py",
        model_dir = f"experiments/{MODEL}/modeling_{{variant}}/xgboost",
        opt_flags = lambda w: _xgb_opt_flags(),
    threads: 4
    benchmark:
        f"benchmarks/xgboost_{{variant}}.tsv"
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
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            {params.opt_flags} \
            --model_directory "{params.model_dir}"
        """

##############################################################################
# RULE modeling_all – build every FIM×residuals modeling variant in one run  #
# Usage:  snakemake -j <N> modeling_all                                       #
##############################################################################
rule modeling_all:
    input:
        expand(
            f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl",
            variant=MODELING_VARIANTS, reg=REG_TYPES,
        ),
        expand(
            f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_mdl_obj.pkl",
            variant=MODELING_VARIANTS,
        ),
        expand(
            f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_mdl_obj.pkl",
            variant=MODELING_VARIANTS,
        ),

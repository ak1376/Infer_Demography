##############################################################################
# ML modeling: combine_features -> linear / random forest / xgboost
##############################################################################

##############################################################################
# RULE build_sfs_dataset – features=observed SFS, targets=sampled params    #
# NOT in rule all. Run explicitly:                                           #
#   snakemake --snakefile Snakefile build_sfs_dataset                       #
##############################################################################
rule build_sfs_dataset:
    input:
        cfg = EXP_CFG,
    output:
        features = f"experiments/{MODEL}/modeling/sfs_datasets/sfs_features_df.pkl",
        targets  = f"experiments/{MODEL}/modeling/sfs_datasets/sfs_targets_df.pkl",
        meta     = f"experiments/{MODEL}/modeling/sfs_datasets/sfs_dataset_meta.json",
    params:
        sim_dir = SIM_BASEDIR,
        out_dir = f"experiments/{MODEL}/modeling/sfs_datasets",
        min_sims = int(CFG.get("build_sfs_dataset_min_sims", 10)),
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/build_sfs_dataset.py \
            --sim-dir  "{params.sim_dir}" \
            --config   "{input.cfg}" \
            --out-dir  "{params.out_dir}" \
            --min-sims {params.min_sims}
        """

##############################################################################
# RULE prepare_sfs_splits – reuse existing split_indices.json + normalize   #
##############################################################################
rule prepare_sfs_splits:
    input:
        features  = f"experiments/{MODEL}/modeling/sfs_datasets/sfs_features_df.pkl",
        targets   = f"experiments/{MODEL}/modeling/sfs_datasets/sfs_targets_df.pkl",
    output:
        ntrain_X = f"experiments/{MODEL}/modeling/sfs_datasets/normalized_train_features.pkl",
        ntrain_y = f"experiments/{MODEL}/modeling/sfs_datasets/normalized_train_targets.pkl",
        ntune_X  = f"experiments/{MODEL}/modeling/sfs_datasets/normalized_tune_features.pkl",
        ntune_y  = f"experiments/{MODEL}/modeling/sfs_datasets/normalized_tune_targets.pkl",
        nval_X   = f"experiments/{MODEL}/modeling/sfs_datasets/normalized_val_features.pkl",
        nval_y   = f"experiments/{MODEL}/modeling/sfs_datasets/normalized_val_targets.pkl",
        meta     = f"experiments/{MODEL}/modeling/sfs_datasets/sfs_splits_meta.json",
    params:
        out_dir   = f"experiments/{MODEL}/modeling/sfs_datasets",
        split_idx = f"experiments/{MODEL}/modeling/datasets/split_indices.json",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        SPLIT_FLAG=""
        if [ -f "{params.split_idx}" ]; then
            SPLIT_FLAG="--split-indices \"{params.split_idx}\""
        fi
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/prepare_sfs_splits.py \
            --features  "{input.features}" \
            --targets   "{input.targets}" \
            --out-dir   "{params.out_dir}" \
            $SPLIT_FLAG
        """

##############################################################################
# RULE combine_features – build datasets (filter, split, normalize)          #
# (robust: discovers existing sims at runtime; skips missing)                #
##############################################################################
rule combine_features:
    input:
        cfg  = EXP_CFG
    wildcard_constraints:
        variant = r"(w|wo)_FIM_(w|wo)_SFSresids"
    output:
        # full post-filtering data
        features_df   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/features_df.pkl",
        targets_df    = f"experiments/{MODEL}/modeling_{{variant}}/datasets/targets_df.pkl",

        # raw splits (temp: modeling rules use normalized_* variants)
        train_X       = temp(f"experiments/{MODEL}/modeling_{{variant}}/datasets/train_features.pkl"),
        train_y       = temp(f"experiments/{MODEL}/modeling_{{variant}}/datasets/train_targets.pkl"),
        tune_X        = temp(f"experiments/{MODEL}/modeling_{{variant}}/datasets/tune_features.pkl"),
        tune_y        = temp(f"experiments/{MODEL}/modeling_{{variant}}/datasets/tune_targets.pkl"),
        val_X         = temp(f"experiments/{MODEL}/modeling_{{variant}}/datasets/val_features.pkl"),
        val_y         = temp(f"experiments/{MODEL}/modeling_{{variant}}/datasets/val_targets.pkl"),

        # normalized splits
        ntrain_X      = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_features.pkl",
        ntrain_y      = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_train_targets.pkl",
        ntune_X       = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_features.pkl",
        ntune_y       = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_tune_targets.pkl",
        nval_X        = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_features.pkl",
        nval_y        = f"experiments/{MODEL}/modeling_{{variant}}/datasets/normalized_val_targets.pkl",

        # split indices + plots/metrics
        split_idx     = f"experiments/{MODEL}/modeling_{{variant}}/datasets/split_indices.json",
        scatter_png   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/features_scatterplot.png",
        mse_val_png   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/mse_bars_val_normalized.png",
        mse_train_png = f"experiments/{MODEL}/modeling_{{variant}}/datasets/mse_bars_train_normalized.png",
        metrics_all   = f"experiments/{MODEL}/modeling_{{variant}}/datasets/metrics_all.json",
        metrics_dadi  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/metrics_dadi.json",
        metrics_moments = f"experiments/{MODEL}/modeling_{{variant}}/datasets/metrics_moments.json",
        metrics_momentsLD = f"experiments/{MODEL}/modeling_{{variant}}/datasets/metrics_momentsLD.json",
        outliers_tsv  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/outliers_removed.tsv",
        outliers_txt  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/outliers_preview.txt",
        empirical_norm_stats = f"experiments/{MODEL}/modeling_{{variant}}/datasets/empirical_norm_stats.json"
    params:
        script = "snakemake_scripts/feature_extraction.py",
        outdir = f"experiments/{MODEL}/modeling_{{variant}}",
        fim_flag   = lambda w: "true" if _variant_flags(w.variant)[0] else "false",
        resid_flag = lambda w: "true" if _variant_flags(w.variant)[1] else "false",
    threads: 1
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --experiment-config "{input.cfg}" \
            --out-dir "{params.outdir}" \
            --use-fim-features {params.fim_flag} \
            --use-residuals {params.resid_flag}
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

##############################################################################
# RAW-FEATURES PIPELINE: observed SFS + MomentsLD means → ensemble          #
# Not in rule all. Not built via combine_features/prepare_sfs_splits (see   #
# their variant wildcard_constraints) — build_raw_features_dataset and      #
# prepare_raw_features_splits below produce this variant's dataset files    #
# directly, but from there it's just modeling_{variant} with               #
# variant="raw_features", so the same linear_regression/random_forest/      #
# xgboost rules train it. Run explicitly, e.g.:                             #
#   snakemake --snakefile Snakefile "experiments/<model>/modeling_raw_features/xgboost/xgb_mdl_obj.pkl"
##############################################################################
RAW_FEAT_DIR = f"experiments/{MODEL}/modeling_raw_features/datasets"
RAW_MDL_DIR  = f"experiments/{MODEL}/modeling_raw_features"

rule build_raw_features_dataset:
    input:
        cfg     = EXP_CFG,
        mv_pkls = expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/means.varcovs.pkl",
            sid=SIM_IDS,
        ),
    output:
        features = f"{RAW_FEAT_DIR}/raw_features_df.pkl",
        targets  = f"{RAW_FEAT_DIR}/raw_targets_df.pkl",
        meta     = f"{RAW_FEAT_DIR}/raw_dataset_meta.json",
    params:
        sim_dir  = SIM_BASEDIR,
        inf_dir  = f"experiments/{MODEL}/inferences",
        out_dir  = RAW_FEAT_DIR,
        min_sims = int(CFG.get("build_sfs_dataset_min_sims", 10)),
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/build_raw_features_dataset.py \
            --sim-dir       "{params.sim_dir}" \
            --inference-dir "{params.inf_dir}" \
            --config        "{input.cfg}" \
            --out-dir       "{params.out_dir}" \
            --min-sims      {params.min_sims}
        """

rule prepare_raw_features_splits:
    input:
        features  = f"{RAW_FEAT_DIR}/raw_features_df.pkl",
        targets   = f"{RAW_FEAT_DIR}/raw_targets_df.pkl",
        split_idx = f"experiments/{MODEL}/modeling/datasets/split_indices.json",
    output:
        ntrain_X = f"{RAW_FEAT_DIR}/normalized_train_features.pkl",
        ntrain_y = f"{RAW_FEAT_DIR}/normalized_train_targets.pkl",
        ntune_X  = f"{RAW_FEAT_DIR}/normalized_tune_features.pkl",
        ntune_y  = f"{RAW_FEAT_DIR}/normalized_tune_targets.pkl",
        nval_X   = f"{RAW_FEAT_DIR}/normalized_val_features.pkl",
        nval_y   = f"{RAW_FEAT_DIR}/normalized_val_targets.pkl",
        meta     = f"{RAW_FEAT_DIR}/raw_splits_meta.json",
    params:
        out_dir  = RAW_FEAT_DIR,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/prepare_sfs_splits.py \
            --features      "{input.features}" \
            --targets       "{input.targets}" \
            --out-dir       "{params.out_dir}" \
            --split-indices "{input.split_idx}"
        """

# raw_features model training is handled by the generic linear_regression /
# random_forest / xgboost rules above with variant="raw_features" — no
# separate rules needed now that RAW_MDL_DIR follows the modeling_{variant}
# convention.

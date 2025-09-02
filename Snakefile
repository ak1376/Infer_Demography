##############################################################################
# CONFIG – Paths and Constants (edit here only)                             #
##############################################################################
import json, math, sys
from pathlib import Path
from snakemake.io import protected

# ── guard against shadow‑import of the Bottleneck C library ────────────────
if "bottleneck" in sys.modules and not hasattr(sys.modules["bottleneck"], "__version__"):
    del sys.modules["bottleneck"]

configfile: "config_files/model_config.yaml"
REG_TYPES = config["linear"]["types"]  # ["standard","ridge","lasso","elasticnet"]

# ── external scripts -------------------------------------------------------
SIM_SCRIPT   = "snakemake_scripts/simulation.py"
INFER_SCRIPT = "snakemake_scripts/moments_dadi_inference.py"
WIN_SCRIPT   = "snakemake_scripts/simulate_window.py"
LD_SCRIPT    = "snakemake_scripts/compute_ld_window.py"
EXP_CFG      = "config_files/experiment_config_split_migration.json"

# ── experiment metadata ----------------------------------------------------
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = CFG["num_draws"]          # number of independent sims
NUM_OPTIMS    = CFG.get("num_optimizations", 3)
TOP_K         = CFG.get("top_k", 2)
NUM_WINDOWS   = 100                       # LD windows per simulation
R_BINS_STR    = "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

SIM_IDS  = [i for i in range(NUM_DRAWS)]
WINDOWS  = range(NUM_WINDOWS)

# ── canonical path builders -----------------------------------------------
SIM_BASEDIR = f"experiments/{MODEL}/simulations"                      # per‑sim artefacts
RUN_DIR     = lambda sid, opt: f"experiments/{MODEL}/runs/run_{sid}_{opt}"
LD_ROOT     = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD"  # use {{sid}} wildcard

opt_pkl   = lambda sid, opt, tool: f"{RUN_DIR(sid, opt)}/inferences/{tool}/fit_params.pkl"
final_pkl = lambda sid, tool: f"experiments/{MODEL}/inferences/sim_{sid}/{tool}/fit_params.pkl"

##############################################################################
# RULE all – final targets the workflow must create                         #
##############################################################################
rule all:
    input:
        # ── raw simulation artefacts ────────────────────────────────────────
        expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",    sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",               sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",   sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/demes.png",             sid=SIM_IDS),

        # ── aggregated optimiser results (produced by rule aggregate_opts) ─
        [final_pkl(sid, "moments") for sid in SIM_IDS],
        [final_pkl(sid, "dadi")    for sid in SIM_IDS],

        # ── LD: per‑window VCFs & statistics, plus best‑fit summary ─────────
        expand(f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",         sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl",  sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/best_fit.pkl",                          sid=SIM_IDS),

        # ── final merge of all three inference types ───────────────────────
        expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl",
            sid=SIM_IDS
        ),

        # ── modeling datasets (after outlier removal, split, normalization) ─
        f"experiments/{MODEL}/modeling/datasets/features_df.pkl",
        f"experiments/{MODEL}/modeling/datasets/targets_df.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",
        f"experiments/{MODEL}/modeling/datasets/features_scatterplot.png", 

        # ── color schemes for plotting ──────────────────────────────────────
        f"experiments/{MODEL}/modeling/color_shades.pkl",
        f"experiments/{MODEL}/modeling/main_colors.pkl", 

        # ── linear regression results ───────────────────────────────────────
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl", reg=REG_TYPES),
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_model_error_{{reg}}.json", reg=REG_TYPES),
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_regression_model_{{reg}}.pkl", reg=REG_TYPES),
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_results_{{reg}}.png", reg=REG_TYPES),

        # ── random forest results ──────────────────────────────────────────
        f"experiments/{MODEL}/modeling/random_forest/random_forest_mdl_obj.pkl",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_model_error.json",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_model.pkl",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_results.png",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_feature_importances.png",

        # ── XGBoost results ───────────────────────────────────────────────
        f"experiments/{MODEL}/modeling/xgboost/xgb_mdl_obj.pkl",
        f"experiments/{MODEL}/modeling/xgboost/xgb_model_error.json",
        f"experiments/{MODEL}/modeling/xgboost/xgb_model.pkl",
        f"experiments/{MODEL}/modeling/xgboost/xgb_results.png",
        f"experiments/{MODEL}/modeling/xgboost/xgb_feature_importances.png",


##############################################################################
# RULE simulate – one complete tree‑sequence + SFS
##############################################################################
rule simulate:
    output:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        tree   = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        fig    = f"{SIM_BASEDIR}/{{sid}}/demes.png",
        done   = protected(f"{SIM_BASEDIR}/{{sid}}/.done"),
    params:
        sim_dir = SIM_BASEDIR,
        cfg     = EXP_CFG,
        model   = MODEL
    threads: 1
    shell:
        r"""
        set -euo pipefail

        python "{SIM_SCRIPT}" \
          --simulation-dir "{params.sim_dir}" \
          --experiment-config "{params.cfg}" \
          --model-type "{params.model}" \
          --simulation-number {wildcards.sid}

        # ensure expected outputs exist, then create sentinel
        test -f "{output.sfs}" && \
        test -f "{output.params}" && \
        test -f "{output.tree}" && \
        test -f "{output.fig}"
        touch "{output.done}"
        """

##############################################################################
# RULE infer_moments  – custom NLopt Poisson SFS optimisation (moments)
##############################################################################
rule infer_moments:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl"   # not read; kept for DAG clarity
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/moments/fit_params.pkl"
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        maxeval  = 800,
        rtol     = 1e-8,
        fix      = ""     # e.g. '--fix N0=10000 --fix m12=0.0'
    threads: 8
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/moments_dadi_inference.py" \
          --mode moments \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "src.simulation:split_migration_model" \
          --outdir "{params.run_dir}/inferences" \
          --maxeval {params.maxeval} \
          --rtol {params.rtol} \
          {params.fix}

        cp "{params.run_dir}/inferences/moments/best_fit.pkl" "{output.pkl}"
        """

##############################################################################
# RULE infer_dadi – custom NLopt Poisson SFS optimisation (dadi)
##############################################################################
rule infer_dadi:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl"   # not read; kept for DAG clarity
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/dadi/fit_params.pkl"
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        maxeval  = 800,
        rtol     = 1e-8,
        fix      = "",     # e.g. '--fix N0=10000 --fix m12=0.0'
        algo     = ""      # '--use-bobyqa' if you want derivative-free for dadi
    threads: 8
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/moments_dadi_inference.py" \
          --mode dadi \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "src.simulation:split_migration_model" \
          --outdir "{params.run_dir}/inferences" \
          --maxeval {params.maxeval} \
          --rtol {params.rtol} \
          {params.algo} \
          {params.fix}

        cp "{params.run_dir}/inferences/dadi/best_fit.pkl" "{output.pkl}"
        """

##############################################################################
# RULE aggregate_opts – keep the TOP_K best fits per simulation             #
##############################################################################
rule aggregate_opts:
    input:
        mom  = lambda w: [opt_pkl(w.sid, o, "moments") for o in range(NUM_OPTIMS)],
        dadi = lambda w: [opt_pkl(w.sid, o, "dadi")    for o in range(NUM_OPTIMS)]
    output:
        mom  = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl",
        dadi = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl"
    run:
        import pickle, numpy as np, pathlib

        def _as_list(x):
            """Return *x* as a list (wrap scalars)."""
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        def merge_keep_best(in_files, out_file):
            params, lls = [], []
            for pkl in in_files:
                d = pickle.load(open(pkl, "rb"))
                params.extend(_as_list(d["best_params"]))
                lls.extend(_as_list(d["best_lls"]))

            keep = np.argsort(lls)[::-1][:TOP_K]            # top‑K by LL
            best = {"best_params": [params[i] for i in keep],
                    "best_lls"   : [lls[i]    for i in keep]}

            pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(best, open(out_file, "wb"))

        merge_keep_best(input.mom,  output.mom)
        merge_keep_best(input.dadi, output.dadi)

##############################################################################
# RULE simulate_window – one VCF window
##############################################################################
rule simulate_window:
    input:
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        done   = f"{SIM_BASEDIR}/{{sid}}/.done"
    output:
        vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz"
    params:
        base_sim   = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        out_winDir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
        rep_idx    = "{win}",
        cfg        = EXP_CFG
    threads: 1
    shell:
        r"""
        python "{WIN_SCRIPT}" \
            --sim-dir      "{params.base_sim}" \
            --rep-index    {params.rep_idx} \
            --config-file  "{params.cfg}" \
            --out-dir      "{params.out_winDir}"
        """

##############################################################################
# RULE ld_window – LD statistics for one window                             #
##############################################################################
rule ld_window:
    input:
        vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
    output:
        pkl    = f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl"
    params:
        sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        bins    = R_BINS_STR,
        cfg    = EXP_CFG

    threads: 4
    resources:
        ld_cores=4
    shell:
        """
        python "{LD_SCRIPT}" \
            --sim-dir      {params.sim_dir} \
            --window-index {wildcards.win} \
            --config-file  {params.cfg} \
            --r-bins       "{params.bins}"
        """

##############################################################################
# RULE optimize_momentsld – aggregate windows & optimise momentsLD          #
##############################################################################
rule optimize_momentsld:
    input:
        pkls = lambda w: expand(
            f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl",
            sid=[w.sid],
            win=WINDOWS
        ),
    output:
        mv   = f"{LD_ROOT}/means.varcovs.pkl",
        boot = f"{LD_ROOT}/bootstrap_sets.pkl",
        pdf  = f"{LD_ROOT}/empirical_vs_theoretical_comparison.pdf",
        best = f"{LD_ROOT}/best_fit.pkl"
    params:
        sim_dir   = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        LD_dir    = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        bins      = R_BINS_STR,
        n_windows = NUM_WINDOWS,
        cfg  = EXP_CFG

    threads: 1
    shell:
        """
        python "snakemake_scripts/LD_inference.py" \
            --run-dir      {params.sim_dir} \
            --output-root  {params.LD_dir} \
            --config-file  {params.cfg}
        """

##############################################################################
# RULE combine_results – merge dadi / moments / moments‑LD fits per sim
##############################################################################
rule combine_results:
    input:
        dadi      = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",
        moments   = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl",
        momentsLD = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/best_fit.pkl"
    output:
        combo = f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl"
    run:
        import pickle, pathlib
        outdir = pathlib.Path(output.combo).parent
        outdir.mkdir(parents=True, exist_ok=True)

        summary = {
            "moments"  : pickle.load(open(input.moments,   "rb")),
            "dadi"     : pickle.load(open(input.dadi,      "rb")),
            "momentsLD": pickle.load(open(input.momentsLD, "rb")),
        }
        pickle.dump(summary, open(output.combo, "wb"))
        print(f"✓ combined → {output.combo}")

##############################################################################
# RULE combine_features – build datasets (outlier-filtered, split, normalized)
##############################################################################
rule combine_features:
    input:
        cfg    = EXP_CFG,
        infs   = expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl",
            sid=SIM_IDS
        ),
        truths = expand(
            f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
            sid=SIM_IDS
        )
    output:
        features_df   = f"experiments/{MODEL}/modeling/datasets/features_df.pkl",
        targets_df    = f"experiments/{MODEL}/modeling/datasets/targets_df.pkl",
        ntrain_X      = f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        ntrain_y      = f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        nval_X        = f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        nval_y        = f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",
        scatter_png   = f"experiments/{MODEL}/modeling/datasets/features_scatterplot.png"
    params:
        script = "snakemake_scripts/feature_extraction.py",
        outdir = f"experiments/{MODEL}/modeling"
    threads: 1
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --experiment-config "{input.cfg}" \
            --out-dir "{params.outdir}"

        # sanity checks
        test -f "{output.features_df}" && \
        test -f "{output.targets_df}"  && \
        test -f "{output.ntrain_X}"    && \
        test -f "{output.ntrain_y}"    && \
        test -f "{output.nval_X}"      && \
        test -f "{output.nval_y}"      && \
        test -f "{output.scatter_png}"
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

        test -f "{output.shades}" && test -f "{output.mains}"
        """

##############################################################################
# RULE linear_regression                                                     #
##############################################################################
rule linear_regression:
    input:
        X_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        y_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        X_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        y_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",
        shades  = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors  = f"experiments/{MODEL}/modeling/main_colors.pkl",
        mdlcfg  = "config_files/model_config.yaml"   # optional
    output:
        obj   = f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl",
        errjs = f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_model_error_{{reg}}.json",
        mdl   = f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_regression_model_{{reg}}.pkl",
        plot  = f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_results_{{reg}}.png"
    params:
        script   = "snakemake_scripts/linear_evaluation.py",
        expcfg   = EXP_CFG,
        alpha    = lambda w: config["linear"].get(w.reg, {}).get("alpha", 0.0),
        l1_ratio = lambda w: config["linear"].get(w.reg, {}).get("l1_ratio", 0.5),
        gridflag = lambda w: "--do_grid_search" if config["linear"].get(w.reg, {}).get("grid_search", False) else ""
    threads: 2
    benchmark:
        f"benchmarks/linear_regression_{{reg}}.tsv"
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --X_train_path "{input.X_train}" \
            --y_train_path "{input.y_train}" \
            --X_val_path   "{input.X_val}" \
            --y_val_path   "{input.y_val}" \
            --experiment_config_path "{params.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            --regression_type "{wildcards.reg}" \
            --alpha {params.alpha} \
            --l1_ratio {params.l1_ratio} \
            {params.gridflag}

        test -f "{output.obj}" && test -f "{output.errjs}" && test -f "{output.mdl}"
        """

##############################################################################
# RULE random_forest                                                         #
##############################################################################
rule random_forest:
    input:
        X_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        y_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        X_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        y_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",
        shades  = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors  = f"experiments/{MODEL}/modeling/main_colors.pkl",
        expcfg  = EXP_CFG,
        mdlcfg  = "config_files/model_config.yaml"
    output:
        obj   = f"experiments/{MODEL}/modeling/random_forest/random_forest_mdl_obj.pkl",
        errjs = f"experiments/{MODEL}/modeling/random_forest/random_forest_model_error.json",
        mdl   = f"experiments/{MODEL}/modeling/random_forest/random_forest_model.pkl",
        plot  = f"experiments/{MODEL}/modeling/random_forest/random_forest_results.png",
        fi    = f"experiments/{MODEL}/modeling/random_forest/random_forest_feature_importances.png"
    params:
        script    = "snakemake_scripts/random_forest.py",
        model_dir = f"experiments/{MODEL}/modeling/random_forest",
        opt_flags = lambda w: " ".join([
            f"--n_estimators {config['rf']['n_estimators']}" \
                if config.get('rf', {}).get('n_estimators') is not None else "",
            f"--max_depth {config['rf']['max_depth']}" \
                if config.get('rf', {}).get('max_depth') is not None else "",
            f"--min_samples_split {config['rf']['min_samples_split']}" \
                if config.get('rf', {}).get('min_samples_split') is not None else "",
            f"--random_state {config['rf']['random_state']}" \
                if config.get('rf', {}).get('random_state') is not None else "",
            f"--n_iter {config['rf']['n_iter']}" \
                if config.get('rf', {}).get('n_iter') is not None else "",
            "--do_random_search" if config.get('rf', {}).get('random_search', False) else ""
        ]).strip()
    threads: 4
    benchmark:
        "benchmarks/random_forest.tsv"
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --X_train_path "{input.X_train}" \
            --y_train_path "{input.y_train}" \
            --X_val_path   "{input.X_val}" \
            --y_val_path   "{input.y_val}" \
            --experiment_config_path "{input.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            --model_directory "{params.model_dir}" \
            {params.opt_flags}

        test -f "{output.obj}"   && \
        test -f "{output.errjs}" && \
        test -f "{output.mdl}"   && \
        test -f "{output.plot}"  && \
        test -f "{output.fi}"
        """

##############################################################################
# RULE xgboost                                                               #
##############################################################################
rule xgboost:
    input:
        X_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        y_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        X_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        y_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",
        shades  = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors  = f"experiments/{MODEL}/modeling/main_colors.pkl",
        expcfg  = EXP_CFG,
        mdlcfg  = "config_files/model_config.yaml"
    output:
        obj   = f"experiments/{MODEL}/modeling/xgboost/xgb_mdl_obj.pkl",
        errjs = f"experiments/{MODEL}/modeling/xgboost/xgb_model_error.json",
        mdl   = f"experiments/{MODEL}/modeling/xgboost/xgb_model.pkl",
        plot  = f"experiments/{MODEL}/modeling/xgboost/xgb_results.png",
        fi    = f"experiments/{MODEL}/modeling/xgboost/xgb_feature_importances.png"
    params:
        script    = "snakemake_scripts/xgboost_evaluation.py",
        model_dir = f"experiments/{MODEL}/modeling/xgboost",
        opt_flags = lambda w: " ".join([
            f"--n_estimators {config['xgb']['n_estimators']}" \
                if config.get('xgb', {}).get('n_estimators') is not None else "",
            f"--max_depth {config['xgb']['max_depth']}" \
                if config.get('xgb', {}).get('max_depth') is not None else "",
            f"--learning_rate {config['xgb']['learning_rate']}" \
                if config.get('xgb', {}).get('learning_rate') is not None else "",
            f"--subsample {config['xgb']['subsample']}" \
                if config.get('xgb', {}).get('subsample') is not None else "",
            f"--colsample_bytree {config['xgb']['colsample_bytree']}" \
                if config.get('xgb', {}).get('colsample_bytree') is not None else "",
            f"--min_child_weight {config['xgb']['min_child_weight']}" \
                if config.get('xgb', {}).get('min_child_weight') is not None else "",
            f"--reg_lambda {config['xgb']['reg_lambda']}" \
                if config.get('xgb', {}).get('reg_lambda') is not None else "",
            f"--reg_alpha {config['xgb']['reg_alpha']}" \
                if config.get('xgb', {}).get('reg_alpha') is not None else "",
            f"--n_iter {config['xgb']['n_iter']}" \
                if config.get('xgb', {}).get('n_iter') is not None else "",
            f"--top_k_features_plot {config['xgb']['top_k_plot']}" \
                if config.get('xgb', {}).get('top_k_plot') is not None else "",
            "--do_random_search" if config.get('xgb', {}).get('do_random_search', False) else ""
        ]).strip()
    threads: 4
    benchmark:
        "benchmarks/xgboost.tsv"
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --X_train_path "{input.X_train}" \
            --y_train_path "{input.y_train}" \
            --X_val_path   "{input.X_val}" \
            --y_val_path   "{input.y_val}" \
            --experiment_config_path "{input.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            --model_directory "{params.model_dir}" \
            {params.opt_flags}

        test -f "{output.obj}"   && \
        test -f "{output.errjs}" && \
        test -f "{output.mdl}"   && \
        test -f "{output.plot}"  && \
        test -f "{output.fi}"
        """

##############################################################################
# Wildcard Constraints                                                      #
##############################################################################
wildcard_constraints:
    opt = "|".join(str(i) for i in range(NUM_OPTIMS))
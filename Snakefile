##############################################################################
# Combined workflow (all stages). Kept so the bash_scripts/ drivers, which
# call --snakefile Snakefile with explicit targets, keep working unchanged.
# Per-stage entry points: simulation.smk, modeling.smk, real_data.smk.
##############################################################################
include: "rules/common.smk"
include: "rules/simulation_rules.smk"
include: "rules/modeling_rules.smk"
include: "rules/real_data_rules.smk"

##############################################################################
# RULE all – final targets the workflow must create
##############################################################################
rule all:
    default_target: True
    input:
        [
            # ======================================================================
            # SIMULATED DATA
            # ======================================================================

            ## ── 1. RAW SIMULATION OUTPUTS ───────────────────────────────────────
            expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",  sid=SIM_IDS),
            expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",             sid=SIM_IDS),
            expand(f"{SIM_BASEDIR}/{{sid}}/demes.png",           sid=SIM_IDS),

            ## ── 2. PER-RUN SFS INFERENCE (sim) ──────────────────────────────────
            expand(
                f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/moments/best_fit.pkl",
                sid=SIM_IDS,
                opt=OPTIMS,
            ),
            # expand(
            #     f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/dadi/best_fit.pkl",
            #     sid=SIM_IDS,
            #     opt=OPTIMS,
            # ),

            # ── 3. CONSOLIDATED SIM INFERENCES ──────────────────────────────────
            expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl", sid=SIM_IDS),
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",    sid=SIM_IDS),
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/cleanup_done.txt",       sid=SIM_IDS),

            ## ── 4. MOMENTS-LD (SIMULATED) ────────────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/best_fit.pkl",
            #     sid=SIM_IDS,
            # ),

            ## ── 5. MOMENTS-LD OPTIMIZATION (always at MomentsLD/best_fit.pkl) ─────
            # expand(
            #     f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/best_fit.pkl",
            #     sid=SIM_IDS,
            # ),

            ## ======================================================================
            ## ACTIVE TARGETS
            ## ======================================================================

            # ── 1. MODELING DATASETS ────────────────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/datasets/features_df.pkl",
            #     variant=MODELING_VARIANTS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/datasets/targets_df.pkl",
            #     variant=MODELING_VARIANTS,
            # ),

            # ── 2. LINEAR REGRESSION ────────────────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl",
            #     variant=MODELING_VARIANTS, reg=REG_TYPES,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_model_error_{{reg}}.json",
            #     variant=MODELING_VARIANTS, reg=REG_TYPES,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/linear_{{reg}}/linear_regression_model_{{reg}}.pkl",
            #     variant=MODELING_VARIANTS, reg=REG_TYPES,
            # ),

            # ── 3. RANDOM FOREST ────────────────────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_mdl_obj.pkl",
            #     variant=MODELING_VARIANTS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_model_error.json",
            #     variant=MODELING_VARIANTS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_model.pkl",
            #     variant=MODELING_VARIANTS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/random_forest/random_forest_feature_importances.png",
            #     variant=MODELING_VARIANTS,
            # ),

            # ── 4. XGBOOST ──────────────────────────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_mdl_obj.pkl",
            #     variant=MODELING_VARIANTS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_model_error.json",
            #     variant=MODELING_VARIANTS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_model.pkl",
            #     variant=MODELING_VARIANTS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling_{{variant}}/xgboost/xgb_feature_importances.png",
            #     variant=MODELING_VARIANTS,
            # ),

            # # ── 5. REAL DATA (DROSOPHILA) ───────────────────────────────────────
            # POLARIZED_VCF,
            # POLARIZED_VCF + ".tbi",
            # UNFOLDED_SFS,

            # # ── 6. REAL DATA: SFS INFERENCE ─────────────────────────────────────
            # f"{REAL_INF_ROOT}/moments/best_fit.pkl",
            # f"{REAL_INF_ROOT}/dadi/best_fit.pkl",

            # # ── 7. REAL DATA: LD ─────────────────────────────────────────────────
            # expand(f"{REAL_LD_ROOT}/LD_stats/LD_stats_window_{{i}}.pkl", i=WINDOWS),
            # f"{REAL_INF_ROOT}/MomentsLD/best_fit.pkl",
        ]

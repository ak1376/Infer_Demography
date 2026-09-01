##############################################################################
# CONFIG – Paths and Constants (edit here only)                              #
##############################################################################
import json, math, sys, os
from pathlib import Path
from snakemake.io import protected, ancient

# Guard against bottleneck shadow import issues
if "bottleneck" in sys.modules and not hasattr(sys.modules["bottleneck"], "__version__"):
    del sys.modules["bottleneck"]
try:
    import bottleneck
    if not hasattr(bottleneck, "__version__"):
        bottleneck.__version__ = "1.0.0"
except ImportError:
    pass

configfile: "config_files/model_config.yaml"

# External scripts
SIM_SCRIPT   = "snakemake_scripts/simulation.py"
INFER_SCRIPT = "snakemake_scripts/moments_dadi_inference.py"
WIN_SCRIPT   = "snakemake_scripts/simulate_window_replicate.py"
LD_SCRIPT    = "snakemake_scripts/compute_ld_window.py"
RESID_SCRIPT = "snakemake_scripts/computing_residuals_from_sfs.py"
EXP_CFG = config["active_experiment_config"]

# Experiment metadata
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = int(CFG["num_draws"])
NUM_OPTIMS    = int(CFG.get("num_optimizations", 3))
NUM_REAL_OPTIMS = int(CFG.get("num_optimizations", 3))
TOP_K         = int(CFG.get("top_k", 2))
NUM_WINDOWS   = int(CFG.get("num_windows", 100))
WINDOW_SIZE   = 10_000_000

# window_mode: "replicates" (default) independently re-simulates each of the
# NUM_WINDOWS windows for a sid at chunk_genome_length bp each
# (simulate_one_window_replicate, one msprime/SLiM run per window). "chunked"
# simulates ONE original tree sequence per sid at genome_length bp and chops
# it into NUM_WINDOWS overlapping windows of chunk_genome_length bp each via
# src/windowing.py::window_trees. chunk_genome_length falls back to
# genome_length when not set in the config (today's un-chunked behavior).
WINDOW_MODE = CFG.get("window_mode", "replicates")
if WINDOW_MODE not in ("replicates", "chunked"):
    raise ValueError(f"window_mode must be 'replicates' or 'chunked', got {WINDOW_MODE!r}")

# Engines to COMPUTE (always); modeling usage is controlled in feature_extraction via config
FIM_ENGINES = CFG.get("fim_engines", ["moments"])

USE_GPU_LD = CFG.get("use_gpu_ld", False)
USE_GPU_DADI = CFG.get("use_gpu_dadi", False)

USE_GS = bool(CFG.get("gram_schmidt", False))

# Make sure these match files that actually exist in your repo
DROSO_DIR        = "real_data_analysis/data/drosophila"
AUTOSOMES        = ["Chr3L"]                                     # Chr2L, Chr2R, Chr3R dropped -- Chr3L only
ANCESTRAL_DIR    = "drosophila_data/dpgp_ancestor"                        # relative to repo root, alongside drosophila_data/data/

# Data now lives in per-chromosome subdirs: {DROSO_DIR}/{chrom}/{polarized,polarized.diploidGT,unfolded.sfs}...
RAW_HAPLOID_VCF  = "drosophila_data/data/Chr3L.vcf.gz"                    # legacy Chr3L alias
REAL_POPFILE     = f"{DROSO_DIR}/popfile.txt"
REAL_VCF         = f"{DROSO_DIR}/Chr3L/polarized.diploidGT.vcf.gz"        # diploid polarized (Chr3L); used by MomentsLD-real
POLARIZED_VCF    = f"{DROSO_DIR}/Chr3L/polarized.vcf.gz"                  # haploid + AA (Chr3L); legacy alias
UNFOLDED_SFS     = f"{DROSO_DIR}/Chr3L/unfolded.sfs.pkl"                  # per-chrom SFS (Chr3L); legacy alias
COMBINED_SFS     = f"{DROSO_DIR}/combined/autosomes.unfolded.sfs.pkl"     # summed autosomal SFS; used by SFS inference
COMBINED_SFS_META = f"{DROSO_DIR}/combined/autosomes.unfolded.sfs.meta.json"  # summed sequence_length across AUTOSOMES
ANCESTRAL_FASTA  = f"{ANCESTRAL_DIR}/chr3L.q30.fa"                        # legacy Chr3L alias

# Per-chromosome path helpers (by-chromosome layout)
def polarized_vcf(chrom):          return f"{DROSO_DIR}/{chrom}/polarized.vcf.gz"
def polarized_diploid_vcf(chrom):  return f"{DROSO_DIR}/{chrom}/polarized.diploidGT.vcf.gz"
def per_chrom_sfs(chrom):          return f"{DROSO_DIR}/{chrom}/unfolded.sfs.pkl"
def ancestral_fasta(chrom):        return f"{ANCESTRAL_DIR}/{chrom.replace('Chr', 'chr', 1)}.q30.fa"

def _resid_vector_fname():
    # which vector do we want to feed into all_inferences.pkl?
    return "residuals_gs_coeffs.npy" if USE_GS else "residuals_flat.npy"

def _resid_vector_regex():
    # for combine_results parsing
    return r"residuals_gs_coeffs\.npy$" if USE_GS else r"residuals_flat\.npy$"


def _normalize_residual_engines(val):
    # accepts "moments", "dadi", "both", list/tuple
    if isinstance(val, str):
        v = val.lower()
        return ["moments", "dadi"] if v in {"both", "all"} else [v]
    if isinstance(val, (list, tuple, set)):
        return [e for e in val if e in {"moments","dadi"}] or ["moments","dadi"]
    return ["moments","dadi"]

RESIDUAL_ENGINES = _normalize_residual_engines(CFG.get("residual_engines", "both"))

# ── Modeling feature-set variants ───────────────────────────────────────────
# Each variant is a distinct feature set produced from the SAME all_inferences.pkl,
# differing only in whether FIM elements / SFS-residual elements are included as
# features. The {variant} wildcard routes datasets + trained models into
#   experiments/<model>/modeling_<variant>/
# so all four can be built in one run with no manual renaming.
MODELING_VARIANTS = [
    "w_FIM_w_SFSresids",
    "w_FIM_wo_SFSresids",
    "wo_FIM_w_SFSresids",
    "wo_FIM_wo_SFSresids",
]

# (use_fim_features, use_residuals) per variant name.
_VARIANT_FLAGS = {
    "w_FIM_w_SFSresids":   (True,  True),
    "w_FIM_wo_SFSresids":  (True,  False),
    "wo_FIM_w_SFSresids":  (False, True),
    "wo_FIM_wo_SFSresids": (False, False),
}

def _variant_flags(variant):
    return _VARIANT_FLAGS[variant]

# CLI-flag builders for random_forest/xgboost — same optuna/manual-override
# knobs regardless of which modeling_{variant} the rule instantiates for
# (including variant="raw_features", see RAW_FEAT_DIR/RAW_MDL_DIR below).
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

# Windows & sims
SIM_IDS  = list(range(NUM_DRAWS))
WINDOWS  = range(NUM_WINDOWS)
OPTIMS  = list(range(NUM_OPTIMS))

# Canonical path builders
SIM_BASEDIR = f"experiments/{MODEL}/simulations"
RUN_DIR     = lambda sid, opt: f"experiments/{MODEL}/runs/run_{sid}_{opt}"
LD_ROOT     = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD"

# Real-data LD windows/LD_stats/aggregated means+varcovs are pure functions of
# the VCF data (chrom, window size, r_bins) -- not of which demographic model
# you're fitting -- so this lives under DROSO_DIR (shared across every model)
# instead of experiments/{MODEL}/...: switching MODEL never re-triggers the
# window split or the (GPU-bound) per-window LD computation. The MomentsLD
# *fit itself* (aggregate_opts_momentsld_real's best_fit.pkl) is genuinely
# model-specific and lives under REAL_INF_ROOT instead -- see that rule below.
REAL_LD_ROOT = f"{DROSO_DIR}/MomentsLD"
# Per-autosome LD decay analysis (independent of the Chr3L inference pipeline above)
REAL_LD_BYCHROM = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_by_chrom"
# Same, but with the real Comeron (R5/dm3) recombination map and 1 Mb windows.
REAL_LD_GENMAP     = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_genmap"
GENMAP_WINDOW_SIZE = 1_000_000
COMERON_XLSX       = f"{DROSO_DIR}/recombination_maps/Comeron_100kb_R5_R6.xlsx"
REAL_RUN_ROOT = f"experiments/{MODEL}/real_data_analysis/runs"
REAL_INF_ROOT = f"experiments/{MODEL}/real_data_analysis/inferences"
REAL_OPTIMS   = list(range(NUM_REAL_OPTIMS))

# Single-chromosome variants of REAL_RUN_ROOT/REAL_INF_ROOT (as opposed to the
# combined-autosome COMBINED_SFS used by infer_engine_real above). {chrom} is
# a real Snakemake wildcard here (double-braced so the f-string leaves it
# literal), constrained below to Chr(2L|2R|3L|3R).
REAL_RUN_ROOT_CHROM = f"experiments/{MODEL}/real_data_analysis/{{chrom}}/runs"
REAL_INF_ROOT_CHROM = f"experiments/{MODEL}/real_data_analysis/{{chrom}}/inferences"

# Number of top replicates the real moments/dadi aggregation keeps.
# Must match the rep count the trained model was built with (moments_*_rep_0..N-1),
# i.e. the sim-side TOP_K -- defaults to TOP_K unless overridden.
REAL_TOP_K    = int(CFG.get("real_top_k", TOP_K))

# ── Real-data prediction (push real fits through a trained model) ───────────
# Generalized over {variant} (the same MODELING_VARIANTS the sim pipeline
# trains) instead of pinning to a single modeling dir via a
# real_predict_modeling_dir config override -- so real-data predictions can
# be compared with/without FIM and SFS-residual features. Each variant's
# outputs live under their own prediction_{variant}/ dir.
REAL_PRED_ROOT = f"experiments/{MODEL}/real_data_analysis/prediction_{{variant}}"

def _real_modeling_dir(variant):
    return f"experiments/{MODEL}/modeling_{variant}"

def _real_train_features(variant):
    return f"{_real_modeling_dir(variant)}/datasets/features_df.pkl"

# model_key wildcard -> trained *_mdl_obj.pkl path, for a given variant
def _real_model_objs(variant):
    d = _real_modeling_dir(variant)
    return {
        "random_forest":     f"{d}/random_forest/random_forest_mdl_obj.pkl",
        "xgboost":           f"{d}/xgboost/xgb_mdl_obj.pkl",
        "linear_standard":   f"{d}/linear_standard/linear_mdl_obj_standard.pkl",
        "linear_ridge":      f"{d}/linear_ridge/linear_mdl_obj_ridge.pkl",
        "linear_lasso":      f"{d}/linear_lasso/linear_mdl_obj_lasso.pkl",
        "linear_elasticnet": f"{d}/linear_elasticnet/linear_mdl_obj_elasticnet.pkl",
    }

# model_key set is the same across variants -- any variant's keys will do.
REAL_MODEL_KEYS = list(_real_model_objs(MODELING_VARIANTS[0]).keys())

wildcard_constraints:
    chrom      = r"Chr(2L|2R|3L|3R)",
    # combine_features narrows this back down (raw_features has its own
    # producer rules, build_raw_features_dataset/prepare_raw_features_splits).
    variant    = r"(w|wo)_FIM_(w|wo)_SFSresids|raw_features",
    ld_variant = r"by_chrom|genmap",
    reg        = r"standard|ridge|lasso|elasticnet",
    opt        = "|".join(str(i) for i in range(NUM_OPTIMS)),
    engine     = "moments|dadi",
    frac_tag   = r"thin\d+|n\d+",
    model_key  = "|".join(REAL_MODEL_KEYS),

# LD r-bins
# 16 log-spaced edges from 1e-6 to 1e-3 (ratio ~1.58x) plus a leading 0,
# matching the moments.LD tutorial's bin density -- finer than the old
# 9-edge (~2-2.5x ratio) set, to reduce quadrature error in compute_theoretical_ld
# for the short-range bins.
R_BINS_STR = "0,1e-06,1.58489e-06,2.51189e-06,3.98107e-06,6.30957e-06,1e-05,1.58489e-05,2.51189e-05,3.98107e-05,6.30957e-05,0.0001,0.000158489,0.000251189,0.000398107,0.000630957,0.001"

# Optional pruning — set "prune_mode": "fraction"|"count" and
# "prune_keep_values" in EXP_CFG to enable.
#   fraction: keep a fixed % of sites per window regardless of density
#             (values are fractions, e.g. [0.15] -> thin15/)
#   count:    cap each window at min(n_sites, N) -- windows already below N
#             are left untouched (values are absolute counts, e.g.
#             [5000, 20000] -> n5000/, n20000/)
PRUNE_MODE   = CFG.get("prune_mode", "off")
PRUNE_VALUES = CFG.get("prune_keep_values", [])
def _frac_tag(f): return f"thin{round(float(f) * 100):02d}"
def _count_tag(n): return f"n{int(n)}"
if PRUNE_MODE == "fraction":
    PRUNE_TAGS = [_frac_tag(v) for v in PRUNE_VALUES]
elif PRUNE_MODE == "count":
    PRUNE_TAGS = [_count_tag(v) for v in PRUNE_VALUES]
else:
    PRUNE_TAGS = []

# What cleanup_optimization_runs (below) waits on / reads for MomentsLD's
# keep-set: pruned mode aggregates per frac_tag under MomentsLD/pruning/<tag>/
# instead of directly under MomentsLD/, so there's one target per tag rather
# than the single unpruned LD_ROOT/best_fit.pkl.
if PRUNE_TAGS:
    MOMENTSLD_AGG_TARGETS = [
        f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{tag}/best_fit.pkl"
        for tag in PRUNE_TAGS
    ]
else:
    MOMENTSLD_AGG_TARGETS = [f"{LD_ROOT}/best_fit.pkl"]

# The single MomentsLD result combine_results (below) folds in per sim. In
# pruned mode there's no unpruned MomentsLD/best_fit.pkl at all -- only
# infer_momentsld_pruned/aggregate_opts_momentsld_pruned ever run -- so this
# must point at the pruned path; PRUNE_TAGS[0] matches the "one canonical
# pruning variant feeds downstream" convention aggregate_ld_stats already
# uses for its --fallback-ld-dir above.
MOMENTSLD_BEST_FIT = (
    MOMENTSLD_AGG_TARGETS[0] if PRUNE_TAGS else f"{LD_ROOT}/best_fit.pkl"
)

##############################################################################
# RULE all – final targets the workflow must create
##############################################################################
rule all:
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
##############################################################################
# RULE simulate – one complete tree‑sequence + SFS
##############################################################################
rule simulate:
    output:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        fig    = f"{SIM_BASEDIR}/{{sid}}/demes.png",
        meta   = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
        # window_mode="chunked": chunk_window (a separate, later job) slices
        # this same tree sequence into windows, so it must persist past this
        # rule's own job. window_mode="replicates" never consumes it, so it
        # stays temp (auto-deleted) there, same as always.
        ts     = (f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees" if WINDOW_MODE == "chunked"
                  else temp(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees")),
        done   = protected(f"{SIM_BASEDIR}/{{sid}}/.done"),
    params:
        sim_dir = SIM_BASEDIR,
        cfg     = EXP_CFG,
        model   = MODEL
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "{SIM_SCRIPT}" \
          --simulation-dir "{params.sim_dir}" \
          --experiment-config "{params.cfg}" \
          --model-type "{params.model}" \
          --simulation-number {wildcards.sid}

        # ensure expected outputs exist, then create sentinel
        test -f "{output.sfs}"    && \
        test -f "{output.params}" && \
        test -f "{output.fig}"    && \
        test -f "{output.meta}"
        touch "{output.done}"
        """


##############################################################################
# RULE infer_moments  – custom NLopt Poisson SFS optimisation (moments)
##############################################################################
rule infer_engine:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",   # not read by moments; kept for DAG clarity
        cfg    = EXP_CFG
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/{{engine}}/best_fit.pkl"
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        fix      = ""     # e.g. '--fix N0=10000 --fix m12=0.0'
    # No internal multi-threaded code path in moments_dadi_inference.py --
    # OMP/MKL threads only apply to whatever BLAS backend numpy uses, and at
    # this grid size (pts_base up to ~40) that overhead outweighs any benefit.
    # threads:1 also lets moments.sh/dadi.sh run several restarts concurrently
    # per array task (via -j) without oversubscribing cores.
    threads: 1
    shell:
        r"""
        set -euo pipefail

        # Skip already-completed opts even if Snakemake scheduled this job (e.g.
        # because the config file's mtime changed on a git pull) — don't waste
        # time/compute re-running an optimization that already finished.
        if [ -s "{output.pkl}" ]; then
            echo "SKIP: {output.pkl} already exists and is non-empty"
            exit 0
        fi

        echo "===== infer_{wildcards.engine} ENV ====="
        echo "sid={wildcards.sid} opt={wildcards.opt}"
        echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-unset}} SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID:-unset}}"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"
        echo "PYCUDA_CACHE_DIR=${{PYCUDA_CACHE_DIR:-unset}}"
        echo "CUDAHOSTCXX=${{CUDAHOSTCXX:-unset}}"
        command -v nvcc >/dev/null 2>&1 && echo "nvcc=$(command -v nvcc)" || echo "nvcc=NOT_FOUND"
        nvidia-smi -L || true

        # Keep threading sane
        export OMP_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}

        # Ensure output dirs exist
        mkdir -p "{params.run_dir}/inferences/{wildcards.engine}"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/moments_dadi_inference.py" \
          --mode {wildcards.engine} \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --ground-truth "{input.params}" \
          --outdir "{params.run_dir}/inferences" \
          --opt-seed {wildcards.opt} {params.fix}
        """

# ── MOMENTS / DADI (sim) ────────────────────────────────────────────────────
rule aggregate_opts_engine:
    input:
        cfg = EXP_CFG,
        # No hard dependency on the full OPTIMS range here on purpose: aggregation
        # should work on however many per-opt files currently exist (discovered via
        # glob in the run: block below), not force every optimization to finish
        # first. aggregate_moments_dadi.sh's FORCE=1 is what triggers a fresh
        # aggregation pass once more opts have completed.
    output:
        pkl = f"experiments/{MODEL}/inferences/sim_{{sid}}/{{engine}}/fit_params.pkl"
    run:
        import pickle, pathlib
        from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

        sid = wildcards.sid
        engine = wildcards.engine
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        records = discover_opt_pkls(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/{engine}/best_fit.pkl",
            rf"/run_{sid}_(\d+)/inferences/{engine}/best_fit\.pkl$",
        )

        best, diag = aggregate_top_k(
            records, TOP_K, min_nonempty=MIN_FILES,
            err_label=f"aggregate_opts_{engine}", err_engine=engine, err_context=f"for sid={sid}",
        )
        best["n_files_found"] = diag["n_records"]
        best["n_nonempty"]    = diag["n_nonempty"]
        best["min_required"]  = int(TOP_K)

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        with open(output.pkl, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ {engine}: found {diag['n_records']} files, aggregated {diag['n_entries']} entries → {output.pkl}")
        print(f"✅ {engine}: kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

# ── CLEANUP RULE: Remove non-top-K optimization runs after all three
#    engines (dadi, moments, MomentsLD) have finished aggregating. Must wait
#    on MomentsLD too -- its top-K opt indices (by LD likelihood) don't
#    necessarily overlap with dadi/moments' (by SFS likelihood), so deleting
#    a run dir just because it wasn't in the dadi/moments keep-set can throw
#    away a restart MomentsLD still needed. ─────────────────────────────────
rule cleanup_optimization_runs:
    input:
        dadi       = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",
        moments    = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl",
        momentsld  = MOMENTSLD_AGG_TARGETS,
    output:
        cleanup_done = f"experiments/{MODEL}/inferences/sim_{{sid}}/cleanup_done.txt"
    run:
        import pickle, pathlib, subprocess

        sid = wildcards.sid

        # --- skip conditions ---
        if NUM_OPTIMS <= 1 or TOP_K >= NUM_OPTIMS:
            pathlib.Path(output.cleanup_done).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(output.cleanup_done).write_text(
                f"Cleanup skipped for simulation {sid} (NUM_OPTIMS={NUM_OPTIMS}, TOP_K={TOP_K}).\n"
            )
            print(f"✅ cleanup skipped sid={sid}")
            return

        # --- existing cleanup logic ---
        with open(input.dadi, "rb") as f:
            dadi_data = pickle.load(f)
        with open(input.moments, "rb") as f:
            moments_data = pickle.load(f)
        momentsld_keep = set()
        for momentsld_path in input.momentsld:
            with open(momentsld_path, "rb") as f:
                momentsld_data = pickle.load(f)
            momentsld_keep |= set((momentsld_data.get("opt_index") or [])[:TOP_K])

        dadi_keep    = set((dadi_data.get("opt_index") or [])[:TOP_K])
        moments_keep = set((moments_data.get("opt_index") or [])[:TOP_K])
        keep_indices = dadi_keep | moments_keep | momentsld_keep

        run_root = pathlib.Path(f"experiments/{MODEL}/runs")
        prefix = f"run_{sid}_"

        cleaned = 0
        for p in run_root.glob(f"{prefix}*"):
            if not p.is_dir():
                continue
            try:
                opt = int(p.name.rsplit("_", 1)[1])
            except Exception:
                continue
            if opt in keep_indices:
                continue

            subprocess.run(["rm", "-rf", str(p)], check=False)
            cleaned += 1

        pathlib.Path(output.cleanup_done).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(output.cleanup_done).write_text(
            f"Cleanup completed for simulation {sid}\n"
            f"Removed {cleaned} optimization directories\n"
            f"Kept optimizations: {sorted(keep_indices)}\n"
        )

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
# RULE simulate_window_replicate / chunk_window – one VCF window
# window_mode="replicates": simulate_window_replicate independently
# re-simulates each window from scratch, at chunk_genome_length bp
# (falls back to genome_length if chunk_genome_length is unset).
# window_mode="chunked": chunk_window chops rule simulate's own
# tree_sequence.trees (genome_length bp, the original/full sequence
# already simulated for the SFS) into NUM_WINDOWS overlapping windows of
# chunk_genome_length bp each (src/windowing.py) – no simulation happens
# in this rule, just slicing. This reuses simulate's tree sequence
# instead of re-simulating a second one, so there's no redundant
# simulation between the SFS path and the LD/windowing path.
# Both branches produce the same output path so downstream rules
# (ld_window) and callers don't need to know which mode is active.
##############################################################################
if WINDOW_MODE == "replicates":
    rule simulate_window_replicate:
        input:
            params   = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
            metafile = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
            done     = f"{SIM_BASEDIR}/{{sid}}/.done"
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
            set -euo pipefail
            PYTHONPATH={workflow.basedir} \
            python "{WIN_SCRIPT}" \
                --sim-dir      "{params.base_sim}" \
                --rep-index    {params.rep_idx} \
                --config-file  "{params.cfg}" \
                --meta-file    "{input.metafile}" \
                --out-dir      "{params.out_winDir}"
            """
else:
    # Two rules instead of one: materialize_sim_vcf writes the WHOLE genome
    # to one bgzipped, indexed VCF (one per simulation, no keep_intervals/
    # simplify at all -- see src.windowing.materialize_full_vcf). chunk_window
    # then does a per-window `bcftools view -r` slice against that finished
    # file. Measured ~4.6x faster per window than the tree-sequence
    # keep_intervals(simplify=True) approach on a test tree sequence, and
    # since slicing is read-only against an already-finished file, every
    # window's chunk_window job is independent again -- safe to run fully in
    # parallel across windows, same as before any of this was batched.
    rule materialize_sim_vcf:
        input:
            trees = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        output:
            # Not temp(): chunk_window's --allowed-rules restriction (see
            # build_windows.sh) means it can't rebuild these if a later
            # retry/rerun of any single window needs them again after
            # Snakemake's temp-cleanup removed them -- MissingInputException,
            # stuck. One whole-genome VCF kept per simulation is cheap
            # relative to everything else in this pipeline.
            vcf_gz  = f"{LD_ROOT}/windows/full_genome.vcf.gz",
            vcf_tbi = f"{LD_ROOT}/windows/full_genome.vcf.gz.tbi",
            samples = f"{LD_ROOT}/windows/samples.txt",
        params:
            out_winDir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
        threads: 1
        run:
            from src.windowing import materialize_full_vcf

            materialize_full_vcf(Path(input.trees), Path(params.out_winDir))

    rule chunk_window:
        input:
            vcf_gz  = f"{LD_ROOT}/windows/full_genome.vcf.gz",
            vcf_tbi = f"{LD_ROOT}/windows/full_genome.vcf.gz.tbi",
        output:
            vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz"
        params:
            out_winDir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
            cfg         = EXP_CFG,
            num_windows = NUM_WINDOWS,
        threads: 1
        run:
            import json
            from src.windowing import window_vcf

            cfg = json.loads(Path(params.cfg).read_text())
            window_size = int(cfg.get("ld_sequence_length", cfg["sequence_length"]))
            recomb_rate = float(cfg["recombination_rate"])

            window_vcf(
                Path(input.vcf_gz),
                Path(params.out_winDir),
                window_size=window_size,
                num_windows=params.num_windows,
                recomb_rate=recomb_rate,
                window_index=int(wildcards.win),
            )

##############################################################################
# RULE ld_window – LD statistics for one window
# prune_mode "off" (default): computes stats directly on the full window
# (unchanged, original behavior).
# prune_mode "fraction"/"count" set: LD pair count scales as
# density^2, so a dense window (many variants) can make the full
# computation prohibitively slow. This branch thins the window first
# (reusing rule prune_window's already-existing pruned VCF) and computes
# stats on THAT instead, writing to the exact same output path - nothing
# downstream (aggregate_ld_stats etc.) needs to know pruning happened.
##############################################################################
if PRUNE_TAGS:
    rule ld_window:
        input:
            vcf_gz = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{PRUNE_TAGS[0]}/windows/window_{{win}}.vcf.gz",
        output:
            pkl = f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl"
        params:
            sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{PRUNE_TAGS[0]}",
            bins    = R_BINS_STR,
            cfg     = EXP_CFG,
        # No threading/multiprocessing code in moments.LD.Parsing.compute_ld_statistics
        # (the CPU path used when use_gpu_ld=false) -- threads:1 (was 4) so
        # LD_stats_windows.sh's batched Snakemake call can actually run
        # multiple windows concurrently per array task instead of one at a
        # time (was previously exactly matching --cpus-per-task=4, so -j
        # never had room for a second job).
        threads: 1
        resources:
            ld_cores = 1,
            gpu      = 1
        shell:
            """
            set -euo pipefail
            python "{LD_SCRIPT}" \
                --sim-dir      {params.sim_dir} \
                --window-index {wildcards.win} \
                --config-file  {params.cfg} \
                --r-bins       "{params.bins}"

            mkdir -p "$(dirname "{output.pkl}")"
            mv "{params.sim_dir}/LD_stats/LD_stats_window_{wildcards.win}.pkl" "{output.pkl}"
            rm -f {params.sim_dir}/windows/window_{wildcards.win}.h5
            """
else:
    rule ld_window:
        input:
            vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
        output:
            pkl    = f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl"
        params:
            sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
            bins    = R_BINS_STR,
            cfg     = EXP_CFG
        # No threading/multiprocessing code in moments.LD.Parsing.compute_ld_statistics
        # (the CPU path used when use_gpu_ld=false) -- threads:1 (was 4) so
        # LD_stats_windows.sh's batched Snakemake call can actually run
        # multiple windows concurrently per array task instead of one at a
        # time (was previously exactly matching --cpus-per-task=4, so -j
        # never had room for a second job).
        threads: 1
        resources:
            ld_cores = 1,
            gpu      = 1
        shell:
            """
            set -euo pipefail
            python "{LD_SCRIPT}" \
                --sim-dir      {params.sim_dir} \
                --window-index {wildcards.win} \
                --config-file  {params.cfg} \
                --r-bins       "{params.bins}"

            # .h5 is written by the LD script but not declared as a Snakemake output;
            # remove it here so it doesn't accumulate across windows.
            rm -f {params.sim_dir}/windows/window_{wildcards.win}.h5
            """

##############################################################################
# RULE prune_window – thin one VCF to a keep-fraction                       #
##############################################################################
rule prune_window:
    input:
        vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
    output:
        pruned_vcf = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/windows/window_{{win}}.vcf.gz"),
    params:
        pruning_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning",
        windows_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
        # frac_tag encodes which pruning mode produced it: "thinNN" (keep NN%
        # of sites) or "nNNN" (cap at NNN sites, min(n_full, NNN) -- see
        # src/prune_vcf.py's --keep-counts mode).
        prune_flag  = lambda w: (
            f"--keep-counts {int(w.frac_tag[1:])}"
            if w.frac_tag.startswith("n")
            else f"--keep-fractions {int(w.frac_tag.replace('thin', '')) / 100}"
        ),
    threads: 1
    shell:
        """
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "src/prune_vcf.py" \
            --vcf            "{params.windows_dir}/window_{wildcards.win}.vcf.gz" \
            --out-dir        "{params.pruning_dir}" \
            {params.prune_flag}                     \
            --no-unpruned                           \
            --workers        1
        """

##############################################################################
# RULE ld_window_pruned – LD stats for one pruned window                    #
##############################################################################
rule ld_window_pruned:
    input:
        vcf_gz = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/windows/window_{{win}}.vcf.gz",
    output:
        pkl = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/LD_stats/LD_stats_window_{{win}}.pkl",
    params:
        sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{w.frac_tag}",
        bins    = R_BINS_STR,
        cfg     = EXP_CFG,
    # See rule ld_window above: no threading/multiprocessing in
    # moments.LD.Parsing.compute_ld_statistics, so threads:1 (was 4) lets
    # LD_stats_windows.sh's batched call run windows concurrently.
    threads: 1
    resources:
        ld_cores = 1,
        gpu      = 1
    shell:
        """
        set -euo pipefail
        python "{LD_SCRIPT}" \
            --sim-dir      {params.sim_dir} \
            --window-index {wildcards.win} \
            --config-file  {params.cfg} \
            --r-bins       "{params.bins}"

        rm -f {params.sim_dir}/windows/window_{wildcards.win}.h5
        """

##############################################################################
# RULE aggregate_ld_stats – aggregate LD windows into means/varcovs + PDF   #
# Works for unpruned-only AND mixed (unpruned primary + pruned fallback).   #
# Runs once per sid; every optimization restart (infer_momentsld, below)    #
# reuses this cached means.varcovs.pkl instead of re-aggregating.           #
##############################################################################
rule aggregate_ld_stats:
    input:
        pkls = lambda w: expand(
            f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl",
            sid=[w.sid],
            win=WINDOWS
        ),
        pruned_pkls = lambda w: (
            [
                f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{PRUNE_TAGS[0]}/LD_stats/LD_stats_window_{win}.pkl"
                for win in WINDOWS
                if not Path(f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/LD_stats/LD_stats_window_{win}.pkl").exists()
            ]
            if PRUNE_TAGS else []
        ),
        cfg = EXP_CFG,
    output:
        mv   = f"{LD_ROOT}/means.varcovs.pkl",
        # Not temp(): aggregate_ld_statistics() checks for this file (alongside
        # means.varcovs.pkl) to short-circuit re-aggregating LD_stats/*.pkl on
        # every downstream infer_momentsld restart. Deleting it right after this
        # rule finished (former temp()) silently broke that cache -- every one
        # of the num_optimizations restarts was re-reading and re-aggregating
        # all window LD_stats from scratch instead of reusing this.
        boot = f"{LD_ROOT}/bootstrap_sets.pkl",
        pdf  = f"{LD_ROOT}/empirical_vs_theoretical_comparison.pdf",
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        pruning_dir = lambda w: (
            f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{PRUNE_TAGS[0]}"
            if PRUNE_TAGS else ""
        ),
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    run:
        import subprocess
        cmd = [
            "python", "snakemake_scripts/LD_inference.py",
            "--run-dir",     params.sim_dir,
            "--output-root", params.output_root,
            "--config-file", params.cfg,
            "--r-bins",      params.bins,
            "--skip-optimize",
        ]
        if params.pruning_dir:
            cmd += ["--fallback-ld-dir", params.pruning_dir]
        env = {**os.environ, "PYTHONPATH": workflow.basedir}
        subprocess.run(cmd, check=True, env=env)

##############################################################################
# RULE infer_momentsld – one LHS/jitter-seeded MomentsLD restart            #
# Mirrors infer_moments/infer_dadi: one Snakemake job per {opt}, each a     #
# single nlopt run from a distinct start point keyed by opt_seed.           #
##############################################################################
rule infer_momentsld:
    input:
        mv  = f"{LD_ROOT}/means.varcovs.pkl",
        cfg = EXP_CFG,
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/MomentsLD/best_fit.pkl"
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        results_dir = lambda w: f"experiments/{MODEL}/runs/run_{w.sid}_{w.opt}/inferences/MomentsLD",
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.results_dir}"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/LD_inference.py" \
            --run-dir     "{params.sim_dir}" \
            --output-root "{params.output_root}" \
            --results-dir "{params.results_dir}" \
            --config-file "{params.cfg}" \
            --r-bins      "{params.bins}" \
            --opt-seed    {wildcards.opt}

        test -f "{output.pkl}"
        """

##############################################################################
# RULE aggregate_opts_momentsld – pick top-K across LHS/jitter restarts     #
# Mirrors aggregate_opts_moments/aggregate_opts_dadi exactly (same TOP_K,   #
# same best_params/best_ll/opt_index list schema).                         #
##############################################################################
rule aggregate_opts_momentsld:
    input:
        cfg = EXP_CFG,
        opts = lambda w: expand(
            f"experiments/{MODEL}/runs/run_{w.sid}_{{opt}}/inferences/MomentsLD/best_fit.pkl",
            opt=OPTIMS,
        ),
    output:
        best = f"{LD_ROOT}/best_fit.pkl",
    run:
        import pickle, pathlib
        from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

        sid = wildcards.sid
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        records = discover_opt_pkls(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/MomentsLD/best_fit.pkl",
            rf"/run_{sid}_(\d+)/inferences/MomentsLD/best_fit\.pkl$",
        )

        best, diag = aggregate_top_k(
            records, TOP_K, min_nonempty=MIN_FILES,
            err_label="aggregate_opts_momentsld", err_engine="MomentsLD", err_context=f"for sid={sid}",
        )
        best["n_files_found"] = diag["n_records"]
        best["n_nonempty"]    = diag["n_nonempty"]
        best["min_required"]  = int(TOP_K)

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        with open(output.best, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ momentsLD: found {diag['n_records']} files, aggregated {diag['n_entries']} entries → {output.best}")
        print(f"✅ momentsLD: kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

##############################################################################
# RULE aggregate_ld_stats_pruned – aggregate PRUNED per-window LD stats into  #
# means/varcovs (pruned-only; mirrors aggregate_ld_stats but reads pruned    #
# LD_stats and never runs the optimizer itself -- that's now a separate,     #
# multi-restart stage below (infer_momentsld_pruned / aggregate_opts_        #
# momentsld_pruned), same split as the unpruned aggregate_ld_stats /         #
# infer_momentsld / aggregate_opts_momentsld pattern.                        #
##############################################################################
rule aggregate_ld_stats_pruned:
    input:
        pruned_pkls = lambda w: expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/LD_stats/LD_stats_window_{{win}}.pkl",
            sid=[w.sid],
            frac_tag=[w.frac_tag],
            win=WINDOWS,
        ),
        cfg = EXP_CFG,
    output:
        mv   = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/means.varcovs.pkl",
        # Not temp() -- see aggregate_ld_stats's identical note above.
        boot = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/bootstrap_sets.pkl",
        pdf  = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/empirical_vs_theoretical_comparison.pdf",
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{w.frac_tag}",
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    run:
        import subprocess
        # output_root drives everything: LD_inference reads output_root/LD_stats/*.pkl
        # (the pruned stats) and writes means/varcovs/bootstrap/pdf there.
        # No --fallback-ld-dir: in pruned-only mode there are no unpruned stats.
        # --skip-optimize: aggregation only here, same as aggregate_ld_stats;
        # optimization is the separate multi-restart stage below.
        cmd = [
            "python", "snakemake_scripts/LD_inference.py",
            "--run-dir",     params.sim_dir,
            "--output-root", params.output_root,
            "--config-file", params.cfg,
            "--r-bins",      params.bins,
            "--skip-optimize",
        ]
        env = {**os.environ, "PYTHONPATH": workflow.basedir}
        subprocess.run(cmd, check=True, env=env)

##############################################################################
# RULE infer_momentsld_pruned – one LHS/jitter-seeded MomentsLD restart,     #
# pruned stats. Mirrors infer_momentsld exactly, keyed additionally by       #
# {frac_tag}.                                                                #
##############################################################################
rule infer_momentsld_pruned:
    input:
        mv  = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/means.varcovs.pkl",
        cfg = EXP_CFG,
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/MomentsLD/pruning/{{frac_tag}}/best_fit.pkl"
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{w.frac_tag}",
        results_dir = lambda w: f"experiments/{MODEL}/runs/run_{w.sid}_{w.opt}/inferences/MomentsLD/pruning/{w.frac_tag}",
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.results_dir}"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/LD_inference.py" \
            --run-dir     "{params.sim_dir}" \
            --output-root "{params.output_root}" \
            --results-dir "{params.results_dir}" \
            --config-file "{params.cfg}" \
            --r-bins      "{params.bins}" \
            --opt-seed    {wildcards.opt}

        test -f "{output.pkl}"
        """

##############################################################################
# RULE aggregate_opts_momentsld_pruned – pick top-K across LHS/jitter        #
# restarts, pruned stats. Mirrors aggregate_opts_momentsld exactly, keyed    #
# additionally by {frac_tag}. Output path is unchanged from the old         #
# optimize_momentsld_mixed target, so MomentsLD.sh's TARGET doesn't need to  #
# change -- only its --allowed-rules list does.                             #
##############################################################################
rule aggregate_opts_momentsld_pruned:
    input:
        cfg = EXP_CFG,
        opts = lambda w: expand(
            f"experiments/{MODEL}/runs/run_{w.sid}_{{opt}}/inferences/MomentsLD/pruning/{w.frac_tag}/best_fit.pkl",
            opt=OPTIMS,
        ),
    output:
        best = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/best_fit.pkl",
    run:
        import pickle, pathlib
        from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

        sid = wildcards.sid
        frac_tag = wildcards.frac_tag
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        records = discover_opt_pkls(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/MomentsLD/pruning/{frac_tag}/best_fit.pkl",
            rf"/run_{sid}_(\d+)/inferences/MomentsLD/pruning/{frac_tag}/best_fit\.pkl$",
        )

        best, diag = aggregate_top_k(
            records, TOP_K, min_nonempty=MIN_FILES,
            err_label="aggregate_opts_momentsld_pruned", err_engine="MomentsLD",
            err_context=f"for sid={sid} frac_tag={frac_tag}",
        )
        best["n_files_found"] = diag["n_records"]
        best["n_nonempty"]    = diag["n_nonempty"]
        best["min_required"]  = int(TOP_K)

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        with open(output.best, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ momentsLD (pruned, {frac_tag}): found {diag['n_records']} files, aggregated {diag['n_entries']} entries → {output.best}")
        print(f"✅ momentsLD (pruned, {frac_tag}): kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

##############################################################################
# RULE compute_fim – observed FIM at best-LL params for {engine}             #
##############################################################################
rule compute_fim:
    input:
        fit = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl",
        sfs = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl"
    output:
        fim  = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.fim.npy"),
        summ = f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.summary.json",
    params:
        script = "snakemake_scripts/compute_fim.py",
        cfg    = EXP_CFG
    threads: 2
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python {params.script} \
            --engine {wildcards.engine} \
            --fit-pkl {input.fit} \
            --sfs {input.sfs} \
            --config {params.cfg} \
            --fim-npy {output.fim} \
            --summary-json {output.summ}
        """


##############################################################################
# RULE sfs_residuals – optimized (best-fit) SFS − observed SFS               #
##############################################################################
rule sfs_residuals:
    input:
        obs_sfs = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        agg_fit = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl",
    output:
        res_arr   = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals.npy",
        res_flat  = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_flat.npy"),
        meta_json = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/meta.json"),
        hist_png  = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_histogram.png",

        # Only required when gram_schmidt=true; otherwise create temp sentinels
        gs_coeffs = (
            temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_gs_coeffs.npy")
            if USE_GS
            else temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/.gs_disabled")
        ),
        gs_basis = (
            f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_gs_basis.npy"
            if USE_GS
            else temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/.gs_basis_disabled")
        ),
    params:
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        inf_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}",
        out_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{w.engine}",
        n_bins   = CFG.get("sfs_n_bins", ""),  # empty string if not specified
        script   = "bash_scripts/run_sfs_residuals.sh",
    threads: 1
    shell:
        r"""
        bash "{params.script}" \
            "{wildcards.engine}" "{params.cfg}" "{params.model_py}" "{input.obs_sfs}" \
            "{params.inf_dir}" "{params.out_dir}" "{USE_GS}" "{RESID_SCRIPT}" \
            "{workflow.basedir}" "{params.n_bins}"
        """


##############################################################################
# RULE combine_results – merge dadi / moments / moments-LD fits per sim      #
# + attach flattened upper-triangular FIM and residual SFS payloads          #
##############################################################################
rule combine_results:
    input:
        cfg       = EXP_CFG,
        dadi      = lambda w: ancient(f"experiments/{MODEL}/inferences/sim_{w.sid}/dadi/fit_params.pkl")
                    if os.path.exists(f"experiments/{MODEL}/inferences/sim_{w.sid}/dadi/fit_params.pkl") else [],
        moments   = lambda w: ancient(f"experiments/{MODEL}/inferences/sim_{w.sid}/moments/fit_params.pkl")
                    if os.path.exists(f"experiments/{MODEL}/inferences/sim_{w.sid}/moments/fit_params.pkl") else [],
        momentsLD = lambda w: ancient(MOMENTSLD_BEST_FIT.format(sid=w.sid))
                    if os.path.exists(MOMENTSLD_BEST_FIT.format(sid=w.sid)) else [],
        # FIM/residuals are always computed (see FIM_ENGINES/RESIDUAL_ENGINES
        # above); use_fim_features/use_residuals only control whether
        # feature_extraction.py later uses them as modeling features.
        fims      = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ],
        resid_vecs = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ],
        resid_meta = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ],
    output:
        combo = f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl"
    run:
        import pickle, pathlib
        from src.combine_payloads import build_fim_payload, build_residual_payload

        if not input.dadi or not input.moments or not input.momentsLD:
            missing = [k for k, v in [("dadi", input.dadi), ("moments", input.moments), ("momentsLD", input.momentsLD)] if not v]
            raise RuntimeError(f"sim_{wildcards.sid}: skipping — prerequisites not ready: {missing}")

        outdir = pathlib.Path(output.combo).parent
        outdir.mkdir(parents=True, exist_ok=True)

        summary = {}
        summary["moments"]   = pickle.load(open(input.moments, "rb"))
        summary["dadi"]      = pickle.load(open(input.dadi, "rb"))
        summary["momentsLD"] = pickle.load(open(input.momentsLD, "rb"))

        fim_payload = build_fim_payload(input.fims)
        if fim_payload:
            summary["FIM"] = fim_payload

        # Either residuals_flat.npy (raw) or residuals_gs_coeffs.npy (GS reduced).
        resid_payload = build_residual_payload(input.resid_vecs)
        if resid_payload:
            summary["SFS_residuals"] = resid_payload

        pickle.dump(summary, open(output.combo, "wb"))
        print(f"✓ combined → {output.combo}")
        
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
        outliers_txt  = f"experiments/{MODEL}/modeling_{{variant}}/datasets/outliers_preview.txt"
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
# RULE annotate_ancestral_allele  (per-chromosome, autosomes)
# Polarize the raw haploid VCF using the DPGP ML-ancestor FASTA.
# FASTA naming differs from the VCF: "Chr2L" -> "chr2L.q30.fa".
# Adds an AA= INFO field; sites where the ancestral base is N or matches
# neither allele are dropped.  Output is bgzipped + tabix-indexed.
##############################################################################
rule annotate_ancestral_allele:
    input:
        vcf   = "drosophila_data/data/{chrom}.vcf.gz",
        tbi   = "drosophila_data/data/{chrom}.vcf.gz.tbi",
        fasta = lambda wc: ancestral_fasta(wc.chrom),
    output:
        vcf = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz",
        tbi = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz.tbi",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.vcf}")"
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/annotate_ancestral_allele.py \
          --input-vcf       "{input.vcf}" \
          --ancestral-fasta "{input.fasta}" \
          --output-vcf      "{output.vcf}"
        """

##############################################################################
# RULE recode_polarized_to_diploid  (per-chromosome, autosomes)
# Recode the AA-annotated haploid VCF to diploid GTs so that MomentsLD can
# use exactly the same sites as the SFS analysis.
##############################################################################
rule recode_polarized_to_diploid:
    input:
        vcf = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz",
        tbi = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz.tbi",
    output:
        vcf = f"{DROSO_DIR}/{{chrom}}/polarized.diploidGT.vcf.gz",
        tbi = f"{DROSO_DIR}/{{chrom}}/polarized.diploidGT.vcf.gz.tbi",
    params:
        script = f"{workflow.basedir}/snakemake_scripts/recode_haploid_to_diploid.py",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.vcf}")"
        out="{output.vcf}"
        tmp_vcf="${{out%.gz}}"                 # uncompressed temp (strip .gz)
        python "{params.script}" "{input.vcf}" "$tmp_vcf"
        bgzip -f "$tmp_vcf"
        tabix -f -p vcf "{output.vcf}"
        """

##############################################################################
# RULE compute_unfolded_sfs  (per-chromosome, autosomes)
# Build the 2D unfolded SFS directly from the polarized haploid VCF.
# Each sample contributes 1 chromosome (no diploid recoding needed).
##############################################################################
rule compute_unfolded_sfs:
    input:
        vcf     = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz",
        tbi     = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz.tbi",
        popfile = REAL_POPFILE,
    output:
        sfs  = f"{DROSO_DIR}/{{chrom}}/unfolded.sfs.pkl",
        # sequence_length here comes straight from this VCF's own ##contig
        # header -- the real-data inference rules read it back out so theta/
        # N_ANC scaling always matches whichever VCF actually built the SFS,
        # instead of a hand-maintained config value that can drift out of sync.
        meta = f"{DROSO_DIR}/{{chrom}}/unfolded.sfs.meta.json",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/compute_unfolded_sfs.py \
          --input-vcf   "{input.vcf}" \
          --popfile     "{input.popfile}" \
          --output-sfs  "{output.sfs}" \
          --output-meta "{output.meta}"
        """

##############################################################################
# RULE combine_autosomal_sfs
# Sum the per-chromosome unfolded SFSs for the four autosomes into one
# genome-wide (autosomal) spectrum (entry-by-entry; fixed-site corners masked).
##############################################################################
rule combine_autosomal_sfs:
    input:
        per_chrom      = expand(f"{DROSO_DIR}/{{chrom}}/unfolded.sfs.pkl", chrom=AUTOSOMES),
        per_chrom_meta = expand(f"{DROSO_DIR}/{{chrom}}/unfolded.sfs.meta.json", chrom=AUTOSOMES),
    output:
        sfs  = COMBINED_SFS,
        meta = COMBINED_SFS_META,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.sfs}")"
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/combine_sfs.py \
          --in-sfs {input.per_chrom} \
          --output-sfs "{output.sfs}" \
          --in-meta {input.per_chrom_meta} \
          --output-meta "{output.meta}"
        """

##############################################################################
# Convenience targets (autosomes)
##############################################################################
rule all_polarized_diploid:
    input:
        expand(f"{DROSO_DIR}/{{chrom}}/polarized.diploidGT.vcf.gz", chrom=AUTOSOMES),

rule all_unfolded_sfs:
    input:
        expand(f"{DROSO_DIR}/{{chrom}}/unfolded.sfs.pkl", chrom=AUTOSOMES),
        COMBINED_SFS,


##############################################################################
# REAL DATA – NLopt Poisson SFS optimisation (moments / dadi)
##############################################################################
rule infer_engine_real:
    input:
        sfs  = COMBINED_SFS,
        meta = COMBINED_SFS_META,
    output:
        pkl = temp(f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/{{engine}}/best_fit.pkl")
    params:
        run_dir  = lambda w: f"{REAL_RUN_ROOT}/run_{w.opt}",
        cfg      = EXP_CFG,
        model_py = (
            f"demes_models:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "demes_models:drosophila_three_epoch"
        ),
    threads: 2
    shell:
        r"""
        set -euo pipefail
        seq_len=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sequence_length'])" "{input.meta}")
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/moments_dadi_inference_real.py \
          --mode {wildcards.engine} \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --outdir "{params.run_dir}/inferences" \
          --opt-seed {wildcards.opt} \
          --real-sequence-length "$seq_len" \
          -v
        """

# ── REAL DATA: MOMENTS / DADI ───────────────────────────────────────────────
rule aggregate_opts_engine_real:
    input:
        runs = lambda w: [f"{REAL_RUN_ROOT}/run_{o}/inferences/{w.engine}/best_fit.pkl"
                          for o in range(NUM_REAL_OPTIMS)]
    output:
        pkl = f"{REAL_INF_ROOT}/{{engine}}/best_fit.pkl"
    run:
        import pickle, pathlib
        from src.aggregate_utils import aggregate_top_k

        records = [(p, i) for i, p in enumerate(input.runs)]

        best, diag = aggregate_top_k(
            records, REAL_TOP_K,
            extra_fields=("theta_hat", "N_ANC_implied_from_theta"),
        )
        best = {"mode": wildcards.engine, **best}

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.pkl, "wb"))

        print(f"✅ [REAL] Aggregated {diag['n_entries']} {wildcards.engine} optimization results → {output.pkl}")


##############################################################################
# REAL DATA (single chromosome) – NLopt Poisson SFS optimisation             #
# (moments / dadi), fit against one chromosome's SFS instead of the         #
# combined-autosome COMBINED_SFS used by infer_engine_real above.           #
##############################################################################
rule infer_engine_real_chrom:
    input:
        sfs  = lambda w: per_chrom_sfs(w.chrom),
        meta = lambda w: f"{DROSO_DIR}/{w.chrom}/unfolded.sfs.meta.json",
    output:
        pkl = temp(f"{REAL_RUN_ROOT_CHROM}/run_{{opt}}/inferences/{{engine}}/best_fit.pkl")
    params:
        run_dir  = lambda w: f"experiments/{MODEL}/real_data_analysis/{w.chrom}/runs/run_{w.opt}",
        cfg      = EXP_CFG,
        model_py = (
            f"demes_models:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "demes_models:drosophila_three_epoch"
        ),
    threads: 2
    shell:
        r"""
        set -euo pipefail
        seq_len=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sequence_length'])" "{input.meta}")
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/moments_dadi_inference_real.py \
          --mode {wildcards.engine} \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --outdir "{params.run_dir}/inferences" \
          --opt-seed {wildcards.opt} \
          --real-sequence-length "$seq_len" \
          -v
        """

rule aggregate_opts_engine_real_chrom:
    input:
        runs = lambda w: [
            f"experiments/{MODEL}/real_data_analysis/{w.chrom}/runs/run_{o}/inferences/{w.engine}/best_fit.pkl"
            for o in range(NUM_REAL_OPTIMS)
        ]
    output:
        pkl = f"{REAL_INF_ROOT_CHROM}/{{engine}}/best_fit.pkl"
    run:
        import pickle, pathlib
        from src.aggregate_utils import aggregate_top_k

        records = [(p, i) for i, p in enumerate(input.runs)]

        best, diag = aggregate_top_k(
            records, REAL_TOP_K,
            extra_fields=("theta_hat", "N_ANC_implied_from_theta"),
        )
        best = {"mode": wildcards.engine, "chrom": wildcards.chrom, **best}

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.pkl, "wb"))

        print(f"✅ [REAL/{wildcards.chrom}] Aggregated {diag['n_entries']} {wildcards.engine} optimization results → {output.pkl}")


##############################################################################
# REAL DATA LD ANALYSIS
##############################################################################

rule split_real_vcf_window:
    input:
        vcf     = REAL_VCF,
        popfile = REAL_POPFILE,
    output:
        vcf_gz = f"{REAL_LD_ROOT}/windows/window_{{i}}.vcf.gz"
    params:
        script      = "snakemake_scripts/split_vcf_windows.py",
        window_size = WINDOW_SIZE,
        num_windows = NUM_WINDOWS
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{REAL_LD_ROOT}/windows"

        python "{params.script}" \
            --input-vcf "{input.vcf}" \
            --popfile "{input.popfile}" \
            --out-dir "{REAL_LD_ROOT}/windows" \
            --window-size "{params.window_size}" \
            --num-windows "{params.num_windows}" \
            --window-index "{wildcards.i}"
        """

rule compute_ld_real:
    input:
        vcf_gz = f"{REAL_LD_ROOT}/windows/window_{{i}}.vcf.gz"
    output:
        pkl = f"{REAL_LD_ROOT}/LD_stats/LD_stats_window_{{i}}.pkl"
    resources:
        gpu = 1 if USE_GPU_LD else 0
    params:
        script = "snakemake_scripts/compute_ld_window.py",
        config = EXP_CFG,
        r_bins  = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{REAL_LD_ROOT}/LD_stats"

        python "{params.script}" \
            --sim-dir "{REAL_LD_ROOT}" \
            --window-index "{wildcards.i}" \
            --config-file "{params.config}" \
            --r-bins "{params.r_bins}"
        """

##############################################################################
# PER-AUTOSOME LD DECAY ANALYSIS
# One windowing/LD/aggregation/decay-comparison chain, parametrised by
# {chrom} AND {ld_variant}:
#   ld_variant="by_chrom" -> flat-rate WINDOW_SIZE (10 Mb) windows, written
#                             under REAL_LD_BYCHROM/{chrom}/
#   ld_variant="genmap"   -> GENMAP_WINDOW_SIZE (1 Mb) windows binned by
#                             genetic distance from the real Comeron (R5/dm3)
#                             recombination map instead of a flat rate,
#                             written under REAL_LD_GENMAP/{chrom}/ — lets you
#                             check whether the cross-autosome curves collapse
#                             once the recombination landscape is accounted for.
# Both variants feed a cross-autosome decay-curve comparison (no inference).
##############################################################################
LD_VARIANT_WINDOW_SIZE = {"by_chrom": WINDOW_SIZE, "genmap": GENMAP_WINDOW_SIZE}


def ld_variant_root(ld_variant, chrom):
    return f"{REAL_LD_BYCHROM if ld_variant == 'by_chrom' else REAL_LD_GENMAP}/{chrom}"


rule build_genetic_map_real:
    input:
        xlsx = COMERON_XLSX,
    output:
        gmap = f"{REAL_LD_GENMAP}/{{chrom}}/genetic_map.txt",
    params:
        script = "snakemake_scripts/build_genetic_map.py",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        python "{params.script}" \
            --xlsx  "{input.xlsx}" \
            --chrom "{wildcards.chrom}" \
            --out   "{output.gmap}"
        """

rule split_real_vcf_window_chrom:
    input:
        vcf     = lambda wc: polarized_diploid_vcf(wc.chrom),
        popfile = REAL_POPFILE,
    output:
        vcf_gz = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_{{ld_variant}}/{{chrom}}/windows/window_{{i}}.vcf.gz"
    params:
        script      = "snakemake_scripts/split_vcf_windows.py",
        window_size = lambda wc: LD_VARIANT_WINDOW_SIZE[wc.ld_variant],
        num_windows = NUM_WINDOWS,
        out_dir     = lambda wc: f"{ld_variant_root(wc.ld_variant, wc.chrom)}/windows",
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.out_dir}"

        python "{params.script}" \
            --input-vcf "{input.vcf}" \
            --popfile "{input.popfile}" \
            --out-dir "{params.out_dir}" \
            --window-size "{params.window_size}" \
            --num-windows "{params.num_windows}" \
            --window-index "{wildcards.i}"
        """

rule compute_ld_real_chrom:
    input:
        vcf_gz = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_{{ld_variant}}/{{chrom}}/windows/window_{{i}}.vcf.gz",
        gmap   = lambda wc: (
            [f"{REAL_LD_GENMAP}/{wc.chrom}/genetic_map.txt"] if wc.ld_variant == "genmap" else []
        ),
    output:
        pkl = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_{{ld_variant}}/{{chrom}}/LD_stats/LD_stats_window_{{i}}.pkl"
    resources:
        gpu = 1 if USE_GPU_LD else 0
    params:
        script  = "snakemake_scripts/compute_ld_window.py",
        config  = EXP_CFG,
        sim_dir = lambda wc: ld_variant_root(wc.ld_variant, wc.chrom),
        r_bins  = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.sim_dir}/LD_stats"

        GMAP_FLAG=""
        if [ -n "{input.gmap}" ]; then
            GMAP_FLAG="--rec-map-file {input.gmap}"
        fi

        python "{params.script}" \
            --sim-dir "{params.sim_dir}" \
            --window-index "{wildcards.i}" \
            --config-file "{params.config}" \
            --r-bins "{params.r_bins}" \
            $GMAP_FLAG
        """

rule aggregate_ld_real_chrom:
    """Aggregate per-window LD stats for one autosome (by_chrom or genmap variant) into means/varcovs."""
    input:
        pkls = lambda w: expand(
            f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_{w.ld_variant}/{w.chrom}/LD_stats/LD_stats_window_{{i}}.pkl",
            i=WINDOWS
        ),
    output:
        mv = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_{{ld_variant}}/{{chrom}}/means.varcovs.pkl",
    params:
        output_root = lambda wc: ld_variant_root(wc.ld_variant, wc.chrom),
        cfg         = EXP_CFG,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/LD_inference.py" \
            --output-root "{params.output_root}" \
            --config-file "{params.cfg}" \
            --skip-optimize
        """

rule compare_ld_decay_autosomes:
    """Overlay LD decay curves across autosomes, one panel per LD statistic, for one variant."""
    input:
        mv = lambda wc: expand(
            f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_{wc.ld_variant}/{{chrom}}/means.varcovs.pkl",
            chrom=AUTOSOMES,
        ),
    output:
        pdf = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_{{ld_variant}}/ld_decay_across_autosomes.pdf",
    params:
        script = "snakemake_scripts/compare_ld_decay_autosomes.py",
        labels = " ".join(AUTOSOMES),
    threads: 1
    shell:
        r"""
        set -euo pipefail
        python "{params.script}" \
            --means {input.mv} \
            --labels {params.labels} \
            --out-pdf "{output.pdf}"
        """

##############################################################################
# REAL DATA: compute_fim_real – observed FIM at best-LL params for {engine}  #
##############################################################################
rule compute_fim_real:
    input:
        fit = lambda w: f"{REAL_INF_ROOT}/{w.engine}/best_fit.pkl",
        sfs = COMBINED_SFS,
    output:
        fim  = temp(f"{REAL_INF_ROOT}/fim/{{engine}}.fim.npy"),
        summ = f"{REAL_INF_ROOT}/fim/{{engine}}.summary.json",
    params:
        script = "snakemake_scripts/compute_fim.py",
        cfg    = EXP_CFG,
    threads: 2
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.fim}")"

        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --engine "{wildcards.engine}" \
            --fit-pkl "{input.fit}" \
            --sfs "{input.sfs}" \
            --config "{params.cfg}" \
            --fim-npy "{output.fim}" \
            --summary-json "{output.summ}"
        """

##############################################################################
# REAL DATA: aggregate_ld_windows_real – aggregate LD windows (once)         #
##############################################################################
rule aggregate_ld_windows_real:
    """
    Aggregate per-window LD stats into means/varcovs and write comparison PDF.
    Runs once; the per-opt optimisation rules consume the means.varcovs.pkl output.
    """
    input:
        pkls = lambda w: expand(
            f"{REAL_LD_ROOT}/LD_stats/LD_stats_window_{{i}}.pkl",
            i=WINDOWS
        ),
    output:
        mv   = f"{REAL_LD_ROOT}/means.varcovs.pkl",
        boot = f"{REAL_LD_ROOT}/bootstrap_sets.pkl",
        pdf  = f"{REAL_LD_ROOT}/empirical_vs_theoretical_comparison.pdf",
    params:
        run_dir     = REAL_INF_ROOT,
        output_root = REAL_LD_ROOT,
        cfg         = EXP_CFG,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.output_root}"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/LD_inference.py" \
            --run-dir       "{params.run_dir}" \
            --output-root   "{params.output_root}" \
            --config-file   "{params.cfg}" \
            --skip-optimize
        """


##############################################################################
# REAL DATA: infer_momentsld_real – one LHS start per opt wildcard           #
##############################################################################
rule infer_momentsld_real:
    input:
        mv       = f"{REAL_LD_ROOT}/means.varcovs.pkl",
        # Seed only: LD inference uses the moments best-fit as an optimization
        # start point. Marked ancient() so regenerating the moments fit (e.g.
        # changing REAL_TOP_K) does not needlessly re-trigger LD inference —
        # the best (rep_0) moments params are unchanged.
        sfs_best = ancient(f"{REAL_INF_ROOT}/moments/best_fit.pkl"),
    output:
        pkl = temp(f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/MomentsLD/best_fit.pkl"),
    params:
        outdir = lambda w: f"{REAL_RUN_ROOT}/run_{w.opt}/inferences/MomentsLD",
        cfg    = EXP_CFG,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.outdir}"

        PYTHONPATH={workflow.basedir} \
        python "src/MomentsLD_real_data.py" \
            --config           "{params.cfg}" \
            --empirical        "{input.mv}" \
            --outdir           "{params.outdir}" \
            --sfs-best-fit-pkl "{input.sfs_best}" \
            --normalization    0 \
            --opt-seed         {wildcards.opt} \
            --verbose

        test -f "{output.pkl}"
        """


##############################################################################
# REAL DATA: aggregate_opts_momentsld_real – pick best across LHS restarts   #
##############################################################################
rule aggregate_opts_momentsld_real:
    input:
        runs = [f"{REAL_RUN_ROOT}/run_{o}/inferences/MomentsLD/best_fit.pkl"
                for o in range(NUM_REAL_OPTIMS)],
    output:
        # Model-specific (unlike REAL_LD_ROOT above) -- this is the fitted
        # MomentsLD result under the active demographic model, so it belongs
        # under REAL_INF_ROOT alongside the moments/dadi best_fit.pkl.
        best = f"{REAL_INF_ROOT}/MomentsLD/best_fit.pkl",
    run:
        import pickle, pathlib
        from src.aggregate_utils import aggregate_top_k

        records = [(p, i) for i, p in enumerate(input.runs)]

        # No theta_hat/N_ANC_implied_from_theta here (unlike moments/dadi real):
        # MomentsLD has no direct access to theta, so there's no absolute-scale
        # implication to carry through.
        best, diag = aggregate_top_k(records, REAL_TOP_K)
        best = {"mode": "momentsLD", **best}

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.best, "wb"))

        print(f"✅ [REAL] Aggregated {diag['n_entries']} MomentsLD optimization results → {output.best}")


##############################################################################
# REAL DATA: sfs_residuals_real – best-fit model SFS − observed SFS          #
##############################################################################
rule sfs_residuals_real:
    input:
        obs_sfs = COMBINED_SFS,
        # (not strictly needed by the script, but keeps DAG honest)
        agg_fit = lambda w: f"{REAL_INF_ROOT}/{w.engine}/best_fit.pkl",
    output:
        res_arr   = f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals.npy",
        res_flat  = temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_flat.npy"),
        meta_json = temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/meta.json"),
        hist_png  = f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_histogram.png",

        # Only required when gram_schmidt=true; otherwise create temp sentinels
        gs_coeffs = (
            temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_gs_coeffs.npy")
            if USE_GS
            else temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/.gs_disabled")
        ),
        gs_basis = (
            f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_gs_basis.npy"
            if USE_GS
            else temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/.gs_basis_disabled")
        ),
    params:
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL not in ["drosophila_three_epoch", "OOA_three_pop_Gutenkunst"]
            else f"src.simulation:{MODEL}"
        ),
        # for real data: inference-dir is the REAL_INF_ROOT, not sim_{sid}
        inf_dir  = REAL_INF_ROOT,
        out_dir  = lambda w: f"{REAL_INF_ROOT}/sfs_residuals/{w.engine}",
        n_bins   = CFG.get("sfs_n_bins", ""),  # empty string if not specified
        script   = "bash_scripts/run_sfs_residuals.sh",
    threads: 1
    shell:
        r"""
        bash "{params.script}" \
            "{wildcards.engine}" "{params.cfg}" "{params.model_py}" "{input.obs_sfs}" \
            "{params.inf_dir}" "{params.out_dir}" "{USE_GS}" "{RESID_SCRIPT}" \
            "{workflow.basedir}" "{params.n_bins}"
        """


##############################################################################
# REAL DATA: combine_results_real – merge fits + attach FIM + residuals      #
##############################################################################
rule combine_results_real:
    input:
        cfg     = EXP_CFG,

        moments = f"{REAL_INF_ROOT}/moments/best_fit.pkl",
        dadi    = f"{REAL_INF_ROOT}/dadi/best_fit.pkl",

        # FIMs (upper-tri flattened) for whatever engines you computed
        fims = lambda w: [
            f"{REAL_INF_ROOT}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ],

        # residual vectors for whichever engines you want to include
        resid_vecs = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ],
        resid_meta = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ],
    output:
        combo = f"{REAL_INF_ROOT}/all_inferences.pkl",
    run:
        import pickle, pathlib
        from src.combine_payloads import build_fim_payload, build_residual_payload

        outdir = pathlib.Path(output.combo).parent
        outdir.mkdir(parents=True, exist_ok=True)

        summary = {}
        summary["moments"] = pickle.load(open(input.moments, "rb"))
        summary["dadi"]    = pickle.load(open(input.dadi, "rb"))

        fim_payload = build_fim_payload(input.fims)
        if fim_payload:
            summary["FIM"] = fim_payload

        resid_payload = build_residual_payload(input.resid_vecs)
        if resid_payload:
            summary["SFS_residuals"] = resid_payload

        pickle.dump(summary, open(output.combo, "wb"))
        print(f"✓ combined REAL → {output.combo}")

##############################################################################
# REAL DATA: build_real_prediction_dataset                                   #
# Assemble the real dadi / moments / MomentsLD fits -- plus the same FIM /    #
# SFS-residual payloads combine_results_real attaches for sims -- into a      #
# single feature row formatted exactly like the training features_df, then    #
# z-score normalize by the priors so it can be pushed through a trained       #
# model. One dataset per {variant} (matches MODELING_VARIANTS) since features #
# differ by whether FIM / SFS-residual columns are included.                 #
#                                                                             #
#   snakemake experiments/<MODEL>/real_data_analysis/prediction_<variant>/real_features_df.pkl
##############################################################################
rule build_real_prediction_dataset:
    input:
        cfg            = EXP_CFG,
        moments        = f"{REAL_INF_ROOT}/moments/best_fit.pkl",
        dadi           = f"{REAL_INF_ROOT}/dadi/best_fit.pkl",
        ld             = f"{REAL_INF_ROOT}/MomentsLD/best_fit.pkl",
        train_features = lambda w: _real_train_features(w.variant),
        fims = lambda w: [
            f"{REAL_INF_ROOT}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ],
        resid_vecs = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ],
        resid_meta = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ],
    output:
        feats = f"{REAL_PRED_ROOT}/real_features_df.pkl",
        raw   = f"{REAL_PRED_ROOT}/real_features_raw_df.pkl",
        meta  = f"{REAL_PRED_ROOT}/real_dataset_meta.json",
    params:
        real_inf_dir = REAL_INF_ROOT,
        # NOT the REAL_PRED_ROOT constant -- that still holds the literal,
        # unsubstituted "{variant}" text; params (unlike input/output) don't
        # get auto-filled from wildcards, so this must build the real path
        # from w.variant directly.
        out_dir      = lambda w: f"experiments/{MODEL}/real_data_analysis/prediction_{w.variant}",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/build_real_prediction_dataset.py \
            --config         "{input.cfg}" \
            --real-inf-dir   "{params.real_inf_dir}" \
            --train-features "{input.train_features}" \
            --out-dir        "{params.out_dir}" \
            --fim-paths      {input.fims} \
            --resid-vec-paths {input.resid_vecs}
        """

##############################################################################
# REAL DATA: predict_real_data – push real features through a trained model  #
# {model_key} ∈ random_forest | xgboost | linear_standard | linear_ridge |    #
#              linear_lasso | linear_elasticnet                               #
# {variant} selects which modeling_{variant}-trained model to use (matches   #
# MODELING_VARIANTS) -- outputs land under the matching prediction_{variant}/ #
#                                                                             #
#   snakemake experiments/<MODEL>/real_data_analysis/prediction_<variant>/predictions_random_forest.json
##############################################################################
rule predict_real_data:
    input:
        feats          = f"{REAL_PRED_ROOT}/real_features_df.pkl",
        model          = lambda w: _real_model_objs(w.variant)[w.model_key],
        train_features = lambda w: _real_train_features(w.variant),
        cfg            = EXP_CFG,
    output:
        json = f"{REAL_PRED_ROOT}/predictions_{{model_key}}.json",
        csv  = f"{REAL_PRED_ROOT}/predictions_{{model_key}}.csv",
    params:
        out_prefix = lambda w: f"experiments/{MODEL}/real_data_analysis/prediction_{w.variant}/predictions_{w.model_key}",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/predict_real_data.py \
            --model-obj      "{input.model}" \
            --real-features  "{input.feats}" \
            --train-features "{input.train_features}" \
            --config         "{input.cfg}" \
            --out-prefix     "{params.out_prefix}" \
            --model-key      "{wildcards.model_key}"
        """

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

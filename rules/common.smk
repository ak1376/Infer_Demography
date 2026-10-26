##############################################################################
# Shared config, constants, path builders and wildcard constraints.
# Included by Snakefile, simulation.smk, modeling.smk and real_data.smk.
##############################################################################

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

USE_GS = bool(CFG.get("gram_schmidt", False))

# Optional real-data region trim (per-chrom), e.g. to drop centromere/telomere-
# proximal ends: config_files/*.json -> "real_data_analysis": {"trim_region":
# {"Chr3L": [447386, 18392988]}}. Keyed by chrom so multiple arms can carry
# independent bounds. Absent -> the whole arm (see real_vcf_for_chrom below).
REAL_DATA_CFG = CFG.get("real_data_analysis", {})
REAL_TRIM_REGION = {
    c: (int(v[0]), int(v[1]))
    for c, v in REAL_DATA_CFG.get("trim_region", {}).items()
}

# All real-data folder names come from src/real_paths.py (shared with the bash
# drivers): short names built from the settings that vary between runs, with
# the rest (exact trim coordinates, r-bins, pairing seed) in a settings.json
# inside each folder.
sys.path.insert(0, workflow.basedir)
from src.real_paths import real_paths, check_settings, write_settings, PSEUDODIPLOID_SEED
RP = real_paths(CFG, CFG["demographic_model"])

# Per-arm input VCF, built from the original DPGP2 VCF (real_data_analysis.
# original_vcf, a "{chrom}" path pattern; read-only) by rule make_real_vcf.
# Derived output only -- never written next to the raw data.
REAL_ORIGINAL_VCF = REAL_DATA_CFG.get("original_vcf")

def _original_vcf(w):
    if not REAL_ORIGINAL_VCF:
        raise ValueError("real_data_analysis.original_vcf is not set in the experiment config")
    return REAL_ORIGINAL_VCF.format(chrom=w.chrom)
REAL_VCF_DIR = "real_data_analysis/data/real_vcf"

def real_vcf_for_chrom(chrom):
    """Input VCF for this chrom, tagged with its trim region (or "whole")."""
    rng = REAL_TRIM_REGION.get(chrom)
    region = f"trim{rng[0]}-{rng[1]}" if rng else "whole"
    return f"{REAL_VCF_DIR}/{chrom}.{region}.vcf.gz"

# real_data_analysis.sfs_projection, e.g. {"CO": 9, "FR": 8}: haplotypes per
# population the SFS is projected to, so sites with a few missing flies still
# count. Absent -> full sample size (any site with a missing fly is skipped).
REAL_SFS_PROJECTION = REAL_DATA_CFG.get("sfs_projection") or {}

# Make sure these match files that actually exist in your repo
DROSO_BASE_DIR   = RP["DROSO_BASE_DIR"]   # region-independent inputs (popfile, pairing, genetic maps)
DROSO_DIR        = RP["DROSO_DIR"]        # processed data: drosophila_trimmed/ (or drosophila/ when untrimmed)
GENMAP_DIR       = RP["GENMAP_DIR"]       # {GENMAP_DIR}/{chrom}/genetic_map.txt (Comeron, region/model-independent)
AUTOSOMES        = ["Chr2L", "Chr2R", "Chr3L", "Chr3R"]          # all autosomal arms (ChrX excluded: not an autosome)
ANCESTRAL_DIR    = "drosophila_data/dpgp_ancestor"                        # relative to repo root, alongside drosophila_data/data/

# popfile.txt (sample->population map) doesn't depend on the genomic region
# analyzed, so it stays on the untagged base dir -- otherwise every new
# trim_region tag would need its own copy of an identical file.
REAL_POPFILE     = f"{DROSO_BASE_DIR}/popfile.txt"
# Pseudo-diploid pairing of the haploid samples (for MomentsLD). Region-
# independent like popfile.txt, so every chrom/trim uses the same pairs.
PSEUDODIPLOID_PAIRS   = RP["PSEUDODIPLOID_PAIRS"]     # tagged .excl-<ids> when real_data_analysis.exclude_samples is set
PSEUDODIPLOID_POPFILE = RP["PSEUDODIPLOID_POPFILE"]
DIPLOID_SUFFIX        = RP["DIPLOID_SUFFIX"]          # "" or ".excl-<ids>" in the diploid VCF names
SFS_SUFFIX            = RP["SFS_SUFFIX"]              # "" or ".excl-<ids>" in the SFS names (same exclude_samples)
EXCLUDE_SAMPLES       = RP["EXCLUDE_SAMPLES"]
COMBINED_SFS     = f"{DROSO_DIR}/combined/autosomes.unfolded{SFS_SUFFIX}.sfs.pkl"     # summed autosomal SFS; used by SFS inference
COMBINED_SFS_META = f"{DROSO_DIR}/combined/autosomes.unfolded{SFS_SUFFIX}.sfs.meta.json"  # summed sequence_length across AUTOSOMES

# Per-chromosome path helpers (by-chromosome layout)
# Phased (a|b) pseudo-diploids: LD stats come from the two known haplotypes.
def polarized_diploid_vcf(chrom):  return f"{DROSO_DIR}/{chrom}/polarized.diploidGT{DIPLOID_SUFFIX}.phased.vcf.gz"
def per_chrom_sfs(chrom):          return f"{DROSO_DIR}/{chrom}/unfolded{SFS_SUFFIX}.sfs.pkl"
def per_chrom_sfs_meta(chrom):     return f"{DROSO_DIR}/{chrom}/unfolded{SFS_SUFFIX}.sfs.meta.json"
def ancestral_fasta(chrom):        return f"{ANCESTRAL_DIR}/{chrom.replace('Chr', 'chr', 1)}.q30.fa"

def _resid_vector_fname():
    # which vector do we want to feed into all_inferences.pkl?
    return "residuals_gs_coeffs.npy" if USE_GS else "residuals_flat.npy"


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

# Which arm(s) the real-data Moments-LD chain runs on, and at what window
# size/count, are config-driven so this can be flipped between e.g. Chr3L-only
# and pooled Chr2L+Chr3L without editing the Snakefile. Deliberately a NEW,
# separate config key from the top-level "num_windows" (used only by the
# simulated-data windowing path above) -- the two must not be conflated.
# num_windows accepts a fixed int (same value applied to every configured arm)
# or the literal string "auto" (per-arm count = arm_span // window_size_bp,
# runtime-determined since the actual VCF POS span isn't known until read --
# mirrors godambe_correction_LRT/Snakefile's `checkpoint tile_nonoverlap`).
# Overlap vs. non-overlap is never chosen explicitly: split_vcf_windows.py's
# existing sliding-window formula produces whichever naturally falls out of
# (window_size, num_windows, actual span) -- e.g. today's Chr3L defaults
# (10 Mb, 100) yield overlapping windows because 100x10Mb can't fit
# non-overlapping in a ~28Mb arm; "auto" at a smaller window_size yields
# non-overlapping windows because the count is chosen to fit exactly.
# Any experiment config JSON without this block falls back to arms=["Chr3L"],
# window_size_bp=10_000_000, num_windows=100 -- byte-identical to the
# pre-existing hardcoded single-arm behavior.
REAL_ARMS       = REAL_DATA_CFG.get("arms", ["Chr3L"])
# "pooled" (default, existing behavior): every configured arm's LD windows are
# flattened together (materialize_real_ld_stats) and fit as ONE combined
# MomentsLD model. "individual": each arm keeps its own windows/LD_stats
# (already arm-separated upstream) AND its own means/varcovs/bootstrap/
# optimization -- see aggregate_ld_windows_real_by_arm/infer_momentsld_real_by_arm/
# aggregate_opts_momentsld_real_by_arm below. Nothing is combined across arms
# in "individual" mode until/unless you explicitly request the pooled target.
REAL_POOLING_MODE = REAL_DATA_CFG.get("pooling_mode", "pooled")
if REAL_POOLING_MODE not in ("pooled", "individual"):
    raise ValueError(f'real_data_analysis.pooling_mode must be "pooled" or "individual", got {REAL_POOLING_MODE!r}')
REAL_WINDOW_BP  = int(REAL_DATA_CFG.get("window_size_bp", 10_000_000))
_raw_real_nw    = REAL_DATA_CFG.get("num_windows", 100)
REAL_NUM_WINDOWS_MODE = "auto" if str(_raw_real_nw).lower() == "auto" else int(_raw_real_nw)
# MomentsLD_real_data.py always needs SOME N_ref to satisfy resolve_n_ref(),
# but only actually USES it for rho-scaling when momentsld_use_scaled_units
# is True -- in absolute-units mode (the current default), rho is rescaled
# from each candidate's own N_ANC instead, so N_ref is otherwise vestigial.
# When scaled units are off, infer_momentsld_real(_by_arm) skip the
# moments/dadi seed fit entirely and pass a placeholder --n-ref instead, so
# MomentsLD can run standalone without requiring moments/dadi to have been
# fit first.
REAL_USE_SCALED_UNITS = bool(CFG.get("momentsld_use_scaled_units", True))
# MomentsLD_real_data.py only reads the moments SFS fit to anchor N_ref (scaled
# units) or to pin a fixed_parameters entry set to "moments_best"; otherwise
# the real MomentsLD rules don't wait on the SFS fit at all.
MOMENTSLD_NEEDS_SFS_FIT = REAL_USE_SCALED_UNITS or any(
    v == "moments_best" for v in CFG.get("fixed_parameters", {}).values()
)
def _momentsld_seed_flag(w, input):
    if not MOMENTSLD_NEEDS_SFS_FIT:
        return "--n-ref 1.0"
    flag = f'--sfs-best-fit-pkl "{input.sfs_best}"'
    return flag if REAL_USE_SCALED_UNITS else f"{flag} --n-ref 1.0"
# LD-run name from arms + window + exclusions, e.g. "Chr3L_100kb" (see
# src/real_paths.py). REAL_LD_ENGINE names the pooled MomentsLD fit subdir,
# REAL_LD_ENGINE_ARM the per-arm one (arm is already in that path).
REAL_LD_ENGINE     = RP["REAL_LD_ENGINE"]
REAL_LD_ENGINE_ARM = RP["REAL_LD_ENGINE_ARM"]

# Real-data LD windows/LD_stats/aggregated means+varcovs are pure functions of
# the VCF data (arms, window size, r_bins) -- not of which demographic model
# you're fitting -- so this lives under DROSO_DIR (shared across every model)
# instead of experiments/{MODEL}/...: switching MODEL never re-triggers the
# window split or the (GPU-bound) per-window LD computation. The MomentsLD
# *fit itself* (aggregate_opts_momentsld_real's best_fit.pkl) is genuinely
# model-specific and lives under REAL_INF_ROOT instead -- see that rule below.
REAL_LD_ROOT = RP["REAL_LD_ROOT"]   # {DROSO_DIR}/ld/{REAL_LD_NAME}
# Real-data LD always converts bp -> genetic distance with the Comeron (R5/dm3)
# recombination map (build_genetic_map_real turns this into GENMAP_DIR files).
COMERON_XLSX       = "drosophila_data/recombination_maps/Comeron_100kb_R5_R6.xlsx"  # input data (like the VCFs), not a pipeline output
# infer_engine_real/aggregate_opts_engine_real fit COMBINED_SFS, which is
# itself trim-tagged (built under DROSO_DIR) -- so these fit outputs need the
# same tag, otherwise a trimmed run would silently overwrite the full-arm
# moments/dadi best_fit.pkl already sitting here. Untagged when trim_region
# is unset, matching every path above.
REAL_FIT_ROOT = RP["REAL_FIT_ROOT"]   # experiments/{MODEL}/real_trimmed (or real_untrimmed)
REAL_RUN_ROOT = RP["REAL_RUN_ROOT"]
REAL_INF_ROOT = RP["REAL_INF_ROOT"]

# Single-chromosome variants of REAL_RUN_ROOT/REAL_INF_ROOT (as opposed to the
# combined-autosome COMBINED_SFS used by infer_engine_real above). {chrom} is
# a real Snakemake wildcard here (double-braced so the f-string leaves it
# literal), constrained below to Chr(2L|2R|3L|3R).
REAL_RUN_ROOT_CHROM = RP["REAL_RUN_ROOT_CHROM"]
REAL_INF_ROOT_CHROM = RP["REAL_INF_ROOT_CHROM"]

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
REAL_PRED_ROOT = f"{REAL_FIT_ROOT}/prediction_{{variant}}"

def _real_modeling_dir(variant):
    return f"experiments/{MODEL}/modeling_{variant}"

def _real_train_features(variant):
    return f"{_real_modeling_dir(variant)}/datasets/features_df.pkl"

def _real_empirical_norm_stats(variant):
    return f"{_real_modeling_dir(variant)}/datasets/empirical_norm_stats.json"

def _real_norm_train_features(variant):
    return f"{_real_modeling_dir(variant)}/datasets/normalized_train_features.pkl"

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

# Model-calibration check (posterior-predictive): simulate at the real-data
# fitted params, one job per (variant, model_key, rep) so replicates can run
# as parallel SLURM array tasks (see bash_scripts/simulation/calibration_simulate.sh) --
# each array task builds just its own replicate_{rep} target.
CALIBRATION_ROOT = f"{REAL_FIT_ROOT}/calibration_{{variant}}/{{model_key}}/replicate_{{rep}}"
NUM_CALIBRATION_REPLICATES = int(CFG.get("calibration_n_replicates", 20))
CALIBRATION_REPS = list(range(NUM_CALIBRATION_REPLICATES))

wildcard_constraints:
    chrom      = r"Chr(2L|2R|3L|3R|X)",
    arm        = r"Chr(2L|2R|3L|3R|X)",
    start      = r"\d+",
    end        = r"\d+",
    # combine_features narrows this back down (raw_features has its own
    # producer rules, build_raw_features_dataset/prepare_raw_features_splits).
    variant    = r"(w|wo)_FIM_(w|wo)_SFSresids|raw_features",
    reg        = r"standard|ridge|lasso|elasticnet",
    opt        = "|".join(str(i) for i in range(NUM_OPTIMS)),
    engine     = "moments|dadi",
    frac_tag   = r"thin\d+|n\d+",
    # TEMPORARY: widened to also allow the raw dadi/moments/momentsLD fits
    # (not just trained ML model keys) as calibration_simulate/calibration_ppc/
    # calibration_ld_ppc model_key values, so those PPC checks can run against
    # the direct SFS/LD fit for comparison against the ML surrogate's PPC.
    # Revert to just REAL_MODEL_KEYS once that comparison is done.
    model_key  = "|".join(REAL_MODEL_KEYS) + "|dadi|moments|momentsLD",
    rep        = r"\d+",

# LD r-bins
# 16 log-spaced edges from 1e-6 to 1e-3 (ratio ~1.58x) plus a leading 0,
# matching the moments.LD tutorial's bin density -- finer than the old
# 9-edge (~2-2.5x ratio) set, to reduce quadrature error in compute_theoretical_ld
# for the short-range bins.
R_BINS_STR = "0,1e-06,1.58489e-06,2.51189e-06,3.98107e-06,6.30957e-06,1e-05,1.58489e-05,2.51189e-05,3.98107e-05,6.30957e-05,0.0001,0.000158489,0.000251189,0.000398107,0.000630957,0.001"

# Refuse to reuse a real-data folder whose settings.json disagrees with the
# current config (e.g. trim coordinates changed but the folder name didn't);
# settings.json is written into whichever of these folders a (non-dry) run
# actually created.
REAL_FOLDER_SETTINGS = real_paths(CFG, CFG["demographic_model"], r_bins=R_BINS_STR)["_settings"]
check_settings(REAL_FOLDER_SETTINGS)

onsuccess:
    write_settings(REAL_FOLDER_SETTINGS)

onerror:
    write_settings(REAL_FOLDER_SETTINGS)

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

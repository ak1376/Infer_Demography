##############################################################################
# Shared by simulation.smk, modeling.smk and real_data.smk (and Snakefile):
# experiment config, sim/optimization counts, residual/FIM settings, LD
# r-bins, modeling variants, and the wildcard patterns used by >1 workflow.
# Anything used by only one workflow lives at the top of its rules/*_rules.smk.
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

# Lets run: blocks and parse-time code import src.* (workflow.basedir is the
# main Snakefile's directory, i.e. the repo root).
sys.path.insert(0, workflow.basedir)

RESID_SCRIPT = "snakemake_scripts/computing_residuals_from_sfs.py"
EXP_CFG = config["active_experiment_config"]

# Experiment metadata
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = int(CFG["num_draws"])
NUM_OPTIMS    = int(CFG.get("num_optimizations", 3))
TOP_K         = int(CFG.get("top_k", 2))
SIM_IDS       = list(range(NUM_DRAWS))

# Engines to COMPUTE (always); modeling usage is controlled in feature_extraction via config
FIM_ENGINES = CFG.get("fim_engines", ["moments"])

USE_GS = bool(CFG.get("gram_schmidt", False))

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

# LD r-bins
# 16 log-spaced edges from 1e-6 to 1e-3 (ratio ~1.58x) plus a leading 0,
# matching the moments.LD tutorial's bin density -- finer than the old
# 9-edge (~2-2.5x ratio) set, to reduce quadrature error in compute_theoretical_ld
# for the short-range bins.
R_BINS_STR = "0,1e-06,1.58489e-06,2.51189e-06,3.98107e-06,6.30957e-06,1e-05,1.58489e-05,2.51189e-05,3.98107e-05,6.30957e-05,0.0001,0.000158489,0.000251189,0.000398107,0.000630957,0.001"

# ── Modeling feature-set variants ───────────────────────────────────────────
# variant name -> feature_extraction.py flags. The {variant} wildcard routes
# datasets + trained models into experiments/<model>/modeling_<variant>/.
#   *_FIM_*_SFSresids: dadi/moments/momentsLD estimates from all_inferences.pkl,
#                      with/without FIM elements and SFS residuals as extra features
#   raw_SFS(_LD):      no-inference baseline -- observed SFS (+ MomentsLD means)
# All variants share one train/tune/val assignment per sim id (split_indices
# in src/feature_extraction_helpers.py), so their errors are directly comparable.
VARIANT_FEATURE_FLAGS = {
    "w_FIM_w_SFSresids":   "--use-fim-features true --use-residuals true",
    "w_FIM_wo_SFSresids":  "--use-fim-features true --use-residuals false",
    "wo_FIM_w_SFSresids":  "--use-fim-features false --use-residuals true",
    "wo_FIM_wo_SFSresids": "--use-fim-features false --use-residuals false",
    "raw_SFS":             "--feature-source raw_sfs",
    "raw_SFS_LD":          "--feature-source raw_sfs_ld",
}
INFERENCE_VARIANTS = [v for v in VARIANT_FEATURE_FLAGS if not v.startswith("raw_")]

# Which variants modeling.smk builds: experiment config "modeling_variants"
# (e.g. add "raw_SFS" for the baseline), default = the four inference variants.
MODELING_VARIANTS = list(CFG.get("modeling_variants", INFERENCE_VARIANTS))
_unknown = set(MODELING_VARIANTS) - set(VARIANT_FEATURE_FLAGS)
if _unknown:
    raise ValueError(f"modeling_variants: unknown {sorted(_unknown)}; choose from {list(VARIANT_FEATURE_FLAGS)}")

wildcard_constraints:
    variant    = "|".join(VARIANT_FEATURE_FLAGS),
    opt        = "|".join(str(i) for i in range(NUM_OPTIMS)),
    engine     = "moments|dadi",

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
WIN_SCRIPT   = "snakemake_scripts/simulate_window.py"
LD_SCRIPT    = "snakemake_scripts/compute_ld_window.py"
RESID_SCRIPT = "snakemake_scripts/computing_residuals_from_sfs.py"
# Experiment config. Defaults to split_migration_growth, but can be overridden
# on the CLI to run a different model without editing this file, e.g.:
#   snakemake --config exp_cfg=config_files/experiment_config_split_migration_growth_both.json <target>
EXP_CFG = config.get("exp_cfg", "config_files/experiment_config_bottleneck.json")

# Experiment metadata
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = int(CFG["num_draws"])
NUM_OPTIMS    = int(CFG.get("num_optimizations", 3))
NUM_REAL_OPTIMS = int(CFG.get("num_optimizations", 3))
TOP_K         = int(CFG.get("top_k", 2))
NUM_WINDOWS   = int(CFG.get("num_windows", 100))
WINDOW_SIZE   = 10_000_000

# Engines to COMPUTE (always); modeling usage is controlled in feature_extraction via config
FIM_ENGINES = CFG.get("fim_engines", ["moments"])

USE_GPU_LD = CFG.get("use_gpu_ld", False)
USE_GPU_DADI = CFG.get("use_gpu_dadi", False)

USE_GS = bool(CFG.get("gram_schmidt", False))

# Make sure these match files that actually exist in your repo
DROSO_DIR        = "real_data_analysis/data/drosophila"
AUTOSOMES        = ["Chr2L", "Chr3L"]                            # Chr3R dropped (In(3R)Payne); Chr2R dropped (anomalous FR diversity / long-range score correlation); X excluded
ANCESTRAL_DIR    = "/sietch_colab/data_share/drosophila_melanogaster/dpgp_ancestor"

# Data now lives in per-chromosome subdirs: {DROSO_DIR}/{chrom}/{polarized,polarized.diploidGT,unfolded.sfs}...
RAW_HAPLOID_VCF  = "drosophila_data/data/Chr2L.vcf.gz"                    # legacy Chr2L alias
REAL_POPFILE     = f"{DROSO_DIR}/popfile.txt"
REAL_VCF         = f"{DROSO_DIR}/Chr2L/polarized.diploidGT.vcf.gz"        # diploid polarized (Chr2L); used by MomentsLD-real
POLARIZED_VCF    = f"{DROSO_DIR}/Chr2L/polarized.vcf.gz"                  # haploid + AA (Chr2L); legacy alias
UNFOLDED_SFS     = f"{DROSO_DIR}/Chr2L/unfolded.sfs.pkl"                  # per-chrom SFS (Chr2L); legacy alias
COMBINED_SFS     = f"{DROSO_DIR}/combined/autosomes.unfolded.sfs.pkl"     # summed autosomal SFS; used by SFS inference
ANCESTRAL_FASTA  = f"{ANCESTRAL_DIR}/chr2L.q30.fa"                        # legacy Chr2L alias

# Per-chromosome path helpers (by-chromosome layout)
def polarized_vcf(chrom):          return f"{DROSO_DIR}/{chrom}/polarized.vcf.gz"
def polarized_diploid_vcf(chrom):  return f"{DROSO_DIR}/{chrom}/polarized.diploidGT.vcf.gz"
def per_chrom_sfs(chrom):          return f"{DROSO_DIR}/{chrom}/unfolded.sfs.pkl"
def ancestral_fasta(chrom):        return f"{ANCESTRAL_DIR}/{chrom.replace('Chr', 'chr', 1)}.q30.fa"

wildcard_constraints:
    chrom = r"Chr(2L|2R|3L|3R)",
    variant = r"(w|wo)_FIM_(w|wo)_SFSresids",
    reg = r"standard|ridge|lasso|elasticnet",


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

def _variant_flags(variant):
    """(use_fim_features, use_residuals) for a variant name like 'w_FIM_wo_SFSresids'."""
    return (variant.startswith("w_FIM_"), variant.endswith("_w_SFSresids"))

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

# Real-data LD mirrors LD_ROOT but without sid
REAL_LD_ROOT = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD"
# Per-autosome LD decay analysis (independent of the Chr2L inference pipeline above)
REAL_LD_BYCHROM = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_by_chrom"
# Same, but with the real Comeron (R5/dm3) recombination map and 1 Mb windows.
REAL_LD_GENMAP     = f"experiments/{MODEL}/real_data_analysis/inferences/MomentsLD_genmap"
GENMAP_WINDOW_SIZE = 1_000_000
COMERON_XLSX       = f"{DROSO_DIR}/recombination_maps/Comeron_100kb_R5_R6.xlsx"
REAL_RUN_ROOT = f"experiments/{MODEL}/real_data_analysis/runs"
REAL_INF_ROOT = f"experiments/{MODEL}/real_data_analysis/inferences"
REAL_OPTIMS   = list(range(NUM_REAL_OPTIMS))

# Number of top replicates the real moments/dadi aggregation keeps.
# Must match the rep count the trained model was built with (moments_*_rep_0..N-1).
# Defaults to keeping every optimization so re-running yields all rep_* slots.
REAL_TOP_K    = int(CFG.get("real_top_k", NUM_REAL_OPTIMS))

# ── Real-data prediction (push real fits through a trained model) ───────────
REAL_PRED_ROOT     = f"experiments/{MODEL}/real_data_analysis/prediction"
# Which trained-model directory to use as the source of models + feature template.
REAL_MODELING_DIR  = CFG.get(
    "real_predict_modeling_dir",
    f"experiments/{MODEL}/modeling_wo_FIM_wo_SFSresids",
)
REAL_TRAIN_FEATURES = f"{REAL_MODELING_DIR}/datasets/features_df.pkl"

# model_key wildcard -> trained *_mdl_obj.pkl path
REAL_MODEL_OBJS = {
    "random_forest":    f"{REAL_MODELING_DIR}/random_forest/random_forest_mdl_obj.pkl",
    "xgboost":          f"{REAL_MODELING_DIR}/xgboost/xgb_mdl_obj.pkl",
    "linear_standard":  f"{REAL_MODELING_DIR}/linear_standard/linear_mdl_obj_standard.pkl",
    "linear_ridge":     f"{REAL_MODELING_DIR}/linear_ridge/linear_mdl_obj_ridge.pkl",
    "linear_lasso":     f"{REAL_MODELING_DIR}/linear_lasso/linear_mdl_obj_lasso.pkl",
    "linear_elasticnet":f"{REAL_MODELING_DIR}/linear_elasticnet/linear_mdl_obj_elasticnet.pkl",
}

# LD r-bins
R_BINS_STR = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"

# Optional pruning — set "prune_keep_fractions": [0.15] in EXP_CFG to enable
PRUNE_FRACS = CFG.get("prune_keep_fractions", [])
def _frac_tag(f): return f"thin{round(float(f) * 100):02d}"
PRUNE_TAGS  = [_frac_tag(f) for f in PRUNE_FRACS]

SIM_IDS  = [i for i in range(NUM_DRAWS)]
WINDOWS  = range(NUM_WINDOWS)

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

            # ## ── 2. PER-RUN SFS INFERENCE (sim) ──────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/moments/fit_params.pkl",
            #     sid=SIM_IDS,
            #     opt=OPTIMS,
            # ),
            # expand(
            #     f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/dadi/fit_params.pkl",
            #     sid=SIM_IDS,
            #     opt=OPTIMS,
            # ),

            # # ── 3. CONSOLIDATED SIM INFERENCES ──────────────────────────────────
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl", sid=SIM_IDS),
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",    sid=SIM_IDS),
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/cleanup_done.txt",       sid=SIM_IDS),

            # ## ── 4. MOMENTS-LD (SIMULATED) ────────────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/best_fit.pkl",
            #     sid=SIM_IDS,
            # ),

            # ## ── 5. MOMENTS-LD OPTIMIZATION (always at MomentsLD/best_fit.pkl) ─────
            # expand(
            #     f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/best_fit.pkl",
            #     sid=SIM_IDS,
            # ),

            # ======================================================================
            # ACTIVE TARGETS
            # ======================================================================

            # # ── 1. MODELING DATASETS ────────────────────────────────────────────
            # f"experiments/{MODEL}/modeling/datasets/features_df.pkl",
            # f"experiments/{MODEL}/modeling/datasets/targets_df.pkl",

            # # ── 2. LINEAR REGRESSION ────────────────────────────────────────────
            # expand(
            #     f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl",
            #     reg=REG_TYPES,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_model_error_{{reg}}.json",
            #     reg=REG_TYPES,
            # ),
            # expand(
            #     f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_regression_model_{{reg}}.pkl",
            #     reg=REG_TYPES,
            # ),

            # # ── 3. RANDOM FOREST ────────────────────────────────────────────────
            # f"experiments/{MODEL}/modeling/random_forest/random_forest_mdl_obj.pkl",
            # f"experiments/{MODEL}/modeling/random_forest/random_forest_model_error.json",
            # f"experiments/{MODEL}/modeling/random_forest/random_forest_model.pkl",
            # f"experiments/{MODEL}/modeling/random_forest/random_forest_feature_importances.png",

            # # ── 4. XGBOOST ──────────────────────────────────────────────────────
            # f"experiments/{MODEL}/modeling/xgboost/xgb_mdl_obj.pkl",
            # f"experiments/{MODEL}/modeling/xgboost/xgb_model_error.json",
            # f"experiments/{MODEL}/modeling/xgboost/xgb_model.pkl",
            # f"experiments/{MODEL}/modeling/xgboost/xgb_feature_importances.png",

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
        ts     = temp(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees"),
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
rule infer_moments:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",   # not read; kept for DAG clarity
        cfg    = EXP_CFG
    output:
        pkl = temp(f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/moments/fit_params.pkl")
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
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
          --model-py "{params.model_py}" \
          --ground-truth "{input.params}" \
          --outdir "{params.run_dir}/inferences" \
          --generate-profiles \

          {params.fix}

        cp "{params.run_dir}/inferences/moments/best_fit.pkl" "{output.pkl}"
        """


##############################################################################
# RULE infer_dadi – custom NLopt Poisson SFS optimisation (dadi)
##############################################################################
rule infer_dadi:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        cfg    = EXP_CFG,
    output:
        pkl = temp(f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/dadi/fit_params.pkl")
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        fix      = "",
    threads: 8
    shell:
        r"""
        set -euo pipefail

        echo "===== infer_dadi ENV ====="
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
        mkdir -p "{params.run_dir}/inferences/dadi"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/moments_dadi_inference.py" \
          --mode dadi \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --outdir "{params.run_dir}/inferences" \
          --ground-truth "{input.params}" \
          {params.fix}

        cp "{params.run_dir}/inferences/dadi/best_fit.pkl" "{output.pkl}"
        """

# ── MOMENTS ONLY ───────────────────────────────────────────────────────────
rule aggregate_opts_moments:
    input:
        cfg = EXP_CFG,
    output:
        mom = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl"
    run:
        import pickle, numpy as np, pathlib, re, glob

        sid = wildcards.sid
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        mom_pkls = sorted(glob.glob(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/moments/fit_params.pkl"
        ))

        def _as_list(x):
            if x is None:
                return []
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        params, lls, opt_ids = [], [], []
        n_readable = 0
        n_nonempty = 0

        for pkl_path in mom_pkls:
            m = re.search(rf"/run_{sid}_(\d+)/inferences/moments/fit_params\.pkl$", pkl_path)
            if not m:
                continue

            opt_idx = int(m.group(1))

            try:
                with open(pkl_path, "rb") as fh:
                    d = pickle.load(fh)
                n_readable += 1
            except Exception as e:
                print(f"WARNING: could not load {pkl_path}: {e}")
                continue

            this_params = _as_list(d.get("best_params"))
            this_lls    = _as_list(d.get("best_ll"))

            if len(this_lls) == 0:
                continue

            n_nonempty += 1
            params.extend(this_params)
            lls.extend(this_lls)
            opt_ids.extend([opt_idx] * len(this_lls))

        if n_nonempty < MIN_FILES:
            raise ValueError(
                f"[aggregate_opts_moments] Need >= {MIN_FILES} non-empty moments optimizations for sid={sid}, "
                f"but got nonempty={n_nonempty} (readable={n_readable}, paths_found={len(mom_pkls)}). "
                f"Not aggregating."
            )

        keep = np.argsort(lls)[::-1][:TOP_K]

        best = {
            "best_params":   [params[i] for i in keep],
            "best_ll":       [lls[i] for i in keep],
            "opt_index":     [opt_ids[i] for i in keep],
            "n_files_found": int(len(mom_pkls)),
            "n_nonempty":    int(n_nonempty),
            "min_required":  int(TOP_K),
        }

        pathlib.Path(output.mom).parent.mkdir(parents=True, exist_ok=True)
        with open(output.mom, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ moments: found {len(mom_pkls)} files, aggregated {len(lls)} entries → {output.mom}")
        print(f"✅ moments: kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

# ── DADI ONLY ──────────────────────────────────────────────────────────────
rule aggregate_opts_dadi:
    input:
        cfg = EXP_CFG,
    output:
        dadi = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl"
    run:
        import pickle, numpy as np, pathlib, re, glob

        sid = wildcards.sid
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        dadi_pkls = sorted(glob.glob(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/dadi/fit_params.pkl"
        ))

        def _as_list(x):
            if x is None:
                return []
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        params, lls, opt_ids = [], [], []
        n_readable = 0
        n_nonempty = 0

        for pkl_path in dadi_pkls:
            m = re.search(rf"/run_{sid}_(\d+)/inferences/dadi/fit_params\.pkl$", pkl_path)
            if not m:
                continue

            opt_idx = int(m.group(1))

            try:
                with open(pkl_path, "rb") as fh:
                    d = pickle.load(fh)
                n_readable += 1
            except Exception as e:
                print(f"WARNING: could not load {pkl_path}: {e}")
                continue

            this_params = _as_list(d.get("best_params"))
            this_lls    = _as_list(d.get("best_ll"))

            if len(this_lls) == 0:
                continue

            n_nonempty += 1
            params.extend(this_params)
            lls.extend(this_lls)
            opt_ids.extend([opt_idx] * len(this_lls))

        if n_nonempty < MIN_FILES:
            raise ValueError(
                f"[aggregate_opts_dadi] Need >= {MIN_FILES} non-empty dadi optimizations for sid={sid}, "
                f"but got nonempty={n_nonempty} (readable={n_readable}, paths_found={len(dadi_pkls)}). "
                f"Not aggregating."
            )

        keep = np.argsort(lls)[::-1][:TOP_K]

        best = {
            "best_params":   [params[i] for i in keep],
            "best_ll":       [lls[i] for i in keep],
            "opt_index":     [opt_ids[i] for i in keep],
            "n_files_found": int(len(dadi_pkls)),
            "n_nonempty":    int(n_nonempty),
            "min_required":  int(TOP_K),
        }

        pathlib.Path(output.dadi).parent.mkdir(parents=True, exist_ok=True)
        with open(output.dadi, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ dadi: found {len(dadi_pkls)} files, aggregated {len(lls)} entries → {output.dadi}")
        print(f"✅ dadi: kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

# ── CLEANUP RULE: Remove non-top-K optimization runs after both aggregations ──
rule cleanup_optimization_runs:
    input:
        dadi    = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",
        moments = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl"
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

        dadi_keep    = set((dadi_data.get("opt_index") or [])[:TOP_K])
        moments_keep = set((moments_data.get("opt_index") or [])[:TOP_K])
        keep_indices = dadi_keep | moments_keep

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
# RULE simulate_window – one VCF window
##############################################################################
rule simulate_window:
    input:
        params   = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        metafile = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
        done     = f"{SIM_BASEDIR}/{{sid}}/.done"
    output:
        vcf_gz = temp(f"{LD_ROOT}/windows/window_{{win}}.vcf.gz")
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
        cfg     = EXP_CFG
    threads: 4
    resources:
        ld_cores = 4
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
        keep_frac   = lambda w: str(int(w.frac_tag.replace("thin", "")) / 100),
    threads: 1
    shell:
        """
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "src/prune_vcf.py" \
            --vcf            "{params.windows_dir}/window_{wildcards.win}.vcf.gz" \
            --out-dir        "{params.pruning_dir}" \
            --keep-fractions "{params.keep_frac}"   \
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
    threads: 4
    resources:
        ld_cores = 4
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
# RULE optimize_momentsld – aggregate windows & optimise momentsLD          #
# Works for unpruned-only AND mixed (unpruned primary + pruned fallback).   #
# Output is always MomentsLD/best_fit.pkl regardless of pruning config.     #
##############################################################################
rule optimize_momentsld:
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
        boot = temp(f"{LD_ROOT}/bootstrap_sets.pkl"),
        pdf  = f"{LD_ROOT}/empirical_vs_theoretical_comparison.pdf",
        best = f"{LD_ROOT}/best_fit.pkl",
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        pruning_dir = lambda w: (
            f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{PRUNE_TAGS[0]}"
            if PRUNE_TAGS else ""
        ),
        cfg = EXP_CFG,
    threads: 1
    run:
        import subprocess
        cmd = [
            "python", "snakemake_scripts/LD_inference.py",
            "--run-dir",     params.sim_dir,
            "--output-root", params.output_root,
            "--config-file", params.cfg,
        ]
        if params.pruning_dir:
            cmd += ["--fallback-ld-dir", params.pruning_dir]
        env = {**os.environ, "PYTHONPATH": workflow.basedir}
        subprocess.run(cmd, check=True, env=env)

##############################################################################
# RULE optimize_momentsld_mixed – optimise MomentsLD from PRUNED stats        #
# Used when pruning is configured (pruned-only). Reads the pruned per-window  #
# LD stats under pruning/{frac_tag}/LD_stats/ as the sole input and writes    #
# best_fit under the same directory. Output path matches MomentsLD.sh's       #
# pruning/{frac_tag}/best_fit.pkl target.                                     #
##############################################################################
rule optimize_momentsld_mixed:
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
        boot = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/bootstrap_sets.pkl"),
        pdf  = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/empirical_vs_theoretical_comparison.pdf",
        best = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/best_fit.pkl",
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{w.frac_tag}",
        cfg = EXP_CFG,
    threads: 1
    run:
        import subprocess
        # output_root drives everything: LD_inference reads output_root/LD_stats/*.pkl
        # (the pruned stats) and writes means/varcovs/bootstrap/pdf/best_fit there.
        # No --fallback-ld-dir: in pruned-only mode there are no unpruned stats.
        cmd = [
            "python", "snakemake_scripts/LD_inference.py",
            "--run-dir",     params.sim_dir,
            "--output-root", params.output_root,
            "--config-file", params.cfg,
        ]
        env = {**os.environ, "PYTHONPATH": workflow.basedir}
        subprocess.run(cmd, check=True, env=env)

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
    threads: 1
    shell:
        r"""
        set -euo pipefail

        # Ensure outdir exists (script also does this, but harmless)
        mkdir -p "{params.out_dir}"

        # Build optional n_bins flag
        N_BINS_ARG=""
        if [ -n "{params.n_bins}" ]; then
            N_BINS_ARG="--n-bins {params.n_bins}"
        fi

        PYTHONPATH={workflow.basedir} \
        python "{RESID_SCRIPT}" \
          --mode {wildcards.engine} \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --observed-sfs "{input.obs_sfs}" \
          --inference-dir "{params.inf_dir}" \
          --outdir "{params.out_dir}" \
          $N_BINS_ARG

        # Base outputs must exist
        test -f "{output.res_arr}"
        test -f "{output.res_flat}"
        test -f "{output.meta_json}"
        test -f "{output.hist_png}"

        # GS outputs: if enabled, require real artifacts; else create sentinels
        if [ "{USE_GS}" = "True" ]; then
            test -f "{output.gs_coeffs}"
            test -f "{output.gs_basis}"
        else
            touch "{output.gs_coeffs}" "{output.gs_basis}"
        fi
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
        momentsLD = lambda w: ancient(f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/best_fit.pkl")
                    if os.path.exists(f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/best_fit.pkl") else [],
        fims      = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ] if CFG.get("use_fim_features", False) else [],
        resid_vecs = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ] if CFG.get("use_residuals", False) else [],
        resid_meta = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ] if CFG.get("use_residuals", False) else [],
    output:
        combo = f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl"
    run:
        import pickle, pathlib, numpy as np, re, os, json

        if not input.dadi or not input.moments or not input.momentsLD:
            missing = [k for k, v in [("dadi", input.dadi), ("moments", input.moments), ("momentsLD", input.momentsLD)] if not v]
            raise RuntimeError(f"sim_{wildcards.sid}: skipping — prerequisites not ready: {missing}")

        outdir = pathlib.Path(output.combo).parent
        outdir.mkdir(parents=True, exist_ok=True)

        cfg_obj = json.loads(open(input.cfg, "r").read())
        use_gs = bool(cfg_obj.get("gram_schmidt", False))

        summary = {}
        summary["moments"]   = pickle.load(open(input.moments, "rb"))
        summary["dadi"]      = pickle.load(open(input.dadi, "rb"))
        summary["momentsLD"] = pickle.load(open(input.momentsLD, "rb"))

        # ---------------- FIM upper-triangles ----------------
        fim_payload = {}
        for fim_path in input.fims:
            eng = re.sub(r".*?/fim/([^.]+)\.fim\.npy$", r"\1", fim_path)
            F = np.load(fim_path)
            iu = np.triu_indices(F.shape[0])
            fim_payload[eng] = {
                "shape": [int(F.shape[0]), int(F.shape[1])],
                "tri_flat": F[iu].astype(float).tolist(),
                "indices": "upper_including_diagonal",
                "order": "row-major",
            }
        if fim_payload:
            summary["FIM"] = fim_payload

        # ---------------- Residual SFS vectors ----------------
        # We attach either:
        #   - residuals_flat.npy (raw) OR
        #   - residuals_gs_coeffs.npy (GS reduced)
        resid_payload = {}
        for vec_path in input.resid_vecs:
            m = re.search(r"/sfs_residuals/([^/]+)/([^/]+)\.npy$", vec_path)
            if not m:
                continue
            eng = m.group(1)
            stem = m.group(2)  # residuals_flat OR residuals_gs_coeffs
            base = os.path.dirname(vec_path)

            vec = np.load(vec_path)

            # full residual array shape (optional)
            arr_path = os.path.join(base, "residuals.npy")
            arr = np.load(arr_path) if os.path.exists(arr_path) else None

            payload = {
                "vector": vec.astype(float).tolist(),
                "vector_dim": int(vec.size),
                "vector_type": (
                    "gram_schmidt_coeffs" if stem == "residuals_gs_coeffs" else "raw_flat_residuals"
                ),
                "full_residual_shape": (list(arr.shape) if arr is not None else None),
                "order": "row-major",
            }

            # If GS: attach GS metadata/basis shapes when available
            if stem == "residuals_gs_coeffs":
                meta_path = os.path.join(base, "meta.json")
                basis_path = os.path.join(base, "residuals_gs_basis.npy")
                if os.path.exists(basis_path):
                    Q = np.load(basis_path)
                    payload["gs_basis_shape"] = [int(Q.shape[0]), int(Q.shape[1])]
                if os.path.exists(meta_path):
                    try:
                        mj = json.loads(open(meta_path, "r").read())
                        payload["gram_schmidt_k"] = mj.get("gram_schmidt_k", None)
                        payload["gram_schmidt_k_effective"] = mj.get("gram_schmidt_k_effective", None)
                        payload["gram_schmidt_basis"] = mj.get("gram_schmidt_basis", None)
                        payload["gram_schmidt_eps"] = mj.get("gram_schmidt_eps", None)
                    except Exception:
                        pass

            resid_payload[eng] = payload

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

        echo "=== AFTER feature_extraction, listing outputs ==="
        ls -lah "{params.outdir}/datasets" || true

        echo "=== Expected files ==="
        for f in \
        "{output.features_df}" \
        "{output.targets_df}" \
        "{output.ntrain_X}" \
        "{output.ntrain_y}" \
        "{output.ntune_X}" \
        "{output.ntune_y}" \
        "{output.nval_X}" \
        "{output.nval_y}" \
        "{output.split_idx}" \
        "{output.scatter_png}" \
        "{output.mse_val_png}" \
        "{output.mse_train_png}" \
        "{output.metrics_all}" \
        "{output.metrics_dadi}" \
        "{output.metrics_moments}" \
        "{output.metrics_momentsLD}" \
        "{output.outliers_tsv}" \
        "{output.outliers_txt}" \
        ; do
        if [ -f "$f" ]; then
            echo "OK   $f"
        else
            echo "MISS $f"
            exit 1
        fi
        done
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
        opt_flags = lambda w: " ".join([
            "--use_optuna" if config.get("rf", {}).get("use_optuna", False) else "",
            f"--n_trials {config['rf']['n_trials']}" if config.get("rf", {}).get("n_trials") is not None else "",
            f"--optuna_timeout {config['rf']['optuna_timeout']}" if config.get("rf", {}).get("optuna_timeout") is not None else "",
            f"--optuna_seed {config['rf']['optuna_seed']}" if config.get("rf", {}).get("optuna_seed") is not None else "",

            f"--final_fit {config['rf']['final_fit']}" if config.get("rf", {}).get("final_fit") is not None else "",

            # manual overrides (bypass optuna if any are set)
            f"--n_estimators {config['rf']['n_estimators']}" if config.get("rf", {}).get("n_estimators") is not None else "",
            f"--max_depth {config['rf']['max_depth']}" if config.get("rf", {}).get("max_depth") is not None else "",
            f"--min_samples_split {config['rf']['min_samples_split']}" if config.get("rf", {}).get("min_samples_split") is not None else "",

            # extra knobs
            f"--min_samples_leaf {config['rf']['min_samples_leaf']}" if config.get("rf", {}).get("min_samples_leaf") is not None else "",
            f"--max_features {config['rf']['max_features']}" if config.get("rf", {}).get("max_features") is not None else "",
            f"--max_samples {config['rf']['max_samples']}" if config.get("rf", {}).get("max_samples") is not None else "",
        ]).strip()

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
        opt_flags = lambda w: " ".join([
            "--use_optuna" if config.get("xgb", {}).get("use_optuna", False) else "",
            f"--n_trials {config['xgb']['n_trials']}" if config.get("xgb", {}).get("n_trials") is not None else "",
            f"--optuna_timeout {config['xgb']['optuna_timeout']}" if config.get("xgb", {}).get("optuna_timeout") is not None else "",
            f"--optuna_seed {config['xgb']['optuna_seed']}" if config.get("xgb", {}).get("optuna_seed") is not None else "",

            f"--final_fit {config['xgb']['final_fit']}" if config.get("xgb", {}).get("final_fit") is not None else "",
            f"--early_stopping_rounds {config['xgb']['early_stopping_rounds']}" if config.get("xgb", {}).get("early_stopping_rounds") is not None else "",

            # manual overrides (bypass optuna if any are set)
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
            --model_directory "{params.model_dir}" \
            {params.opt_flags}

        test -f "{output.obj}"   && \
        test -f "{output.errjs}" && \
        test -f "{output.mdl}"   && \
        test -f "{output.plot}"  && \
        test -f "{output.fi}"
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


rule recode_haploid_to_diploid_Chr2L:
    """
    Recode haploid-coded GTs in Chr2L.vcf.gz to diploid-coded GTs,
    BGZF-compress, and tabix-index.
    """
    input:
        vcf="drosophila_data/data/Chr2L.vcf.gz",
        tbi="drosophila_data/data/Chr2L.vcf.gz.tbi",
    output:
        vcf="real_data_analysis/data/drosophila/Chr2L/diploidGT.vcf.gz",
        tbi="real_data_analysis/data/drosophila/Chr2L/diploidGT.vcf.gz.tbi",
    params:
        script=f"{workflow.basedir}/snakemake_scripts/recode_haploid_to_diploid.py",
        tmp_vcf="real_data_analysis/data/drosophila/Chr2L/diploidGT.vcf",  # Changed to match output dir
    threads: 1
    shell:
        r"""
        set -euo pipefail

        # Ensure output directory exists
        mkdir -p "$(dirname "{params.tmp_vcf}")"
        
        python "{params.script}" "{input.vcf}" "{params.tmp_vcf}"
        bgzip -f "{params.tmp_vcf}"
        tabix -f -p vcf "{output.vcf}"
        """

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
        sfs = f"{DROSO_DIR}/{{chrom}}/unfolded.sfs.pkl",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/compute_unfolded_sfs.py \
          --input-vcf  "{input.vcf}" \
          --popfile    "{input.popfile}" \
          --output-sfs "{output.sfs}"
        """

##############################################################################
# RULE combine_autosomal_sfs
# Sum the per-chromosome unfolded SFSs for the four autosomes into one
# genome-wide (autosomal) spectrum (entry-by-entry; fixed-site corners masked).
##############################################################################
rule combine_autosomal_sfs:
    input:
        per_chrom = expand(f"{DROSO_DIR}/{{chrom}}/unfolded.sfs.pkl", chrom=AUTOSOMES),
    output:
        sfs = COMBINED_SFS,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.sfs}")"
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/combine_sfs.py \
          --in-sfs {input.per_chrom} \
          --output-sfs "{output.sfs}"
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
# REAL DATA – NLopt Poisson SFS optimisation (moments)
##############################################################################
rule infer_moments_real:
    input:
        sfs = COMBINED_SFS,
    output:
        pkl = temp(f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/moments/best_fit.pkl")
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
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/moments_dadi_inference_real.py \
          --mode moments \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --outdir "{params.run_dir}/inferences" \
          --opt-seed {wildcards.opt} \
          -v
        """



##############################################################################
# REAL DATA – NLopt Poisson SFS optimisation (dadi)
##############################################################################
rule infer_dadi_real:
    input:
        sfs = COMBINED_SFS,
    output:
        pkl = temp(f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/dadi/best_fit.pkl")
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
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/moments_dadi_inference_real.py \
          --mode dadi \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --outdir "{params.run_dir}/inferences" \
          --opt-seed {wildcards.opt} \
          -v
        """

# ── REAL DATA: MOMENTS ONLY ────────────────────────────────────────────────
rule aggregate_opts_moments_real:
    input:
        mom = [f"{REAL_RUN_ROOT}/run_{o}/inferences/moments/best_fit.pkl" for o in range(NUM_REAL_OPTIMS)]
    output:
        mom = f"{REAL_INF_ROOT}/moments/best_fit.pkl"
    run:
        import pickle, numpy as np, pathlib

        def _as_list(x):
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        params, lls, opt_ids = [], [], []
        thetas, nancs = [], []

        for opt_idx, pkl in enumerate(input.mom):
            d = pickle.load(open(pkl, "rb"))
            this_params = _as_list(d["best_params"])
            this_lls    = _as_list(d["best_ll"])

            params.extend(this_params)
            lls.extend(this_lls)
            opt_ids.extend([opt_idx] * len(this_lls))

            # optional useful metadata (safe even if missing)
            thetas.extend(_as_list(d.get("theta_hat", np.nan)))
            nancs.extend(_as_list(d.get("N_ANC_implied_from_theta", np.nan)))

        keep = np.argsort(lls)[::-1][:REAL_TOP_K]

        best = {
            "mode": "moments",
            "best_params": [params[i] for i in keep],
            "best_ll":     [lls[i]    for i in keep],
            "opt_index":   [opt_ids[i] for i in keep],
            "theta_hat":   [thetas[i] for i in keep],
            "N_ANC_implied_from_theta": [nancs[i] for i in keep],
        }

        pathlib.Path(output.mom).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.mom, "wb"))

        print(f"✅ [REAL] Aggregated {len(params)} moments optimization results → {output.mom}")


# ── REAL DATA: DADI ONLY ───────────────────────────────────────────────────
rule aggregate_opts_dadi_real:
    input:
        dadi = [f"{REAL_RUN_ROOT}/run_{o}/inferences/dadi/best_fit.pkl" for o in range(NUM_REAL_OPTIMS)]
    output:
        dadi = f"{REAL_INF_ROOT}/dadi/best_fit.pkl"
    run:
        import pickle, numpy as np, pathlib

        def _as_list(x):
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        params, lls, opt_ids = [], [], []
        thetas, nancs = [], []

        for opt_idx, pkl in enumerate(input.dadi):
            d = pickle.load(open(pkl, "rb"))
            this_params = _as_list(d["best_params"])
            this_lls    = _as_list(d["best_ll"])

            params.extend(this_params)
            lls.extend(this_lls)
            opt_ids.extend([opt_idx] * len(this_lls))

            thetas.extend(_as_list(d.get("theta_hat", np.nan)))
            nancs.extend(_as_list(d.get("N_ANC_implied_from_theta", np.nan)))

        keep = np.argsort(lls)[::-1][:REAL_TOP_K]

        best = {
            "mode": "dadi",
            "best_params": [params[i] for i in keep],
            "best_ll":     [lls[i]    for i in keep],
            "opt_index":   [opt_ids[i] for i in keep],
            "theta_hat":   [thetas[i] for i in keep],
            "N_ANC_implied_from_theta": [nancs[i] for i in keep],
        }

        pathlib.Path(output.dadi).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.dadi, "wb"))

        print(f"✅ [REAL] Aggregated {len(params)} dadi optimization results → {output.dadi}")


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
# Same window/LD machinery as the rules above, but parametrised by {chrom} and
# written under REAL_LD_BYCHROM/{chrom}/ so each autosome is analysed
# independently. Feeds a cross-autosome decay-curve comparison (no inference).
##############################################################################
rule split_real_vcf_window_chrom:
    input:
        vcf     = lambda wc: polarized_diploid_vcf(wc.chrom),
        popfile = REAL_POPFILE,
    output:
        vcf_gz = f"{REAL_LD_BYCHROM}/{{chrom}}/windows/window_{{i}}.vcf.gz"
    params:
        script      = "snakemake_scripts/split_vcf_windows.py",
        window_size = WINDOW_SIZE,
        num_windows = NUM_WINDOWS,
        out_dir     = f"{REAL_LD_BYCHROM}/{{chrom}}/windows",
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
        vcf_gz = f"{REAL_LD_BYCHROM}/{{chrom}}/windows/window_{{i}}.vcf.gz"
    output:
        pkl = f"{REAL_LD_BYCHROM}/{{chrom}}/LD_stats/LD_stats_window_{{i}}.pkl"
    resources:
        gpu = 1 if USE_GPU_LD else 0
    params:
        script  = "snakemake_scripts/compute_ld_window.py",
        config  = EXP_CFG,
        sim_dir = f"{REAL_LD_BYCHROM}/{{chrom}}",
        r_bins  = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.sim_dir}/LD_stats"

        python "{params.script}" \
            --sim-dir "{params.sim_dir}" \
            --window-index "{wildcards.i}" \
            --config-file "{params.config}" \
            --r-bins "{params.r_bins}"
        """

rule aggregate_ld_chrom:
    """Aggregate per-window LD stats for one autosome into means/varcovs."""
    input:
        pkls = lambda w: expand(
            f"{REAL_LD_BYCHROM}/{w.chrom}/LD_stats/LD_stats_window_{{i}}.pkl",
            i=WINDOWS
        ),
    output:
        mv = f"{REAL_LD_BYCHROM}/{{chrom}}/means.varcovs.pkl",
    params:
        output_root = f"{REAL_LD_BYCHROM}/{{chrom}}",
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
        test -f "{output.mv}"
        """

rule compare_ld_decay_autosomes:
    """Overlay LD decay curves across autosomes, one panel per LD statistic."""
    input:
        mv = expand(f"{REAL_LD_BYCHROM}/{{chrom}}/means.varcovs.pkl", chrom=AUTOSOMES),
    output:
        pdf = f"{REAL_LD_BYCHROM}/ld_decay_across_autosomes.pdf",
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
# PER-AUTOSOME LD DECAY WITH THE REAL COMERON (R5/dm3) RECOMBINATION MAP
# Same as the by_chrom rules above, but (1) windows are 1 Mb (faster) and
# (2) LD stats are binned by genetic distance from the Comeron map instead of a
# flat rate. Lets you check whether the cross-autosome curves collapse once the
# recombination landscape is accounted for.
##############################################################################
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

rule split_real_vcf_window_genmap:
    input:
        vcf     = lambda wc: polarized_diploid_vcf(wc.chrom),
        popfile = REAL_POPFILE,
    output:
        vcf_gz = f"{REAL_LD_GENMAP}/{{chrom}}/windows/window_{{i}}.vcf.gz"
    params:
        script      = "snakemake_scripts/split_vcf_windows.py",
        window_size = GENMAP_WINDOW_SIZE,
        num_windows = NUM_WINDOWS,
        out_dir     = f"{REAL_LD_GENMAP}/{{chrom}}/windows",
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

rule compute_ld_real_genmap:
    input:
        vcf_gz = f"{REAL_LD_GENMAP}/{{chrom}}/windows/window_{{i}}.vcf.gz",
        gmap   = f"{REAL_LD_GENMAP}/{{chrom}}/genetic_map.txt",
    output:
        pkl = f"{REAL_LD_GENMAP}/{{chrom}}/LD_stats/LD_stats_window_{{i}}.pkl"
    resources:
        gpu = 1 if USE_GPU_LD else 0
    params:
        script  = "snakemake_scripts/compute_ld_window.py",
        config  = EXP_CFG,
        sim_dir = f"{REAL_LD_GENMAP}/{{chrom}}",
        r_bins  = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.sim_dir}/LD_stats"

        python "{params.script}" \
            --sim-dir "{params.sim_dir}" \
            --window-index "{wildcards.i}" \
            --config-file "{params.config}" \
            --r-bins "{params.r_bins}" \
            --rec-map-file "{input.gmap}"
        """

rule aggregate_ld_genmap:
    """Aggregate the Comeron-mapped per-window LD stats for one autosome."""
    input:
        pkls = lambda w: expand(
            f"{REAL_LD_GENMAP}/{w.chrom}/LD_stats/LD_stats_window_{{i}}.pkl",
            i=WINDOWS
        ),
    output:
        mv = f"{REAL_LD_GENMAP}/{{chrom}}/means.varcovs.pkl",
    params:
        output_root = f"{REAL_LD_GENMAP}/{{chrom}}",
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
        test -f "{output.mv}"
        """

rule compare_ld_decay_genmap:
    """Overlay Comeron-mapped LD decay curves across autosomes."""
    input:
        mv = expand(f"{REAL_LD_GENMAP}/{{chrom}}/means.varcovs.pkl", chrom=AUTOSOMES),
    output:
        pdf = f"{REAL_LD_GENMAP}/ld_decay_across_autosomes.pdf",
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
        sfs = "real_data_analysis/data/drosophila/drosophila.sfs.pkl",
    output:
        fim  = temp(f"{REAL_INF_ROOT}/fim/{{engine}}.fim.npy"),
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
            --fim-npy "{output.fim}"
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

        test -f "{output.mv}"
        test -f "{output.boot}"
        test -f "{output.pdf}"
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
        best = f"{REAL_LD_ROOT}/best_fit.pkl",
    run:
        import pickle, numpy as np, pathlib

        records = []
        for opt_idx, pkl in enumerate(input.runs):
            d = pickle.load(open(pkl, "rb"))
            records.append((float(d["best_ll"]), opt_idx, d))

        records.sort(key=lambda t: t[0], reverse=True)
        best_ll, best_opt, best_d = records[0]

        out = dict(best_d)
        out["opt_index"] = best_opt

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(out, open(output.best, "wb"))

        print(f"✅ [REAL MomentsLD] Best run: opt={best_opt}  ll={best_ll:.6f}  → {output.best}")


##############################################################################
# REAL DATA: sfs_residuals_real – best-fit model SFS − observed SFS          #
##############################################################################
rule sfs_residuals_real:
    input:
        obs_sfs = "real_data_analysis/data/drosophila/drosophila.sfs.pkl",
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
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.out_dir}"

        # Build optional n_bins flag
        N_BINS_ARG=""
        if [ -n "{params.n_bins}" ]; then
            N_BINS_ARG="--n-bins {params.n_bins}"
        fi

        PYTHONPATH={workflow.basedir} \
        python "{RESID_SCRIPT}" \
          --mode "{wildcards.engine}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --observed-sfs "{input.obs_sfs}" \
          --inference-dir "{params.inf_dir}" \
          --outdir "{params.out_dir}" \
          $N_BINS_ARG

        # Base outputs must exist
        test -f "{output.res_arr}"
        test -f "{output.res_flat}"
        test -f "{output.meta_json}"
        test -f "{output.hist_png}"

        # GS outputs: if enabled, require real artifacts; else create sentinels
        if [ "{USE_GS}" = "True" ]; then
            test -f "{output.gs_coeffs}"
            test -f "{output.gs_basis}"
        else
            touch "{output.gs_coeffs}" "{output.gs_basis}"
        fi
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
        import json, os, re, pickle, pathlib
        import numpy as np

        outdir = pathlib.Path(output.combo).parent
        outdir.mkdir(parents=True, exist_ok=True)

        cfg_obj = json.loads(open(input.cfg, "r").read())
        use_gs = bool(cfg_obj.get("gram_schmidt", False))

        summary = {}
        summary["moments"] = pickle.load(open(input.moments, "rb"))
        summary["dadi"]    = pickle.load(open(input.dadi, "rb"))

        # ---------------- FIM upper-triangles ----------------
        fim_payload = {}
        for fim_path in input.fims:
            eng = re.sub(r".*?/fim/([^.]+)\.fim\.npy$", r"\1", fim_path)
            F = np.load(fim_path)
            iu = np.triu_indices(F.shape[0])
            fim_payload[eng] = {
                "shape": [int(F.shape[0]), int(F.shape[1])],
                "tri_flat": F[iu].astype(float).tolist(),
                "indices": "upper_including_diagonal",
                "order": "row-major",
            }
        if fim_payload:
            summary["FIM"] = fim_payload

        # ---------------- Residual SFS vectors ----------------
        resid_payload = {}
        for vec_path in input.resid_vecs:
            m = re.search(r"/sfs_residuals/([^/]+)/([^/]+)\.npy$", vec_path)
            if not m:
                continue
            eng = m.group(1)
            stem = m.group(2)  # residuals_flat OR residuals_gs_coeffs
            base = os.path.dirname(vec_path)

            vec = np.load(vec_path)

            arr_path = os.path.join(base, "residuals.npy")
            arr = np.load(arr_path) if os.path.exists(arr_path) else None

            payload = {
                "vector": vec.astype(float).tolist(),
                "vector_dim": int(vec.size),
                "vector_type": (
                    "gram_schmidt_coeffs" if stem == "residuals_gs_coeffs" else "raw_flat_residuals"
                ),
                "full_residual_shape": (list(arr.shape) if arr is not None else None),
                "order": "row-major",
            }

            if stem == "residuals_gs_coeffs":
                meta_path = os.path.join(base, "meta.json")
                basis_path = os.path.join(base, "residuals_gs_basis.npy")
                if os.path.exists(basis_path):
                    Q = np.load(basis_path)
                    payload["gs_basis_shape"] = [int(Q.shape[0]), int(Q.shape[1])]
                if os.path.exists(meta_path):
                    try:
                        mj = json.loads(open(meta_path, "r").read())
                        payload["gram_schmidt_k"] = mj.get("gram_schmidt_k", None)
                        payload["gram_schmidt_k_effective"] = mj.get("gram_schmidt_k_effective", None)
                        payload["gram_schmidt_basis"] = mj.get("gram_schmidt_basis", None)
                        payload["gram_schmidt_eps"] = mj.get("gram_schmidt_eps", None)
                    except Exception:
                        pass

            resid_payload[eng] = payload

        if resid_payload:
            summary["SFS_residuals"] = resid_payload

        pickle.dump(summary, open(output.combo, "wb"))
        print(f"✓ combined REAL → {output.combo}")

##############################################################################
# REAL DATA: build_real_prediction_dataset                                   #
# Assemble the real dadi / moments / MomentsLD fits into a single feature     #
# row formatted exactly like the training features_df, then z-score normalize #
# by the priors so it can be pushed through a trained model.                  #
#                                                                             #
#   snakemake build_real_prediction_dataset                                   #
##############################################################################
rule build_real_prediction_dataset:
    input:
        cfg            = EXP_CFG,
        moments        = f"{REAL_INF_ROOT}/moments/best_fit.pkl",
        dadi           = f"{REAL_INF_ROOT}/dadi/best_fit.pkl",
        ld             = f"{REAL_LD_ROOT}/best_fit.pkl",
        train_features = REAL_TRAIN_FEATURES,
    output:
        feats = f"{REAL_PRED_ROOT}/real_features_df.pkl",
        raw   = f"{REAL_PRED_ROOT}/real_features_raw_df.pkl",
        meta  = f"{REAL_PRED_ROOT}/real_dataset_meta.json",
    params:
        real_inf_dir = REAL_INF_ROOT,
        out_dir      = REAL_PRED_ROOT,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/build_real_prediction_dataset.py \
            --config         "{input.cfg}" \
            --real-inf-dir   "{params.real_inf_dir}" \
            --train-features "{input.train_features}" \
            --out-dir        "{params.out_dir}"

        test -f "{output.feats}" && test -f "{output.raw}" && test -f "{output.meta}"
        """

##############################################################################
# REAL DATA: predict_real_data – push real features through a trained model  #
# {model_key} ∈ random_forest | xgboost | linear_standard | linear_ridge |    #
#              linear_lasso | linear_elasticnet                               #
#                                                                             #
#   snakemake experiments/<MODEL>/real_data_analysis/prediction/predictions_random_forest.json
##############################################################################
rule predict_real_data:
    input:
        feats          = f"{REAL_PRED_ROOT}/real_features_df.pkl",
        model          = lambda w: REAL_MODEL_OBJS[w.model_key],
        train_features = REAL_TRAIN_FEATURES,
        cfg            = EXP_CFG,
    output:
        json = f"{REAL_PRED_ROOT}/predictions_{{model_key}}.json",
        csv  = f"{REAL_PRED_ROOT}/predictions_{{model_key}}.csv",
    params:
        out_prefix = lambda w: f"{REAL_PRED_ROOT}/predictions_{w.model_key}",
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

        test -f "{output.json}" && test -f "{output.csv}"
        """

##############################################################################
# RAW-FEATURES PIPELINE: observed SFS + MomentsLD means → ensemble         #
# Not in rule all. Run explicitly, e.g.:                                    #
#   snakemake --snakefile Snakefile raw_features_xgboost                   #
##############################################################################
RAW_FEAT_DIR = f"experiments/{MODEL}/modeling/raw_features_datasets"
RAW_MDL_DIR  = f"experiments/{MODEL}/modeling/raw_features_modeling"

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

rule raw_features_linear_regression:
    input:
        ntrain_X = f"{RAW_FEAT_DIR}/normalized_train_features.pkl",
        ntrain_y = f"{RAW_FEAT_DIR}/normalized_train_targets.pkl",
        ntune_X  = f"{RAW_FEAT_DIR}/normalized_tune_features.pkl",
        ntune_y  = f"{RAW_FEAT_DIR}/normalized_tune_targets.pkl",
        nval_X   = f"{RAW_FEAT_DIR}/normalized_val_features.pkl",
        nval_y   = f"{RAW_FEAT_DIR}/normalized_val_targets.pkl",
        shades   = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors   = f"experiments/{MODEL}/modeling/main_colors.pkl",
        mdlcfg   = "config_files/model_config.yaml",
    output:
        obj   = f"{RAW_MDL_DIR}/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl",
        errjs = f"{RAW_MDL_DIR}/linear_{{reg}}/linear_model_error_{{reg}}.json",
        mdl   = f"{RAW_MDL_DIR}/linear_{{reg}}/linear_regression_model_{{reg}}.pkl",
        plot  = f"{RAW_MDL_DIR}/linear_{{reg}}/linear_results_{{reg}}.png",
    params:
        expcfg    = EXP_CFG,
        model_dir = f"{RAW_MDL_DIR}/linear_{{reg}}",
        alpha     = lambda w: config["linear"].get(w.reg, {}).get("alpha", 0.0),
        l1_ratio  = lambda w: config["linear"].get(w.reg, {}).get("l1_ratio", 0.5),
        gridflag  = lambda w: "--do_grid_search" if config["linear"].get(w.reg, {}).get("grid_search", False) else "",
    threads: 2
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/linear_evaluation.py \
            --X_train_path "{input.ntrain_X}" \
            --y_train_path "{input.ntrain_y}" \
            --X_tune_path  "{input.ntune_X}" \
            --y_tune_path  "{input.ntune_y}" \
            --X_val_path   "{input.nval_X}" \
            --y_val_path   "{input.nval_y}" \
            --experiment_config_path "{params.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            --model_directory        "{params.model_dir}" \
            --regression_type "{wildcards.reg}" \
            --alpha {params.alpha} \
            --l1_ratio {params.l1_ratio} {params.gridflag}
        """

rule raw_features_random_forest:
    input:
        ntrain_X = f"{RAW_FEAT_DIR}/normalized_train_features.pkl",
        ntrain_y = f"{RAW_FEAT_DIR}/normalized_train_targets.pkl",
        ntune_X  = f"{RAW_FEAT_DIR}/normalized_tune_features.pkl",
        ntune_y  = f"{RAW_FEAT_DIR}/normalized_tune_targets.pkl",
        nval_X   = f"{RAW_FEAT_DIR}/normalized_val_features.pkl",
        nval_y   = f"{RAW_FEAT_DIR}/normalized_val_targets.pkl",
        shades   = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors   = f"experiments/{MODEL}/modeling/main_colors.pkl",
        expcfg   = EXP_CFG,
        mdlcfg   = "config_files/model_config.yaml",
    output:
        obj   = f"{RAW_MDL_DIR}/random_forest/random_forest_mdl_obj.pkl",
        errjs = f"{RAW_MDL_DIR}/random_forest/random_forest_model_error.json",
        mdl   = f"{RAW_MDL_DIR}/random_forest/random_forest_model.pkl",
        plot  = f"{RAW_MDL_DIR}/random_forest/random_forest_results.png",
        fi    = f"{RAW_MDL_DIR}/random_forest/random_forest_feature_importances.png",
    params:
        model_dir = f"{RAW_MDL_DIR}/random_forest",
        opt_flags = lambda w: " ".join([
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
        ]).strip(),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/random_forest.py \
            --X_train_path "{input.ntrain_X}" \
            --y_train_path "{input.ntrain_y}" \
            --X_tune_path  "{input.ntune_X}" \
            --y_tune_path  "{input.ntune_y}" \
            --X_val_path   "{input.nval_X}" \
            --y_val_path   "{input.nval_y}" \
            --experiment_config_path "{input.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            --model_directory        "{params.model_dir}" \
            {params.opt_flags}
        """

rule raw_features_xgboost:
    input:
        ntrain_X = f"{RAW_FEAT_DIR}/normalized_train_features.pkl",
        ntrain_y = f"{RAW_FEAT_DIR}/normalized_train_targets.pkl",
        ntune_X  = f"{RAW_FEAT_DIR}/normalized_tune_features.pkl",
        ntune_y  = f"{RAW_FEAT_DIR}/normalized_tune_targets.pkl",
        nval_X   = f"{RAW_FEAT_DIR}/normalized_val_features.pkl",
        nval_y   = f"{RAW_FEAT_DIR}/normalized_val_targets.pkl",
        shades   = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors   = f"experiments/{MODEL}/modeling/main_colors.pkl",
        expcfg   = EXP_CFG,
        mdlcfg   = "config_files/model_config.yaml",
    output:
        obj   = f"{RAW_MDL_DIR}/xgboost/xgb_mdl_obj.pkl",
        errjs = f"{RAW_MDL_DIR}/xgboost/xgb_model_error.json",
        mdl   = f"{RAW_MDL_DIR}/xgboost/xgb_model.pkl",
        plot  = f"{RAW_MDL_DIR}/xgboost/xgb_results.png",
        fi    = f"{RAW_MDL_DIR}/xgboost/xgb_feature_importances.png",
    params:
        model_dir = f"{RAW_MDL_DIR}/xgboost",
        opt_flags = lambda w: " ".join([
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
        ]).strip(),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/xgboost_evaluation.py \
            --X_train_path "{input.ntrain_X}" \
            --y_train_path "{input.ntrain_y}" \
            --X_tune_path  "{input.ntune_X}" \
            --y_tune_path  "{input.ntune_y}" \
            --X_val_path   "{input.nval_X}" \
            --y_val_path   "{input.nval_y}" \
            --experiment_config_path "{input.expcfg}" \
            --model_config_path      "{input.mdlcfg}" \
            --color_shades_file      "{input.shades}" \
            --main_colors_file       "{input.colors}" \
            --model_directory        "{params.model_dir}" \
            {params.opt_flags}
        """

##############################################################################
# Wildcard Constraints                                                      #
##############################################################################
wildcard_constraints:
    opt       = "|".join(str(i) for i in range(NUM_OPTIMS)),
    engine    = "moments|dadi",
    frac_tag  = r"thin\d+",
    model_key = "|".join(REAL_MODEL_OBJS.keys()),

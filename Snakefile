##############################################################################
# CONFIG – Paths and Constants (edit here only)                              #
##############################################################################
import json, math, sys, os
from pathlib import Path
from snakemake.io import protected

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
EXP_CFG      = "config_files/experiment_config_drosophila_three_epoch.json"

# Experiment metadata
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = int(CFG["num_draws"])
NUM_OPTIMS    = int(CFG.get("num_optimizations", 3))
TOP_K         = int(CFG.get("top_k", 2))
NUM_WINDOWS   = int(CFG.get("num_windows", 100))

# Engines to COMPUTE (always); modeling usage is controlled in feature_extraction via config
FIM_ENGINES = CFG.get("fim_engines", ["moments"])

def _normalize_residual_engines(val):
    # accepts "moments", "dadi", "both", list/tuple
    if isinstance(val, str):
        v = val.lower()
        return ["moments", "dadi"] if v in {"both", "all"} else [v]
    if isinstance(val, (list, tuple, set)):
        return [e for e in val if e in {"moments","dadi"}] or ["moments","dadi"]
    return ["moments","dadi"]

RESIDUAL_ENGINES = _normalize_residual_engines(CFG.get("residual_engines", "both"))

# Regressors
REG_TYPES = config["linear"]["types"]  # e.g., ["standard","ridge","lasso","elasticnet"]

# Windows & sims
SIM_IDS  = list(range(NUM_DRAWS))
WINDOWS  = range(NUM_WINDOWS)

# Canonical path builders
SIM_BASEDIR = f"experiments/{MODEL}/simulations"
RUN_DIR     = lambda sid, opt: f"experiments/{MODEL}/runs/run_{sid}_{opt}"
LD_ROOT     = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD"

opt_pkl   = lambda sid, opt, tool: f"{RUN_DIR(sid, opt)}/inferences/{tool}/fit_params.pkl"
final_pkl = lambda sid, tool: f"experiments/{MODEL}/inferences/sim_{sid}/{tool}/fit_params.pkl"

# LD r-bins
R_BINS_STR = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"


SIM_IDS  = [i for i in range(NUM_DRAWS)]
WINDOWS  = range(NUM_WINDOWS)

# ── canonical path builders -----------------------------------------------
SIM_BASEDIR = f"experiments/{MODEL}/simulations"                      # per‑sim artefacts
RUN_DIR     = lambda sid, opt: f"experiments/{MODEL}/runs/run_{sid}_{opt}"
LD_ROOT     = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD"  # use {{sid}} wildcard

opt_pkl   = lambda sid, opt, tool: f"{RUN_DIR(sid, opt)}/inferences/{tool}/fit_params.pkl"
final_pkl = lambda sid, tool: f"experiments/{MODEL}/inferences/sim_{sid}/{tool}/fit_params.pkl"

import os, json, pathlib

def ensure_dir(p):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)

def exists(p):
    return os.path.exists(p)

def dump_json(obj, path):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


##############################################################################
# RULE all – final targets the workflow must create                          #
##############################################################################
rule all:
    input:
        # Simulation artifacts
        expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",  sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",             sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/demes.png",           sid=SIM_IDS),

        # Aggregated optimizer results
        [final_pkl(sid, "moments") for sid in SIM_IDS],
        [final_pkl(sid, "dadi")    for sid in SIM_IDS],

        # LD artifacts
        expand(f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",        sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl", sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/best_fit.pkl",                         sid=SIM_IDS),

        # FIM (always computed)
        expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.fim.npy",
            sid=SIM_IDS, engine=FIM_ENGINES
        ),

        # Residuals (always computed)
        expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_flat.npy",
            sid=SIM_IDS, engine=RESIDUAL_ENGINES
        ),

        # Combined per-sim inference blobs (include FIM/residuals payloads)
        expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl", sid=SIM_IDS),

        # Modeling datasets
        f"experiments/{MODEL}/modeling/datasets/features_df.pkl",
        f"experiments/{MODEL}/modeling/datasets/targets_df.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",
        f"experiments/{MODEL}/modeling/datasets/features_scatterplot.png",

        # Colors
        f"experiments/{MODEL}/modeling/color_shades.pkl",
        f"experiments/{MODEL}/modeling/main_colors.pkl",

        # Models
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_mdl_obj_{{reg}}.pkl", reg=REG_TYPES),
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_model_error_{{reg}}.json", reg=REG_TYPES),
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_regression_model_{{reg}}.pkl", reg=REG_TYPES),
        expand(f"experiments/{MODEL}/modeling/linear_{{reg}}/linear_results_{{reg}}.png", reg=REG_TYPES),

        f"experiments/{MODEL}/modeling/random_forest/random_forest_mdl_obj.pkl",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_model_error.json",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_model.pkl",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_results.png",
        f"experiments/{MODEL}/modeling/random_forest/random_forest_feature_importances.png",

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
# RULE infer_moments – always write status.json; fit_params.pkl only on success
##############################################################################
rule infer_moments:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl"
    output:
        status = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/moments/status.json"
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        fix      = ""
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.run_dir}/inferences"

        # Run and capture exit code
        set +e
        PYTHONPATH={workflow.basedir} python "snakemake_scripts/moments_dadi_inference.py" \
            --mode moments \
            --sfs-file "{input.sfs}" \
            --config "{params.cfg}" \
            --model-py "{params.model_py}" \
            --ground-truth "{input.params}" \
            --outdir "{params.run_dir}/inferences" \
            {params.fix}
        rc=$?
        set -e

        if [ $rc -eq 0 ] && [ -f "{params.run_dir}/inferences/moments/best_fit.pkl" ]; then
            mkdir -p "{params.run_dir}/inferences/moments"
            cp "{params.run_dir}/inferences/moments/best_fit.pkl" "{params.run_dir}/inferences/moments/fit_params.pkl"
            python -c 'import json,pathlib; p="{output.status}"; pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True); open(p,"w").write(json.dumps(dict(status="ok", reason=None)))'
        else
            python -c 'import json,pathlib; p="{output.status}"; pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True); open(p,"w").write(json.dumps(dict(status="failed", reason="optimize_error_or_missing_output")))'
        fi
        """

##############################################################################
# RULE infer_dadi – always write status.json; fit_params.pkl only on success
##############################################################################
rule infer_dadi:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl"
    output:
        status = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/dadi/status.json"
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        fix      = ""
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.run_dir}/inferences"

        set +e
        PYTHONPATH={workflow.basedir} python "snakemake_scripts/moments_dadi_inference.py" \
            --mode dadi \
            --sfs-file "{input.sfs}" \
            --config "{params.cfg}" \
            --model-py "{params.model_py}" \
            --ground-truth "{input.params}" \
            --outdir "{params.run_dir}/inferences" \
            {params.fix}
        rc=$?
        set -e

        if [ $rc -eq 0 ] && [ -f "{params.run_dir}/inferences/dadi/best_fit.pkl" ]; then
            mkdir -p "{params.run_dir}/inferences/dadi"
            cp "{params.run_dir}/inferences/dadi/best_fit.pkl" "{params.run_dir}/inferences/dadi/fit_params.pkl"
            python -c 'import json,pathlib; p="{output.status}"; pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True); open(p,"w").write(json.dumps(dict(status="ok", reason=None)))'
        else
            python -c 'import json,pathlib; p="{output.status}"; pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True); open(p,"w").write(json.dumps(dict(status="failed", reason="optimize_error_or_missing_output")))'
        fi
        """

# ── MOMENTS ONLY ───────────────────────────────────────────────────────────
rule aggregate_opts_moments:
    input:
        statuses = lambda w: [f"{RUN_DIR(w.sid, o)}/inferences/moments/status.json" for o in range(NUM_OPTIMS)]
    output:
        mom    = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl",
        status = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/aggregate.status"
    run:
        import pickle, numpy as np, pathlib, json, os
        sid = wildcards.sid
        # collect existing pkls from all opts
        pkls = [opt_pkl(sid, o, "moments") for o in range(NUM_OPTIMS)]
        pkls = [p for p in pkls if os.path.exists(p)]

        pathlib.Path(output.mom).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(output.status).parent.mkdir(parents=True, exist_ok=True)

        if not pkls:
            # No successes: write an empty aggregate payload but still mark status
            with open(output.mom, "wb") as f:
                pickle.dump({"best_params": [], "best_ll": []}, f)
            with open(output.status, "w") as f:
                json.dump({"n_success": 0}, f)
        else:
            def _as_list(x):
                return x if isinstance(x, (list, tuple, np.ndarray)) else [x]
            params, lls = [], []
            for p in pkls:
                d = pickle.load(open(p, "rb"))
                params.extend(_as_list(d.get("best_params", [])))
                lls.extend(_as_list(d.get("best_ll", [])))
            if len(lls) == 0:
                keep_params, keep_lls = [], []
                n_success = 0
            else:
                keep = np.argsort(lls)[::-1][:TOP_K]
                keep_params = [params[i] for i in keep]
                keep_lls    = [lls[i]    for i in keep]
                n_success   = len(pkls)

            with open(output.mom, "wb") as f:
                pickle.dump({"best_params": keep_params, "best_ll": keep_lls}, f)
            with open(output.status, "w") as f:
                json.dump({"n_success": int(n_success)}, f)

# ── DADI ONLY ──────────────────────────────────────────────────────────────
rule aggregate_opts_dadi:
    input:
        statuses = lambda w: [f"{RUN_DIR(w.sid, o)}/inferences/dadi/status.json" for o in range(NUM_OPTIMS)]
    output:
        dadi   = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",
        status = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/aggregate.status"
    run:
        import pickle, numpy as np, pathlib, json, os
        sid = wildcards.sid
        pkls = [opt_pkl(sid, o, "dadi") for o in range(NUM_OPTIMS)]
        pkls = [p for p in pkls if os.path.exists(p)]

        pathlib.Path(output.dadi).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(output.status).parent.mkdir(parents=True, exist_ok=True)

        if not pkls:
            with open(output.dadi, "wb") as f:
                pickle.dump({"best_params": [], "best_ll": []}, f)
            with open(output.status, "w") as f:
                json.dump({"n_success": 0}, f)
        else:
            def _as_list(x):
                import numpy as np
                return x if isinstance(x, (list, tuple, np.ndarray)) else [x]
            params, lls = [], []
            for p in pkls:
                d = pickle.load(open(p, "rb"))
                params.extend(_as_list(d.get("best_params", [])))
                lls.extend(_as_list(d.get("best_ll", [])))
            if len(lls) == 0:
                keep_params, keep_lls = [], []
                n_success = 0
            else:
                keep = np.argsort(lls)[::-1][:TOP_K]
                keep_params = [params[i] for i in keep]
                keep_lls    = [lls[i]    for i in keep]
                n_success   = len(pkls)

            with open(output.dadi, "wb") as f:
                pickle.dump({"best_params": keep_params, "best_ll": keep_lls}, f)
            with open(output.status, "w") as f:
                json.dump({"n_success": int(n_success)}, f)

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
# RULE compute_fim – produce placeholder if no valid fit
##############################################################################
rule compute_fim:
    input:
        agg_status = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/aggregate.status",
        sfs        = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl"
    output:
        fim  = f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.fim.npy",
        summ = f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.summary.json"
    params:
        fit_pkl = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl",
        script  = "snakemake_scripts/compute_fim.py",
        cfg     = EXP_CFG
    threads: 2
    run:
        import json, numpy as np, os, pathlib, pickle
        pathlib.Path(output.fim).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(output.summ).parent.mkdir(parents=True, exist_ok=True)

        ok = os.path.exists(params.fit_pkl)
        best_params_ok = False
        if ok:
            try:
                d = pickle.load(open(params.fit_pkl, "rb"))
                best_params_ok = bool(d.get("best_params"))
            except Exception:
                best_params_ok = False

        if not (ok and best_params_ok):
            # write minimal placeholder (0x0 FIM) and a summary noting skip
            np.save(output.fim, np.zeros((0,0), dtype=float))
            with open(output.summ, "w") as f:
                json.dump({"skipped": True, "reason": "no_valid_fit"}, f, indent=2)
        else:
            # run the real FIM computation
            shell("""
                PYTHONPATH={workflow.basedir} \
                python {params.script} \
                    --engine {wildcards.engine} \
                    --fit-pkl {params.fit_pkl} \
                    --sfs {input.sfs} \
                    --config {params.cfg} \
                    --fim-npy {output.fim} \
                    --summary-json {output.summ}
            """)


##############################################################################
# RULE sfs_residuals – produce placeholder if no valid fit
##############################################################################
rule sfs_residuals:
    input:
        obs_sfs   = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        agg_status = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/aggregate.status"
    output:
        res_arr   = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals.npy",
        res_flat  = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_flat.npy",
        meta_json = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/meta.json",
        hist_png  = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_histogram.png"
    params:
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        inf_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}",
        out_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{w.engine}",
        fit_pkl  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl"
    threads: 1
    run:
        import json, os, numpy as np, pathlib, pickle
        pathlib.Path(params.out_dir).mkdir(parents=True, exist_ok=True)

        ok = os.path.exists(params.fit_pkl)
        best_params_ok = False
        if ok:
            try:
                d = pickle.load(open(params.fit_pkl, "rb"))
                best_params_ok = bool(d.get("best_params"))
            except Exception:
                best_params_ok = False

        if not (ok and best_params_ok):
            # write placeholders so downstream can proceed
            np.save(output.res_arr, np.array([], dtype=float))
            np.save(output.res_flat, np.array([], dtype=float))
            with open(output.meta_json, "w") as f:
                json.dump({"skipped": True, "reason": "no_valid_fit"}, f, indent=2)
            # create an empty histogram image
            shell("python - <<'PY'\nimport matplotlib.pyplot as plt\nplt.figure(); plt.title('No residuals (no valid fit)'); plt.savefig('{hist}');\nPY".format(hist=output.hist_png))
        else:
            shell(r"""
                set -euo pipefail
                PYTHONPATH={workflow.basedir} \
                python "{RESID_SCRIPT}" \
                  --mode {wildcards.engine} \
                  --config "{params.cfg}" \
                  --model-py "{params.model_py}" \
                  --observed-sfs "{input.obs_sfs}" \
                  --inference-dir "{params.inf_dir}" \
                  --outdir "{params.out_dir}"
                test -f "{output.res_arr}"   && \
                test -f "{output.res_flat}"  && \
                test -f "{output.meta_json}" && \
                test -f "{output.hist_png}"
            """)

##############################################################################
# RULE combine_results – robust: no memmap in pickle, optional momentsLD
##############################################################################
rule combine_results:
    input:
        cfg        = EXP_CFG,
        mom_status = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/moments/aggregate.status",
        dadi_status= lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/dadi/aggregate.status",
        # NOTE: momentsLD removed from inputs to keep this rule optional wrt that file
    output:
        combo = f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl"
    run:
        import os, json, pickle, pathlib
        import numpy as np

        sid = wildcards.sid
        pathlib.Path(output.combo).parent.mkdir(parents=True, exist_ok=True)

        def safe_load_pickle(path, default):
            if not os.path.exists(path):
                return default
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return default

        summary = {}

        # moments
        mom_pkl = f"experiments/{MODEL}/inferences/sim_{sid}/moments/fit_params.pkl"
        summary["moments"] = safe_load_pickle(
            mom_pkl, {"best_params": np.array([], dtype=np.float32), "best_ll": np.array([], dtype=np.float32)}
        )

        # dadi
        dadi_pkl = f"experiments/{MODEL}/inferences/sim_{sid}/dadi/fit_params.pkl"
        summary["dadi"] = safe_load_pickle(
            dadi_pkl, {"best_params": np.array([], dtype=np.float32), "best_ll": np.array([], dtype=np.float32)}
        )

        # momentsLD (optional)
        mld_pkl = f"experiments/{MODEL}/inferences/sim_{sid}/MomentsLD/best_fit.pkl"
        if os.path.exists(mld_pkl):
            summary["momentsLD"] = safe_load_pickle(mld_pkl, {})
        else:
            summary["momentsLD"] = {}

        # -------- FIM payloads (force ndarrays; pack upper-triangle) --------
        fim_dir = f"experiments/{MODEL}/inferences/sim_{sid}/fim"
        fim_payload = {}
        fim_nan_total = 0
        if os.path.isdir(fim_dir):
            for name in os.listdir(fim_dir):
                if not name.endswith(".fim.npy"):
                    continue
                eng = name[:-8]  # strip ".fim.npy"
                path = os.path.join(fim_dir, name)

                try:
                    # default np.load returns a regular ndarray (not a memmap)
                    F = np.load(path, allow_pickle=False)
                    if F.ndim != 2 or F.shape[0] != F.shape[1]:
                        # skip corrupt/non-square matrices
                        continue

                    # stats *before* packing, to help you debug NaNs
                    n_nan = int(np.isnan(F).sum())
                    fim_nan_total += n_nan

                    iu = np.triu_indices(F.shape[0])
                    tri_flat = np.asarray(F[iu], dtype=np.float32)  # real ndarray copy, no memmap

                    fim_payload[eng] = {
                        "shape": [int(F.shape[0]), int(F.shape[1])],
                        "tri_flat": tri_flat,  # ndarray(float32)
                        "indices": "upper_including_diagonal",
                        "order": "row-major",
                    }
                except Exception:
                    # if that engine's FIM is unreadable/corrupt, skip it
                    continue

        if fim_payload:
            summary["FIM"] = fim_payload

        # -------- Residuals payloads (force ndarrays; keep flat) --------
        resid_root = f"experiments/{MODEL}/inferences/sim_{sid}/sfs_residuals"
        resid_payload = {}
        if os.path.isdir(resid_root):
            for eng in os.listdir(resid_root):
                base = os.path.join(resid_root, eng)
                flat_path = os.path.join(base, "residuals_flat.npy")
                arr_path  = os.path.join(base, "residuals.npy")
                if os.path.exists(flat_path):
                    try:
                        flat = np.load(flat_path, allow_pickle=False)
                        flat = np.asarray(flat, dtype=np.float32)  # ensure ndarray (not memmap)

                        if os.path.exists(arr_path):
                            arr = np.load(arr_path, allow_pickle=False)
                            shape = list(arr.shape)
                        else:
                            shape = [0] if flat.size == 0 else [int(flat.size)]

                        resid_payload[eng] = {
                            "shape": shape,
                            "flat": flat,  # ndarray(float32)
                            "order": "row-major",
                        }
                    except Exception:
                        # skip bad residuals for this engine
                        continue

        if resid_payload:
            summary["SFS_residuals"] = resid_payload

        # write combined pickle (arrays are real ndarrays; safe to unpickle anywhere)
        with open(output.combo, "wb") as f:
            pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

        # light log to STDOUT to help you spot NaNs and counts
        print(json.dumps({
            "sim": int(sid),
            "fim_engines": len(summary.get("FIM", {})),
            "fim_total_nan": fim_nan_total
        }))
        print(f"✓ combined → {output.combo}")


##############################################################################
# RULE combine_features – build datasets (filter, split, normalize)          #
##############################################################################
rule combine_features:
    input:
        cfg  = EXP_CFG,
        infs = expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl", sid=SIM_IDS),
        truths = expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl", sid=SIM_IDS)
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
    opt    = "|".join(str(i) for i in range(NUM_OPTIMS)),
    engine = "moments|dadi"

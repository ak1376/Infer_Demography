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
EXP_CFG = "config_files/experiment_config_OOA_three_pop.json"

# Experiment metadata
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = int(CFG["num_draws"])
NUM_OPTIMS    = int(CFG.get("num_optimizations", 3))
NUM_REAL_OPTIMS = int(CFG.get("num_optimizations", 3))
TOP_K         = int(CFG.get("top_k", 2))
NUM_WINDOWS   = int(CFG.get("num_windows", 100))

# Engines to COMPUTE (always); modeling usage is controlled in feature_extraction via config
FIM_ENGINES = CFG.get("fim_engines", ["moments"])

USE_GPU_LD = CFG.get("use_gpu_ld", False)
USE_GPU_DADI = CFG.get("use_gpu_dadi", False)

USE_GS = bool(CFG.get("gram_schmidt", False))

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

# LD r-bins
R_BINS_STR = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"


SIM_IDS  = [i for i in range(NUM_DRAWS)]
WINDOWS  = range(NUM_WINDOWS)

# ##############################################################################
# # RULE all – final targets the workflow must create                          #
# ##############################################################################
rule all:
    input:
        [
            # ---------------- SIMULATED DATA PIPELINE ----------------
            # expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",  sid=SIM_IDS),
            # expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",             sid=SIM_IDS),
            # expand(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees", sid=SIM_IDS),
            # expand(f"{SIM_BASEDIR}/{{sid}}/demes.png",           sid=SIM_IDS),

            # Aggregated optimizer results (simulated)
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl", sid=SIM_IDS),
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",   sid=SIM_IDS),

            # Cleanup completion markers (simulated)
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/cleanup_done.txt", sid=SIM_IDS),

            # LD artifacts (simulated; best-fit only)
            # expand(f"{LD_ROOT}/best_fit.pkl", sid=SIM_IDS),

            # FIM (simulated)
            # expand(
            #     f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.fim.npy",
            #     sid=SIM_IDS, engine=FIM_ENGINES
            # ),

            # Residuals (simulated)
            # expand(
            #     f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_flat.npy",
            #     sid=SIM_IDS, engine=RESIDUAL_ENGINES
            # ),

            # Combined per-sim inference blobs (simulated)
            # expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl", sid=SIM_IDS),

            # ---------------- MODELING (on simulated inferences) ----------------
            # f"experiments/{MODEL}/modeling/datasets/features_df.pkl",
            # f"experiments/{MODEL}/modeling/datasets/targets_df.pkl",

            # ---------------- REAL DATA ----------------
            # Real-data SFS
            f"experiments/{MODEL}/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.sfs.pkl",

            # REAL DATA: per-optimization SFS inferences
            *expand(
                f"experiments/{MODEL}/real_data_analysis/runs/run_{{opt}}/inferences/moments/fit_params.pkl",
                opt=OPTIMS,
            ),
            *expand(
                f"experiments/{MODEL}/real_data_analysis/runs/run_{{opt}}/inferences/dadi/fit_params.pkl",
                opt=OPTIMS,
            ),

            # REAL DATA: aggregated inferences
            f"experiments/{MODEL}/real_data_analysis/inferences/moments/fit_params.pkl",
            f"experiments/{MODEL}/real_data_analysis/inferences/dadi/fit_params.pkl",

            *expand(f"{REAL_LD_ROOT}/windows/window_{{i}}.vcf.gz", i=WINDOWS),
            *expand(f"{REAL_LD_ROOT}/LD_stats/LD_stats_window_{{i}}.pkl", i=WINDOWS)

        ]

##############################################################################
# RULE simulate – one complete tree‑sequence + SFS
##############################################################################
rule simulate:
    output:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        tree   = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        fig    = f"{SIM_BASEDIR}/{{sid}}/demes.png",
        meta   = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
        done   = protected(f"{SIM_BASEDIR}/{{sid}}/.done"),
    params:
        sim_dir = SIM_BASEDIR,
        cfg     = EXP_CFG,
        model   = MODEL
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=/projects/kernlab/akapoor/Infer_Demography \
        python "{SIM_SCRIPT}" \
          --simulation-dir "{params.sim_dir}" \
          --experiment-config "{params.cfg}" \
          --model-type "{params.model}" \
          --simulation-number {wildcards.sid}

        # ensure expected outputs exist, then create sentinel
        test -f "{output.sfs}"    && \
        test -f "{output.params}" && \
        test -f "{output.tree}"   && \
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
    output:
        mom = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl"
    run:
        import pickle, numpy as np, pathlib, re, glob

        sid = wildcards.sid

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

            # counts as "contributing" only if it provides at least one LL
            if len(this_lls) == 0:
                continue

            n_nonempty += 1
            params.extend(this_params)
            lls.extend(this_lls)
            opt_ids.extend([opt_idx] * len(this_lls))

        # ---- FAIL HARD if fewer than TOP_K contributing opt pkls ----
        if n_nonempty < TOP_K:
            raise ValueError(
                f"[aggregate_opts_moments] Need >= {TOP_K} non-empty moments optimizations for sid={sid}, "
                f"but got nonempty={n_nonempty} (readable={n_readable}, paths_found={len(mom_pkls)}). "
                f"Not aggregating."
            )

        keep = np.argsort(lls)[::-1][:TOP_K]
        best = {
            "best_params":   [params[i] for i in keep],
            "best_ll":       [lls[i]    for i in keep],
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
    output:
        dadi = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl"
    run:
        import pickle, numpy as np, pathlib, re, glob

        sid = wildcards.sid

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

        # ---- FAIL HARD if fewer than TOP_K contributing opt pkls ----
        if n_nonempty < TOP_K:
            raise ValueError(
                f"[aggregate_opts_dadi] Need >= {TOP_K} non-empty dadi optimizations for sid={sid}, "
                f"but got nonempty={n_nonempty} (readable={n_readable}, paths_found={len(dadi_pkls)}). "
                f"Not aggregating."
            )

        keep = np.argsort(lls)[::-1][:TOP_K]
        best = {
            "best_params":   [params[i] for i in keep],
            "best_ll":       [lls[i]    for i in keep],
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

            print(f"🗑️ removing sid={sid} opt={opt}: {p}")
            subprocess.run(["rm", "-rf", str(p)], check=False)
            cleaned += 1

        pathlib.Path(output.cleanup_done).parent.mkdir(parents=True, exist_ok=True)
        with open(output.cleanup_done, "w") as f:
            f.write(f"Cleanup completed for simulation {sid}\n")
            f.write(f"Removed {cleaned} optimization directories\n")
            f.write(f"Kept optimizations: {sorted(keep_indices)}\n")

        print(f"✅ cleanup sid={sid}: removed {cleaned}, kept {sorted(keep_indices)}")

##############################################################################
# RULE simulate_window – one VCF window
##############################################################################
rule simulate_window:
    input:
        params   = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        metafile = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
        done     = f"{SIM_BASEDIR}/{{sid}}/.done"
    output:
        vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
        trees  = f"{LD_ROOT}/windows/window_{{win}}.trees"
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
        trees  = f"{LD_ROOT}/windows/window_{{win}}.trees"
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
        # Run the LD computation and capture exit code
        python "{LD_SCRIPT}" \
            --sim-dir      {params.sim_dir} \
            --window-index {wildcards.win} \
            --config-file  {params.cfg} \
            --r-bins       "{params.bins}" \
            --use-gpu

        
        EXIT_CODE=$?
        
        # If successful, clean up intermediate files
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ LD computation successful, cleaning up intermediate files for window {wildcards.win}"
            rm -vf {params.sim_dir}/windows/window_{wildcards.win}.h5
            rm -vf {params.sim_dir}/windows/window_{wildcards.win}.trees
            rm -vf {params.sim_dir}/windows/window_{wildcards.win}.vcf.gz
            echo "🧹 Cleanup completed for window {wildcards.win}"
        else
            echo "❌ LD computation failed (exit code $EXIT_CODE), preserving intermediate files for debugging"
            exit $EXIT_CODE
        fi
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
        r"""
        PYTHONPATH=/projects/kernlab/akapoor/Infer_Demography \
        python "snakemake_scripts/LD_inference.py" \
            --run-dir      {params.sim_dir} \
            --output-root  {params.LD_dir} \
            --config-file  {params.cfg}
        """

##############################################################################
# RULE compute_fim – observed FIM at best-LL params for {engine}             #
##############################################################################
rule compute_fim:
    input:
        fit = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl",
        sfs = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl"
    output:
        fim  = f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.fim.npy",
        summ = f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.summary.json"
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
        agg_fit = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl"
    output:
        res_arr   = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals.npy",
        res_flat  = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_flat.npy",
        meta_json = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/meta.json",
        hist_png  = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_histogram.png",
        # Only required when gram_schmidt=true
        gs_coeffs = (
            f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_gs_coeffs.npy"
            if USE_GS
            else temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/.gs_disabled")
        ),
        # Optional reproducibility artifacts (only if enabled)
        gs_basis = (
            f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_gs_basis.npy"
            if USE_GS
            else temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/.gs_basis_disabled")
        ),
        gs_recon = (
            f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_gs_reconstruction.npy"
            if USE_GS
            else temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/.gs_recon_disabled")
        )
    params:
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        inf_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}",
        out_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{w.engine}",
        n_bins   = CFG.get("sfs_n_bins", "")  # empty string if not specified
    threads: 1
    shell:
        r"""
        set -euo pipefail
        
        # Build n_bins argument conditionally
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

        # ensure base outputs exist
        test -f "{output.res_arr}"   && \
        test -f "{output.res_flat}"  && \
        test -f "{output.meta_json}" && \
        test -f "{output.hist_png}"

        # if GS enabled, ensure GS artifacts exist
        if [ "{USE_GS}" = "True" ]; then
          test -f "{output.gs_coeffs}"
          # basis + reconstruction are written by the script when GS is enabled
          test -f "{output.gs_basis}"
          test -f "{output.gs_recon}"
        fi
        """

##############################################################################
# RULE combine_results – merge dadi / moments / moments-LD fits per sim      #
# + attach flattened upper-triangular FIM and residual SFS payloads          #
##############################################################################
rule combine_results:
    input:
        cfg       = EXP_CFG,
        dadi      = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/dadi/fit_params.pkl",
        moments   = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/moments/fit_params.pkl",
        momentsLD = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/best_fit.pkl",
        fims      = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ],
        # 👇 pick which residual vector to include based on config
        resid_vecs = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ],
        # also pull meta so changes to GS settings trigger recombine
        resid_meta = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ],
    output:
        combo = f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl"
    run:
        import pickle, pathlib, numpy as np, re, os, json

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
        features_df   = f"experiments/{MODEL}/modeling/datasets/features_df.pkl",
        targets_df    = f"experiments/{MODEL}/modeling/datasets/targets_df.pkl",

        # raw splits
        train_X       = f"experiments/{MODEL}/modeling/datasets/train_features.pkl",
        train_y       = f"experiments/{MODEL}/modeling/datasets/train_targets.pkl",
        tune_X        = f"experiments/{MODEL}/modeling/datasets/tune_features.pkl",
        tune_y        = f"experiments/{MODEL}/modeling/datasets/tune_targets.pkl",
        val_X         = f"experiments/{MODEL}/modeling/datasets/validation_features.pkl",
        val_y         = f"experiments/{MODEL}/modeling/datasets/validation_targets.pkl",

        # normalized splits
        ntrain_X      = f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        ntrain_y      = f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        ntune_X       = f"experiments/{MODEL}/modeling/datasets/normalized_tune_features.pkl",
        ntune_y       = f"experiments/{MODEL}/modeling/datasets/normalized_tune_targets.pkl",
        nval_X        = f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        nval_y        = f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",

        # split indices + plots/metrics
        split_idx     = f"experiments/{MODEL}/modeling/datasets/split_indices.json",
        scatter_png   = f"experiments/{MODEL}/modeling/datasets/features_scatterplot.png",
        mse_val_png   = f"experiments/{MODEL}/modeling/datasets/mse_bars_val_normalized.png",
        mse_train_png = f"experiments/{MODEL}/modeling/datasets/mse_bars_train_normalized.png",
        metrics_all   = f"experiments/{MODEL}/modeling/datasets/metrics_all.json",
        metrics_dadi  = f"experiments/{MODEL}/modeling/datasets/metrics_dadi.json",
        metrics_moments = f"experiments/{MODEL}/modeling/datasets/metrics_moments.json",
        metrics_momentsLD = f"experiments/{MODEL}/modeling/datasets/metrics_momentsLD.json",
        outliers_tsv  = f"experiments/{MODEL}/modeling/datasets/outliers_removed.tsv",
        outliers_txt  = f"experiments/{MODEL}/modeling/datasets/outliers_preview.txt"
    params:
        script = "snakemake_scripts/feature_extraction.py",
        outdir = f"experiments/{MODEL}/modeling"
    threads: 1
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        /home/akapoor/miniforge3/envs/snakemake-env/bin/python "{params.script}" \
            --experiment-config "{input.cfg}" \
            --out-dir "{params.outdir}"

        # sanity checks
        test -f "{output.features_df}"   && \
        test -f "{output.targets_df}"    && \
        test -f "{output.train_X}"       && \
        test -f "{output.train_y}"       && \
        test -f "{output.tune_X}"        && \
        test -f "{output.tune_y}"        && \
        test -f "{output.val_X}"         && \
        test -f "{output.val_y}"         && \
        test -f "{output.ntrain_X}"      && \
        test -f "{output.ntrain_y}"      && \
        test -f "{output.ntune_X}"       && \
        test -f "{output.ntune_y}"       && \
        test -f "{output.nval_X}"        && \
        test -f "{output.nval_y}"        && \
        test -f "{output.split_idx}"     && \
        test -f "{output.scatter_png}"   && \
        test -f "{output.mse_val_png}"   && \
        test -f "{output.mse_train_png}" && \
        test -f "{output.metrics_all}"
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

        # ✅ add tune
        X_tune  = f"experiments/{MODEL}/modeling/datasets/normalized_tune_features.pkl",
        y_tune  = f"experiments/{MODEL}/modeling/datasets/normalized_tune_targets.pkl",

        X_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_features.pkl",
        y_val   = f"experiments/{MODEL}/modeling/datasets/normalized_validation_targets.pkl",
        shades  = f"experiments/{MODEL}/modeling/color_shades.pkl",
        colors  = f"experiments/{MODEL}/modeling/main_colors.pkl",
        mdlcfg  = "config_files/model_config.yaml"
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
            --X_tune_path  "{input.X_tune}" \
            --y_tune_path  "{input.y_tune}" \
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
        X_tune  = f"experiments/{MODEL}/modeling/datasets/normalized_tune_features.pkl",
        y_tune  = f"experiments/{MODEL}/modeling/datasets/normalized_tune_targets.pkl",
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
        "benchmarks/random_forest.tsv"
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
        X_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_features.pkl",
        y_train = f"experiments/{MODEL}/modeling/datasets/normalized_train_targets.pkl",
        X_tune  = f"experiments/{MODEL}/modeling/datasets/normalized_tune_features.pkl",
        y_tune  = f"experiments/{MODEL}/modeling/datasets/normalized_tune_targets.pkl",
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
        "benchmarks/xgboost.tsv"
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
# RULE download_1000G_data – Download and prepare 1000 Genomes data         #
##############################################################################
rule download_1000G_data:
    output:
        vcf      = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.vcf.gz",
        tbi      = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.vcf.gz.tbi",
        yri      = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI.samples",
        ceu      = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/CEU.samples",
        chb      = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/CHB.samples",
        popfile  = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.popfile",
        done     = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/.download_done"
    shell:
        r"""
        set -euo pipefail

        # Run the download and preparation script, passing the output directory
        bash bash_scripts/download_and_prepare_1000G_YRI_CEU_CHB.sh \
          "$(dirname {output.vcf})"

        # Verify all expected outputs exist
        test -f "{output.vcf}" && \
        test -f "{output.tbi}" && \
        test -f "{output.yri}" && \
        test -f "{output.ceu}" && \
        test -f "{output.chb}" && \
        test -f "{output.popfile}"

        # Create completion marker
        touch "{output.done}"

        echo "✓ 1000 Genomes data download and preparation complete"
        """



##############################################################################
# RULE compute_real_data_sfs – build 3D SFS from YRI/CEU/CHB VCF             #
##############################################################################
rule compute_real_data_sfs:
    input:
        vcf     = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.vcf.gz",
        popfile = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.popfile",
    output:
        sfs = f"experiments/{MODEL}/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.sfs.pkl"
    params:
        config = "config_files/experiment_config_OOA_three_pop.json",
    shell:
        r"""
        set -euo pipefail

        python snakemake_scripts/real_data_sfs.py \
          --input-vcf {input.vcf} \
          --popfile   {input.popfile} \
          --config    {params.config} \
          --output-sfs {output.sfs}
        """



##############################################################################
# REAL DATA – NLopt Poisson SFS optimisation (moments)
##############################################################################
rule infer_moments_real:
    """
    Run moments-based SFS inference on the real CEU/YRI SFS.
    Uses the same model + config, but stores results in runs/run_real_{opt}.
    """
    input:
        sfs = f"experiments/{MODEL}/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.sfs.pkl",
    output:
        # One run directory per opt, mirroring simulations:
        pkl = f"experiments/{MODEL}/real_data_analysis/runs/run_{{opt}}/inferences/moments/fit_params.pkl"
    params:
        run_dir  = lambda w: f"experiments/{MODEL}/real_data_analysis/runs/run_{w.opt}",
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        fix      = ""  # you can plug in real-data fixes here if you want
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
          --outdir "{params.run_dir}/inferences" \
          --generate-profiles \
          {params.fix}

        cp "{params.run_dir}/inferences/moments/best_fit.pkl" "{output.pkl}"
        """

##############################################################################
# REAL DATA – NLopt Poisson SFS optimisation (dadi)
##############################################################################
rule infer_dadi_real:
    """
    Run dadi-based SFS inference on the real CEU/YRI SFS.
    Stores results in runs/run_real_{opt}.
    """
    input:
        sfs = f"experiments/{MODEL}/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.sfs.pkl",
    output:
        pkl = f"experiments/{MODEL}/real_data_analysis/runs/run_{{opt}}/inferences/dadi/fit_params.pkl"
    params:
        run_dir  = lambda w: f"experiments/{MODEL}/real_data_analysis/runs/run_{w.opt}",
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        fix      = ""  # e.g. '--fix N0=10000' if you want to constrain N0 for real data
    threads: 8
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/moments_dadi_inference.py" \
          --mode dadi \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --outdir "{params.run_dir}/inferences" \
          {params.fix}

        cp "{params.run_dir}/inferences/dadi/best_fit.pkl" "{output.pkl}"
        """

# ── REAL DATA: MOMENTS ONLY ────────────────────────────────────────────────
rule aggregate_opts_moments_real:
    input:
        mom = [f"experiments/{MODEL}/real_data_analysis/runs/run_{o}/inferences/moments/fit_params.pkl"
               for o in range(NUM_REAL_OPTIMS)]
    output:
        mom = f"experiments/{MODEL}/real_data_analysis/inferences/moments/fit_params.pkl"
    run:
        import pickle, numpy as np, pathlib

        def _as_list(x):
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        params, lls, opt_ids = [], [], []

        for opt_idx, pkl in enumerate(input.mom):
            d = pickle.load(open(pkl, "rb"))
            this_params = _as_list(d["best_params"])
            this_lls    = _as_list(d["best_ll"])

            params.extend(this_params)
            lls.extend(this_lls)
            opt_ids.extend([opt_idx] * len(this_lls))

        keep = np.argsort(lls)[::-1][:TOP_K]

        best = {
            "best_params": [params[i] for i in keep],
            "best_ll":     [lls[i]    for i in keep],
            "opt_index":   [opt_ids[i] for i in keep],
        }

        pathlib.Path(output.mom).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.mom, "wb"))

        print(f"✅ [REAL] Aggregated {len(params)} moments optimization results → {output.mom}")
        print(f"✅ [REAL] Kept top-{TOP_K} moments optimizations (opts={sorted(set(best['opt_index']))})")


# ── REAL DATA: DADI ONLY ───────────────────────────────────────────────────
rule aggregate_opts_dadi_real:
    input:
        dadi = [f"experiments/{MODEL}/real_data_analysis/runs/run_{o}/inferences/dadi/fit_params.pkl"
                for o in range(NUM_REAL_OPTIMS)]
    output:
        dadi = f"experiments/{MODEL}/real_data_analysis/inferences/dadi/fit_params.pkl"
    run:
        import pickle, numpy as np, pathlib

        def _as_list(x):
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        params, lls, opt_ids = [], [], []

        for opt_idx, pkl in enumerate(input.dadi):
            d = pickle.load(open(pkl, "rb"))
            this_params = _as_list(d["best_params"])
            this_lls    = _as_list(d["best_ll"])

            params.extend(this_params)
            lls.extend(this_lls)
            opt_ids.extend([opt_idx] * len(this_lls))

        keep = np.argsort(lls)[::-1][:TOP_K]

        best = {
            "best_params": [params[i] for i in keep],
            "best_ll":     [lls[i]    for i in keep],
            "opt_index":   [opt_ids[i] for i in keep],
        }

        pathlib.Path(output.dadi).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.dadi, "wb"))

        print(f"✅ [REAL] Aggregated {len(params)} dadi optimization results → {output.dadi}")
        print(f"✅ [REAL] Kept top-{TOP_K} dadi optimizations (opts={sorted(set(best['opt_index']))})")



##############################################################################
# REAL DATA LD ANALYSIS
##############################################################################

# One job per window
rule split_real_vcf_window:
    input:
        vcf     = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.chr1.no_exons.vcf.gz",
        popfile = "experiments/OOA_three_pop/real_data_analysis/data/data_chr1_YRI_CEU_CHB/YRI_CEU_CHB.popfile"
    output:
        vcf_gz = f"{REAL_LD_ROOT}/windows/window_{{i}}.vcf.gz"
    params:
        script      = "snakemake_scripts/split_vcf_windows.py",
        window_size = 10_000_000,
        num_windows = NUM_WINDOWS
    shell:
        """
        python {params.script} \
            --input-vcf {input.vcf} \
            --popfile {input.popfile} \
            --out-dir {REAL_LD_ROOT}/windows \
            --window-size {params.window_size} \
            --num-windows {params.num_windows} \
            --window-index {wildcards.i}
        """

rule split_real_vcf:
    input:
        expand(f"{REAL_LD_ROOT}/windows/window_{{i}}.vcf.gz", i=WINDOWS)

rule compute_ld_real:
    input:
        vcf_gz = f"{REAL_LD_ROOT}/windows/window_{{i}}.vcf.gz"
    output:
        pkl = f"{REAL_LD_ROOT}/LD_stats/LD_stats_window_{{i}}.pkl"
    params:
        script = "snakemake_scripts/compute_ld_window.py",
        config = EXP_CFG,
        r_bins = "0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"
    shell:
        """
        python {params.script} \
            --sim-dir {REAL_LD_ROOT} \
            --window-index {wildcards.i} \
            --config-file {params.config} \
            --r-bins "{params.r_bins}" \
            --use-gpu

        """

rule real_ld:
    input:
        expand(f"{REAL_LD_ROOT}/LD_stats/LD_stats_window_{{i}}.pkl", i=WINDOWS)
        
##############################################################################
# Wildcard Constraints                                                      #
##############################################################################
wildcard_constraints:
    opt    = "|".join(str(i) for i in range(NUM_OPTIMS)),
    engine = "moments|dadi"

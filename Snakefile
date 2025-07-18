##############################################################################
# CONFIG – Paths and Constants (edit here only)                             #
##############################################################################
import json, math, sys
from pathlib import Path

# ── guard against shadow‑import of the Bottleneck C library ────────────────
if "bottleneck" in sys.modules and not hasattr(sys.modules["bottleneck"], "__version__"):
    del sys.modules["bottleneck"]

# ── external scripts -------------------------------------------------------
SIM_SCRIPT   = "snakemake_scripts/simulation.py"
INFER_SCRIPT = "snakemake_scripts/moments_dadi_inference.py"
WIN_SCRIPT   = "snakemake_scripts/simulate_window.py"
LD_SCRIPT    = "snakemake_scripts/compute_ld_window.py"
EXP_CFG      = "config_files/experiment_config_split_isolation.json"

# ── experiment metadata ----------------------------------------------------
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = CFG["num_draws"]          # number of independent sims
NUM_OPTIMS    = CFG.get("num_optimizations", 3)
TOP_K         = CFG.get("top_k", 2)
NUM_WINDOWS   = 100                       # LD windows per simulation
R_BINS_STR    = "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

_pad     = int(math.log10(NUM_DRAWS - 1)) + 1
SIM_IDS  = [f"{i:0{_pad}d}" for i in range(NUM_DRAWS)]  # 00, 01 …
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
        # simulation + moments/dadi inference summary files
        expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",             sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/demes.png",           sid=SIM_IDS),
        [final_pkl(sid, "moments") for sid in SIM_IDS],
        [final_pkl(sid, "dadi")    for sid in SIM_IDS],
        # LD windows + LD‑stats + best‑fit results
        expand(f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",    sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl", sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/best_fit.pkl", sid=SIM_IDS), 
        expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl",
            sid=SIM_IDS
        ),


##############################################################################
# RULE simulate – one complete tree‑sequence + SFS                          #
##############################################################################
rule simulate:
    output:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        tree   = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        fig    = f"{SIM_BASEDIR}/{{sid}}/demes.png",
    params:
        sim_dir = SIM_BASEDIR,
        cfg     = EXP_CFG,
        model   = MODEL,
    threads: 1
    shell:
        """
        python "{SIM_SCRIPT}" \
            --simulation-dir {params.sim_dir} \
            --experiment-config {params.cfg} \
            --model-type {params.model} \
            --simulation-number {wildcards.sid}
        """

##############################################################################
# RULE infer_sfs – one optimiser start (moments + dadi)                     #
##############################################################################
rule infer_sfs:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        cfg    = EXP_CFG
    output:
        mom  = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/moments/fit_params.pkl",
        dadi = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/dadi/fit_params.pkl"
    params:
        run_dir = lambda w: RUN_DIR(w.sid, w.opt)
    threads: 4
    shell:
        """
        python "{INFER_SCRIPT}" \
            --run-dir        "{params.run_dir}" \
            --config-file    "{input.cfg}" \
            --sfs            "{input.sfs}" \
            --sampled-params "{input.params}" \
            --rep-index      {wildcards.opt} \
            --run-moments    True \
            --run-dadi       True
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
        def merge_keep_best(in_files, out_file):
            params, lls = [], []
            for pkl in in_files:
                d = pickle.load(open(pkl, "rb"))
                params.extend(d["best_params"])
                lls.extend(d["best_lls"])
            keep = np.argsort(lls)[::-1][:TOP_K]
            best = {"best_params": [params[i] for i in keep],
                    "best_lls"   : [lls[i]    for i in keep]}
            pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(best, open(out_file, "wb"))
        merge_keep_best(input.mom,  output.mom)
        merge_keep_best(input.dadi, output.dadi)

##############################################################################
# RULE simulate_window – one VCF window                                     #
##############################################################################
rule simulate_window:
    input:
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        cfg    = EXP_CFG
    output:
        vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz"
    params:
        base_sim   = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        out_winDir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
        rep_idx    = "{win}"
    threads: 1
    shell:
        """
        python "{WIN_SCRIPT}" \
            --sim-dir      {params.base_sim} \
            --rep-index    {params.rep_idx} \
            --config-file  {input.cfg} \
            --out-dir      {params.out_winDir}
        """

##############################################################################
# RULE ld_window – LD statistics for one window                             #
##############################################################################
rule ld_window:
    input:
        vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
        cfg    = EXP_CFG
    output:
        pkl    = f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl"
    params:
        sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        bins    = R_BINS_STR
    threads: 1
    shell:
        """
        python "{LD_SCRIPT}" \
            --sim-dir      {params.sim_dir} \
            --window-index {wildcards.win} \
            --config-file  {input.cfg} \
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
        cfg  = EXP_CFG
    output:
        mv   = f"{LD_ROOT}/means.varcovs.pkl",
        boot = f"{LD_ROOT}/bootstrap_sets.pkl",
        pdf  = f"{LD_ROOT}/empirical_vs_theoretical_comparison.pdf",
        best = f"{LD_ROOT}/best_fit.pkl"
    params:
        sim_dir   = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        LD_dir    = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        bins      = R_BINS_STR,
        n_windows = NUM_WINDOWS
    threads: 1
    shell:
        """
        python "snakemake_scripts/LD_inference.py" \
            --sim-dir      {params.sim_dir} \
            --LD_dir       {params.LD_dir} \
            --config-file  {input.cfg} \
            --num-windows  {params.n_windows} \
            --r-bins       "{params.bins}"
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
# Wildcard Constraints                                                      #
##############################################################################
wildcard_constraints:
    opt = "|".join(str(i) for i in range(NUM_OPTIMS))
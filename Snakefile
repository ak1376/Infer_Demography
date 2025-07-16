##############################################################################
# CONFIG – Paths and Constants (Edit These)
##############################################################################
import json, math, sys
from pathlib import Path

# Clean shadowed bottleneck import (if present)
if "bottleneck" in sys.modules and not hasattr(sys.modules["bottleneck"], "__version__"):
    del sys.modules["bottleneck"]

# Scripts
SIM_SCRIPT    = "snakemake_scripts/simulation.py"
INFER_SCRIPT  = "snakemake_scripts/moments_dadi_inference.py"
WIN_SCRIPT    = "snakemake_scripts/simulate_window.py"
LD_SCRIPT     = "snakemake_scripts/compute_ld_window.py"
EXP_CFG       = "config_files/experiment_config_bottleneck.json"

# Load config
CFG         = json.loads(Path(EXP_CFG).read_text())
MODEL       = CFG["demographic_model"]
NUM_DRAWS   = CFG["num_draws"]
NUM_OPTIMS  = CFG.get("num_optimizations", 3)
TOP_K       = CFG.get("top_k", 2)
NUM_WINDOWS = 100
R_BINS_STR  = "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

# IDs
_pad     = int(math.log10(NUM_DRAWS - 1)) + 1
SIM_IDS  = [f"{i:0{_pad}d}" for i in range(NUM_DRAWS)]
WINDOWS  = range(NUM_WINDOWS)

# Paths
SIM_BASEDIR = f"experiments/{MODEL}/simulations"
RUN_DIR     = lambda sid, opt: f"experiments/{MODEL}/runs/run_{sid}_{opt}"
LD_ROOT     = lambda sid: f"experiments/{MODEL}/inferences/sim_{sid}/MomentsLD"

opt_pkl   = lambda sid, opt, tool: f"{RUN_DIR(sid, opt)}/inferences/{tool}/fit_params.pkl"
final_pkl = lambda sid, tool: f"experiments/{MODEL}/inferences/sim_{sid}/{tool}/fit_params.pkl"

##############################################################################
# RULE all – Everything
##############################################################################
rule all:
    input:
        # simulation + inference
        expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/demes.png", sid=SIM_IDS),
        [final_pkl(sid, "moments") for sid in SIM_IDS],
        [final_pkl(sid, "dadi")    for sid in SIM_IDS],

        # LD: windows and LD stats
        expand(f"{LD_ROOT}/sim_{{sid}}/windows/window_{{win}}.vcf.gz", sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/sim_{{sid}}/LD_stats/LD_stats_window_{{win}}.pkl", sid=SIM_IDS, win=WINDOWS),
        expand(f"{LD_ROOT}/sim_{{sid}}/best_fit.pkl", sid=SIM_IDS)

##############################################################################
rule simulate:
    output:
        sfs     = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params  = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        tree    = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        fig     = f"{SIM_BASEDIR}/{{sid}}/demes.png",
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

rule simulate_window:
    input:
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        cfg    = EXP_CFG
    output:
        vcf_gz = f"{LD_ROOT}/sim_{{sid}}/windows/window_{{win}}.vcf.gz"
    params:
        base_sim   = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        out_winDir = lambda w: f"{LD_ROOT}/sim_{w.sid}/windows",
        rep_idx    = lambda w: int(w.win)
    threads: 1
    shell:
        """
        python "{WIN_SCRIPT}" \
            --sim-dir      {params.base_sim} \
            --rep-index    {params.rep_idx} \
            --config-file  {input.cfg} \
            --out-dir      {params.out_winDir}
        """

rule ld_window:
    input:
        vcf_gz = f"{LD_ROOT}/sim_{{sid}}/windows/window_{{win}}.vcf.gz",
        cfg    = EXP_CFG
    output:
        pkl    = f"{LD_ROOT}/sim_{{sid}}/LD_stats/LD_stats_window_{{win}}.pkl"
    params:
        sim_dir = lambda w: f"{LD_ROOT}/sim_{w.sid}",
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

rule optimize_momentsld:
    input:
        pkls = lambda w: expand(f"{LD_ROOT}/sim_{w.sid}/LD_stats/LD_stats_window_{{win}}.pkl", win=WINDOWS),
        cfg  = EXP_CFG
    output:
        mv    = f"{LD_ROOT}/sim_{{sid}}/means.varcovs.pkl",
        boot  = f"{LD_ROOT}/sim_{{sid}}/bootstrap_sets.pkl",
        pdf   = f"{LD_ROOT}/sim_{{sid}}/empirical_vs_theoretical_comparison.pdf",
        best  = f"{LD_ROOT}/sim_{{sid}}/best_fit.pkl"
    params:
        sim_dir   = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        LD_dir    = lambda w: f"{LD_ROOT}/sim_{w.sid}",
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
# Wildcard Constraints
##############################################################################
wildcard_constraints:
    opt = "|".join(str(i) for i in range(NUM_OPTIMS))

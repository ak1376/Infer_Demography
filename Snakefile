##############################################################################
# CONFIG – adjust paths & constants once
##############################################################################

import sys
# strip accidental shadowing of the Bottleneck package
if "bottleneck" in sys.modules and not hasattr(sys.modules["bottleneck"], "__version__"):
    del sys.modules["bottleneck"]

SIM_SCRIPT  = "snakemake_scripts/simulation.py"          # base simulator
WIN_SCRIPT  = "snakemake_scripts/simulate_window.py"     # 1 window ➜ VCF
LD_SCRIPT   = "snakemake_scripts/compute_ld_window.py"   # 1 window ➜ LD.pkl
EXP_CFG     = "config_files/experiment_config_split_isolation.json"

SIM_BASEDIR = "ld_experiments/split_isolation/simulations"    # simulations/<sid>
LD_ROOT     = "MomentsLD/LD_stats"                       # final LD folder

NUM_WINDOWS = 100                                        # per simulation
R_BINS_STR  = "0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

##############################################################################
# derive simulation IDs from experiment config
##############################################################################
import json, math
from pathlib import Path

with open(EXP_CFG) as f:
    cfg = json.load(f)

n_draws = cfg["num_draws"]
pad     = int(math.log10(n_draws - 1)) + 1
SIM_IDS = [f"{i:0{pad}d}" for i in range(n_draws)]

# NEW – names used later in expand()
SIMS     = SIM_IDS                 # just an alias
WINDOWS  = range(NUM_WINDOWS)      # 0 … 99

##############################################################################
# RULE 0 – “all”: every artefact we need (sim -> VCFs -> LD pickles)
##############################################################################
rule all:
    input:
        # basic simulation artefacts
        expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",            sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/demes.png",sid=SIM_IDS),

        # every VCF window
        expand(f"{LD_ROOT}/sim_{{sid}}/windows/window_{{win}}.vcf.gz",
            sid=SIM_IDS, win=WINDOWS),

        # every LD-statistics pickle
        expand(f"{LD_ROOT}/sim_{{sid}}/LD_stats/LD_stats_window_{{win}}.pkl",
            sid=SIM_IDS, win=WINDOWS),

        # final LD statistics artefacts
        expand(f"{LD_ROOT}/sim_{{sid}}/best_fit.pkl", sid=SIM_IDS)


##############################################################################
# RULE 1 – simulate one full data set (tree sequence + SFS)
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
        model   = cfg["demographic_model"],
    # log:
    #     f"{SIM_BASEDIR}/{{sid}}/simulate.log"
    threads: 1
    shell:
        """
        python "{SIM_SCRIPT}" \
            --simulation-dir      {params.sim_dir} \
            --experiment-config   {params.cfg} \
            --model-type          {params.model} \
            --simulation-number   {wildcards.sid} \
        """

##############################################################################
# RULE 2 – simulate one replicate (window_{win:04d}.vcf.gz)
##############################################################################
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
            --out-dir      {params.out_winDir} \
        """

# #############################################################################
# RULE 3 – compute LD statistics for one window
# #############################################################################
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
            --r-bins       "{params.bins}" \
        """

##############################################################################
# RULE 4 – combine LD windows & run optimisation for one simulation
##############################################################################
rule optimize_momentsld:
    input:
        pkls = lambda w: expand(
            f"{LD_ROOT}/sim_{w.sid}/LD_stats/LD_stats_window_{{win}}.pkl",
            win=WINDOWS),
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

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

SIM_BASEDIR = "experiments/split_isolation/simulations"    # simulations/<sid>
DADI_ROOT    = "dadi"      # dadi root
MOMENTS_ROOT = "moments"   # moments root



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
OPTS = cfg["num_optimizations"]

# NEW – names used later in expand()
SIMS     = SIM_IDS                 # just an alias

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

        expand(
            f"experiments/{cfg['demographic_model']}/runs/run_{{sid}}_{{opt}}/"
            "inferences/moments/fit_params.pkl",
            sid = SIM_IDS,
            opt = OPTS
        ),
        expand(
            f"experiments/{cfg['demographic_model']}/runs/run_{{sid}}_{{opt}}/"
            "inferences/dadi/fit_params.pkl",
            sid = SIM_IDS,
            opt = OPTS
        )


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
# RULE 2 – run ONE moments & dadi optimisation replicate
##############################################################################

rule infer_sfs:
    """Run one optimiser start (moments + dadi) for one simulation."""
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        cfg    = EXP_CFG
    output:
        mom = f"experiments/{cfg['demographic_model']}/runs/"
              f"run_{{sid}}_{{opt}}/inferences/moments/fit_params.pkl",
        dadi = f"experiments/{cfg['demographic_model']}/runs/"
               f"run_{{sid}}_{{opt}}/inferences/dadi/fit_params.pkl"
    params:
        run_dir = lambda w: (
            f"experiments/{cfg['demographic_model']}/runs/"
            f"run_{w.sid}_{w.opt}"
        )
    threads: 4           # moments & dadi are single‑threaded; adjust as you like
    shell:
        """
        python snakemake_scripts/moments_dadi_inference.py \
               --run-dir        "{params.run_dir}" \
               --config-file    "{input.cfg}"      \
               --sfs            "{input.sfs}"      \
               --sampled-params "{input.params}"   \
               --rep-index      {wildcards.opt}    \
               --run-moments    True               \
               --run-dadi       True
        """

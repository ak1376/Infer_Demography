##############################################################################
#  CONFIG – edit only the three paths below if you move files around
##############################################################################

SIM_SCRIPT   = "snakemake_scripts/simulation.py"
INFER_SCRIPT = "snakemake_scripts/moments_dadi_inference.py"
EXP_CFG      = "config_files/experiment_config_bottleneck.json"

##############################################################################
#  Read experiment config ----------------------------------------------------
##############################################################################
import json, math
from pathlib import Path

CFG = json.loads(Path(EXP_CFG).read_text())

MODEL            = CFG["demographic_model"]          # "split_isolation"
NUM_DRAWS        = CFG["num_draws"]                  # 10
NUM_OPTIMISERS   = CFG["num_optimizations"]          # 3
TOP_K            = CFG["top_k"]                      # 2

# simulation IDs 000 … 009
_pad = int(math.log10(NUM_DRAWS - 1)) + 1
SIM_IDS = [f"{i:0{_pad}d}" for i in range(NUM_DRAWS)]

##############################################################################
#  Helper‑path builders ------------------------------------------------------
##############################################################################
SIM_BASEDIR = f"experiments/{MODEL}/simulations"    # simulations/<sid>

def run_dir(sid, opt):
    return f"experiments/{MODEL}/runs/run_{sid}_{opt}"

def opt_pkl(sid, opt, tool):
    return f"{run_dir(sid, opt)}/inferences/{tool}/fit_params.pkl"

def final_pkl(sid, tool):
    return f"experiments/{MODEL}/inferences/sim_{sid}/{tool}/fit_params.pkl"

##############################################################################
#  RULE all – every file we ultimately want
##############################################################################
rule all:
    input:
        # simulation artefacts
        [f"{SIM_BASEDIR}/{sid}/sampled_params.pkl"    for sid in SIM_IDS] +
        [f"{SIM_BASEDIR}/{sid}/SFS.pkl"               for sid in SIM_IDS] +
        [f"{SIM_BASEDIR}/{sid}/tree_sequence.trees"   for sid in SIM_IDS] +
        [f"{SIM_BASEDIR}/{sid}/demes.png"             for sid in SIM_IDS] +
        # combined (best‑k) fit pickles
        [final_pkl(sid, "moments") for sid in SIM_IDS] +
        [final_pkl(sid, "dadi")    for sid in SIM_IDS]

##############################################################################
#  RULE 1 – simulate one full data set (tree‑sequence + SFS)
##############################################################################
rule simulate:
    output:
        sfs   = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params= f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        tree  = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        fig   = f"{SIM_BASEDIR}/{{sid}}/demes.png",
    params:
        sim_dir = SIM_BASEDIR,
        cfg     = EXP_CFG,
        model   = MODEL,
        
    threads: 1
    shell:
        """
        python "{SIM_SCRIPT}" \
            --simulation-dir    {params.sim_dir} \
            --experiment-config {params.cfg}     \
            --model-type        {params.model}   \
            --simulation-number {wildcards.sid}
        """

##############################################################################
#  RULE 2 – one optimiser start (moments + dadi)  → run_<sid>_<opt>
##############################################################################
rule infer_sfs:
    """Run a single optimiser start for one simulation."""
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        cfg    = EXP_CFG
    output:
        mom  = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/moments/fit_params.pkl",
        dadi = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/dadi/fit_params.pkl"
    params:
        run_dir = lambda w: run_dir(w.sid, w.opt)
    threads: 4
    shell:
        """
        python "{INFER_SCRIPT}"                       \
               --run-dir        "{params.run_dir}"    \
               --config-file    "{input.cfg}"         \
               --sfs            "{input.sfs}"         \
               --sampled-params "{input.params}"      \
               --rep-index      {wildcards.opt}       \
               --run-moments    True                  \
               --run-dadi       True
        """

##############################################################################
#  RULE 3 – aggregate NUM_OPTIMISERS runs and keep TOP_K best
##############################################################################
rule aggregate_opts:
    input:
        mom  = lambda w: [opt_pkl(w.sid, o, "moments") for o in range(NUM_OPTIMISERS)],
        dadi = lambda w: [opt_pkl(w.sid, o, "dadi")    for o in range(NUM_OPTIMISERS)]
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
            keep = np.argsort(lls)[::-1][:TOP_K]          # best log‑likelihoods
            best = {"best_params": [params[i] for i in keep],
                    "best_lls"   : [lls[i]    for i in keep]}
            pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(best, open(out_file, "wb"))
            print(f"✓ aggregated → {out_file}")

        merge_keep_best(input.mom,  output.mom)
        merge_keep_best(input.dadi, output.dadi)

##############################################################################
#  Wild‑card constraints – tell Snakemake valid “opt” values
##############################################################################
wildcard_constraints:
    opt = "|".join(str(i) for i in range(NUM_OPTIMISERS))
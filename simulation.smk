##############################################################################
# Simulated data: simulate -> SFS/LD inference -> combine_results
# Usage: snakemake -s simulation.smk -j <N>
##############################################################################
include: "rules/common.smk"

rule all:
    input:
        expand(f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl", sid=SIM_IDS),

include: "rules/simulation_rules.smk"

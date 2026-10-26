##############################################################################
# ML modeling: reads experiments/<model>/inferences/sim_*/all_inferences.pkl
# (produced by simulation.smk) and trains every modeling variant.
# Usage: snakemake -s modeling.smk -j <N>
##############################################################################
include: "rules/common.smk"
include: "rules/modeling_rules.smk"

rule all:
    default_target: True
    input:
        rules.modeling_all.input,

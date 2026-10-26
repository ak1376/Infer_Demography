##############################################################################
# Real data: VCF prep -> SFS/LD inference -> predictions from the models
# trained by modeling.smk (read from experiments/<model>/modeling_<variant>/).
# Usage: snakemake -s real_data.smk -j <N>
##############################################################################
include: "rules/common.smk"
include: "rules/real_data_rules.smk"

rule all:
    default_target: True
    input:
        expand(rules.predict_real_data.output.json, variant=MODELING_VARIANTS, model_key=REAL_MODEL_KEYS),

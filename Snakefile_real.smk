# --- user config ---
REPS   = range(100)
VCF    = "/sietch_colab/akapoor/Infer_Demography/OOA/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.vcf"
POP    = "/sietch_colab/akapoor/Infer_Demography/OOA/1KG.YRI.CEU.popfile.txt"
CFG    = "/sietch_colab/akapoor/Infer_Demography/config_files/experiment_config_split_migration.json"
MODEL  = "src.simulation:split_migration_model"
OUTDIR = "/sietch_colab/akapoor/Infer_Demography/real_data_analysis"
POP_IDS = ["YRI","CEU"]
NS      = [20,24]

rule all:
    input:
        expand(f"{OUTDIR}/runs/rep_{{r}}.pkl", r=REPS),
        expand(f"{OUTDIR}/runs/rep_{{r}}.png", r=REPS)

rule dadi_fit_rep:
    input:
        vcf=VCF,
        popfile=POP,
        cfg=CFG
    output:
        pkl_out=f"{OUTDIR}/runs/rep_{{r}}.pkl",
        png_out=f"{OUTDIR}/runs/rep_{{r}}.png"
    params:
        runsdir=OUTDIR + "/runs",
        model=MODEL,
        pop_ids=" ".join(POP_IDS),
        ns=" ".join(map(str, NS)),
        seed_base=lambda wc: 1337 + int(wc.r),  # unique deterministic seed per job
        perturb_fold=0.5,
        extra="--folded --log10-plot"
    log:
        out=f"{OUTDIR}/runs/rep_{{r}}.out.log",
        err=f"{OUTDIR}/runs/rep_{{r}}.err.log"
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.runsdir}"
        export MPLBACKEND=Agg
        # If needed to import src.simulation from repo root, uncomment:
        # export PYTHONPATH="{params.runsdir}/../..:$PYTHONPATH"

        python real_data_dadi_analysis.py \
          --vcf {input.vcf} \
          --popfile {input.popfile} \
          --pop-ids {params.pop_ids} \
          --ns {params.ns} \
          --config {input.cfg} \
          --model-py {params.model} \
          --n-reps 1 \
          --seed-base {params.seed_base} \
          --perturb-fold {params.perturb_fold} \
          --out-pkl {output.pkl_out} \
          --save-plot {output.png_out} \
          {params.extra} \
          1> {log.out} 2> {log.err}
        """

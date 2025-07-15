#!/bin/bash
#SBATCH --job-name=win-array
#SBATCH --array=0-99                 # one task per window/replicate
#SBATCH --output=logs/win_%A_%a.out
#SBATCH --error=logs/win_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# -------------------------------------------------------------------------
# 1. read run‑wide settings from the same experiment‑config JSON
# -------------------------------------------------------------------------
CONFIG_JSON="/projects/kernlab/akapoor/Infer_Demography/config_files/experiment_config_split_isolation.json"

DEM_MODEL=$(jq -r '.demographic_model'      "$CONFIG_JSON")
SEED     =$(jq -r '.seed'                   "$CONFIG_JSON")
N_DRAWS  =$(jq -r '.num_draws'              "$CONFIG_JSON")   # e.g. 10 sims
WIN_PER_SIM=100                                            # keep in sync!

# -------------------------------------------------------------------------
# 2. decide which simulation ID this array task belongs to
#    (here: split the 0‑99 replicate index into   sim‑id   +   window‑id)
# -------------------------------------------------------------------------
SIM_ID=$(printf "%0${#N_DRAWS}d"  $(( SLURM_ARRAY_TASK_ID / WIN_PER_SIM )))
WIN_ID=$(( SLURM_ARRAY_TASK_ID % WIN_PER_SIM ))

SIM_BASE="ld_experiments/${DEM_MODEL}/simulations/${SIM_ID}"
OUT_ROOT="MomentsLD/LD_stats/sim_${SIM_ID}"

echo ">>>  model=${DEM_MODEL}   sim=${SIM_ID}   window=${WIN_ID}"

# -------------------------------------------------------------------------
# 3. let Snakemake build the exact VCF target for *this* window
#    (Snakemake does all dependency work: simulation → VCF → LD pickle …)
# -------------------------------------------------------------------------

snakemake  \
  --snakefile  /projects/kernlab/akapoor/Infer_Demography/Snakefile \
  --directory  /gpfs/projects/kernlab/akapoor/Infer_Demography \
  --rerun-incomplete \
  --cores     "$SLURM_CPUS_PER_TASK" \
  "${OUT_ROOT}/windows/window_${WIN_ID}.vcf.gz"

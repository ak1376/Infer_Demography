#!/bin/bash
#SBATCH --job-name=ld_optimize
#SBATCH --array=0-9                      # adjust to last simulation index
#SBATCH --output=logs/opt_%A_%a.out
#SBATCH --error=logs/opt_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=akapoor@uoregon.edu

# ----------------------------------------------------------------------
# SINGLE place to point at the config JSON you are optimising for
EXP_CFG="/projects/kernlab/akapoor/Infer_Demography/config_files/experiment_config.json"
# ----------------------------------------------------------------------

# —‑ 1) extract the model name once -------------------------------------
MODEL=$(jq -r '.demographic_model' "${EXP_CFG}")
if [[ -z "${MODEL}" || "${MODEL}" == "null" ]]; then
    echo "Could not read .demographic_model from ${EXP_CFG}" >&2
    exit 1
fi
echo "Demographic model: ${MODEL}"

# —‑ 2) build paths that depend on the model ----------------------------
SIM_BASEDIR="/projects/kernlab/akapoor/Infer_Demography/ld_experiments/${MODEL}/simulations"
LD_ROOT="/projects/kernlab/akapoor/Infer_Demography/MomentsLD/LD_stats"

# constants taken from your Snakefile
NUM_WINDOWS=100
RBINS="0,1e-6,3.2e-6,1e-5,3.2e-5,1e-4,3.2e-4,1e-3"

# zero‑padded simulation ID for this array element
SID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")

SIM_DIR="${SIM_BASEDIR}/${SID}"
LD_DIR="${LD_ROOT}/sim_${SID}"

echo "Optimising simulation ${SID}"
echo "SIM_DIR = ${SIM_DIR}"
echo "LD_DIR  = ${LD_DIR}"
echo "----------------------------------"

python /projects/kernlab/akapoor/Infer_Demography/snakemake_scripts/LD_inference.py \
       --sim-dir      "${SIM_DIR}" \
       --LD_dir       "${LD_DIR}" \
       --config-file  "${EXP_CFG}" \
       --num-windows  "${NUM_WINDOWS}" \
       --r-bins       "${RBINS}"

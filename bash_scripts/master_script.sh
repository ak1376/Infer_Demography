#!/bin/bash
#SBATCH --job-name=main_pipeline
#SBATCH --output=logs/pipeline_main.out
#SBATCH --error=logs/pipeline_main.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

set -euo pipefail
mkdir -p logs

##############################################################################
# 1. single place to define the config file for **all** stages
##############################################################################
CFG_PATH="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_drosophila_three_epoch.json"
export CFG_PATH        # every sbatch inherits this

##############################################################################
# 2. unlock Snakemake repo‑wide (harmless if no lock)
##############################################################################
snakemake --unlock || true

##############################################################################
# 3. helper: submit job & return its ID
##############################################################################
submit() {
    sbatch --parsable --export=ALL "$@"
}

##############################################################################
# 4. submit stage scripts with afterok dependencies
##############################################################################
echo "Using config: $CFG_PATH"
echo "Submitting pipeline …"

if ! squeue -u "$USER" -h -o "%j" | grep -q snakemake; then
    snakemake --directory /projects/kernlab/akapoor/Infer_Demography --unlock || true
fi

# 4.1 run simulations (first stage)
sim_id=$(submit bash_scripts/running_simulation.sh)
echo "running_simulation.sh  → $sim_id"

# 4.2 windows generation depends on simulations
win_id=$(submit --dependency=afterok:$sim_id bash_scripts/simulate_windows.sh)
echo "simulate_windows.sh    → $win_id"

# 4.3 LD‑stats windows depends on windows
ld_id=$(submit --dependency=afterok:$win_id bash_scripts/LD_stats_windows.sh)
echo "LD_stats_windows.sh    → $ld_id"

# 4.4 MomentsLD optimisation depends on LD‑stats
momLD_id=$(submit --dependency=afterok:$ld_id bash_scripts/MomentsLD.sh)
echo "MomentsLD.sh           → $momLD_id"

# 4.5 moments & dadi each depend only on simulations
mom_id=$(submit --dependency=afterok:$sim_id bash_scripts/moments.sh)
dadi_id=$(submit --dependency=afterok:$sim_id bash_scripts/dadi.sh)
echo "moments.sh             → $mom_id"
echo "dadi.sh                → $dadi_id"

# 4.6 aggregate moments+dadi after both finish
agg_id=$(submit --dependency=afterok:$mom_id:$dadi_id bash_scripts/aggregate_moments_dadi.sh)
echo "aggregate_moments_dadi → $agg_id"

# 4.7 final combine after MomentsLD **and** aggregate‑moments‑dadi
comb_id=$(submit --dependency=afterok:$momLD_id:$agg_id bash_scripts/run_combine.sh)
echo "run_combine.sh         → $comb_id"

# 4.8 build feature / target matrices after all‑inferences are ready
feat_id=$(submit --dependency=afterok:$comb_id bash_scripts/aggregate_features.sh)
echo "aggregate_features.sh  → $feat_id"

echo "Pipeline submitted. Final job ID (aggregate_features): $feat_id"

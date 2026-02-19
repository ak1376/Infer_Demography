#!/bin/bash
#SBATCH --job-name=main_pipeline
#SBATCH --output=logs/pipeline_main.out
#SBATCH --error=logs/pipeline_main.err
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

REPO="/projects/kernlab/akapoor/Infer_Demography"
mkdir -p "$REPO/logs"
cd "$REPO"

CFG_PATH="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_split_migration_growth.json"
export CFG_PATH

snakemake --directory "$REPO" --unlock || true

submit() { sbatch --parsable --export=ALL "$@"; }
dep_afterany() { echo "--dependency=afterany:$1"; }

RUN_MOMENTS_DADI_MODE=""

echo "Using config: $CFG_PATH"
echo "Submitting pipeline from: $PWD"

sim_id=$(submit bash_scripts/running_simulation.sh); [[ -n "$sim_id" ]]

win_id=$(submit $(dep_afterany "$sim_id") bash_scripts/simulate_windows.sh); [[ -n "$win_id" ]]
ld_id=$(submit  $(dep_afterany "$win_id") bash_scripts/LD_stats_windows.sh); [[ -n "$ld_id" ]]
momLD_id=$(submit $(dep_afterany "$ld_id") bash_scripts/MomentsLD.sh); [[ -n "$momLD_id" ]]

if [[ "$RUN_MOMENTS_DADI_MODE" == "parallel" ]]; then
  mom_id=$(submit $(dep_afterany "$ld_id") bash_scripts/moments.sh); [[ -n "$mom_id" ]]
  dadi_id=$(submit $(dep_afterany "$ld_id") bash_scripts/dadi.sh); [[ -n "$dadi_id" ]]
else
  mom_id=$(submit $(dep_afterany "$ld_id") bash_scripts/moments.sh); [[ -n "$mom_id" ]]
  dadi_id=$(submit $(dep_afterany "$mom_id") bash_scripts/dadi.sh); [[ -n "$dadi_id" ]]
fi

agg_id=$(submit --dependency=afterany:$mom_id:$dadi_id bash_scripts/aggregate_moments_dadi.sh); [[ -n "$agg_id" ]]
comb_id=$(submit --dependency=afterany:$momLD_id:$agg_id bash_scripts/run_combine.sh); [[ -n "$comb_id" ]]
feat_id=$(submit --dependency=afterany:$comb_id bash_scripts/aggregate_features.sh); [[ -n "$feat_id" ]]

echo "Final job ID (aggregate_features): $feat_id"

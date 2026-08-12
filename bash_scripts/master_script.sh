#!/bin/bash
#SBATCH --job-name=main_pipeline
#SBATCH --output=logs/pipeline_main.out
#SBATCH --error=logs/pipeline_main.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

REPO="${REPO:-/projects/kernlab/akapoor/Infer_Demography}"
mkdir -p "$REPO/logs"
cd "$REPO"

source "$REPO/bash_scripts/lib_active_config.sh"
source "$REPO/bash_scripts/lib_array_size.sh"
CFG_PATH="$(resolve_cfg_path "$REPO")"
export CFG_PATH

snakemake --directory "$REPO" --unlock || true

# Sizes needed to submit each array stage with its correct --array range
# directly, instead of letting it self-resubmit. Every one of these stage
# scripts contains "if no SLURM_ARRAY_TASK_ID: sbatch --array=... \"$0\"; exit 0"
# for standalone convenience — but that means a job submitted bare (as
# submit() below used to do) exits in seconds after spawning the REAL array
# job, so --dependency=afterany on ITS id was satisfied almost immediately,
# not when the real work finished. Passing --array explicitly here makes the
# id we capture the real array job's id, so afterany actually waits.
NUM_DRAWS=$(jq -r '.num_draws' "$CFG_PATH")
NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG_PATH")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG_PATH")

submit() { sbatch --parsable --export=ALL "$@"; }
submit_array() {
  local spec="$1"; shift
  sbatch --parsable --export=ALL --array="$spec" "$@"
}
dep_afterany() { echo "--dependency=afterany:$1"; }

RUN_MOMENTS_DADI_MODE=""

echo "Using config: $CFG_PATH"
echo "Submitting pipeline from: $PWD"

# --- 1. simulate ---
export BATCH_SIZE=1
sim_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  bash_scripts/running_simulation.sh); [[ -n "$sim_id" ]]

# --- 2. build LD windows ---
export BATCH_SIZE=50
win_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_WINDOWS ))" "$BATCH_SIZE" 200)" \
  $(dep_afterany "$sim_id") bash_scripts/build_windows.sh); [[ -n "$win_id" ]]

# --- 3. LD stats ---
export BATCH_SIZE=50
ld_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_WINDOWS ))" "$BATCH_SIZE")" \
  $(dep_afterany "$win_id") bash_scripts/LD_stats_windows.sh); [[ -n "$ld_id" ]]

# --- 4. Moments-LD optimization ---
export BATCH_SIZE=1
momLD_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  $(dep_afterany "$ld_id") bash_scripts/MomentsLD.sh); [[ -n "$momLD_id" ]]

# --- 5/6. moments + dadi SFS inference ---
export BATCH_SIZE=50
if [[ "$RUN_MOMENTS_DADI_MODE" == "parallel" ]]; then
  mom_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_OPTIMS ))" "$BATCH_SIZE" 100)" \
    $(dep_afterany "$ld_id") bash_scripts/moments.sh); [[ -n "$mom_id" ]]
  dadi_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_OPTIMS ))" "$BATCH_SIZE")" \
    $(dep_afterany "$ld_id") bash_scripts/dadi.sh); [[ -n "$dadi_id" ]]
else
  mom_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_OPTIMS ))" "$BATCH_SIZE" 100)" \
    $(dep_afterany "$ld_id") bash_scripts/moments.sh); [[ -n "$mom_id" ]]
  dadi_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_OPTIMS ))" "$BATCH_SIZE")" \
    $(dep_afterany "$mom_id") bash_scripts/dadi.sh); [[ -n "$dadi_id" ]]
fi

# --- 7. aggregate moments+dadi top-K, cleanup ---
export BATCH_SIZE=1
agg_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  --dependency=afterany:$mom_id:$dadi_id bash_scripts/aggregate_moments_dadi.sh); [[ -n "$agg_id" ]]

# --- 8. combine_results ---
export BATCH_SIZE=1
comb_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  --dependency=afterany:$momLD_id:$agg_id bash_scripts/run_combine.sh); [[ -n "$comb_id" ]]

# --- 9. build modeling dataset (single job, no array) ---
feat_id=$(submit --dependency=afterany:$comb_id bash_scripts/aggregate_features.sh); [[ -n "$feat_id" ]]

echo "Final job ID (aggregate_features): $feat_id"

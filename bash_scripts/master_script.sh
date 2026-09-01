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
USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG_PATH")

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

# --- 2. materialize per-sim whole-genome VCFs (window_mode="chunked" only;
#         materialize_windows.sh no-ops immediately otherwise) ---
export BATCH_SIZE=1
mat_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  $(dep_afterany "$sim_id") bash_scripts/materialize_windows.sh); [[ -n "$mat_id" ]]

# --- 3. build LD windows ---
export BATCH_SIZE=50
# Overridable: tune concurrency to your account's actual QOS/TRES headroom
# (check with `sacctmgr show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU`
# for the QOS your kern/preempt/kerngpu partitions resolve to) rather than
# editing this literal each time.
MAX_CONCURRENT_WINDOWS="${MAX_CONCURRENT_WINDOWS:-1500}"
win_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_WINDOWS ))" "$BATCH_SIZE" "$MAX_CONCURRENT_WINDOWS")" \
  $(dep_afterany "$mat_id") bash_scripts/build_windows.sh); [[ -n "$win_id" ]]

# --- 4. LD stats ---
# submit_array passes --array explicitly, so LD_stats_windows.sh's own
# dispatcher block (which normally picks partition/gres from use_gpu_ld
# when SLURM_ARRAY_TASK_ID is unset) never runs here -- SLURM_ARRAY_TASK_ID
# is already set on this very first submission. Pick the partition/gres
# here instead, mirroring that block, or every run silently falls back to
# the file's #SBATCH defaults (GPU partitions) regardless of use_gpu_ld.
export BATCH_SIZE=50
if [[ "$USE_GPU_LD" == "true" ]]; then
  LD_STATS_SBATCH_OPTS=(--partition=kerngpu,gpulong,gpu --gres=gpu:1)
else
  LD_STATS_SBATCH_OPTS=(--partition=kern,preempt --gres=gpu:0)
fi
ld_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_WINDOWS ))" "$BATCH_SIZE")" \
  $(dep_afterany "$win_id") "${LD_STATS_SBATCH_OPTS[@]}" bash_scripts/LD_stats_windows.sh); [[ -n "$ld_id" ]]

# --- 5. Moments-LD optimization ---
# Split into prep -> restarts -> aggregate so the num_optimizations restarts
# per sim actually run in parallel across the array (NUM_DRAWS * NUM_OPTIMS)
# instead of serially inside one per-sim task.
export BATCH_SIZE=1
momLD_prep_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  $(dep_afterany "$ld_id") bash_scripts/MomentsLD_prep.sh); [[ -n "$momLD_prep_id" ]]

export BATCH_SIZE=50
momLD_id=$(submit_array "$(array_spec "$(( NUM_DRAWS * NUM_OPTIMS ))" "$BATCH_SIZE" 5000)" \
  $(dep_afterany "$momLD_prep_id") bash_scripts/MomentsLD.sh); [[ -n "$momLD_id" ]]

export BATCH_SIZE=1
momLD_agg_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  $(dep_afterany "$momLD_id") bash_scripts/aggregate_momentsld.sh); [[ -n "$momLD_agg_id" ]]

# --- 6/7. moments + dadi SFS inference ---
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

# --- 8. aggregate moments+dadi top-K, cleanup ---
# Must also wait on momLD_agg_id: cleanup_optimization_runs deletes any
# run_{sid}_{opt} dir not in the union of dadi/moments/MomentsLD top-K, so
# it can't run until MomentsLD has already picked its own keep-set.
export BATCH_SIZE=1
agg_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  --dependency=afterany:$mom_id:$dadi_id:$momLD_agg_id bash_scripts/aggregate_moments_dadi.sh); [[ -n "$agg_id" ]]

# --- 8b/8c. FIM + SFS-residuals (always computed, regardless of
#            use_fim_features/use_residuals -- those flags only control
#            whether feature_extraction.py later uses them as features) ---
export BATCH_SIZE=1
fim_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  $(dep_afterany "$agg_id") bash_scripts/compute_fim.sh); [[ -n "$fim_id" ]]
resid_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  $(dep_afterany "$agg_id") bash_scripts/compute_residuals.sh); [[ -n "$resid_id" ]]

# --- 9. combine_results ---
export BATCH_SIZE=1
comb_id=$(submit_array "$(array_spec "$NUM_DRAWS" "$BATCH_SIZE")" \
  --dependency=afterany:$momLD_agg_id:$agg_id:$fim_id:$resid_id bash_scripts/run_combine.sh); [[ -n "$comb_id" ]]

# --- 10. build modeling dataset (single job, no array) ---
feat_id=$(submit --dependency=afterany:$comb_id bash_scripts/aggregate_features.sh); [[ -n "$feat_id" ]]

echo "Final job ID (aggregate_features): $feat_id"

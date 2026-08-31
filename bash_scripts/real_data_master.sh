#!/bin/bash
#SBATCH --job-name=real_data_pipeline
#SBATCH --output=logs/real_data_pipeline.out
#SBATCH --error=logs/real_data_pipeline.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# Orchestrates the full real-data (Drosophila) analysis, mirroring
# master_script.sh's stage-chaining style for the simulated pipeline --
# every multi-restart stage (SFS inference, MomentsLD) is split into an
# array job that runs each restart in its own SLURM task, followed by a
# separate small aggregate job, exactly like moments.sh/dadi.sh +
# aggregate_moments_dadi.sh and MomentsLD_prep.sh/MomentsLD.sh +
# aggregate_momentsld.sh do for the simulated pipeline:
#
#   A.  real_data_prep.sh          -- polarize VCFs, per-chrom + combined SFS
#   B1. real_sfs_inference.sh      -- moments/dadi restarts, ARRAY (engine x opt)
#   B2. real_aggregate_sfs.sh      -- top-k moments/dadi best_fit
#   C.  real_ld_windows.sh         -- per-window LD stats (Chr3L), ARRAY
#   D1. real_momentsld_prep.sh     -- aggregate LD windows -> means.varcovs.pkl
#   D2. real_momentsld.sh          -- MomentsLD restarts, ARRAY (opt)
#   D3. real_aggregate_momentsld.sh -- top-k MomentsLD best_fit
#   E.  real_fim_residuals.sh      -- FIM + SFS residuals at best-fit params
#   F.  real_combine_predict.sh    -- all_inferences.pkl + push through trained models
#
# B1 and C only depend on A, so they run in parallel. D1 only needs C.
# D2 needs D1 (means.varcovs.pkl) AND B2 (moments best_fit as an
# optimization seed). E only needs B2. F needs B2, D3, and E.

set -euo pipefail

REPO="${REPO:-/projects/kernlab/akapoor/Infer_Demography}"
mkdir -p "$REPO/logs"
cd "$REPO"

source "$REPO/bash_scripts/lib_active_config.sh"
source "$REPO/bash_scripts/lib_array_size.sh"
CFG_PATH="$(resolve_cfg_path "$REPO")"
export CFG_PATH

snakemake --directory "$REPO" --unlock || true

NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG_PATH")
NUM_REAL_OPTIMS=$(jq -r '.num_optimizations // 3' "$CFG_PATH")
USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG_PATH")

submit() { sbatch --parsable --export=ALL "$@"; }
submit_array() {
  local spec="$1"; shift
  sbatch --parsable --export=ALL --array="$spec" "$@"
}
dep_afterany() { echo "--dependency=afterany:$1"; }

echo "Using config: $CFG_PATH"
echo "Submitting real-data pipeline from: $PWD"

# --- A. prep: polarize VCFs, per-chrom + combined SFS ---
prep_id=$(submit bash_scripts/real_data_prep.sh); [[ -n "$prep_id" ]]

# --- B1/B2. moments/dadi SFS inference (combined autosomes) ---
export BATCH_SIZE=10
sfs_id=$(submit_array "$(array_spec "$(( 2 * NUM_REAL_OPTIMS ))" "$BATCH_SIZE")" \
  $(dep_afterany "$prep_id") bash_scripts/real_sfs_inference.sh); [[ -n "$sfs_id" ]]
sfs_agg_id=$(submit $(dep_afterany "$sfs_id") bash_scripts/real_aggregate_sfs.sh); [[ -n "$sfs_agg_id" ]]

# --- C. LD windows (Chr3L), array job ---
export BATCH_SIZE=20
if [[ "$USE_GPU_LD" == "true" ]]; then
  LD_SBATCH_OPTS=(--partition=kerngpu,gpulong,gpu --gres=gpu:1)
else
  LD_SBATCH_OPTS=(--partition=kern,preempt --gres=gpu:0)
fi
ld_id=$(submit_array "$(array_spec "$NUM_WINDOWS" "$BATCH_SIZE")" \
  $(dep_afterany "$prep_id") "${LD_SBATCH_OPTS[@]}" bash_scripts/real_ld_windows.sh); [[ -n "$ld_id" ]]

# --- D1/D2/D3. aggregate LD windows -> MomentsLD restarts -> top-k ---
momld_prep_id=$(submit $(dep_afterany "$ld_id") bash_scripts/real_momentsld_prep.sh); [[ -n "$momld_prep_id" ]]

export BATCH_SIZE=10
momld_id=$(submit_array "$(array_spec "$NUM_REAL_OPTIMS" "$BATCH_SIZE")" \
  --dependency=afterany:$momld_prep_id:$sfs_agg_id bash_scripts/real_momentsld.sh); [[ -n "$momld_id" ]]

momld_agg_id=$(submit $(dep_afterany "$momld_id") bash_scripts/real_aggregate_momentsld.sh); [[ -n "$momld_agg_id" ]]

# --- E. FIM + SFS residuals (needs B2's moments/dadi best_fit) ---
fim_resid_id=$(submit $(dep_afterany "$sfs_agg_id") bash_scripts/real_fim_residuals.sh); [[ -n "$fim_resid_id" ]]

# --- F. combine + predict (needs B2, D3, E) ---
final_id=$(submit --dependency=afterany:$sfs_agg_id:$momld_agg_id:$fim_resid_id bash_scripts/real_combine_predict.sh); [[ -n "$final_id" ]]

echo "Final job ID (real_combine_predict): $final_id"

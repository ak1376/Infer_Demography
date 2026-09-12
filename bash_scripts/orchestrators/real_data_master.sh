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
#   A.  real_data_prep.sh          -- polarize VCFs, per-arm (+ combined, if
#                                      pooled) SFS
#   B1. real_sfs_inference.sh      -- moments/dadi restarts, ARRAY
#                                      (pooled: engine x opt;
#                                       individual: arm x engine x opt)
#   B2. real_aggregate_sfs.sh      -- top-k moments/dadi best_fit
#                                      (pooled: one; individual: per arm)
#   C.  real_ld_windows.sh         -- per-window LD stats, per arm, ARRAY
#                                      (same either way -- arm-count-agnostic)
#   D1. real_momentsld_prep.sh     -- aggregate LD windows -> means.varcovs.pkl
#                                      (pooled: one combined; individual: per arm)
#   D2. real_momentsld.sh          -- MomentsLD restarts, ARRAY
#                                      (pooled: opt; individual: arm x opt)
#   D3. real_aggregate_momentsld.sh -- top-k MomentsLD best_fit
#                                      (pooled: one; individual: per arm)
#   E.  real_fim_residuals.sh      -- FIM + SFS residuals at best-fit params
#                                      (pooled fit only, regardless of mode)
#   F.  real_combine_predict.sh    -- all_inferences.pkl + push through trained models
#                                      (pooled fit only, regardless of mode)
#
# B1 and C only depend on A, so they run in parallel. D1 only needs C.
# D2 needs D1 (means.varcovs.pkl) AND B2 (moments best_fit as an
# optimization seed). E and F always need the POOLED moments/dadi fit (B2),
# so in pooling_mode="individual" -- where B2 only builds per-arm fits, not
# the pooled one -- stages E/F are skipped entirely rather than run against
# missing inputs; there is no per-arm FIM/residuals/prediction path yet.

set -euo pipefail

REPO="${REPO:-/projects/kernlab/akapoor/Infer_Demography}"
mkdir -p "$REPO/logs"
cd "$REPO"

source "$REPO/bash_scripts/lib/lib_active_config.sh"
source "$REPO/bash_scripts/lib/lib_real_data_config.sh"
source "$REPO/bash_scripts/lib/lib_array_size.sh"
CFG_PATH="$(resolve_cfg_path "$REPO")"
export CFG_PATH

snakemake --directory "$REPO" --unlock || true

NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG_PATH")
USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG_PATH")

load_real_data_config "$CFG_PATH"
NUM_REAL_OPTIMS=$(jq -r '.num_optimizations // 3' "$CFG_PATH")
N_ARMS=${#REAL_ARMS[@]}

echo "REAL_POOLING_MODE=$REAL_POOLING_MODE  REAL_ARMS=${REAL_ARMS[*]}"

submit() { sbatch --parsable --export=ALL "$@"; }
submit_array() {
  local spec="$1"; shift
  sbatch --parsable --export=ALL --array="$spec" "$@"
}
dep_afterany() { echo "--dependency=afterany:$1"; }

echo "Using config: $CFG_PATH"
echo "Submitting real-data pipeline from: $PWD"

# --- A. prep: polarize VCFs, per-arm (+ combined, if pooled) SFS ---
prep_id=$(submit bash_scripts/real_data/real_data_prep.sh); [[ -n "$prep_id" ]]

# --- B1/B2. moments/dadi SFS inference ---
# pooled: fit against the combined-autosome SFS (2*NUM_REAL_OPTIMS tasks).
# individual: fit each arm's own SFS independently, no pooling
# (2*NUM_REAL_OPTIMS*N_ARMS tasks).
if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
  SFS_TOTAL_TASKS=$(( 2 * NUM_REAL_OPTIMS ))
else
  SFS_TOTAL_TASKS=$(( 2 * NUM_REAL_OPTIMS * N_ARMS ))
fi
export BATCH_SIZE=10
sfs_id=$(submit_array "$(array_spec "$SFS_TOTAL_TASKS" "$BATCH_SIZE")" \
  $(dep_afterany "$prep_id") bash_scripts/real_data/real_sfs_inference.sh); [[ -n "$sfs_id" ]]
sfs_agg_id=$(submit $(dep_afterany "$sfs_id") bash_scripts/real_data/real_aggregate_sfs.sh); [[ -n "$sfs_agg_id" ]]

# --- C. LD windows, per arm, array job (same either way) ---
# --- D1. aggregate LD windows -> means.varcovs.pkl ---
# pooled: one combined means.varcovs.pkl across every configured arm.
# individual: one means.varcovs.pkl PER ARM, no cross-arm combining.
# If already built for the current arm-set/genmap/pooling-mode, LD windows +
# aggregation are already done (Snakemake's own rerun-triggers=mtime would
# confirm this fine-grained, but it can't be consulted here before
# submission) -- skip C/D1 entirely rather than resubmit a job that has
# nothing left to do.
ld_id=""
momld_prep_id=""

d1_done() {
  if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
    [[ -s "${REAL_LD_ROOT}/means.varcovs.pkl" ]]
  else
    local arm
    for arm in "${REAL_ARMS[@]}"; do
      [[ -s "${REAL_LD_ROOT}/${arm}/means.varcovs.pkl" ]] || return 1
    done
    return 0
  fi
}

if d1_done; then
  echo "means.varcovs.pkl already exists under ${REAL_LD_ROOT} (mode=$REAL_POOLING_MODE) -- skipping stage C/D1 (LD windows + aggregation)."
else
  export BATCH_SIZE=20
  if [[ "$USE_GPU_LD" == "true" ]]; then
    LD_SBATCH_OPTS=(--partition=kerngpu,gpulong,gpu --gres=gpu:1)
  else
    LD_SBATCH_OPTS=(--partition=kern,preempt --gres=gpu:0)
  fi
  if [[ "$NUM_WINDOWS" =~ ^[0-9]+$ ]]; then
    # Literal window count: real_ld_windows.sh's own per-arm window count
    # will always equal this value regardless of VCF span (see
    # split_vcf_windows.py), so it's safe to pre-size the array here.
    ld_id=$(submit_array "$(array_spec "$NUM_WINDOWS" "$BATCH_SIZE")" \
      $(dep_afterany "$prep_id") "${LD_SBATCH_OPTS[@]}" bash_scripts/real_data/real_ld_windows.sh); [[ -n "$ld_id" ]]
  else
    # num_windows: "auto" -- the real per-arm window count depends on the
    # VCF's actual span, which isn't known until stage A (real_data_prep.sh)
    # has actually run, so it can't be pre-sized here before submission.
    # Fall back to a plain (non-array) submission and let
    # real_ld_windows.sh's own self-dispatch logic size and resubmit the
    # real array job once the VCF exists. NOTE: the job ID captured below is
    # that quick dispatcher, not the real array job, so
    # --dependency=afterany:$ld_id for stage D1 may fire before LD windows
    # are actually done in this mode -- rerun D1 by hand
    # (bash_scripts/real_data/real_momentsld_prep.sh) if it fails/finishes early.
    ld_id=$(submit $(dep_afterany "$prep_id") "${LD_SBATCH_OPTS[@]}" bash_scripts/real_data/real_ld_windows.sh); [[ -n "$ld_id" ]]
  fi
  momld_prep_id=$(submit $(dep_afterany "$ld_id") bash_scripts/real_data/real_momentsld_prep.sh); [[ -n "$momld_prep_id" ]]
fi

# --- D2/D3. MomentsLD restarts -> top-k ---
# pooled: NUM_REAL_OPTIMS tasks, one combined fit.
# individual: NUM_REAL_OPTIMS*N_ARMS tasks, one independent fit per arm.
# Depend on sfs_agg_id always (moments best_fit is the optimization seed);
# only add momld_prep_id to the dependency list if it was actually submitted.
if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
  MOMLD_TOTAL_TASKS=$NUM_REAL_OPTIMS
else
  MOMLD_TOTAL_TASKS=$(( NUM_REAL_OPTIMS * N_ARMS ))
fi

momld_deps="afterany:$sfs_agg_id"
[[ -n "$momld_prep_id" ]] && momld_deps="afterany:$momld_prep_id:$sfs_agg_id"

export BATCH_SIZE=10
momld_id=$(submit_array "$(array_spec "$MOMLD_TOTAL_TASKS" "$BATCH_SIZE")" \
  --dependency=$momld_deps bash_scripts/real_data/real_momentsld.sh); [[ -n "$momld_id" ]]

momld_agg_id=$(submit $(dep_afterany "$momld_id") bash_scripts/real_data/real_aggregate_momentsld.sh); [[ -n "$momld_agg_id" ]]

# --- E/F. FIM + SFS residuals, then combine + predict ---
# Both stages only work against the POOLED moments/dadi fit -- there is no
# per-arm FIM/residuals/prediction path yet. In pooling_mode="individual",
# stage B2 only builds per-arm fits (not the pooled one), so E/F would run
# against a missing input; skip them entirely rather than fail or silently
# build the pooled fit as a side effect.
if [[ "$REAL_POOLING_MODE" == "pooled" ]]; then
  fim_resid_id=$(submit $(dep_afterany "$sfs_agg_id") bash_scripts/real_data/real_fim_residuals.sh); [[ -n "$fim_resid_id" ]]

  final_id=$(submit --dependency=afterany:$sfs_agg_id:$momld_agg_id:$fim_resid_id bash_scripts/real_data/real_combine_predict.sh); [[ -n "$final_id" ]]

  echo "Final job ID (real_combine_predict): $final_id"
else
  echo "pooling_mode=individual -- skipping stages E (FIM/residuals) and F (combine/predict);" \
       "no per-arm path for those yet. Last stage submitted: real_aggregate_momentsld ($momld_agg_id)."
fi

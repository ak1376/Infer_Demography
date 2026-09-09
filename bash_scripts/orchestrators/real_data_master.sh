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

source "$REPO/bash_scripts/lib/lib_active_config.sh"
source "$REPO/bash_scripts/lib/lib_array_size.sh"
CFG_PATH="$(resolve_cfg_path "$REPO")"
export CFG_PATH

snakemake --directory "$REPO" --unlock || true

NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG_PATH")
NUM_REAL_OPTIMS=$(jq -r '.num_optimizations // 3' "$CFG_PATH")
USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG_PATH")

# Must match the Snakefile's DROSO_DIR / REAL_TAG / REAL_LD_ENGINE / REAL_LD_ROOT
# (and bash_scripts/real_data/real_ld_windows.sh's copy of the same logic).
DROSO_DIR="real_data_analysis/data/drosophila"
readarray -t REAL_ARMS < <(jq -r '.real_data_analysis.arms // ["Chr3L"] | .[]' "$CFG_PATH")
REAL_USE_GENMAP=$(jq -r '.real_data_analysis.use_genmap // false' "$CFG_PATH")
REAL_TAG=$(IFS=_; echo "${REAL_ARMS[*]}")
[[ "$REAL_USE_GENMAP" == "true" ]] && REAL_TAG="${REAL_TAG}_genmap"
REAL_LD_ROOT="${DROSO_DIR}/MomentsLD_${REAL_TAG}"

submit() { sbatch --parsable --export=ALL "$@"; }
submit_array() {
  local spec="$1"; shift
  sbatch --parsable --export=ALL --array="$spec" "$@"
}
dep_afterany() { echo "--dependency=afterany:$1"; }

echo "Using config: $CFG_PATH"
echo "Submitting real-data pipeline from: $PWD"

# --- A. prep: polarize VCFs, per-chrom + combined SFS ---
prep_id=$(submit bash_scripts/real_data/real_data_prep.sh); [[ -n "$prep_id" ]]

# --- B1/B2. moments/dadi SFS inference (combined autosomes) ---
export BATCH_SIZE=10
sfs_id=$(submit_array "$(array_spec "$(( 2 * NUM_REAL_OPTIMS ))" "$BATCH_SIZE")" \
  $(dep_afterany "$prep_id") bash_scripts/real_data/real_sfs_inference.sh); [[ -n "$sfs_id" ]]
sfs_agg_id=$(submit $(dep_afterany "$sfs_id") bash_scripts/real_data/real_aggregate_sfs.sh); [[ -n "$sfs_agg_id" ]]

# --- C. LD windows (Chr3L), array job ---
# --- D1. aggregate LD windows -> means.varcovs.pkl ---
# If the aggregated means.varcovs.pkl for the current arm-set/genmap tag
# already exists, LD windows + aggregation are already done (Snakemake's own
# rerun-triggers=mtime would confirm this fine-grained, but it can't be
# consulted here before submission) -- skip C/D1 entirely rather than
# resubmit a job that has nothing left to do.
ld_id=""
momld_prep_id=""

if [[ -s "${REAL_LD_ROOT}/means.varcovs.pkl" ]]; then
  echo "means.varcovs.pkl already exists under ${REAL_LD_ROOT} -- skipping stage C/D1 (LD windows + aggregation)."
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
# Depend on sfs_agg_id always (moments best_fit is the optimization seed);
# only add momld_prep_id to the dependency list if it was actually submitted.
momld_deps="afterany:$sfs_agg_id"
[[ -n "$momld_prep_id" ]] && momld_deps="afterany:$momld_prep_id:$sfs_agg_id"

export BATCH_SIZE=10
momld_id=$(submit_array "$(array_spec "$NUM_REAL_OPTIMS" "$BATCH_SIZE")" \
  --dependency=$momld_deps bash_scripts/real_data/real_momentsld.sh); [[ -n "$momld_id" ]]

momld_agg_id=$(submit $(dep_afterany "$momld_id") bash_scripts/real_data/real_aggregate_momentsld.sh); [[ -n "$momld_agg_id" ]]

# --- E. FIM + SFS residuals (needs B2's moments/dadi best_fit) ---
fim_resid_id=$(submit $(dep_afterany "$sfs_agg_id") bash_scripts/real_data/real_fim_residuals.sh); [[ -n "$fim_resid_id" ]]

# --- F. combine + predict (needs B2, D3, E) ---
final_id=$(submit --dependency=afterany:$sfs_agg_id:$momld_agg_id:$fim_resid_id bash_scripts/real_data/real_combine_predict.sh); [[ -n "$final_id" ]]

echo "Final job ID (real_combine_predict): $final_id"

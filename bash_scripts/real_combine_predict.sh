#!/bin/bash
#SBATCH --job-name=real_combine_predict
#SBATCH --output=logs/real_combine_predict_%j.out
#SBATCH --error=logs/real_combine_predict_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage F (final): merge moments/dadi/MomentsLD fits + FIM + residuals into
# all_inferences.pkl, build the real feature row, and push it through every
# trained model found under real_predict_modeling_dir. Requires stages
# A-E (real_data_prep, real_sfs_inference, real_ld_windows, real_momentsld,
# real_fim_residuals) to have already finished.
#
# Rules run: combine_results_real, build_real_prediction_dataset,
#            predict_real_data
#
# predict_real_data needs a trained model (from the sim modeling pipeline)
# for each model_key -- this script silently skips any model_key whose
# trained *_mdl_obj.pkl doesn't exist yet rather than failing the whole job.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")
REAL_PREDICT_MODELING_DIR=$(jq -r '.real_predict_modeling_dir // empty' "$CFG")
if [[ -z "$REAL_PREDICT_MODELING_DIR" ]]; then
    REAL_PREDICT_MODELING_DIR="experiments/${MODEL}/modeling_wo_FIM_wo_SFSresids"
fi

REAL_INF_ROOT="experiments/${MODEL}/real_data_analysis/inferences"
REAL_PRED_ROOT="experiments/${MODEL}/real_data_analysis/prediction"

echo "MODEL=$MODEL"
echo "Combining results -> ${REAL_INF_ROOT}/all_inferences.pkl"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules combine_results_real \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "${REAL_INF_ROOT}/all_inferences.pkl"

echo "Building real prediction dataset -> ${REAL_PRED_ROOT}/real_features_df.pkl"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules build_real_prediction_dataset \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "${REAL_PRED_ROOT}/real_features_df.pkl"

# Must match the Snakefile's REAL_MODEL_OBJS dict keys/paths.
declare -A REAL_MODEL_OBJS=(
    [random_forest]="${REAL_PREDICT_MODELING_DIR}/random_forest/random_forest_mdl_obj.pkl"
    [xgboost]="${REAL_PREDICT_MODELING_DIR}/xgboost/xgb_mdl_obj.pkl"
    [linear_standard]="${REAL_PREDICT_MODELING_DIR}/linear_standard/linear_mdl_obj_standard.pkl"
    [linear_ridge]="${REAL_PREDICT_MODELING_DIR}/linear_ridge/linear_mdl_obj_ridge.pkl"
    [linear_lasso]="${REAL_PREDICT_MODELING_DIR}/linear_lasso/linear_mdl_obj_lasso.pkl"
    [linear_elasticnet]="${REAL_PREDICT_MODELING_DIR}/linear_elasticnet/linear_mdl_obj_elasticnet.pkl"
)

for model_key in "${!REAL_MODEL_OBJS[@]}"; do
    model_pkl="${REAL_MODEL_OBJS[$model_key]}"
    if [[ ! -f "$ROOT/$model_pkl" ]]; then
        echo "SKIP: $model_key (no trained model at $model_pkl)"
        continue
    fi
    echo "Predicting with $model_key -> ${REAL_PRED_ROOT}/predictions_${model_key}.json"
    snakemake \
        --snakefile "$SNAKEFILE" \
        --directory "$ROOT" \
        --nolock \
        --keep-going \
        --rerun-incomplete \
        --rerun-triggers mtime \
        --allowed-rules predict_real_data \
        -j "${SLURM_CPUS_PER_TASK:-2}" \
        "${REAL_PRED_ROOT}/predictions_${model_key}.json"
done

echo "real_combine_predict finished."

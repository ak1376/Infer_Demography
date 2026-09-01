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
# all_inferences.pkl, then build a real feature row and push it through every
# trained model, once per modeling {variant} (w/wo FIM x w/wo SFS-residuals --
# same four variants the sim pipeline trains, see Snakefile's
# MODELING_VARIANTS) so real-data predictions can be compared across variants
# instead of being pinned to a single hardcoded modeling dir. Requires stages
# A-E (real_data_prep, real_sfs_inference, real_ld_windows, real_momentsld,
# real_fim_residuals) to have already finished.
#
# Rules run: combine_results_real, build_real_prediction_dataset,
#            predict_real_data
#
# predict_real_data needs a trained model (from the sim modeling pipeline)
# for each (variant, model_key) -- this script silently skips any combination
# whose trained *_mdl_obj.pkl doesn't exist yet rather than failing the job.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")

REAL_INF_ROOT="experiments/${MODEL}/real_data_analysis/inferences"

# Must match the Snakefile's MODELING_VARIANTS.
MODELING_VARIANTS=(
    w_FIM_w_SFSresids
    w_FIM_wo_SFSresids
    wo_FIM_w_SFSresids
    wo_FIM_wo_SFSresids
)
MODEL_KEYS=(
    random_forest
    xgboost
    linear_standard
    linear_ridge
    linear_lasso
    linear_elasticnet
)

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

for variant in "${MODELING_VARIANTS[@]}"; do
    REAL_PRED_ROOT="experiments/${MODEL}/real_data_analysis/prediction_${variant}"
    MODELING_DIR="experiments/${MODEL}/modeling_${variant}"

    echo "[$variant] Building real prediction dataset -> ${REAL_PRED_ROOT}/real_features_df.pkl"
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

    for model_key in "${MODEL_KEYS[@]}"; do
        case "$model_key" in
            linear_*) model_pkl="${MODELING_DIR}/${model_key}/linear_mdl_obj_${model_key#linear_}.pkl" ;;
            xgboost)  model_pkl="${MODELING_DIR}/xgboost/xgb_mdl_obj.pkl" ;;
            *)        model_pkl="${MODELING_DIR}/${model_key}/${model_key}_mdl_obj.pkl" ;;
        esac

        if [[ ! -f "$ROOT/$model_pkl" ]]; then
            echo "SKIP: [$variant] $model_key (no trained model at $model_pkl)"
            continue
        fi

        echo "[$variant] Predicting with $model_key -> ${REAL_PRED_ROOT}/predictions_${model_key}.json"
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
done

echo "real_combine_predict finished."

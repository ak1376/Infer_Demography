#!/bin/bash
#SBATCH --job-name=real_fim_resid
#SBATCH --output=logs/real_fim_resid_%j.out
#SBATCH --error=logs/real_fim_resid_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Stage E: observed FIM + SFS-residuals at the real moments/dadi best-fit
# params, against the combined autosomal SFS. real_sfs_inference.sh must
# have already produced the real moments/dadi best_fit.pkl.
#
# Rules run: compute_fim_real, sfs_residuals_real
#
# Engine lists are config-driven (fim_engines, residual_engines), mirroring
# compute_fim.sh / the Snakefile's FIM_ENGINES / RESIDUAL_ENGINES.

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"

MODEL=$(jq -r '.demographic_model' "$CFG")

mapfile -t FIM_ENGINES < <(jq -r '(.fim_engines // ["moments"]) | if type=="array" then .[] else . end' "$CFG")

RESID_RAW=$(jq -r 'if (.residual_engines // "both") | type == "array"
                    then (.residual_engines | join(" "))
                    else (.residual_engines // "both") end' "$CFG")
case "$RESID_RAW" in
    both|all) RESIDUAL_ENGINES=(moments dadi) ;;
    moments|dadi) RESIDUAL_ENGINES=("$RESID_RAW") ;;
    *) read -ra RESIDUAL_ENGINES <<< "$RESID_RAW" ;;
esac

REAL_INF_ROOT="experiments/${MODEL}/real_data_analysis/inferences"
USE_GS=$(jq -r '.gram_schmidt // false' "$CFG")
RESID_FNAME="residuals_flat.npy"
[[ "$USE_GS" == "true" ]] && RESID_FNAME="residuals_gs_coeffs.npy"

TARGETS=()
for eng in "${FIM_ENGINES[@]}"; do
    TARGETS+=("${REAL_INF_ROOT}/fim/${eng}.fim.npy")
done
for eng in "${RESIDUAL_ENGINES[@]}"; do
    TARGETS+=("${REAL_INF_ROOT}/sfs_residuals/${eng}/${RESID_FNAME}")
done

echo "MODEL=$MODEL  FIM_ENGINES=${FIM_ENGINES[*]}  RESIDUAL_ENGINES=${RESIDUAL_ENGINES[*]}"
echo "Targets: ${TARGETS[*]}"

snakemake \
    --snakefile "$SNAKEFILE" \
    --directory "$ROOT" \
    --nolock \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --allowed-rules compute_fim_real sfs_residuals_real \
    -j "${SLURM_CPUS_PER_TASK:-2}" \
    "${TARGETS[@]}"

echo "real_fim_residuals finished."

#!/bin/bash
#SBATCH --job-name=real_sfs_profile
#SBATCH --output=logs/real_sfs_profile_%A_%a.out
#SBATCH --error=logs/real_sfs_profile_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Moments SFS profile likelihoods for the real data, per configured arm --
# the SLURM version of the Snakefile's sfs_profile_point_real_by_arm /
# sfs_profile_real_by_arm rules. Grids come from the config:
#   "real_data_analysis": {"sfs_profile_likelihood": {"grids": {"T": [0.02, 2, 13]}, "n_starts": 8}}
# (scaled units: sizes are N_*/N_ANC, T is T/(2 N_ANC), migration is 2 N_ANC m).
# Each grid value pins the parameter and re-fits everything else from
# n_starts starts (start 0 warm-starts from the arm's AGGREGATED moments best
# fit -- run real_aggregate_sfs.sh first -- the others are LHS draws).
#
# STAGE=points (default): self-resubmitting array, one (arm, param, grid point,
#   start) fit per task; already-finished fits are skipped.
#   Rule run: sfs_profile_point_real_by_arm
# STAGE=plot: once every point is done, collect each profile into
#   profiles/moments/<param>/profile_<param>.{png,tsv}.
#   Rule run: sfs_profile_real_by_arm
#
# Usage:
#   sbatch bash_scripts/real_data/real_sfs_profile.sh              # all fits
#   STAGE=plot sbatch bash_scripts/real_data/real_sfs_profile.sh   # afterwards

set -euo pipefail
mkdir -p logs

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
source "$ROOT/bash_scripts/lib/lib_real_data_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/Snakefile"
STAGE="${STAGE:-points}"

load_real_data_config "$CFG"

readarray -t PARAMS < <(jq -r '.real_data_analysis.sfs_profile_likelihood.grids // {} | keys[]' "$CFG")
N_STARTS=$(jq -r '.real_data_analysis.sfs_profile_likelihood.n_starts // 8' "$CFG")
if [[ ${#PARAMS[@]} -eq 0 ]]; then
    echo "ERROR: no real_data_analysis.sfs_profile_likelihood.grids in $CFG" >&2
    exit 1
fi

# Every (arm, param, grid point, start) fit, in a fixed order.
TASKS=()
for arm in "${REAL_ARMS[@]}"; do
    for p in "${PARAMS[@]}"; do
        n=$(jq -r --arg p "$p" '.real_data_analysis.sfs_profile_likelihood.grids[$p][2]' "$CFG")
        for ((k = 0; k < n; k++)); do
            for ((s = 0; s < N_STARTS; s++)); do
                TASKS+=("${REAL_FIT_ROOT}/${arm}/profiles/moments/${p}/pt${k}/start${s}/moments/best_fit.pkl")
            done
        done
    done
done

run_snakemake() {
    snakemake \
        --snakefile "$SNAKEFILE" \
        --directory "$ROOT" \
        --nolock \
        --keep-going \
        --rerun-incomplete \
        --rerun-triggers mtime \
        --config active_experiment_config="$CFG" \
        --allowed-rules "$1" \
        -j "${SLURM_CPUS_PER_TASK:-2}" \
        "${@:2}"
}

if [[ "$STAGE" == "plot" ]]; then
    TARGETS=()
    for arm in "${REAL_ARMS[@]}"; do
        for p in "${PARAMS[@]}"; do
            TARGETS+=("${REAL_FIT_ROOT}/${arm}/profiles/moments/${p}/profile_${p}.png")
        done
    done
    echo "arms: ${REAL_ARMS[*]}  params: ${PARAMS[*]}"
    printf 'target: %s\n' "${TARGETS[@]}"
    run_snakemake sfs_profile_real_by_arm "${TARGETS[@]}"
    echo "real_sfs_profile (plot) finished."
    exit 0
fi

# STAGE=points
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    for arm in "${REAL_ARMS[@]}"; do
        best="$(real_data_chrom_path "$REAL_INF_ROOT_CHROM_TMPL" "$arm")/moments/best_fit.pkl"
        if [[ ! -s "$ROOT/$best" ]]; then
            echo "ERROR: $best not found -- aggregate the arm's moments runs first (real_aggregate_sfs.sh)" >&2
            exit 1
        fi
    done
    echo "arms: ${REAL_ARMS[*]}  params: ${PARAMS[*]}  starts per point: $N_STARTS  total fits: ${#TASKS[@]}"
    echo "Submitting array 0..$(( ${#TASKS[@]} - 1 ))"
    STAGE=points ROOT="$ROOT" sbatch --array=0-"$(( ${#TASKS[@]} - 1 ))" "$0" "$@"
    exit 0
fi

TARGET="${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [[ -s "$ROOT/$TARGET" ]]; then
    echo "SKIP (already done): $TARGET"
    exit 0
fi
echo "Task $SLURM_ARRAY_TASK_ID -> $TARGET"
run_snakemake sfs_profile_point_real_by_arm "$TARGET"
echo "real_sfs_profile task $SLURM_ARRAY_TASK_ID finished."

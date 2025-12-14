#!/bin/bash
#SBATCH --job-name=combine_inf
#SBATCH --output=logs/combine_%A_%a.out
#SBATCH --error=logs/combine_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# 0. paths & experiment constants
CFG="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_bottleneck.json"

ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

NUM_DRAWS=$(jq -r '.num_draws'     "$CFG")
MODEL=$(jq -r '.demographic_model' "$CFG")

# 1. (re)submit with the correct --array range if none was provided
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    sbatch --array=0-$(( NUM_DRAWS - 1 )) "$0" "$@"
    exit 0
fi

# 2. decode sid from array index (no padding)
sid="$SLURM_ARRAY_TASK_ID"
echo "combine_results: sid=$sid  (folder sim_$sid)"

SFS_FILE="experiments/${MODEL}/simulations/${sid}/SFS.pkl"

# high-level targets for THIS sim
TARGET_COMBO="experiments/${MODEL}/inferences/sim_${sid}/all_inferences.pkl"
TARGET_DADI="experiments/${MODEL}/inferences/sim_${sid}/dadi/fit_params.pkl"
TARGET_MOM="experiments/${MODEL}/inferences/sim_${sid}/moments/fit_params.pkl"
TARGET_LD="experiments/${MODEL}/inferences/sim_${sid}/MomentsLD/best_fit.pkl"

# 3. Check simulation exists
if [[ ! -f "$SFS_FILE" ]]; then
    echo "ERROR: Simulation file $SFS_FILE doesn't exist for sim_${sid}!"
    echo "This script assumes simulations are already done. Run simulation first."
    exit 1
fi

echo "INFO: Simulation exists for sim_${sid}: $SFS_FILE"

# 4. Clean up metadata for simulation files to mark them as complete
echo "INFO: Marking simulation files as complete in Snakemake metadata..."
snakemake --cleanup-metadata \
  "experiments/${MODEL}/simulations/${sid}/SFS.pkl" \
  "experiments/${MODEL}/simulations/${sid}/sampled_params.pkl" \
  "experiments/${MODEL}/simulations/${sid}/tree_sequence.trees" \
  "experiments/${MODEL}/simulations/${sid}/demes.png" \
  "experiments/${MODEL}/simulations/${sid}/bgs.meta.json" \
  "experiments/${MODEL}/simulations/${sid}/.done" 2>/dev/null || true

# 5. If the combined inference blob already exists, skip this sim
if [[ -f "$TARGET_COMBO" ]]; then
    echo "INFO: Combined inferences already exist for sim_${sid}:"
    echo "  - $TARGET_COMBO ($(stat -c %y "$TARGET_COMBO"))"
    echo "INFO: Skipping sim_${sid} - full inference already complete."
    exit 0
fi

echo "INFO: Full inference NOT complete for sim_${sid}."
[[ ! -f "$TARGET_DADI" ]] && echo "  - Missing: $TARGET_DADI"
[[ ! -f "$TARGET_MOM"  ]] && echo "  - Missing: $TARGET_MOM"
[[ ! -f "$TARGET_LD"   ]] && echo "  - Missing: $TARGET_LD"
[[ ! -f "$TARGET_COMBO" ]] && echo "  - Missing: $TARGET_COMBO"

# 6. Decide whether windows need to be (re)generated
#    Use a BASH ARRAY for allowed rules (no commas!)
ALLOWED_RULES=(
  infer_dadi
  aggregate_opts_dadi
  infer_moments
  aggregate_opts_moments
  ld_window
  optimize_momentsld
  compute_fim
  sfs_residuals
  combine_results
)

# if LD windows are missing, also allow simulate_window
if [[ ! -d "$WINDOWS_DIR" ]] || [[ $(ls -1 "$WINDOWS_DIR"/*.vcf.gz 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "INFO: Simulation windows missing. Allowing simulate_window rule."
    ALLOWED_RULES+=(simulate_window)
else
    echo "INFO: Simulation windows exist. Restricting to inference rules only."
fi

echo "INFO: Allowed rules: ${ALLOWED_RULES[*]}"

# 7. Run full pipeline (dadi + moments + LD + FIM + residuals + combined blob) for this sim only
snakemake \
  -j "$SLURM_CPUS_PER_TASK" \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --rerun-triggers mtime \
  --nolock \
  --allowed-rules "${ALLOWED_RULES[@]}" \
  --keep-going \
  "$TARGET_COMBO"

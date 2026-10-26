#!/bin/bash
#SBATCH --job-name=postproc_features
#SBATCH --output=logs/postprocessing_features_%j.out
#SBATCH --error=logs/postprocessing_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

set -euo pipefail
ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib/lib_active_config.sh"
CFG_PATH="$(resolve_cfg_path "$ROOT")"
SNAKEFILE="$ROOT/modeling.smk"
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

# Same list modeling.smk builds: the experiment config's "modeling_variants",
# defaulting to the four inference variants (see rules/common.smk).
mapfile -t MODELING_VARIANTS < <(jq -r '(.modeling_variants // ["w_FIM_w_SFSresids","w_FIM_wo_SFSresids","wo_FIM_w_SFSresids","wo_FIM_wo_SFSresids"])[]' "$CFG_PATH")

# One combine_features job writes every dataset file for its variant.
DATA_TARGETS=()
for variant in "${MODELING_VARIANTS[@]}"; do
  DATA_TARGETS+=("experiments/${MODEL}/modeling_${variant}/datasets/features_df.pkl")
done

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --cores "$SLURM_CPUS_PER_TASK" \
  --nolock --rerun-incomplete \
  --allowed-rules combine_features \
  --forcerun combine_features \
  -- \
  "${DATA_TARGETS[@]}"
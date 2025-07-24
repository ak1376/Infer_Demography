#!/bin/bash
#SBATCH --job-name=postprocessing_features
#SBATCH --output=logs/postprocessing_features_%j.out
#SBATCH --error=logs/postprocessing_features_%j.err
#SBATCH --time=02:00:00               # adjust as needed
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue                               # Requeue on preemption
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

##############################################################################
# Environment                                                               #
##############################################################################
set -euo pipefail
: "${CFG_PATH:?CFG_PATH is not defined}"      # exported by your master script
ROOT="/projects/kernlab/akapoor/Infer_Demography"
SNAKEFILE="$ROOT/Snakefile"

# Pull MODEL from the config so we can point Snakemake at the right outputs
MODEL=$(jq -r '.demographic_model' "$CFG_PATH")

##############################################################################
# Run Snakemake (only rule combine_features)                                #
##############################################################################
snakemake --snakefile "$SNAKEFILE" \
          --directory  "$ROOT"     \
          --nolock                \
          --rerun-incomplete      \
          -j "$SLURM_CPUS_PER_TASK" \
          combine_features

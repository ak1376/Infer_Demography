#!/bin/bash
#SBATCH --job-name=sim_job_array                # Job name
#SBATCH --array=0-4999                            # Array range (adjust based on the number of tasks and batch size)
#SBATCH --output=logs/simulation_%A_%a.out      # Standard output log file (%A is job ID, %a is the array index)
#SBATCH --error=logs/simulation_%A_%a.err       # Standard error log file
#SBATCH --time=6:00:00                          # Time limit
#SBATCH --cpus-per-task=8                       # Number of CPU cores per task
#SBATCH --mem=8G                                # Memory per task
#SBATCH --partition=kern,preempt,kerngpu        # Partitions to submit the job to
#SBATCH --account=kernlab                       # Account to use
#SBATCH --requeue                               # Requeue on preemption
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Define batch size and total number of tasks
BATCH_SIZE=1
TOTAL_TASKS=5000

# the master script exports CFG_PATH; abort if it is not set
# : "${CFG_PATH:?CFG_PATH is not defined}"
CFG_PATH="/home/akapoor/kernlab/Infer_Demography/config_files/experiment_config_drosophila_three_epoch.json"
EXPERIMENT_CONFIG_FILE="$CFG_PATH"

# Extract the values from the JSON config
DEMOGRAPHIC_MODEL=$(jq -r '.demographic_model' $EXPERIMENT_CONFIG_FILE)
DADI_ANALYSIS=$(jq -r '.dadi_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_ANALYSIS=$(jq -r '.moments_analysis' $EXPERIMENT_CONFIG_FILE)
MOMENTS_LD_ANALYSIS=$(jq -r '.momentsLD_analysis' $EXPERIMENT_CONFIG_FILE)
SEED=$(jq -r '.seed' $EXPERIMENT_CONFIG_FILE)
NUM_SIMS_PRETRAIN=$(jq -r '.num_sims_pretrain' $EXPERIMENT_CONFIG_FILE)
K=$(jq -r '.k' $EXPERIMENT_CONFIG_FILE)
TOP_VALUES_K=$(jq -r '.top_values_k' $EXPERIMENT_CONFIG_FILE)

# Function to convert lowercase true/false to True/False
capitalize_bool() {
    if [ "$1" == "true" ]; then
        echo "True"
    elif [ "$1" == "false" ]; then
        echo "False"
    else
        echo "$1"
    fi
}

DADI_ANALYSIS=$(capitalize_bool $DADI_ANALYSIS)
MOMENTS_ANALYSIS=$(capitalize_bool $MOMENTS_ANALYSIS)
MOMENTS_LD_ANALYSIS=$(capitalize_bool $MOMENTS_LD_ANALYSIS)

# Only measure the time for the full execution of the entire job array
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # Start timing at the beginning of the first job array task
    overall_start_time=$(date +%s)
    echo "Overall start time recorded: $overall_start_time"
fi

# Calculate the starting and ending TASK_IDs for this batch
BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH_END=$(((SLURM_ARRAY_TASK_ID + 1) * BATCH_SIZE - 1))

# Ensure that BATCH_END does not exceed TOTAL_TASKS
if [ "$BATCH_END" -ge "$TOTAL_TASKS" ]; then
    BATCH_END=$((TOTAL_TASKS - 1))
fi

# build a handy prefix once …
EXP_ROOT="/projects/kernlab/akapoor/Infer_Demography/experiments/${DEMOGRAPHIC_MODEL}/simulations"

# Run Snakemake rule for each task in the batch
for TASK_ID in $(seq $BATCH_START $BATCH_END); do
    PAD_ID=$(printf "$TASK_ID")   # 0 → 00, 1 → 01 …

    # Unlock any stale locks before running
    snakemake --unlock \
      --snakefile /projects/kernlab/akapoor/Infer_Demography/Snakefile \
      --directory /projects/kernlab/akapoor/Infer_Demography

    snakemake -j "$SLURM_CPUS_PER_TASK" \
      --snakefile /projects/kernlab/akapoor/Infer_Demography/Snakefile \
      --directory /projects/kernlab/akapoor/Infer_Demography \
      --rerun-incomplete \
      "experiments/${DEMOGRAPHIC_MODEL}/simulations/${PAD_ID}/tree_sequence.trees"
done

# Calculate the overall elapsed time only at the last task in the array
if [ "$SLURM_ARRAY_TASK_ID" -eq $((TOTAL_TASKS / BATCH_SIZE - 1)) ]; then
    overall_end_time=$(date +%s)
    echo "Overall end time recorded: $overall_end_time"
    
    # Ensure both start and end times are set before calculating elapsed time
    if [[ -n "$overall_start_time" && -n "$overall_end_time" ]]; then
        # Calculate and print the overall elapsed time
        overall_elapsed_time=$((overall_end_time - overall_start_time))
        echo "Total time taken for the entire job array: $overall_elapsed_time seconds"
    else
        echo "Error: Start time or end time not set properly."
    fi
fi
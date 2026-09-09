#!/usr/bin/env bash
# Shared arithmetic for sizing a SLURM --array range from a total task count
# and a batch size. Used by master_script.sh to submit each pipeline stage
# with its correct --array range directly (rather than letting the stage
# script self-resubmit), so the job ID master captures is the REAL array
# job — required for --dependency=afterany to wait for the whole stage to
# finish rather than a dispatcher that exits in seconds.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib_array_size.sh"
#   n=$(array_upper_bound "$TOTAL_TASKS" "$BATCH_SIZE")   # -> "N" for --array=0-N
#   spec=$(array_spec "$TOTAL_TASKS" "$BATCH_SIZE" [THROTTLE])  # -> "0-N" or "0-N%THROTTLE"

array_upper_bound() {
    local total="$1" batch="$2"
    echo $(( (total + batch - 1) / batch - 1 ))
}

array_spec() {
    local total="$1" batch="$2" throttle="${3:-}"
    local n
    n=$(array_upper_bound "$total" "$batch")
    if [[ -n "$throttle" ]]; then
        echo "0-${n}%${throttle}"
    else
        echo "0-${n}"
    fi
}
#!/bin/bash
#SBATCH --job-name=pipeline_concurrent
#SBATCH --output=logs/pipeline_concurrent_%j.out
#SBATCH --error=logs/pipeline_concurrent_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=kern
#SBATCH --account=kernlab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

# ---------------------------------------------------------------------------
# Concurrent, quota-aware version of master_script.sh.
#
# master_script.sh queues EVERY stage at once (later stages held back with
# --dependency). Pending tasks count toward the per-user submit limit
# (QOS "normal": MaxSubmitPU=30001, every array task counts), so at large
# num_draws the later sbatch calls are rejected and the pipeline is cut off.
#
# This script instead stays alive (a long, cheap 1-CPU job on the
# non-preemptible kern partition) and submits each stage only when
#   (a) the stages it reads from have finished, and
#   (b) your queued+running task count leaves room for the whole stage.
# Independent branches run at the same time:
#
#   simulate ─┬─ materialize → windows → LD stats → MomentsLD prep → MomentsLD ─┐
#             ├─ moments (needs only SFS.pkl + sampled_params.pkl) ──────────────┼─ aggregate → diagnostics → combine → features
#             └─ dadi    (needs only SFS.pkl + sampled_params.pkl) ──────────────┘
#
# Like master_script.sh's afterany, a stage starts once the previous one has
# FINISHED, whether or not every task succeeded; a per-stage state summary
# is printed so failures are visible in the log.
#
# Submit:      sbatch bash_scripts/orchestrators/master_script_concurrent.sh
# Resume after this job dies: just resubmit it. Stages whose jobs are still
#   in the queue (matched by job name) are waited on instead of resubmitted,
#   and SKIP_STAGES skips stages that already finished, e.g.
#   SKIP_STAGES=sim,mat,win sbatch bash_scripts/orchestrators/master_script_concurrent.sh
#   Stage keys: sim mat win ld momld_prep momld moments dadi agg diag combine features
#   (Don't run two pipelines for different configs at once: adoption matches by job name.)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${REPO:-/projects/kernlab/akapoor/Infer_Demography}"
mkdir -p "$REPO/logs"
cd "$REPO"

source "$REPO/bash_scripts/lib/lib_active_config.sh"
source "$REPO/bash_scripts/lib/lib_array_size.sh"
CFG_PATH="$(resolve_cfg_path "$REPO")"
export CFG_PATH

snakemake -s "$REPO/simulation.smk" --directory "$REPO" --unlock 2>/dev/null || true

NUM_DRAWS=$(jq -r '.num_draws' "$CFG_PATH")
NUM_WINDOWS=$(jq -r '.num_windows // 100' "$CFG_PATH")
NUM_OPTIMS=$(jq -r '.num_optimizations' "$CFG_PATH")
USE_GPU_LD=$(jq -r '.use_gpu_ld // false' "$CFG_PATH")

# ---- knobs (all overridable from the environment) -------------------------
SUBMIT_LIMIT="${SUBMIT_LIMIT:-29500}"   # stay a little under MaxSubmitPU=30001
POLL="${POLL:-120}"                     # seconds between squeue checks
SKIP_STAGES="${SKIP_STAGES:-}"

# Items per array task. Per-item runtimes from past sacct records:
# materialize ~4 min, MomentsLD prep ~2, aggregate ~2, diagnostics ~3,
# combine ~0.5 -> 10 per task keeps each task well under its time limit.
# The 50-batched stages are already 25-90 min/task; LD stats has had tasks
# near its 16 h limit, so don't raise those.
BS_SIM="${BS_SIM:-1}"
BS_MAT="${BS_MAT:-10}"
BS_WIN="${BS_WIN:-50}"
BS_LD="${BS_LD:-50}"
BS_MOMLD_PREP="${BS_MOMLD_PREP:-10}"
BS_MOMLD="${BS_MOMLD:-50}"
BS_MOMENTS="${BS_MOMENTS:-50}"
BS_DADI="${BS_DADI:-50}"
BS_AGG="${BS_AGG:-10}"
BS_DIAG="${BS_DIAG:-10}"
BS_COMBINE="${BS_COMBINE:-10}"

# Max tasks of one stage RUNNING at once (the %N in --array); "" = no cap.
THR_WIN="${THR_WIN:-${MAX_CONCURRENT_WINDOWS:-1500}}"
THR_MOMLD="${THR_MOMLD:-5000}"
THR_MOMENTS="${THR_MOMENTS:-500}"      # was 100 in master_script.sh
THR_DADI="${THR_DADI:-}"

if [[ "$USE_GPU_LD" == "true" ]]; then
  LD_OPTS=(--partition=kerngpu,gpulong,gpu --gres=gpu:1)
else
  LD_OPTS=(--partition=kern,preempt --gres=gpu:0)
fi

LOCK="$REPO/logs/.pipeline_submit.lock.${SLURM_JOB_ID:-$$}"
: > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log() { echo "[$(date '+%F %T')] $*"; }

# Tasks you currently have queued or running (each array task counts).
queued_count() {
  local out
  until out=$(squeue -h -u "$USER" -r -o %i 2>/dev/null); do sleep 30; done
  wc -l <<<"$out"
}

# True while any task of array/job $1 is still queued or running. A failed
# squeue call counts as "still active" so a slurmctld hiccup can't make
# us start the next stage early.
job_active() {
  local out
  out=$(squeue -h -u "$USER" -o %F 2>/dev/null) || return 0
  grep -qx "$1" <<<"$out"
}

wait_job() {
  local key="$1" id="$2"
  while job_active "$id"; do sleep "$POLL"; done
  log "$key: job $id finished — task states: $(sacct -j "$id" -X -n -o State 2>/dev/null \
        | awk '{print $1}' | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}')"
}

skipped() { [[ ",$SKIP_STAGES," == *",$1,"* ]]; }

# run_stage KEY SCRIPT TOTAL_ITEMS BATCH THROTTLE [extra sbatch args...]
# TOTAL_ITEMS=0 submits a single non-array job.
run_stage() {
  local key="$1" script="$2" total="$3" batch="$4" thr="$5"; shift 5
  if skipped "$key"; then log "$key: skipped (SKIP_STAGES)"; return 0; fi

  # Resume support: if this stage's jobs are already queued from an earlier
  # run of this script, wait on them instead of submitting a duplicate.
  local name existing
  name=$(grep -m1 -oP '(?<=--job-name=)\S+' "$script")
  existing=$(squeue -h -u "$USER" -n "$name" -o %F 2>/dev/null | sort -u | tr '\n' ' ')
  if [[ -n "${existing// /}" ]]; then
    log "$key: adopting already-queued job(s) $existing(name $name)"
    for id in $existing; do wait_job "$key" "$id"; done
    return 0
  fi

  local ntasks=1 array_args=()
  if (( total > 0 )); then
    ntasks=$(( (total + batch - 1) / batch ))
    array_args=(--array="$(array_spec "$total" "$batch" "$thr")")
  fi

  # One submitter at a time, so two branches can't both see the same free
  # room and overshoot the limit together.
  local id
  id=$(
    {
      flock 9
      while (( $(queued_count) + ntasks > SUBMIT_LIMIT )); do sleep "$POLL"; done
      BATCH_SIZE="$batch" sbatch --parsable --export=ALL "${array_args[@]}" "$@" "$script"
    } 9>>"$LOCK"
  )
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { log "$key: sbatch failed (got '$id')"; return 1; }
  log "$key: submitted job $id ($ntasks tasks, batch=$batch${thr:+, max $thr running})"
  wait_job "$key" "$id"
}

S="bash_scripts/simulation"
M="bash_scripts/modeling"

log "config: $CFG_PATH  (num_draws=$NUM_DRAWS windows=$NUM_WINDOWS optims=$NUM_OPTIMS gpu_ld=$USE_GPU_LD)"
log "submit limit: $SUBMIT_LIMIT  poll: ${POLL}s  skip: ${SKIP_STAGES:-none}"

# --- 1. simulate ---
run_stage sim "$S/running_simulation.sh" "$NUM_DRAWS" "$BS_SIM" ""

# --- 2. three independent branches ---
branch_ld() {
  run_stage mat        "$S/materialize_windows.sh" "$NUM_DRAWS"                    "$BS_MAT"        ""
  run_stage win        "$S/build_windows.sh"       "$(( NUM_DRAWS * NUM_WINDOWS ))" "$BS_WIN"        "$THR_WIN"
  run_stage ld         "$S/LD_stats_windows.sh"    "$(( NUM_DRAWS * NUM_WINDOWS ))" "$BS_LD"         ""  "${LD_OPTS[@]}"
  run_stage momld_prep "$S/MomentsLD_prep.sh"      "$NUM_DRAWS"                    "$BS_MOMLD_PREP" ""
  run_stage momld      "$S/MomentsLD.sh"           "$(( NUM_DRAWS * NUM_OPTIMS ))"  "$BS_MOMLD"      "$THR_MOMLD"
}
branch_ld &                                                                                        pid_ld=$!
run_stage moments "$S/moments.sh" "$(( NUM_DRAWS * NUM_OPTIMS ))" "$BS_MOMENTS" "$THR_MOMENTS" &   pid_mom=$!
run_stage dadi    "$S/dadi.sh"    "$(( NUM_DRAWS * NUM_OPTIMS ))" "$BS_DADI"    "$THR_DADI" &      pid_dadi=$!

fail=0
wait "$pid_ld"   || { log "LD branch failed to submit a stage"; fail=1; }
wait "$pid_mom"  || { log "moments failed to submit";           fail=1; }
wait "$pid_dadi" || { log "dadi failed to submit";              fail=1; }
(( fail == 0 )) || { log "stopping before aggregation; fix and resubmit with SKIP_STAGES"; exit 1; }

# --- 3. aggregate → diagnostics → combine → features ---
run_stage agg      "$S/aggregate_opts.sh"     "$NUM_DRAWS" "$BS_AGG"     ""
run_stage diag     "$S/fit_diagnostics.sh"    "$NUM_DRAWS" "$BS_DIAG"    ""
run_stage combine  "$M/run_combine.sh"        "$NUM_DRAWS" "$BS_COMBINE" ""
run_stage features "$M/aggregate_features.sh" 0            1             ""

log "pipeline complete"

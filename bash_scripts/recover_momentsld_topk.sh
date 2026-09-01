#!/bin/bash
# ONE-OFF RECOVERY SCRIPT -- delete after use.
#
# cleanup_optimization_runs deleted experiments/{MODEL}/runs/run_{sid}_{opt}
# directories that weren't in the dadi/moments top-K, without regard to
# what MomentsLD still needed from them. This rebuilds
# experiments/{MODEL}/inferences/sim_{sid}/MomentsLD/pruning/{frac_tag}/best_fit.pkl
# from whatever run_{sid}_*/inferences/MomentsLD/pruning/{frac_tag}/best_fit.pkl
# files survived, taking the top-K of *whatever remains* instead of enforcing
# the usual aggregate_min_replicates floor (which would just raise and refuse
# to run). This experiment runs in pruned MomentsLD mode (prune_mode != off),
# so per-run output lives under .../MomentsLD/pruning/<frac_tag>/, not
# directly under .../MomentsLD/ -- frac tags are derived from the config
# exactly like aggregate_momentsld.sh does.
#
# Usage:
#   bash bash_scripts/recover_momentsld_topk.sh              # all sim ids, top 5
#   TOP_K=3 bash bash_scripts/recover_momentsld_topk.sh       # different K
#   FORCE=1 bash bash_scripts/recover_momentsld_topk.sh       # overwrite existing best_fit.pkl too
#   SIDS="0 1 2" bash bash_scripts/recover_momentsld_topk.sh  # only these sim ids

set -euo pipefail

TOP_K="${TOP_K:-5}"
FORCE="${FORCE:-0}"

ROOT="${ROOT:-/projects/kernlab/akapoor/Infer_Demography}"
source "$ROOT/bash_scripts/lib_active_config.sh"
CFG="$(resolve_cfg_path "$ROOT")"
cd "$ROOT"

MODEL=$(jq -r '.demographic_model' "$CFG")
NUM_DRAWS=$(jq -r '.num_draws' "$CFG")
SIDS="${SIDS:-$(seq 0 $(( NUM_DRAWS - 1 )))}"

PRUNE_MODE=$(jq -r '.prune_mode // "off"' "$CFG")
PRUNE_VALUES_JSON=$(jq -c '.prune_keep_values // []' "$CFG")

if [[ "$PRUNE_MODE" != "fraction" && "$PRUNE_MODE" != "count" ]]; then
    echo "ERROR: prune_mode=$PRUNE_MODE in $CFG -- this experiment doesn't look like pruned MomentsLD" >&2
    echo "mode. Check .../runs/<any_dir>/inferences/MomentsLD/ by hand instead." >&2
    exit 1
fi

echo "MODEL=$MODEL  TOP_K=$TOP_K  FORCE=$FORCE  PRUNE_MODE=$PRUNE_MODE  PRUNE_VALUES=$PRUNE_VALUES_JSON"

PYTHONPATH="$ROOT" python3 -u - "$MODEL" "$TOP_K" "$FORCE" "$PRUNE_MODE" "$PRUNE_VALUES_JSON" -- $SIDS <<'PYEOF'
import json, pathlib, pickle, re, sys, time
from src.aggregate_utils import aggregate_top_k

model, top_k, force, prune_mode, prune_values_json = sys.argv[1:6]
top_k = int(top_k)
force = force == "1"
sids = sys.argv[sys.argv.index("--") + 1:]
sids_wanted = set(sids)

# Mirror the Snakefile's own _frac_tag/_count_tag exactly (Snakefile:249-254)
# so these tags are guaranteed to match the directories the pipeline itself
# created -- the bash/jq formula aggregate_momentsld.sh uses for the same
# purpose doesn't zero-pad ("thin5" vs Snakefile's "thin05"), which would
# silently mismatch for prune values under 10%.
prune_values = json.loads(prune_values_json)
if prune_mode == "fraction":
    frac_tags = [f"thin{round(float(v) * 100):02d}" for v in prune_values]
else:
    frac_tags = [f"n{int(v)}" for v in prune_values]

# Single pass over experiments/{model}/runs/ instead of one glob.glob() per
# sid -- with NUM_DRAWS sids, re-scanning that whole directory NUM_DRAWS
# times over a cluster filesystem is what made the naive version crawl.
t0 = time.time()
runs_root = pathlib.Path(f"experiments/{model}/runs")
name_re = re.compile(r"^run_(\d+)_(\d+)$")

print(f"Scanning {runs_root} for frac_tags={frac_tags} ...")
by_frac_sid = {ft: {} for ft in frac_tags}
n_entries = 0
for entry in runs_root.iterdir():
    n_entries += 1
    if n_entries % 200000 == 0:
        print(f"  ...{n_entries} entries scanned so far ({time.time()-t0:.0f}s)")
    m = name_re.match(entry.name)
    if not m:
        continue
    sid, opt = m.group(1), int(m.group(2))
    if sid not in sids_wanted:
        continue
    for ft in frac_tags:
        pkl = entry / "inferences" / "MomentsLD" / "pruning" / ft / "best_fit.pkl"
        if pkl.is_file():
            by_frac_sid[ft].setdefault(sid, []).append((str(pkl), opt))

for ft in frac_tags:
    print(f"  frac_tag={ft}: {sum(len(v) for v in by_frac_sid[ft].values())} surviving files "
          f"across {len(by_frac_sid[ft])} sids")
print(f"Scan done: {n_entries} run dirs ({time.time()-t0:.0f}s)")

n_done = n_skipped_existing = n_skipped_empty = 0

for i, sid in enumerate(sids):
    if i and i % 500 == 0:
        print(f"  ...processed {i}/{len(sids)} sids ({time.time()-t0:.0f}s)")

    for ft in frac_tags:
        out = pathlib.Path(f"experiments/{model}/inferences/sim_{sid}/MomentsLD/pruning/{ft}/best_fit.pkl")
        if out.exists() and not force:
            n_skipped_existing += 1
            continue

        records = by_frac_sid[ft].get(sid, [])
        if not records:
            n_skipped_empty += 1
            continue

        # No min_nonempty floor here on purpose: take the top-K of whatever
        # survived cleanup, even if that's fewer than the usual required minimum.
        best, diag = aggregate_top_k(records, top_k)

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(best, fh)

        kept = sorted(set(best.get("opt_index", [])))
        flag = "" if diag["n_nonempty"] >= top_k else "  (** fewer than top_k survived **)"
        print(f"sid={sid} frac_tag={ft}: found {diag['n_records']} surviving files, "
              f"{diag['n_nonempty']} non-empty -> kept opts={kept}{flag} -> {out}")
        n_done += 1

print(f"\nDone in {time.time()-t0:.0f}s. wrote={n_done} skipped_existing={n_skipped_existing} skipped_no_survivors={n_skipped_empty}")
PYEOF

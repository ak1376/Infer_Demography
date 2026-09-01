#!/bin/bash
# ONE-OFF RECOVERY SCRIPT -- delete after use.
#
# cleanup_optimization_runs deleted experiments/{MODEL}/runs/run_{sid}_{opt}
# directories that weren't in the dadi/moments top-K, without regard to
# what MomentsLD still needed from them. This rebuilds
# experiments/{MODEL}/inferences/sim_{sid}/MomentsLD/best_fit.pkl from
# whatever run_{sid}_*/inferences/MomentsLD/best_fit.pkl files survived,
# taking the top-K of *whatever remains* instead of enforcing the usual
# aggregate_min_replicates floor (which would just raise and refuse to run).
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

echo "MODEL=$MODEL  TOP_K=$TOP_K  FORCE=$FORCE"

PYTHONPATH="$ROOT" python3 - "$MODEL" "$TOP_K" "$FORCE" $SIDS <<'PYEOF'
import pathlib, pickle, sys
from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

model, top_k, force, sids = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4:]

n_done = n_skipped_existing = n_skipped_empty = 0

for sid in sids:
    out = pathlib.Path(f"experiments/{model}/inferences/sim_{sid}/MomentsLD/best_fit.pkl")
    if out.exists() and not force:
        n_skipped_existing += 1
        continue

    records = discover_opt_pkls(
        f"experiments/{model}/runs/run_{sid}_*/inferences/MomentsLD/best_fit.pkl",
        rf"/run_{sid}_(\d+)/inferences/MomentsLD/best_fit\.pkl$",
    )
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
    print(f"sid={sid}: found {diag['n_records']} surviving files, "
          f"{diag['n_nonempty']} non-empty -> kept opts={kept}{flag} -> {out}")
    n_done += 1

print(f"\nDone. wrote={n_done} skipped_existing={n_skipped_existing} skipped_no_survivors={n_skipped_empty}")
PYEOF

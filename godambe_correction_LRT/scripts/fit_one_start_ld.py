#!/usr/bin/env python3
# godambe_correction_LRT/scripts/fit_one_start_ld.py

"""
ONE LHS start of an LD fit (for Snakemake per-start parallelism). Runs LHS row
`--opt-index` for the model chosen with `--model` (null or alt) and writes just
that start's result. A separate collect step (collect_best_fit.py) picks the
best across all starts.

Reuses run_one_start from fit_simple_ld.py so the optimization is identical.
Same rule fits either model -- just pass a different --model and --out.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_simple_ld import run_one_start, resolve_model, NULL, _ALIASES
# run_one_start((opt_index, n_opt, mv, model_name)) -> result dict


def main():
    ap = argparse.ArgumentParser(description="One LHS start of the null/alt LD fit.")
    ap.add_argument("--mv", type=Path, required=True)
    ap.add_argument("--opt-index", type=int, required=True)
    ap.add_argument("--n-opt", type=int, default=100, help="LHS grid size (must be stable)")
    ap.add_argument("--model", default=NULL,
                    help="split_migration_growth (null) or split_migration_growth_both "
                         "(alt); 'null'/'alt' aliases ok (default %(default)s)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model_name = _ALIASES.get(args.model, args.model)
    resolve_model(model_name)          # validate early
    with open(args.mv, "rb") as f:
        mv = pickle.load(f)

    result = run_one_start((args.opt_index, args.n_opt, mv, model_name))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    print(f"start {args.opt_index}: LL={result['best_lls']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()

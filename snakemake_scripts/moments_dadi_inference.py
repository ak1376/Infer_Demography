#!/usr/bin/env python3
# snakemake_scripts/moments_dadi_inference.py
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import sys
from pathlib import Path                                                   # NEW

# ------------------------------------------------------------------- add this
ROOT = Path(__file__).resolve().parents[1]        # project root (…/Infer_Demography)
SRC  = ROOT / "src"
if str(SRC) not in sys.path:                      # prepend once, idempotent
    sys.path.insert(0, str(SRC))
# -----------------------------------------------------------------------------

import argparse, json, pickle
from typing import Dict, Any     
#  now the local imports work
from moments_inference import fit_model as moments_fit_model
from dadi_inference    import fit_model as dadi_fit_model
from simulation        import bottleneck_model, split_isolation_model, \
                              split_migration_model, drosophila_three_epoch

# ───────────────────────── CLI ─────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--run-dir",        required=True, type=Path)
p.add_argument("--config-file",    required=True, type=Path)
p.add_argument("--sfs",            required=True, type=Path)
p.add_argument("--sampled-params", required=True, type=Path)
p.add_argument("--rep-index",      required=True, type=int)
p.add_argument("--run-moments",    type=str, default="True")
p.add_argument("--run-dadi",       type=str, default="True")
args = p.parse_args()

run_moments = args.run_moments.lower() in {"true", "1", "yes"}
run_dadi    = args.run_dadi.lower()    in {"true", "1", "yes"}

cfg  : Dict[str, Any] = json.loads(args.config_file.read_text())
sfs  = pickle.load(args.sfs.open("rb"))
pars = pickle.load(args.sampled_params.open("rb"))

demo_lookup = {
    "bottleneck"            : bottleneck_model,
    "split_isolation"       : split_isolation_model,
    "split_migration"       : split_migration_model,
    "drosophila_three_epoch": drosophila_three_epoch,
}
demo_model = demo_lookup[cfg["demographic_model"]]

start_dict = {k: (lo + hi) / 2 for k, (lo, hi) in cfg["priors"].items()}

# make sure output folders exist
mom_out  = args.run_dir / "inferences" / "moments"
dadi_out = args.run_dir / "inferences" / "dadi"
mom_out.mkdir(parents=True, exist_ok=True)
dadi_out.mkdir(parents=True, exist_ok=True)

if run_moments:
    vecs, lls = moments_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=demo_model,
        experiment_config={**cfg, "log_dir": str(mom_out / "logs")},
        sampled_params=pars,
    )

    vec       = vecs[0]
    opt_params = dict(zip(start_dict.keys(), vec))

    pickle.dump({"best_params": opt_params, "best_lls": lls[0]},
                (mom_out / "fit_params.pkl").open("wb"))
if run_dadi:
    vecs, lls = dadi_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=demo_model,
        experiment_config={**cfg, "log_dir": str(dadi_out / "logs")},
        sampled_params=pars,
    )
    vec       = vecs[0]                     # the 1-element array
    opt_params = dict(zip(start_dict.keys(), vec))

    pickle.dump({"best_params": opt_params, "best_lls": lls[0]},
                (dadi_out / "fit_params.pkl").open("wb"))
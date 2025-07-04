#!/usr/bin/env python3
"""
split_migration.py – multi-replicate, directory-aware
=====================================================
Each replicate:

  1. Draws parameters from uniform `priors` (CLI pins override);
  2. Simulates, runs moments & dadi inference, stores fits (+ LL);
  3. Writes to  split_migration/runs/run_XXXX/…

After all replicates finish it makes two log-likelihood-coloured
scatter-plots (moments & dadi).
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import demesdraw, moments, numpy as np
from tqdm import tqdm

# ── local imports ───────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from simulation        import split_migration_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit_model, save_scatterplots
from dadi_inference    import fit_model as dadi_fit_model

# ── constants ───────────────────────────────────────────────────────────
PARAM_NAMES: List[str] = ["N0", "N1", "N2", "m12", "m21", "t_split"]

# ── helpers ─────────────────────────────────────────────────────────────
def _sample_one(priors: Dict[str, list[float]],
                rng:    np.random.Generator) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for k, (lo, hi) in priors.items():
        v = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            v = int(round(v))
        d[k] = v
    return d

def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    return [
        {**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
        for v, ll in zip(vecs, lls)
    ]

# ── one replicate ───────────────────────────────────────────────────────
def run_one(cfg: Dict[str, Any],
            params: Dict[str, float],
            out:    Path) -> Tuple[List[dict], List[dict]]:
    data_dir = out / "data";                 data_dir.mkdir(parents=True)
    mom_dir  = out / "inferences" / "moments"
    dadi_dir = out / "inferences" / "dadi"
    mom_dir.mkdir(parents=True)
    dadi_dir.mkdir(parents=True)

    # 1. demography figure
    g = split_migration_model(params)
    ax = demesdraw.tubes(g); ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_split_migration_model.png",
                dpi=300, bbox_inches="tight"); plt.close(ax.figure)

    # 2. simulation
    ts, g_sim = simulation(params, "split_migration", cfg)
    sfs = create_SFS(ts)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "split_migration_SFS.pkl").open("wb"))
    ts.dump(data_dir / "split_migration_tree_sequence.trees")

    # 3. inference
    start = moments.Misc.perturb_params([params[p] for p in PARAM_NAMES], 0.1)
    fits_mom,  lls_mom  = moments_fit_model(sfs, start, g_sim, cfg)
    fits_dadi, lls_dadi = dadi_fit_model(   sfs, start, g_sim, cfg)

    mom_dicts  = _attach_ll(fits_mom,  lls_mom)
    dadi_dicts = _attach_ll(fits_dadi, lls_dadi)

    pickle.dump(mom_dicts,  (mom_dir  / "fit_params.pkl").open("wb"))
    pickle.dump(dadi_dicts, (dadi_dir / "fit_params.pkl").open("wb"))
    return mom_dicts, dadi_dicts

# ── main driver ─────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("Split-migration multi-replicate runner")
    p.add_argument("--experiment_config", required=True, type=Path)
    p.add_argument("--num_draws", type=int, default=None,
                   help="Number of replicates (defaults to JSON)")
    for n in PARAM_NAMES:                      # CLI pins
        p.add_argument(f"--{n}", type=float, default=None)
    args = p.parse_args()

    cfg      = json.loads(args.experiment_config.read_text())
    priors   = cfg["priors"]
    rng      = np.random.default_rng(cfg.get("seed"))
    n_draw   = args.num_draws or cfg.get("num_draws", 1)

    base      = Path(cfg["demographic_model"]); base.mkdir(exist_ok=True)
    runs_root = base / "runs"; runs_root.mkdir(exist_ok=True)

    all_true, all_mom, all_dadi = [], [], []

    for idx in tqdm(range(1, n_draw + 1), desc="replicates"):
        params = _sample_one(priors, rng)
        for k in PARAM_NAMES:                 # apply CLI overrides
            pin = getattr(args, k)
            if pin is not None:
                params[k] = pin
        run_dir = runs_root / f"run_{idx:04d}"
        mom, dadi = run_one(cfg, params, run_dir)
        all_true.extend([params] * len(mom))
        all_mom.extend(mom)
        all_dadi.extend(dadi)

    # ── global scatter-plots ───────────────────────────────────────────
    inf_dir = base / "inferences"; inf_dir.mkdir(exist_ok=True)
    save_scatterplots(
        true_vecs=all_true,
        est_vecs=all_mom,
        ll_vec=[d["loglik"] for d in all_mom],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_moments_vs_true.png",
        label="moments",
    )
    save_scatterplots(
        true_vecs=all_true,
        est_vecs=all_dadi,
        ll_vec=[d["loglik"] for d in all_dadi],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_dadi_vs_true.png",
        label="dadi",
    )

if __name__ == "__main__":
    main()

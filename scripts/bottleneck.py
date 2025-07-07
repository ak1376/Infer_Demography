#!/usr/bin/env python3
"""
bottleneck.py – multi-replicate runner (harmonised style)
=========================================================
Workflow:

  1. Draw ground-truth parameters from JSON *priors*
     (CLI flags can pin any of them);
  2. Simulate data with msprime + demes;
  3. Run **moments** inference, starting from a
     10 %-jittered vector centred on the prior means;
  4. Save everything under  bottleneck/runs/run_XXXX/ …;
  5. After all replicates finish, write a scatter-plot
     coloured by log-likelihood.

JSON keys used
--------------
genome_length, mutation_rate, recombination_rate, demographic_model,
num_samples, seed, num_optimizations, top_k, num_draws, priors
"""

from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing   import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import demesdraw, moments, numpy as np
from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from simulation        import bottleneck_model, simulation, create_SFS
from moments_inference import (
    fit_model as moments_fit_model,
    save_scatterplots,
)
from dadi_inference    import fit_model as dadi_fit_model  # kept for later

# ── constants ──────────────────────────────────────────────────────────
PARAM_NAMES = [
    "N0",
    "N_bottleneck",
    "N_recover",
    "t_bottleneck_start",
    "t_bottleneck_end",
]

# ── helper functions ───────────────────────────────────────────────────
def _sample_one(priors: Dict[str, list[float]],
                rng:    np.random.Generator) -> Dict[str, float]:
    """Draw a single parameter vector from uniform priors."""
    d: Dict[str, float] = {}
    for k, (lo, hi) in priors.items():
        v = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            v = int(round(v))
        d[k] = v

    # ensure start > end for a proper bottleneck
    if d["t_bottleneck_start"] <= d["t_bottleneck_end"]:
        d["t_bottleneck_start"], d["t_bottleneck_end"] = (
            d["t_bottleneck_end"], d["t_bottleneck_start"]
        )
    return d


def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    """Attach log-likelihoods to optimised vectors."""
    return [
        {**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
        for v, ll in zip(vecs, lls)
    ]


# ── one replicate ───────────────────────────────────────────────────────
def run_one(
    cfg:    Dict[str, Any],
    params: Dict[str, float],
    out:    Path,
) -> Tuple[List[dict], List[dict]]:
    """
    Simulate + infer for one parameter set.
    Returns two lists of dicts: (moments fits, dadi fits).
    """

    # ----- directory scaffolding ---------------------------------------
    data_dir     = out / "data";                    data_dir.mkdir(parents=True)
    mom_dir      = out / "inferences" / "moments"
    dadi_dir     = out / "inferences" / "dadi"
    mom_log_dir  = out / "inferences" / "logs" / "moments"
    dadi_log_dir = out / "inferences" / "logs" / "dadi"
    for d in (mom_dir, dadi_dir, mom_log_dir, dadi_log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ----- 1. demography figure ----------------------------------------
    g = bottleneck_model(params)
    ax = demesdraw.tubes(g); ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_bottleneck_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # ----- 2. simulation ------------------------------------------------
    ts, _ = simulation(params, "bottleneck", cfg)
    sfs   = create_SFS(ts)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "bottleneck_SFS.pkl").open("wb"))
    ts.dump(data_dir / "bottleneck_tree_sequence.trees")

    # ----- 3. optimisation starts --------------------------------------
    # start_dict = {k: v for k, v in zip(PARAM_NAMES, params.values())}
    means      = [(lo + hi) / 2 for lo, hi in cfg["priors"].values()]
    start_dict = {k: v for k, v in zip(PARAM_NAMES, means)}
    start_dict = {
        k: v * np.random.uniform(0.9, 1.1)          # ±10 % jitter
        for k, v in start_dict.items()
    }

    # ----- 4. moments inference ----------------------------------------
    fits_mom, lls_mom = moments_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=bottleneck_model,
        experiment_config={**cfg, "log_dir": str(mom_log_dir)},
        sampled_params=params,  # attach ground truth
    )

    # ----- 5. dadi inference (optional – currently commented) ----------
    # fits_dadi, lls_dadi = dadi_fit_model(
    #     sfs,
    #     start_dict=start_dict,
    #     demo_model=bottleneck_model,
    #     experiment_config={**cfg, "log_dir": str(dadi_log_dir)},
    #     sampled_params=params,  # attach ground truth
    # )

    mom_dicts  = _attach_ll(fits_mom,  lls_mom)
    dadi_dicts = []  # _attach_ll(fits_dadi, lls_dadi)

    pickle.dump(mom_dicts,  (mom_dir  / "fit_params.pkl").open("wb"))
    # pickle.dump(dadi_dicts, (dadi_dir / "fit_params.pkl").open("wb"))
    return mom_dicts, dadi_dicts


# ── main driver ────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("Bottleneck multi-replicate runner")
    p.add_argument("--experiment_config", required=True, type=Path)
    p.add_argument("--num_draws", type=int, default=None,
                   help="How many replicates (default from JSON)")
    for n in PARAM_NAMES:                            # CLI pins
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
        params = _sample_one(priors, rng)            # ground truth
        for k in PARAM_NAMES:                        # CLI overrides
            pin = getattr(args, k)
            if pin is not None:
                params[k] = pin

        run_dir = runs_root / f"run_{idx:04d}"
        mom, dadi = run_one(cfg, params, run_dir)

        all_true.extend([params] * len(mom))
        all_mom.extend(mom)
        all_dadi.extend(dadi)

    # ----- scatter-plots -----------------------------------------------
    inf_dir = base / "inferences"; inf_dir.mkdir(exist_ok=True)
    save_scatterplots(
        true_vecs=all_true,
        est_vecs=all_mom,
        ll_vec   =[d["loglik"] for d in all_mom],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_moments_vs_true.png",
        label="moments",
    )
    # save_scatterplots(
    #     true_vecs=all_true,
    #     est_vecs=all_dadi,
    #     ll_vec   =[d["loglik"] for d in all_dadi],
    #     param_names=PARAM_NAMES,
    #     outfile=inf_dir / "scatter_dadi_vs_true.png",
    #     label="dadi",
    # )


if __name__ == "__main__":
    main()

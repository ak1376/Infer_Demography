#!/usr/bin/env python3
"""
split_isolation.py – Ray-parallel, toggle-able Moments / Dadi
=============================================================

Outputs
-------
experiments/split_isolation/runs/run_0001/…
experiments/split_isolation/inferences/scatter_{moments,dadi}_vs_true.png
"""

from __future__ import annotations
import argparse, json, pickle, sys, os
import demesdraw, moments, ray

# ── project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC          = PROJECT_ROOT / "src"
EXPERIMENTS  = PROJECT_ROOT / "experiments"
sys.path.append(str(SRC))                         # import local modules
import demesdraw, moments, numpy as np
from tqdm import tqdm
import ray   
                                        # ← NEW
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # one level above scripts/
SRC_DIR      = PROJECT_ROOT / "src"       
# ─────────────────────────── local imports ──────────────────────────────
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from simulation        import split_isolation_model, simulation, create_SFS
from moments_inference import fit_model as moments_fit, save_scatterplots
from dadi_inference    import fit_model as dadi_fit

PARAM_NAMES = ["N0", "N1", "N2", "m", "t_split"]

# ── Ray initialisation (no working-dir clone, just PYTHONPATH) ─────────
ray.init(
    logging_level="ERROR",
    runtime_env={"env_vars": {"PYTHONPATH": f"{PROJECT_ROOT}/src"}},
    ignore_reinit_error=True,
    num_cpus=5,  # adjust based on your system; 5 is a good default for 8-core CPUs
)

# ── helpers ────────────────────────────────────────────────────────────
def _sample_priors(priors: Dict[str, list[float]],
                   rng: np.random.Generator) -> Dict[str, float]:
    return {k: rng.uniform(*b) for k, b in priors.items()}

def _midpoint(priors: Dict[str, list[float]]) -> Dict[str, float]:
    return {k: (lo + hi) / 2 for k, (lo, hi) in priors.items()}

def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    return [{**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
            for v, ll in zip(vecs, lls)]

# ── Ray remote tasks ───────────────────────────────────────────────────
@ray.remote
def _simulate(params: Dict[str, float],
              cfg: Dict[str, Any]) -> Tuple[Any, "moments.Spectrum"]:
    ts, _ = simulation(params, "split_isolation", cfg)
    return ts, create_SFS(ts)

@ray.remote
def _moments(sfs, cfg, start_dict, params):
    return moments_fit(
        sfs,
        start_dict=start_dict,
        demo_model=split_isolation_model,
        experiment_config={**cfg, "log_dir": f"{cfg['log_dir']}/moments"}
# ───────────────────────────── helpers ──────────────────────────────────
def _sample_one(priors: Dict[str, list[float]],
                rng:    np.random.Generator) -> Dict[str, float]:
    draw: Dict[str, float] = {}
    for k, (lo, hi) in priors.items():
        v = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            v = int(round(v))
        draw[k] = v
    return draw

def _attach_ll(vecs: List[np.ndarray], lls: List[float]) -> List[dict]:
    return [
        {**dict(zip(PARAM_NAMES, v.tolist())), "loglik": ll}
        for v, ll in zip(vecs, lls)
    ]

# ────────────────────────── one replicate (Ray task) ────────────────────
@ray.remote
def run_one(cfg: Dict[str, Any],
            params: Dict[str, float],
            run_dir: str,                   # Path objects → str for Ray
            seed: int) -> Tuple[List[dict], List[dict]]:
    """Simulate + infer for one parameter set; returns (moments, dadi)."""
    run_dir = Path(run_dir)
    rng     = np.random.default_rng(seed)

    data_dir     = run_dir / "data";                    data_dir.mkdir(parents=True, exist_ok=True)
    mom_dir      = run_dir / "inferences" / "moments"
    dadi_dir     = run_dir / "inferences" / "dadi"
    mom_log_dir  = run_dir / "inferences" / "logs" / "moments"
    dadi_log_dir = run_dir / "inferences" / "logs" / "dadi"
    for d in (mom_dir, dadi_dir, mom_log_dir, dadi_log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. demography figure ----------------------------------------------
    g = split_isolation_model(params)
    ax = demesdraw.tubes(g); ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_split_isolation_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # 2. simulation -----------------------------------------------------
    ts, _ = simulation(params, "split_isolation", cfg)
    sfs = create_SFS(ts)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "split_isolation_SFS.pkl").open("wb"))
    ts.dump(data_dir / "split_isolation_tree_sequence.trees")

    # 3. independent start vector ---------------------------------------
    mean   = [(lo + hi) / 2 for lo, hi in cfg["priors"].values()]
    # start = mean.copy()  # use the mean of the prior as the start vector
    start  = moments.Misc.perturb_params(mean, 0.1)
    start_dict = {k: v for k, v in zip(PARAM_NAMES, mean)}

    # 4. inference -------------------------------------------------------
    fits_mom, lls_mom = moments_fit_model(
        sfs,
        start_dict=start_dict,
        demo_model=split_isolation_model,
        experiment_config={**cfg, "log_dir": str(mom_log_dir)},
    )

@ray.remote
def _dadi(sfs, cfg, start_dict, params):
    return dadi_fit(
        sfs,
        start_dict=start_dict,
        demo_model=split_isolation_model,
        experiment_config={**cfg, "log_dir": f"{cfg['log_dir']}/dadi"}
    )

# ── one replicate (Ray worker) ─────────────────────────────────────────
@ray.remote
def _run_one(idx: int, params: Dict[str, float],
             cfg: Dict[str, Any], base_dir: str,
             run_mom: bool, run_dadi: bool):

    run_dir  = Path(base_dir) / f"run_{idx:04d}"
    data_dir = run_dir / "data"; data_dir.mkdir(parents=True, exist_ok=True)
        experiment_config={**cfg, "log_dir": str(dadi_log_dir)},
    )

    mom_dicts  = _attach_ll(fits_mom,  lls_mom)
    dadi_dicts = _attach_ll(fits_dadi, lls_dadi)

    ts, sfs = ray.get(_simulate.remote(params, cfg))

    ax = demesdraw.tubes(split_isolation_model(params))
    ax.set_xlabel("Time"); ax.set_ylabel("N")
    plt.savefig(data_dir / "demes_split_isolation_model.png",
                dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    pickle.dump(params, (data_dir / "sampled_params.pkl").open("wb"))
    pickle.dump(sfs,    (data_dir / "split_isolation_SFS.pkl").open("wb"))
    ts.dump(data_dir / "split_isolation_tree_sequence.trees")
# ───────────────────────────── main driver ──────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser("Split-isolation (Ray-parallel) runner")
    p.add_argument("--experiment_config", required=True, type=Path)
    p.add_argument("--num_draws", type=int, default=None,
                   help="How many replicates (default from JSON)")
    for name in PARAM_NAMES:                               # CLI pins
        p.add_argument(f"--{name}", type=float, default=None)
    args = p.parse_args()

    cfg      = json.loads(args.experiment_config.read_text())
    priors   = cfg["priors"]
    rng_main = np.random.default_rng(cfg.get("seed"))
    n_draw   = args.num_draws or cfg.get("num_draws", 1)

    start_dict = _midpoint(cfg["priors"])
    mom_ref = dadi_ref = None
    if run_mom:  mom_ref  = _moments.remote(sfs, cfg, start_dict, params)
    if run_dadi: dadi_ref = _dadi.remote(sfs, cfg, start_dict, params)

    mom_dicts = dadi_dicts = []
    if mom_ref:
        fits, lls = ray.get(mom_ref)
        mom_dicts = _attach_ll(fits, lls)
        (run_dir / "inferences/moments").mkdir(parents=True, exist_ok=True)
        pickle.dump(mom_dicts, (run_dir / "inferences/moments/fit_params.pkl").open("wb"))
    if dadi_ref:
        fits, lls = ray.get(dadi_ref)
        dadi_dicts = _attach_ll(fits, lls)
        (run_dir / "inferences/dadi").mkdir(parents=True, exist_ok=True)
        pickle.dump(dadi_dicts, (run_dir / "inferences/dadi/fit_params.pkl").open("wb"))

    return mom_dicts, dadi_dicts, params

# ── CLI driver ─────────────────────────────────────────────────────────
def str2bool(s: str) -> bool:
    return s.lower() not in {"false", "0", "no"}

def main():
    cli = argparse.ArgumentParser("Split-isolation runner (Ray)")
    cli.add_argument("--experiment_config", required=True, type=Path)
    cli.add_argument("--num_draws", type=int)
    cli.add_argument("--run_moments", type=str2bool, default=True)
    cli.add_argument("--run_dadi",    type=str2bool, default=False)
    args = cli.parse_args()

    cfg    = json.loads(args.experiment_config.read_text())
    priors = cfg["priors"]
    draws  = args.num_draws or cfg.get("num_draws", 1)
    rng    = np.random.default_rng(cfg.get("seed"))

    runs_root = (EXPERIMENTS / cfg["demographic_model"] / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)

    futures = [_run_one.remote(i,
                               _sample_priors(priors, rng),
                               cfg, str(runs_root),
                               args.run_moments, args.run_dadi)
               for i in range(1, draws + 1)]

    all_true, all_mom, all_dadi = [], [], []
    for mom, dadi, true_par in ray.get(futures):
        if mom:  all_mom.extend(mom)
        if dadi: all_dadi.extend(dadi)
        all_true.extend([true_par] * max(len(mom), len(dadi), 1))

    inf_dir = (EXPERIMENTS / cfg["demographic_model"] / "inferences")
    inf_dir.mkdir(parents=True, exist_ok=True)

    if all_mom:
        save_scatterplots(
            all_true, all_mom,
            [d["loglik"] for d in all_mom],
            PARAM_NAMES,
            inf_dir / "scatter_moments_vs_true.png",
            label="moments",
        )
    if all_dadi:
        save_scatterplots(
            all_true, all_dadi,
            [d["loglik"] for d in all_dadi],
            PARAM_NAMES,
            inf_dir / "scatter_dadi_vs_true.png",
            label="dadi",
        )

if __name__ == "__main__" and os.getenv("RUNNING_RAY_WORKER") != "1":
    # ─── launch Ray -----------------------------------------------------
    ray.init(
        runtime_env={
            # ship the entire repo so every worker sees the same files
            "working_dir": str(PROJECT_ROOT),

            # and make “src/” importable
            "env_vars": {"PYTHONPATH": str(SRC_DIR)}
        }
    )
    futures = []
    true_vecs: list[dict[str, float]] = []

    for idx in range(1, n_draw + 1):
        params = _sample_one(priors, rng_main)     # ground truth
        for k in PARAM_NAMES:                      # CLI overrides
            manual = getattr(args, k)
            if manual is not None:
                params[k] = manual

        run_dir = runs_root / f"run_{idx:04d}"
        seed_i  = cfg.get("seed", 0) + idx

        futures.append(
            run_one.remote(cfg, params, str(run_dir), seed_i)
        )
        true_vecs.append(params)

    # gather results -----------------------------------------------------
    results = ray.get(futures)      # blocks until all replicates finish

    all_mom, all_dadi = [], []
    for (mom, dadi) in results:
        all_mom.extend(mom)
        all_dadi.extend(dadi)

    # save scatter-plots --------------------------------------------------
    inf_dir = base / "inferences"; inf_dir.mkdir(exist_ok=True)
    rep_mom   = len(all_mom)  // n_draw        # an int
    rep_dadi  = len(all_dadi) // n_draw

    save_scatterplots(
        true_vecs=true_vecs * rep_mom,         # list * int
        est_vecs=all_mom,
        ll_vec=[d["loglik"] for d in all_mom],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_moments_vs_true.png",
        label="moments",
    )

    save_scatterplots(
        true_vecs=true_vecs * rep_dadi,
        est_vecs=all_dadi,
        ll_vec=[d["loglik"] for d in all_dadi],
        param_names=PARAM_NAMES,
        outfile=inf_dir / "scatter_dadi_vs_true.png",
        label="dadi",
    )

if __name__ == "__main__":
    main()

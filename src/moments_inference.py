#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import moments
import nlopt
import numdifftools as nd


# ───────────────────────── diffusion helper ────────────────────────────
def _diffusion_sfs(
    log_space_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    param_names: List[str],
    sampled_demes: List[str],
    haploid_sizes: List[int],
    experiment_config: Dict[str, Any],
    *,
    return_graph: bool = False,   # <--- ADD
):
    real_space_vec = 10 ** log_space_vec
    p_dict = {k: float(v) for k, v in zip(param_names, real_space_vec)}

    graph = demo_model(p_dict)

    muL = float(experiment_config["mutation_rate"]) * float(experiment_config["genome_length"])
    N0 = float(p_dict[param_names[0]])
    theta = 4.0 * N0 * muL

    fs = moments.Spectrum.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=theta,
    )

    if return_graph:
        return fs, graph, p_dict, theta, muL   # <--- RETURN EXTRA STUFF
    return fs


# ───────────────────────── optimisation wrapper ────────────────────────
def fit_model(
    sfs: moments.Spectrum,
    start_vec: np.ndarray,
    demo_model: Callable[[Dict[str, float]], Any],
    experiment_config: Dict[str, Any],
    param_order: Optional[List[str]] = None,
    verbose: bool = False,
    rtol: float = 1e-8,
    eps: float = 1e-300,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run a **single** moments optimisation (nlopt.LD_LBFGS) in log10 space.
    Returns:
      - best_params: list with one vector of fitted REAL-space params (in param_order)
      - best_lls:    list with one max log-likelihood value
    """

    assert isinstance(sfs, moments.Spectrum), "sfs must be a moments.Spectrum"

    # ---- parameter order (single source of truth) ----
    if param_order is None:
        param_order = list(experiment_config["parameter_order"])
    param_names = list(param_order)

    # ---- bounds ----
    priors = experiment_config.get("priors", experiment_config.get("parameters", {}))
    if not priors:
        raise ValueError("experiment_config must include 'priors' (or 'parameters').")

    lb_full = np.array([float(priors[p][0]) for p in param_names], dtype=float)
    ub_full = np.array([float(priors[p][1]) for p in param_names], dtype=float)

    if np.any(lb_full <= 0) or np.any(ub_full <= 0):
        bad = [p for p, lo, hi in zip(param_names, lb_full, ub_full) if lo <= 0 or hi <= 0]
        raise ValueError(f"All bounds must be positive for log10 optimization. Bad: {bad}")

    # ---- SFS axis / demes order ----
    sampled_demes = list(getattr(sfs, "pop_ids", []))
    if not sampled_demes:
        raise ValueError("Observed SFS has no pop_ids; cannot infer sampled_demes order.")

    # DEBUG STUFF
    # print("sfs.pop_ids:", getattr(sfs, "pop_ids", None))
    # print("sfs.shape:", sfs.shape)
    # print("config num_samples keys:", list(experiment_config["num_samples"].keys()))
    # print("config demographic_model expects demes named:", ["YRI","CEU"])  # or whatever your model uses
    # print("folded?", getattr(sfs, "folded", None))

    haploid_sizes = [n - 1 for n in sfs.shape]

    # ---- objective: Poisson composite log-likelihood ----
    def loglikelihood(log10_params: np.ndarray) -> float:
        exp_sfs = _diffusion_sfs(
            log_space_vec=log10_params,
            demo_model=demo_model,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            experiment_config=experiment_config,
        )
        # log(exp + eps) avoids -inf when exp has zeros
        return float(np.sum(np.log(np.asarray(exp_sfs) + eps) * np.asarray(sfs) - np.asarray(exp_sfs)))

    grad_fn = nd.Gradient(loglikelihood, n=1, step=1e-4)

    def objective(log10_params: np.ndarray, grad: np.ndarray) -> float:
        ll = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"loglik: {ll:.6g}  log10_params: {log10_params}")
        return ll  # maximize

    # ---- optimizer setup ----
    start_vec = np.asarray(start_vec, dtype=float)
    if start_vec.shape != (len(param_names),):
        raise ValueError(f"start_vec shape {start_vec.shape} != ({len(param_names)},)")

    opt = nlopt.opt(nlopt.LD_LBFGS, start_vec.size)
    opt.set_lower_bounds(np.log10(lb_full))
    opt.set_upper_bounds(np.log10(ub_full))
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)



    # --- DEBUG: expected under ground truth vs observed ---
    gt_path = experiment_config.get("ground_truth_path", None)  # or pass gt dict directly
    # if you already loaded gt_params in runner, pass it in; otherwise load here.

    # Suppose you can load gt dict here for now:
    import pickle
    from pathlib import Path
    gt = pickle.load(open("/sietch_colab/akapoor/Infer_Demography/experiments/IM_asymmetric/simulations/0/sampled_params.pkl","rb"))

    x_true = np.log10(np.array([gt[p] for p in param_names], dtype=float))

    exp_true, g_true, p_true, theta_used, muL = _diffusion_sfs(
        log_space_vec=x_true,
        demo_model=demo_model,
        param_names=param_names,
        sampled_demes=sampled_demes,
        haploid_sizes=haploid_sizes,
        experiment_config=experiment_config,
        return_graph=True,
    )


    # obs = np.asarray(sfs).ravel()
    # exp = np.asarray(exp_true).ravel()

    # # normalize to probabilities
    # obs_p = obs / obs.sum()
    # exp_p = exp / exp.sum()

    # # compare shapes (ignore overall scale)
    # print("L1 shape diff:", np.sum(np.abs(obs_p - exp_p)))
    # print("corr(log):", np.corrcoef(np.log(obs_p + 1e-300), np.log(exp_p + 1e-300))[0,1])

    # obs = np.asarray(sfs)
    # exp = np.asarray(exp_true)

    # print("DEBUG sum(obs):", obs.sum())
    # print("DEBUG sum(exp_true):", exp.sum())
    # print("DEBUG ratio exp/obs:", exp.sum() / obs.sum())

    # # scatter sanity
    # mask = (obs > 0) & (exp > 0)
    # print("DEBUG obs>0:", np.sum(obs > 0), " exp>0:", np.sum(exp > 0), " both>0:", np.sum(mask))
    # print("DEBUG exp==0 where obs>0:", np.sum((obs > 0) & (exp == 0)))

    # -------------------- NEW DEBUG: theta scaling check --------------------
    # 1) Expected SFS at theta=1 (per-unit-theta SFS)
    exp_theta1 = moments.Spectrum.from_demes(
        g_true,
        sampled_demes=sampled_demes,
        sample_sizes=haploid_sizes,
        theta=1.0,
    )
    base_sum = float(np.asarray(exp_theta1).sum())
    obs_sum  = float(np.asarray(sfs).sum())
    print("DEBUG sum(exp_theta1):", base_sum)

    # 2) Effective theta needed to match observed total mass
    theta_eff = obs_sum / base_sum
    print("DEBUG theta_eff to match obs sum:", theta_eff)

    # 3) Effective Ne implied by theta_eff = 4*Ne*muL
    Ne_eff = theta_eff / (4.0 * muL)
    N0 = float(p_true[param_names[0]])  # your assumed reference Ne (e.g. N_anc)
    print("DEBUG muL:", muL)
    print("DEBUG Ne_eff implied by moments-from_demes:", Ne_eff)
    print("DEBUG N0 used in your theta:", N0)
    print("DEBUG ratio Ne_eff/N0:", Ne_eff / N0)
    print("DEBUG theta_used (your code):", theta_used)
    print("DEBUG theta_eff/theta_used:", theta_eff / theta_used)
    # ------------------------------------------------------------------------

    # ---- run optimization ----
    x0 = np.log10(start_vec)
    xhat = opt.optimize(x0)

    ll_hat = loglikelihood(xhat)
    fitted_real = 10 ** xhat

    # ---- optional: save fitted expected SFS ----
    out_dir = experiment_config.get("out_dir", None)
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fitted_sfs = _diffusion_sfs(
            log_space_vec=xhat,
            demo_model=demo_model,
            param_names=param_names,
            sampled_demes=sampled_demes,
            haploid_sizes=haploid_sizes,
            experiment_config=experiment_config,
        )
        np.save(out_path / "expected_sfs_fitted.npy", np.asarray(fitted_sfs))

        # also save fitted params for sanity
        fitted_dict = {p: float(v) for p, v in zip(param_names, fitted_real)}
        (out_path / "fitted_params.json").write_text(
            __import__("json").dumps(fitted_dict, indent=2, sort_keys=False)
        )

    return [fitted_real], [ll_hat]

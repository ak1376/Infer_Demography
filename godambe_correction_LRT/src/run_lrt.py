#!/usr/bin/env python3
# godambe_correction_LRT/src/run_lrt.py

"""
Logic for the Godambe-corrected composite-likelihood LRT for CO growth, one arm.

Two parameterizations are supported (choose with --alt-model on the CLI):

  co_grow_from_anc  (Option A, DEFAULT, identifiable)
      null co_const_at_anc : CO constant at N_ANC          (6 params)
      alt  co_grow_from_anc: CO grows N_ANC -> N_CO1        (7 params)
      N_CO0 is tied to N_ANC, so there is no redundant deep-time CO size and the
      Godambe H is well conditioned. df = 1 (freeing N_CO1 from N_ANC).

  split_migration_growth_both  (original, UNIDENTIFIABLE -- for diagnosis only)
      null split_migration_growth : CO constant at free N_CO   (7 params)
      alt  split_migration_growth_both: free N_CO0 AND N_CO1   (8 params)
      N_CO0 is redundant with N_ANC -> singular H -> negative adjust. Kept so the
      conditioning diagnostic can demonstrate the breakdown.

Pipeline
  raw statistic   D      = 2 * (ll_alt - ll_null)
  adjustment      c      = df / trace(J H^-1)      (moments.LD.Godambe.LRT_adjust)
  corrected stat  D_adj  = c * D
  p-value         one-sided growth (boundary):  0.5 * chi2.sf(D_adj, 1)

Always prints the conditioning of H (cond(H), eigenvalue spread, trace(J H^-1)):
if H is ill-conditioned the corrected test is not trustworthy. --validate-only
checks the model_func scaling against a hand-built theory curve (needs just the
null fit).

CLI wrapper: godambe_correction_LRT/snakemake_scripts/run_lrt.py
"""

import copy
import pickle
import logging

import numpy as np
import moments
from moments.LD import Inference
from moments.LD.Godambe import (
    _get_statistics_and_remove_normalization,
    _get_godambe,
)

import compute_J_ld as cj                        # R_BINS, POPULATIONS, NORMALIZATION, ll_from_theory
from bootstrap_ld import average_ld_structure
from momentsld_inference import (
    MODELS, MODEL_FUNCS, NULL, ALT, CO_CONST, CO_GROW,
)

logging.getLogger().setLevel(logging.WARNING)

POPULATIONS = cj.POPULATIONS
NORMALIZATION = cj.NORMALIZATION                 # 0

# For each alt model: the matching null model, how to set the alt-only params to
# their null values when embedding the null fit (embed: alt_param -> null_param),
# and the nested (freed) params (len = df). Default fit-dir tags too.
TESTS = {
    CO_GROW: {
        "null_model": CO_CONST,
        "embed":  {"N_CO1": "N_ANC"},            # null: CO constant at N_ANC
        "nested": ["N_CO1"],
        "null_tag": "coconst", "alt_tag": "cogrow",
    },
    ALT: {
        "null_model": NULL,
        "embed":  {"N_CO0": "N_CO", "N_CO1": "N_CO"},   # null: CO constant at free N_CO
        "nested": ["N_CO1"],
        "null_tag": "simple", "alt_tag": "complex",
    },
}


def make_model_func(alt_model):
    """Native model_func(params, rho, theta) for moments bin_stats / LRT_adjust.
    Returns RAW LDstats (LRT_adjust applies sigmaD2 itself). The model wrapper
    (e.g. co_grow_from_anc_model) applies any parameter tying internally.
    """
    names = MODELS[alt_model]
    build = MODEL_FUNCS[alt_model]

    def model_func(params, rho=None, theta=0.001, **kwargs):
        graph = build(dict(zip(names, params)))
        if rho is None:
            rho = [0.0, 1.0]                     # names()-only path; values irrelevant
        return moments.Demes.LD(graph, sampled_demes=POPULATIONS, rho=rho)

    return model_func, names


def theory_for(param_dict, alt_model):
    """σD²-normalized theory curve (hand Simpson binning) -- the validation ref."""
    build = MODEL_FUNCS[alt_model]
    graph = build(param_dict)
    ref = float(param_dict["N_ANC"])
    rho_edges = 4.0 * ref * np.asarray(cj.R_BINS)
    ld_edges = moments.Demes.LD(graph, sampled_demes=POPULATIONS, rho=rho_edges)
    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=POPULATIONS, rho=rho_mids)
    ld_bins = [(ld_edges[i] + ld_edges[i + 1] + 4 * ld_mids[i]) / 6.0
               for i in range(len(rho_mids))]
    ld_bins.append(ld_edges[-1])
    ld_stats = moments.LD.LDstats(ld_bins, num_pops=ld_edges.num_pops,
                                  pop_ids=ld_edges.pop_ids)
    return moments.LD.Inference.sigmaD2(ld_stats)


def embed_null(null_params, alt_model):
    """Null MLE expressed in the alt parameterization (+ Ne appended)."""
    spec = TESTS[alt_model]
    names = MODELS[alt_model]
    embed = spec["embed"]
    d = {}
    for n in names:
        if n in embed:
            d[n] = float(null_params[embed[n]])  # alt-only param -> its null value
        else:
            d[n] = float(null_params[n])
    p0 = [d[n] for n in names] + [float(null_params["N_ANC"])]   # last = Ne
    nested_indices = [names.index(x) for x in spec["nested"]]
    return p0, d, nested_indices


def build_all_boot(ld_stats_dir, n_boot, seed):
    windows = {}
    for p in ld_stats_dir.glob("LD_stats_window_*.pkl"):
        s = pickle.load(p.open("rb"))
        if isinstance(s, dict) and s.get("empty"):
            continue
        windows[int(p.stem.split("_")[-1])] = s
    if not windows:
        raise RuntimeError(f"No (non-empty) LD tiles in {ld_stats_dir}")
    win_list = list(windows.values())
    ld_names, h_names = win_list[0]["stats"]
    nn = len(win_list)
    rng = np.random.default_rng(seed)
    all_boot = [
        average_ld_structure([win_list[i] for i in rng.integers(0, nn, size=nn)],
                             ld_names, h_names)
        for _ in range(n_boot)
    ]
    return all_boot, nn


def validate_model_func(model_func, alt_model, p0, alt_dict):
    rs = np.asarray(cj.R_BINS)
    rho = 4.0 * p0[-1] * rs
    y = Inference.bin_stats(model_func, p0[:-1], rho=rho)
    y = Inference.sigmaD2(y, normalization=NORMALIZATION)
    ref = theory_for(alt_dict, alt_model)
    diffs = []
    for a, b in zip(y, ref):
        a = np.asarray(a, float); b = np.asarray(b, float)
        denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
        diffs.append(np.max(np.abs(a - b) / denom))
    max_rel = float(np.max(diffs))
    print(f"--- model_func validation (bin_stats vs hand theory) ---")
    print(f"  max relative diff: {max_rel:.3e}  {'PASS' if max_rel < 1e-6 else 'FAIL'}")
    return max_rel < 1e-6


def diagnose_godambe(model_func, all_boot, p0, means, varcovs, eps, r_edges,
                     normalization, pass_Ne):
    rs = np.asarray(r_edges)
    ms, vcs, boots = copy.deepcopy(means), copy.deepcopy(varcovs), copy.deepcopy(all_boot)
    statistics, ms, vcs, boots = _get_statistics_and_remove_normalization(
        model_func, p0, ms, vcs, boots, normalization, pass_Ne)

    def pass_func(params, statistics):
        rho = 4 * params[-1] * rs
        y = Inference.bin_stats(model_func, params if pass_Ne else params[:-1], rho=rho)
        y = Inference.sigmaD2(y, normalization=normalization)
        y = Inference.remove_nonpresent_statistics(y, statistics)
        return y

    _, H, J, _ = _get_godambe(pass_func, boots, p0, ms, vcs, eps, statistics, log=False)
    H = np.asarray(H); J = np.asarray(J)
    Hs = 0.5 * (H + H.T)
    eig = np.sort(np.abs(np.linalg.eigvalsh(Hs)))
    cond = np.linalg.cond(H)
    tr = float(np.trace(J @ np.linalg.inv(H)))
    print("\n--- FULL-matrix Godambe (what moments.LD.Godambe.LRT_adjust does) ---")
    print(f"  H is {H.shape[0]}x{H.shape[0]} (all params incl. Ne)")
    print(f"  cond(H)               = {cond:.3e}   ({'ILL-CONDITIONED' if cond > 1e8 else 'ok'})")
    print(f"  |eig(H)| min / max    = {eig[0]:.3e} / {eig[-1]:.3e}   ratio {eig[-1]/max(eig[0],1e-300):.2e}")
    print(f"  trace(J H^-1)         = {tr:.4g}   ->  full-matrix adjust = {1.0/tr if tr != 0 else float('inf'):.4g}"
          "  [WRONG: not restricted to tested param]")
    return cond, tr


def restricted_adjust(model_func, all_boot, p0, nested_indices, means, varcovs,
                      eps, r_edges, normalization, pass_Ne, log=False):
    """SFS-style Godambe LRT adjustment: H and J restricted to the NESTED
    (tested) parameters, holding the rest fixed at p0 -- exactly what
    moments.Godambe.LRT_adjust (SFS) does via its diff_func.

    moments.LD.Godambe.LRT_adjust instead builds H, J over the FULL parameter
    matrix; that matrix is indefinite at the null-embedded p0 (the nuisance
    directions are not at a max there), so trace(J H^-1) can go negative and the
    adjustment is nonsensical. Restricting to the tested direction avoids this and
    correctly reduces to adjust=1 for a true (non-composite) likelihood.

    adjust = len(nested_indices) / trace(J_nested H_nested^-1).
    """
    rs = np.asarray(r_edges)
    ms, vcs, boots = copy.deepcopy(means), copy.deepcopy(varcovs), copy.deepcopy(all_boot)
    statistics, ms, vcs, boots = _get_statistics_and_remove_normalization(
        model_func, p0, ms, vcs, boots, normalization, pass_Ne)

    def pass_func(params, statistics):
        rho = 4 * params[-1] * rs
        y = Inference.bin_stats(model_func, params if pass_Ne else params[:-1], rho=rho)
        y = Inference.sigmaD2(y, normalization=normalization)
        return Inference.remove_nonpresent_statistics(y, statistics)

    p0arr = np.asarray(p0, dtype=float)

    def diff_func(diff_params, statistics):                 # vary ONLY nested params
        full = p0arr.copy()
        full[nested_indices] = diff_params
        return pass_func(full, statistics)

    p_nested = p0arr[nested_indices]
    _, H, J, _ = _get_godambe(diff_func, boots, p_nested, ms, vcs, eps, statistics, log=log)
    H = np.asarray(H); J = np.asarray(J)
    tr = float(np.trace(J @ np.linalg.inv(H)))
    adjust = len(nested_indices) / tr
    print("\n--- RESTRICTED Godambe (tested param only -- the correct adjustment) ---")
    print(f"  H_nested = {H.ravel()}   J_nested = {J.ravel()}")
    print(f"  trace(J H^-1)         = {tr:.4g}   ->  adjust = {adjust:.4g}")
    return adjust, tr

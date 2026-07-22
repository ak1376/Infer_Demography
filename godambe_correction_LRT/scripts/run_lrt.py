#!/usr/bin/env python3
# godambe_correction_LRT/scripts/run_lrt.py

"""
Godambe-corrected composite-likelihood LRT for CO growth, one arm.

Two parameterizations are supported (choose with --alt-model):

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
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import copy
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import moments
from moments.LD import Inference
from moments.LD.Godambe import (
    _get_statistics_and_remove_normalization,
    _get_godambe,
)
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def main():
    ap = argparse.ArgumentParser(description="Godambe-corrected CO-growth LRT for one arm.")
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--alt-model", default=CO_GROW, choices=[CO_GROW, ALT],
                    help="which alt model / parameterization (default %(default)s)")
    ap.add_argument("--blocksize", type=int, default=100_000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--null-fit", type=Path, default=None,
                    help="override null best_fit.pkl (default: <arm>/fit_<null_tag>/best_fit.pkl)")
    ap.add_argument("--alt-fit", type=Path, default=None,
                    help="override alt best_fit.pkl (default: <arm>/fit_<alt_tag>/best_fit.pkl)")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    spec = TESTS[args.alt_model]
    arm = args.arm_dir
    model_func, alt_names = make_model_func(args.alt_model)

    null_path = args.null_fit or (arm / f"fit_{spec['null_tag']}" / "best_fit.pkl")
    null = pickle.load(null_path.open("rb"))
    null_params = null["best_params"]
    ll_null = float(null["best_lls"])
    p0, alt_dict, nested_indices = embed_null(null_params, args.alt_model)

    print(f"arm            : {arm.name}")
    print(f"test           : null={spec['null_model']}  ->  alt={args.alt_model}")
    print(f"p0 ({len(p0)}-vec, +Ne): {['%.4g' % v for v in p0]}")
    print(f"nested_indices : {nested_indices}  (df={len(nested_indices)})\n")

    ok = validate_model_func(model_func, args.alt_model, p0, alt_dict)
    if args.validate_only:
        return
    if not ok:
        raise SystemExit("model_func validation FAILED -- fix scaling first.")

    alt_path = args.alt_fit or (arm / f"fit_{spec['alt_tag']}" / "best_fit.pkl")
    if not alt_path.exists():
        print(f"\n[waiting] {alt_path} not present yet.")
        return
    alt = pickle.load(alt_path.open("rb"))
    p_alt = alt["best_params"]
    ll_alt = float(alt["best_lls"])

    mv = pickle.load((arm / "overlap" / "means.varcovs.pkl").open("rb"))
    ld_dir = arm / "sweep" / f"bs{args.blocksize}" / "LD_stats"
    all_boot, nn = build_all_boot(ld_dir, args.n_boot, args.seed)
    print(f"\nall_boot: {len(all_boot)} replicates from {nn} tiles @ {args.blocksize} bp")

    # full-matrix version (moments LD) -- shown only to document the breakdown
    cond, tr_full = diagnose_godambe(model_func, all_boot, p0, mv["means"], mv["varcovs"],
                                     args.eps, np.asarray(cj.R_BINS), NORMALIZATION, pass_Ne=False)
    # restricted version (SFS-style) -- the one we actually use
    c, tr = restricted_adjust(
        model_func, all_boot, p0, nested_indices, mv["means"], mv["varcovs"],
        args.eps, np.asarray(cj.R_BINS), NORMALIZATION, pass_Ne=False)

    D = 2.0 * (ll_alt - ll_null)
    D_adj = c * D
    df = len(nested_indices)
    p_two = float(chi2.sf(D_adj, df=df)) if D_adj > 0 else 1.0
    p_one = 0.5 * p_two

    # growth direction (for co_grow_from_anc: N_CO1 vs N_ANC; else N_CO1 vs N_CO0)
    n1 = float(p_alt.get("N_CO1"))
    n0 = float(p_alt.get("N_CO0", p_alt.get("N_ANC")))
    grew = n1 > n0
    G_CO = np.log(n1 / n0) / float(p_alt["T"])

    print("\n" + "=" * 64)
    print(f"ll_null={ll_null:.4f}  ll_alt={ll_alt:.4f}")
    print(f"raw   D      = 2*(ll_alt-ll_null) = {D:.4f}")
    print(f"adjust c     = df/trace(J H^-1)   = {c:.4g}   (restricted; SFS-style)"
          + ("   [INVALID: adjust<=0]" if c <= 0 else ""))
    print(f"corrected    D_adj = c*D          = {D_adj:.4f}")
    print(f"CO size      : {n0:.4g} -> N_CO1={n1:.4g}  ({'GREW' if grew else 'shrank/flat'}, "
          f"G_CO={G_CO:.3g}/gen)")
    print(f"p-value (two-sided chi2_{df})       = {p_two:.4g}")
    print(f"p-value (one-sided, growth test)  = {p_one:.4g}"
          + ("" if grew else "   [MLE is not growth -> growth not supported]"))
    print("=" * 64)

    out = arm / f"lrt_result_{spec['alt_tag']}.pkl"
    with out.open("wb") as f:
        pickle.dump({"alt_model": args.alt_model, "D": D, "adjust": float(c),
                     "D_adj": D_adj, "cond_H": float(cond), "trace_JHinv": tr,
                     "p_two_sided": p_two, "p_one_sided": p_one,
                     "ll_null": ll_null, "ll_alt": ll_alt, "grew": grew,
                     "G_CO": float(G_CO), "blocksize": args.blocksize}, f)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

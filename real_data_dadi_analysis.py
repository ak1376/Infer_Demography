#!/usr/bin/env python3
"""
real_data_dadi_analysis.py – single-run dadi optimisation on real data (no Demes)
* Poisson objective with θ re-estimated each eval: θ̂ = sum(data)/sum(model).
* Adds 2×2 plot: observed, model (θ̂·model), residuals, residual histogram.
"""

from pathlib import Path
from collections import OrderedDict
import argparse, importlib, json, pickle, datetime
import numpy as np
import matplotlib.pyplot as plt                      # <-- NEW
import dadi
import nlopt
import numdifftools as nd
import src.simulation

# ── Dadi-compatible split migration model with YRI/CEU demes ──
def split_migration_model_dadi(params):
    N0, YRI, CEU, m12, m21, t_split = params
    import demes
    b = demes.Builder()
    b.add_deme("N0", epochs=[dict(start_size=N0, end_time=t_split)])
    b.add_deme("YRI", ancestors=["N0"], epochs=[dict(start_size=YRI)])
    b.add_deme("CEU", ancestors=["N0"], epochs=[dict(start_size=CEU)])
    b.add_migration(source="YRI", dest="CEU", rate=m12)
    b.add_migration(source="CEU", dest="YRI", rate=m21)
    g = b.resolve()
    return g

# ── Wrapper for split_migration_model to match optimizer signature ──
def split_migration_model_wrapper(params, ns, pts, pop_ids=None):
    graph = split_migration_model_dadi(params)
    if pop_ids is None:
        pop_ids = ["YRI", "CEU"]
    fs = dadi.Spectrum.from_demes(
        graph,
        sample_sizes=ns,
        sampled_demes=pop_ids,
        pts=pts
    )
    return fs

# ── SFS loading helpers ───────────────────────────────────────────
def load_sfs_from_vcf(vcf: Path, popfile: Path, pop_ids: list[str], ns: list[int], folded: bool) -> dadi.Spectrum:
    dd = dadi.Misc.make_data_dict_vcf(str(vcf), str(popfile))
    fs = dadi.Spectrum.from_data_dict(dd, pop_ids, ns)
    return fs.fold() if folded else fs

def load_sfs_from_file(sfs_path: Path) -> dadi.Spectrum:
    if sfs_path.suffix == ".fs":
        return dadi.Spectrum.from_file(str(sfs_path))
    return pickle.loads(sfs_path.read_bytes())

def dynamic_pts(ns: list[int]) -> list[int]:
    nmax = max(ns)
    return [nmax + 20, nmax + 40, nmax + 60]

# ── NEW: model + θ̂ helper (matches objective’s scaling) ─────────
def expected_with_theta_hat(model_func, params, ns, pts_l, folded, sfs_obs: dadi.Spectrum):
    func_ex = dadi.Numerics.make_extrap_func(model_func)
    model = func_ex(params, ns, pts_l)
    if folded:
        model = model.fold()
    total_model = float(np.sum(model))
    if not np.isfinite(total_model) or total_model <= 0:
        return None, None
    theta_hat = float(np.sum(sfs_obs)) / total_model
    expected = theta_hat * model
    return expected, theta_hat

# ── NEW: plotting function for 2×2 panel ─────────────────────────
def plot_sfs_comparison(obs: dadi.Spectrum,
                        exp: dadi.Spectrum,
                        out_path: Path | None = None,
                        title: str | None = None,
                        log10: bool = False):
    """
    obs: observed 2D SFS (dadi.Spectrum)
    exp: expected 2D SFS (already scaled by θ̂)
    """
    # Convert to plain arrays; mask monomorphic corners (0,0) and (n1,n2) if present
    O = np.array(obs, dtype=float)
    E = np.array(exp, dtype=float)
    eps = 1e-12

    # Standardized residuals: (O - E) / sqrt(E)
    std_res = (O - E) / np.sqrt(np.maximum(E, eps))
    std_res_flat = std_res[np.isfinite(std_res)].ravel()

    # Optionally log-transform counts for display only
    disp_O = np.log10(O + 1.0) if log10 else O
    disp_E = np.log10(E + 1.0) if log10 else E

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.25, wspace=0.25)

    # Observed
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(disp_O.T, origin="lower", aspect="auto")
    ax1.set_title("Observed SFS" + (" (log10)" if log10 else ""))
    ax1.set_xlabel("Pop 1 derived counts"); ax1.set_ylabel("Pop 2 derived counts")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Expected
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(disp_E.T, origin="lower", aspect="auto")
    ax2.set_title("Model SFS (θ̂·model)" + (" (log10)" if log10 else ""))
    ax2.set_xlabel("Pop 1 derived counts"); ax2.set_ylabel("Pop 2 derived counts")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Residuals
    ax3 = fig.add_subplot(gs[1, 0])
    # Diverging colormap centered at 0
    vmax = np.nanpercentile(np.abs(std_res), 98)
    im3 = ax3.imshow(std_res.T, origin="lower", aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax3.set_title("Standardized residuals (O−E)/√E")
    ax3.set_xlabel("Pop 1 derived counts"); ax3.set_ylabel("Pop 2 derived counts")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Histogram of residuals
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(std_res_flat, bins=30)
    ax4.set_title("Residuals histogram")
    ax4.set_xlabel("Standardized residual"); ax4.set_ylabel("Frequency")

    if title:
        fig.suptitle(title, y=0.995, fontsize=12)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"saved plot -> {out_path}")
    else:
        plt.show()
    plt.close(fig)

# ── main fitting (unchanged core) ─────────────────────────────────
def fit_model(
    sfs: dadi.Spectrum,
    start_dict: dict[str, float],
    model_func,                       # (params, ns, pts) -> Spectrum
    priors: dict[str, list[float]],
    fixed_params: dict[str, float] | None = None,
    folded: bool = False,
):
    param_names = list(start_dict.keys())
    p0          = np.array([start_dict[p] for p in param_names], dtype=float)
    lower_b     = np.array([priors[p][0] for p in param_names], dtype=float)
    upper_b     = np.array([priors[p][1] for p in param_names], dtype=float)

    ns = list(sfs.sample_sizes)
    pts_l = dynamic_pts(ns)
    func_ex = dadi.Numerics.make_extrap_func(model_func)

    fixed_params = fixed_params or {}
    free_idx  = [i for i, n in enumerate(param_names) if n not in fixed_params]
    fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
    for name, val in fixed_params.items():
        lo, hi = priors[name]
        if not (lo <= val <= hi):
            raise ValueError(f"Fixed param {name}={val} outside bounds [{lo}, {hi}]")
    fixed_values_log10 = np.zeros(len(param_names))
    for i in fixed_idx:
        fixed_values_log10[i] = np.log10(max(float(fixed_params[param_names[i]]), 1e-300))

    seed        = dadi.Misc.perturb_params(p0, fold=0.1)
    start_log10 = np.log10(np.maximum(seed, 1e-300))
    lower_log10 = np.log10(np.maximum(lower_b, 1e-300))
    upper_log10 = np.log10(np.maximum(upper_b, 1e-300))

    free_start = start_log10[free_idx]
    free_lower = lower_log10[free_idx]
    free_upper = upper_log10[free_idx]

    def expand_params_log10(free_params_log10: np.ndarray) -> np.ndarray:
        full = np.zeros(len(param_names))
        full[free_idx]  = free_params_log10
        full[fixed_idx] = fixed_values_log10[fixed_idx]
        return full

    def objective_function(free_params_log10, gradient):
        full_log10 = expand_params_log10(free_params_log10)
        params = 10 ** full_log10
        try:
            model = func_ex(params, ns, pts_l)
            if folded:
                model = model.fold()
            total_model = np.sum(model)
            if total_model <= 0 or not np.isfinite(total_model):
                return -np.inf
            theta_hat = np.sum(sfs) / total_model
            expected = theta_hat * model
            m = np.maximum(expected, 1e-300)
            ll = np.sum(sfs * np.log(m) - m)
            if gradient.size > 0:
                pass
            print(f"[LL={ll:.6g}] log10_free={np.array2string(np.asarray(free_params_log10), precision=4)}")
            return ll
        except Exception as e:
            print("Error in objective:", e)
            return -np.inf

    print("▶ dadi custom NLopt optimisation started –", datetime.datetime.now().isoformat(timespec='seconds'))
    print("  lower bounds:", lower_b.tolist())
    print("  upper bounds:", upper_b.tolist())
    if fixed_params:
        print("  fixing parameters:", fixed_params)

    opt = nlopt.opt(nlopt.LN_COBYLA, len(free_start))
    opt.set_lower_bounds(free_lower)
    opt.set_upper_bounds(free_upper)
    opt.set_max_objective(objective_function)
    opt.set_ftol_rel(1e-8)
    opt.set_maxeval(10000)

    try:
        best_free_log10 = opt.optimize(free_start)
        best_ll = opt.last_optimum_value()
        status = opt.last_optimize_result()
    except Exception as e:
        print("Optimization failed:", e)
        best_free_log10 = free_start
        best_ll = objective_function(free_start, np.array([]))
        status = nlopt.FAILURE

    best_full_log10 = expand_params_log10(best_free_log10)
    best_params = 10 ** best_full_log10

    print("✔ finished –", datetime.datetime.now().isoformat(timespec='seconds'))
    print("  status :", status)
    print("  LL     :", best_ll)
    print("  params :", {k: v for k, v in zip(param_names, best_params)})

    # return both for plotting downstream
    return [best_params], [best_ll], ns, pts_l, func_ex

# ── CLI ───────────────────────────────────────────────────────────
def main():
    cli = argparse.ArgumentParser("Standalone dadi single-fit on real data (adds comparison plot)")
    src = cli.add_mutually_exclusive_group(required=True)
    src.add_argument("--sfs-file", type=Path, help="Pickled Spectrum or .fs")
    src.add_argument("--vcf",      type=Path, help="VCF/BCF (optionally gzipped)")
    cli.add_argument("--popfile",  type=Path, help="Popfile (VCF mode)")
    cli.add_argument("--pop-ids",  nargs="+", help="Pop IDs in order (VCF mode)")
    cli.add_argument("--ns",       nargs="+", type=int, help="Sample sizes per pop (VCF mode)")
    cli.add_argument("--folded",   action="store_true", help="Fold data & model")
    cli.add_argument("--config",   required=True, type=Path, help="JSON with 'priors' and optional 'start'")
    cli.add_argument("--model-py", required=True, type=str,
                     help="python:module.function -> returns fs given (params, ns, pts)")
    cli.add_argument("--fixed-json", type=Path, help="JSON mapping of fixed params")
    # NEW: plotting
    cli.add_argument("--save-plot", type=Path, help="Where to save the 2x2 comparison figure (PNG)")
    cli.add_argument("--show-plot", action="store_true", help="Show the figure instead of only saving")
    cli.add_argument("--log10-plot", action="store_true", help="Display log10 counts in SFS panels")

    args = cli.parse_args()

    # load observed SFS
    if args.sfs_file:
        sfs = load_sfs_from_file(args.sfs_file)
    else:
        if not (args.popfile and args.pop_ids and args.ns):
            raise SystemExit("--vcf requires --popfile, --pop-ids, and --ns")
        sfs = load_sfs_from_vcf(args.vcf, args.popfile, args.pop_ids, args.ns, args.folded)

    # priors & starts
    cfg    = json.loads(args.config.read_text())
    priors = cfg["priors"]
    start  = cfg.get("start", {k: 0.5*(lo+hi) for k,(lo,hi) in priors.items()})

    # model import
    mod_path, func_name = args.model_py.split(":")
    model_func = getattr(importlib.import_module(mod_path), func_name)

    # fixed params
    fixed = json.loads(args.fixed_json.read_text()) if args.fixed_json else None

    # If using split_migration_model, swap in wrapper
    if func_name == "split_migration_model":
        def model_func(params, ns, pts):
            return split_migration_model_wrapper(params, ns, pts, pop_ids=args.pop_ids)

    # fit
    (best_params_list, best_ll_list, ns, pts_l, func_ex) = fit_model(
        sfs=sfs,
        start_dict=start,
        model_func=model_func,
        priors=priors,
        fixed_params=fixed,
        folded=args.folded,
    )

    # ── NEW: build expected with θ̂ and plot ──
    best_params = best_params_list[0]
    exp_sfs, theta_hat = expected_with_theta_hat(model_func, best_params, ns, pts_l, args.folded, sfs)
    if exp_sfs is not None:
        title = f"Fit comparison (LL={best_ll_list[0]:.3f}, θ̂={theta_hat:.3g})"
        out = args.save_plot
        if out or args.show_plot:
            plot_sfs_comparison(sfs, exp_sfs, out_path=out, title=title, log10=args.log10_plot)
    else:
        print("Could not form expected SFS for plotting (θ̂ undefined).")

if __name__ == "__main__":
    main()

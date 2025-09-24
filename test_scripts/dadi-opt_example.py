#!/usr/bin/env python3
import numpy as np
import msprime
import demes
import nlopt
import numdifftools as nd
import matplotlib.pyplot as plt

# ========= USER TOGGLES =========
BACKEND = "dadi"   # "moments" or "dadi"
USE_BOBYQA = False    # False → LD_LBFGS (finite-diff grad); True → LN_BOBYQA (derivative-free)
VERBOSE = True
RTOL = 1e-8
MAXEVAL = 800
EPS = 1e-300          # guard for log(0)
# =================================

# Lazy import of the chosen backend
if BACKEND == "moments":
    import moments as sfs_backend
elif BACKEND == "dadi":
    import dadi as sfs_backend
else:
    raise ValueError("BACKEND must be 'moments' or 'dadi'")

# ------------------------------
# Model (msprime → demes → SFS)
# ------------------------------
def msprime_model(N_ANC, N_A, N_B, T, mAB, mBA):
    demogr = msprime.Demography()
    demogr.add_population(name="A",   initial_size=N_A)
    demogr.add_population(name="B",   initial_size=N_B)
    demogr.add_population(name="ANC", initial_size=N_ANC)
    demogr.add_population_split(time=T, ancestral="ANC", derived=["A", "B"])
    # Asymmetric migration between A and B (no migration for ANC)
    demogr.migration_matrix = np.array([
        [0,   mAB, 0],
        [mBA, 0,   0],
        [0,   0,   0],
    ])
    return demogr

def _default_pts(sample_sizes):
    """
    For dadi only: finite-difference grid. Keep modest for speed.
    """
    n_max = int(max(sample_sizes))
    return [n_max + 10, n_max + 20, n_max + 30]

def expected_sfs(log10_params, sample_sizes, mu_times_L, pts=None):
    """
    Build the *expected counts* SFS for Poisson LL in the chosen BACKEND.
    - sample_sizes: haploid counts [nA, nB]
    - mu_times_L: μ⋅L (per-base μ times sequence length)
    θ scaling INCLUDED: 4*N_ANC*(μ⋅L)
    """
    N_ANC, N_A, N_B, T, mAB, mBA = 10 ** np.asarray(log10_params, float)
    demogr = msprime_model(N_ANC, N_A, N_B, T, mAB, mBA)
    g = demogr.to_demes()

    if BACKEND == "moments":
        # moments doesn’t need pts; we’ll multiply by theta ourselves for parity
        fs = sfs_backend.Spectrum.from_demes(
            g, sampled_demes=["A", "B"], sample_sizes=sample_sizes
        )
    else:  # dadi
        if pts is None:
            pts = _default_pts(sample_sizes)
        fs = sfs_backend.Spectrum.from_demes(
            g, sampled_demes=["A", "B"], sample_sizes=sample_sizes, pts=pts
        )

    # θ scaling → expected *counts* (Poisson objective needs counts)
    fs = fs * (4.0 * N_ANC * mu_times_L)
    return fs

# ------------------------------
# Likelihood & helpers
# ------------------------------
def _get_sample_sizes_from_obs(observed_sfs):
    ns = getattr(observed_sfs, "sample_sizes", None)
    if ns is None:
        # moments/dadi Spectrum shape is (nA+1, nB+1) for polarized
        ns = [n - 1 for n in observed_sfs.shape]
    return list(map(int, ns))

def _make_pts_for_obs(observed_sfs):
    ns = _get_sample_sizes_from_obs(observed_sfs)
    return _default_pts(ns) if BACKEND == "dadi" else None

def poisson_loglikelihood(log10_params, observed_sfs, mu_times_L, pts=None, eps=EPS):
    """
    Poisson composite log-likelihood:
        sum( log(model) * obs - model )
    Assumes observed_sfs are RAW counts and polarization/folding matches.
    """
    ns = _get_sample_sizes_from_obs(observed_sfs)
    model = expected_sfs(log10_params, ns, mu_times_L, pts=pts)
    if getattr(observed_sfs, "folded", False):
        model = model.fold()
    model = np.maximum(model, eps)   # avoid log(0)
    return np.sum(np.log(model) * observed_sfs - model)

# ------------------------------
# Optimization (NLopt)
# ------------------------------
def optimize_poisson_nlopt(
    start_values, lower_bounds, upper_bounds,
    observed_sfs, mu_times_L,
    verbose=VERBOSE, rtol=RTOL, maxeval=MAXEVAL
):
    """
    Maximize Poisson LL in log10-parameter space with NLopt.
    Default: LD_LBFGS (bounded) with finite-diff gradient.
    Toggle to LN_BOBYQA (derivative-free) via USE_BOBYQA.
    """
    # Sanity on Spectrum type (moments or dadi)
    assert hasattr(observed_sfs, "folded"), "observed_sfs must be a Spectrum-like object"

    start_values = np.asarray(start_values, float)
    lb = np.log10(np.asarray(lower_bounds, float))
    ub = np.log10(np.asarray(upper_bounds, float))
    x0 = np.log10(start_values)

    pts = _make_pts_for_obs(observed_sfs)

    def obj_value(xlog10):
        return poisson_loglikelihood(xlog10, observed_sfs, mu_times_L, pts=pts)

    # Finite-difference gradient for L-BFGS
    grad_fn = nd.Gradient(obj_value, step=1e-4)

    def objective(xlog10, grad):
        ll = obj_value(xlog10)
        if grad.size > 0 and not USE_BOBYQA:
            grad[:] = grad_fn(xlog10)  # (ignored by BOBYQA if toggled on)
        if verbose:
            print(f"[LL={ll:.6g}] log10_params={np.array2string(xlog10, precision=4)}")
        return ll  # MAXIMIZE

    # Optimizer choice
    opt = nlopt.opt(nlopt.LN_BOBYQA if USE_BOBYQA else nlopt.LD_LBFGS, x0.size)

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)
    opt.set_xtol_rel(rtol)
    opt.set_maxeval(maxeval)

    try:
        x_hat = opt.optimize(x0)
        status = opt.last_optimize_result()   # status code (int)
        best_ll = opt.last_optimum_value()    # best objective value (the LL)
    except Exception as e:
        x_hat = opt.last_optimum()
        best_ll = opt.last_optimum_value() if np.isfinite(opt.last_optimum_value()) else obj_value(x0)
        status = nlopt.FAILURE
        if verbose:
            print(f"[WARN] nlopt exception: {e}. Recovered best-so-far.")

    fitted_params = 10 ** x_hat
    ns = _get_sample_sizes_from_obs(observed_sfs)
    fitted_sfs = expected_sfs(x_hat, ns, mu_times_L, pts=pts)
    if getattr(observed_sfs, "folded", False):
        fitted_sfs = fitted_sfs.fold()
    return fitted_params, fitted_sfs, best_ll, status

# ------------------------------
# 1D Poisson LL profile
# ------------------------------
def parameter_grid_loglik_poisson(
    param_values, what_to_vary, grid_of_values, observed_sfs, mu_times_L
):
    param_values = np.asarray(param_values, float)
    loglik_surface = []
    pts = _make_pts_for_obs(observed_sfs)

    for p in grid_of_values:
        test = param_values.copy()
        test[what_to_vary] = float(p)
        ll = poisson_loglikelihood(np.log10(test), observed_sfs, mu_times_L, pts=pts)
        loglik_surface.append(ll)

    return np.array(loglik_surface)

# ------------------------------
# Sanity check: msprime vs expected
# ------------------------------
def sanity_check(num_reps=50):
    """
    Monte Carlo (msprime) branch-SFS vs backend expected SFS for the same demography.
    Uses μ to convert branch SFS to expected site-SFS density for comparison.
    """
    seqlen = 1e6
    mu = 1e-8
    params = np.array([1e4, 1e3, 2e4, 1e4, 1e-6, 1e-5])
    demogr = msprime_model(*params)

    tsg = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        sequence_length=seqlen,
        recombination_rate=1e-8,
        demography=demogr,
        random_seed=1,
        num_replicates=num_reps,
    )

    mc_sfs = None
    for ts in tsg:
        tmp = ts.allele_frequency_spectrum(
            sample_sets=[list(ts.samples(population=p)) for p in [0, 1]],
            mode="branch",
            span_normalise=False,
            polarised=True,
        ) * mu
        mc_sfs = tmp if mc_sfs is None else (mc_sfs + tmp)
    mc_sfs /= num_reps

    sample_sizes = [mc_sfs.shape[0] - 1, mc_sfs.shape[1] - 1]
    exp_sfs = expected_sfs(np.log10(params), sample_sizes, mu * seqlen)

    plt.figure(figsize=(4,4), dpi=140)
    plt.plot(exp_sfs, mc_sfs, "o", markersize=3)
    mx = float(np.mean(exp_sfs))
    plt.axline((mx, mx), (mx + 1, mx + 1), linestyle="dashed")
    plt.xlabel(f"{BACKEND} expected SFS (Poisson-scaled)")
    plt.ylabel("msprime MC SFS")
    plt.xscale("log"); plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"sanity-check-{BACKEND}.png")
    plt.close()

# ------------------------------
# Example run
# ------------------------------
if __name__ == "__main__":
    # 1) Sanity check
    sanity_check()

    # 2) Simulate an observed dataset
    true_pars = np.array([1e4, 1e3, 2e4, 2e4, 1e-6, 1e-5])  # [N_ANC, N1, N2, T, m12, m21]
    seqlen = 5e7
    mu = 1e-8

    ts = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        sequence_length=seqlen,
        recombination_rate=1e-8,
        demography=msprime_model(*true_pars),
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=2)

    obs_array = ts.allele_frequency_spectrum(
        sample_sets=[list(ts.samples(population=p)) for p in [0, 1]],
        mode="site",
        span_normalise=False,
        polarised=True,
    )

    # Wrap observed SFS in the appropriate Spectrum class for Poisson (raw counts)
    if BACKEND == "moments":
        observed_sfs = sfs_backend.Spectrum(obs_array)
    else:  # dadi
        observed_sfs = sfs_backend.Spectrum(obs_array)

    # 3) Optimize
    lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8, 1e-8])
    ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3, 1e-3])
    st = np.sqrt(lb * ub)  # geometric mid

    fitted_pars, fitted_sfs, best_ll, status = optimize_poisson_nlopt(
        st, lb, ub, observed_sfs, mu * seqlen, verbose=VERBOSE, rtol=RTOL, maxeval=MAXEVAL
    )
    print("backend:", BACKEND, "  USE_BOBYQA:", USE_BOBYQA)
    print("fitted:", fitted_pars)
    print("true  :", true_pars)
    print("best LL:", best_ll, " status:", status)

    # 4) 1D Poisson LL profiles
    param_names = ["N_ANC", "N1", "N2", "T", "m12", "m21"]
    grids = []
    for i, name in enumerate(param_names):
        center = fitted_pars[i]
        lo = lb[i]
        hi = ub[i]
        # wider for sizes+time; narrower for migration
        if i <= 3:
            gmin = max(lo, center / 50.0)
            gmax = min(hi, center * 50.0)
        else:
            gmin = max(lo, center / 100.0)
            gmax = min(hi, center * 100.0)
        gmin = max(gmin, 1e-12)
        if gmax <= gmin * 1.0001:
            gmax = min(hi, gmin * 10.0)
        grids.append(np.logspace(np.log10(gmin), np.log10(gmax), 41))

    for i, (name, grid) in enumerate(zip(param_names, grids)):
        ll_curve = parameter_grid_loglik_poisson(
            param_values=fitted_pars,
            what_to_vary=i,
            grid_of_values=grid,
            observed_sfs=observed_sfs,
            mu_times_L=mu * seqlen,
        )
        plt.figure(figsize=(6,4), dpi=150)
        plt.plot(grid, ll_curve, "-o", linewidth=1.0, markersize=3, label="profile LL")
        plt.axvline(x=true_pars[i], linestyle="--", label="true")
        plt.axvline(x=fitted_pars[i], linestyle="-.", label="MLE")
        plt.xscale("log")
        plt.ylabel("Poisson composite log-likelihood")
        plt.xlabel(name)
        plt.legend(frameon=False)
        plt.tight_layout()
        outpng = f"loglik_surface_{BACKEND}_{name}.png"
        plt.savefig(outpng)
        plt.close()
        print(f"==== saved {outpng} ====")

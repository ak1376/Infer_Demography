#!/usr/bin/env python3
import numpy as np
import msprime
import demes
import dadi
import nlopt
import numdifftools as nd
import matplotlib.pyplot as plt


# ------------------------------
# Model (msprime → demes → dadi)
# ------------------------------
def msprime_model(N_ANC, N_A, N_B, T, mAB, mBA):
    demogr = msprime.Demography()
    demogr.add_population(name="A",   initial_size=N_A)
    demogr.add_population(name="B",   initial_size=N_B)
    demogr.add_population(name="ANC", initial_size=N_ANC)
    demogr.add_population_split(time=T, ancestral="ANC", derived=["A", "B"])
    # Asymmetric migration between A and B (no migration with ANC)
    demogr.migration_matrix = np.array([
        [0,   mAB, 0],
        [mBA, 0,   0],
        [0,   0,   0],
    ])
    return demogr


def _default_pts(sample_sizes):
    """
    dadi needs a finite-difference grid. Keep it modest for speed.
    sample_sizes are haploid counts [nA, nB].
    """
    n_max = int(max(sample_sizes))
    return [n_max + 10, n_max + 20, n_max + 30]


def expected_sfs_dadi(log10_params, sample_sizes, mutation_rate, pts=None):
    """
    Build the *expected counts* SFS for Poisson LL.
    - sample_sizes: haploid counts [nA, nB]
    - mutation_rate: μ⋅L  (per-base μ times sequence length)
    - θ scaling INCLUDED: 4*N_ANC*(μ⋅L)
    """
    if pts is None:
        pts = _default_pts(sample_sizes)

    N_ANC, N_A, N_B, T, mAB, mBA = 10 ** np.asarray(log10_params, float)
    demogr = msprime_model(N_ANC, N_A, N_B, T, mAB, mBA)

    fs = dadi.Spectrum.from_demes(
        demogr.to_demes(),
        sampled_demes=["A", "B"],
        sample_sizes=sample_sizes,
        pts=pts,
    )
    # θ scaling → expected *counts* (Poisson objective needs counts)
    fs = fs * (4.0 * N_ANC * mutation_rate)
    return fs


# ------------------------------
# Likelihood & Optimization
# ------------------------------
def _get_sample_sizes_from_obs(observed_sfs):
    ns = getattr(observed_sfs, "sample_sizes", None)
    if ns is None:
        ns = [n - 1 for n in observed_sfs.shape]
    return list(map(int, ns))


def _make_pts_for_obs(observed_sfs):
    return _default_pts(_get_sample_sizes_from_obs(observed_sfs))


def poisson_loglikelihood_dadi(log10_params, observed_sfs, mutation_rate, pts=None, eps=1e-300):
    """
    Poisson composite log-likelihood:
        sum( log(model) * obs - model )
    Assumes observed_sfs are RAW counts and polarization matches.
    """
    ns = _get_sample_sizes_from_obs(observed_sfs)
    model = expected_sfs_dadi(log10_params, ns, mutation_rate, pts=pts)
    if observed_sfs.folded:
        model = model.fold()
    model = np.maximum(model, eps)  # avoid log(0)
    return np.sum(np.log(model) * observed_sfs - model)


def optimize_poisson_nlopt(
    start_values,
    lower_bounds,
    upper_bounds,
    observed_sfs,
    mutation_rate,     # μ⋅L
    verbose=False,
    rtol=1e-8,
    maxeval=800,
):
    """
    Maximize Poisson LL in log10-parameter space with NLopt.
    Default: LD_LBFGS (bounded) with finite-diff gradient.
    Toggle to LN_BOBYQA by changing ONE line where the opt is created.
    """
    assert isinstance(observed_sfs, dadi.Spectrum)
    start_values = np.asarray(start_values, float)
    lb = np.log10(np.asarray(lower_bounds, float))
    ub = np.log10(np.asarray(upper_bounds, float))
    x0 = np.log10(start_values)

    pts = _make_pts_for_obs(observed_sfs)

    def obj_value(xlog10):
        return poisson_loglikelihood_dadi(xlog10, observed_sfs, mutation_rate, pts=pts)

    # Finite-diff gradient for L-BFGS
    grad_fn = nd.Gradient(obj_value, step=1e-4)

    def objective(xlog10, grad):
        ll = obj_value(xlog10)
        if grad.size > 0:
            grad[:] = grad_fn(xlog10)  # ignored by LN_BOBYQA if you toggle
        if verbose:
            print(f"[LL={ll:.6g}] log10_params={np.array2string(xlog10, precision=4)}")
        return ll  # MAXIMIZE

    # --------------------------
    # Optimizer choice (TOGGLE)
    # --------------------------
    # Default (matches your moments script): bounded L-BFGS
    opt = nlopt.opt(nlopt.LD_LBFGS, x0.size)
    # Safer derivative-free alternative:
    # opt = nlopt.opt(nlopt.LN_BOBYQA, x0.size)

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
    fitted_sfs = expected_sfs_dadi(x_hat, ns, mutation_rate, pts=pts)
    if observed_sfs.folded:
        fitted_sfs = fitted_sfs.fold()
    return fitted_params, fitted_sfs, best_ll, status


def parameter_grid_loglik_poisson(
    param_values,
    what_to_vary: int,
    grid_of_values: np.ndarray,
    observed_sfs,
    mutation_rate,
):
    """
    1D profile Poisson LL over grid for a chosen parameter index.
    """
    param_values = np.asarray(param_values, float)
    loglik_surface = []
    pts = _make_pts_for_obs(observed_sfs)

    for p in grid_of_values:
        test = param_values.copy()
        test[what_to_vary] = float(p)
        ll = poisson_loglikelihood_dadi(np.log10(test), observed_sfs, mutation_rate, pts=pts)
        loglik_surface.append(ll)

    return np.array(loglik_surface)


# ------------------------------
# Sanity check: msprime vs dadi
# ------------------------------
def sanity_check(num_reps=100):
    """
    Monte Carlo (msprime) branch-SFS vs dadi expected SFS for the same demography.
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

    # dadi expected (polarized, same sample sizes)
    sample_sizes = [mc_sfs.shape[0] - 1, mc_sfs.shape[1] - 1]
    exp_sfs = expected_sfs_dadi(np.log10(params), sample_sizes, mu * seqlen)

    # Plot sanity scatter (log-log)
    plt.figure(figsize=(4,4), dpi=140)
    plt.plot(exp_sfs, mc_sfs, "o", color="black", markersize=3)
    mx = exp_sfs.mean()
    plt.axline((mx, mx), (mx + 1, mx + 1), linestyle="dashed", color="red")
    plt.xlabel("dadi expected SFS (Poisson-scaled)")
    plt.ylabel("msprime MC SFS")
    plt.xscale("log"); plt.yscale("log")
    plt.tight_layout()
    plt.savefig("sanity-check.png")
    plt.close()


# ------------------------------
# Example run
# ------------------------------
if __name__ == "__main__":
    sanity_check()

    # Generate observed data once
    true_pars = np.array([1e4, 1e3, 2e4, 2e4, 1e-6, 1e-5])  # [N_ANC, N_A, N_B, T, mAB, mBA]
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

    obs = ts.allele_frequency_spectrum(
        sample_sets=[list(ts.samples(population=p)) for p in [0, 1]],
        mode="site",
        span_normalise=False,
        polarised=True,
    )
    observed_sfs = dadi.Spectrum(obs)  # RAW counts (Poisson)

    # Optimize
    lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8, 1e-8])
    ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3, 1e-3])
    st = np.sqrt(lb * ub)  # geometric mid

    fitted_pars, fitted_sfs, best_ll, status = optimize_poisson_nlopt(
        st, lb, ub, observed_sfs, mu * seqlen, verbose=True, rtol=1e-8, maxeval=800
    )
    print("fitted:", fitted_pars)
    print("true  :", true_pars)
    print("best LL:", best_ll, " status:", status)

    # --- 1D Poisson log-likelihood profiles for ALL parameters ---
    param_names = ["N_ANC", "N1", "N2", "T", "m12", "m21"]
    grids = []
    for i, name in enumerate(param_names):
        center = fitted_pars[i]
        lo = lb[i]
        hi = ub[i]
        # wider range for sizes & time; narrower for migration
        if i <= 3:
            gmin = max(lo, center / 50.0)
            gmax = min(hi, center * 50.0)
        else:
            gmin = max(lo, center / 100.0)
            gmax = min(hi, center * 100.0)
        # guard against degenerate ranges
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
            mutation_rate=mu * seqlen,
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
        outpng = f"loglik_surface_{name}.png"
        plt.savefig(outpng)
        plt.close()
        print(f"==== saved {outpng} ====")

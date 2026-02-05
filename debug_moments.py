import numpy as np
import msprime
import demes
import moments
import nlopt
import numdifftools as nd
import matplotlib.pyplot as plt


# def msprime_model_old(N_ANC, N_A, N_B, T, mAB, mBA):
#     demogr = msprime.Demography()
#     demogr.add_population(name="A", initial_size=N_A)
#     demogr.add_population(name="B", initial_size=N_B)
#     demogr.add_population(name="ANC", initial_size=N_ANC)
#     demogr.add_population_split(time=T, ancestral="ANC", derived=["A", "B"])
#     demogr.migration_matrix = np.array([[0, mAB, 0], [mBA, 0, 0], [0, 0, 0]])
#     return demogr

def pop_id(ts, name: str) -> int:
    import json
    for pid in range(ts.num_populations):
        md = ts.population(pid).metadata
        if isinstance(md, (bytes, bytearray)):
            md = json.loads(md.decode())
        if isinstance(md, dict) and md.get("name") == name:
            return pid
    raise KeyError(f"Population {name!r} not found")


def msprime_model(N_ANC, N_A, N_B, T, mAB, mBA):
    b = demes.Builder()

    b.add_deme("ANC", epochs=[dict(start_size=float(N_ANC), end_time=float(T))])
    b.add_deme("A", ancestors=["ANC"], epochs=[dict(start_size=float(N_A))])
    b.add_deme("B", ancestors=["ANC"], epochs=[dict(start_size=float(N_B))])

    if mAB > 0:
        b.add_migration(source="A", dest="B", rate=float(mAB))
    if mBA > 0:
        b.add_migration(source="B", dest="A", rate=float(mBA))

    dg = b.resolve()
    demogr = msprime.Demography.from_demes(dg)
    return demogr

    

def expected_sfs(log10_params, sample_size, mutation_rate):
    N_ANC, N_A, N_B, T, mAB, mBA = 10 ** log10_params
    demogr = msprime_model(N_ANC, N_A, N_B, T, mAB, mBA)
    return moments.Spectrum.from_demes(
        demogr.to_demes(),
        sampled_demes=["A", "B"],
        sample_sizes=sample_size,
        theta=4 * N_ANC * mutation_rate,
    )


def optimize_lbfgs(
    start_values,
    lower_bounds,
    upper_bounds,
    observed_sfs,
    mutation_rate,  # scaled by sequence length, e.g. per_base_mutation_rate * sequence_length
    verbose=False,
    rtol=1e-8,
):
    assert isinstance(observed_sfs, moments.Spectrum)
    sample_size = [n - 1 for n in observed_sfs.shape]
    
    def loglikelihood(log10_params):
        exp_sfs = expected_sfs(log10_params, sample_size, mutation_rate)
        loglik = np.sum(np.log(exp_sfs) * observed_sfs - exp_sfs)
        return loglik
    
    def gradient(log10_params):
        return nd.Gradient(loglikelihood, n=1, step=1e-4)(log10_params)
    
    def negative_loss(log10_params, grad):
        loglik = loglikelihood(log10_params)
        if grad.size > 0:
            grad[:] = gradient(log10_params)
        if verbose:
            print(f"loglik: {loglik}, params: {log10_params}")
        return loglik

    opt = nlopt.opt(nlopt.LD_LBFGS, start_values.size)
    opt.set_lower_bounds(np.log10(lower_bounds))
    opt.set_upper_bounds(np.log10(upper_bounds))
    opt.set_max_objective(negative_loss)
    opt.set_ftol_rel(rtol)

    fitted_log10_params = opt.optimize(np.log10(start_values))
    fitted_sfs = expected_sfs(fitted_log10_params, sample_size, mutation_rate)
    max_loglik = opt.last_optimize_result()

    return 10 ** fitted_log10_params, fitted_sfs, max_loglik


def parameter_grid_loglik(
    param_values,
    what_to_vary: int,
    grid_of_values: np.ndarray,
    observed_sfs,
    mutation_rate,  # scaled by sequence length, e.g. per_base_mutation_rate * sequence_length
):
    assert isinstance(observed_sfs, moments.Spectrum)
    sample_size = [n - 1 for n in observed_sfs.shape]
    
    def loglikelihood(log10_params):
        exp_sfs = expected_sfs(log10_params, sample_size, mutation_rate)
        loglik = np.sum(np.log(exp_sfs) * observed_sfs - exp_sfs)
        return loglik

    loglik_surface = []
    for p in grid_of_values:
        log10_pars = np.log10(param_values)
        log10_pars[what_to_vary] = np.log10(p)
        loglik_surface.append(loglikelihood(log10_pars))

    return loglik_surface


#### sanity check

def sanity_check(num_reps=100):
    """
    Check that msprime model spec (that is, what we simulate data from)
    match moments model spec (that is, what we calculate expectation from)
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
            mode='branch',
            span_normalise=False,
            polarised=True,
        ) * mu
        if mc_sfs is None:
            mc_sfs = tmp
        else:
            mc_sfs += tmp
    mc_sfs /= num_reps
    exp_sfs = expected_sfs(np.log10(params), [n - 1 for n in mc_sfs.shape], mu * seqlen)
    import matplotlib.pyplot as plt
    plt.plot(exp_sfs, mc_sfs, "o", color="black", markersize=4)
    mx = exp_sfs.mean()
    plt.axline((mx, mx), (mx+1,mx+1), linestyle="dashed", color="red")
    plt.xlabel("moments sfs")
    plt.ylabel("msprime sfs")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("sanity-check.png")
    plt.clf()

# sanity_check()

#### now test it out

true_pars = np.array([1e4, 1e3, 2e4, 2e4, 1e-6, 1e-5])
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

A = pop_id(ts, "A")
B = pop_id(ts, "B")

obs_sfs = moments.Spectrum(
    ts.allele_frequency_spectrum(
        sample_sets=[list(ts.samples(population=A)), list(ts.samples(population=B))],
        mode="site",
        span_normalise=False,
        polarised=True,
    )
)



# optimize it
lb = np.array([5e2, 5e2, 5e2, 5e2, 1e-8, 1e-8])
ub = np.array([5e4, 5e4, 5e4, 5e4, 1e-3, 1e-3])
st = lb / 2 + ub / 2

fitted_pars, _, _ = optimize_lbfgs(st, lb, ub, obs_sfs, mu * seqlen, verbose=True)
print(fitted_pars)
print(true_pars)



param_names = ["N_ANC", "N_A", "N_B", "T", "mAB", "mBA"]

# grids: log-spaced for all (works fine since all params are positive)
# (you can tweak per-parameter ranges if you want tighter plots)
grid_specs = {
    "N_ANC": (np.log10(lb[0]), np.log10(ub[0]), 51),
    "N_A":   (np.log10(lb[1]), np.log10(ub[1]), 51),
    "N_B":   (np.log10(lb[2]), np.log10(ub[2]), 51),
    "T":     (np.log10(lb[3]), np.log10(ub[3]), 51),
    "mAB":   (np.log10(lb[4]), np.log10(ub[4]), 51),
    "mBA":   (np.log10(lb[5]), np.log10(ub[5]), 51),
}

for j, name in enumerate(param_names):
    lo, hi, n = grid_specs[name]
    grid = np.logspace(lo, hi, n)

    loglik_surface = parameter_grid_loglik(
        fitted_pars,
        what_to_vary=j,
        grid_of_values=grid,
        observed_sfs=obs_sfs,
        mutation_rate=mu * seqlen,
    )

    plt.figure()
    plt.plot(grid, loglik_surface, "-o", markersize=3)  # default color
    plt.axvline(x=true_pars[j], linestyle="--")        # true value marker (default color)
    plt.xscale("log")
    plt.ylabel("Loglikelihood")
    plt.xlabel(name)
    plt.title(f"1D loglik profile at MLE: vary {name}")

    plt.tight_layout()
    plt.savefig(f"loglik_surface_{name}.png")
    plt.close()
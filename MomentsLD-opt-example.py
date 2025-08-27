#!/usr/bin/env python3
"""
Isolation-with-migration example using MomentsLD:
- Single source of truth = Demes graph
- msprime sims + LD parsing in parallel with Ray
- Expected σD² LD from the same Demes graph
- Composite Gaussian log-likelihood fitting (LD-only) via NLOpt L-BFGS
- 1D log-likelihood surface plots for each parameter (saved as PNGs)
"""

import os
import time
import copy
import numpy as np
import msprime
import demes
import moments
import nlopt
import numdifftools as nd
import ray
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
os.system("rm -f ./data/*.vcf.gz ./data/*.h5")

# --------------------------------------------------------------------------------------
# Demes is the single source of truth
# --------------------------------------------------------------------------------------
def _make_demes_graph(N_ANC, N1, N2, T, m12, m21):
    b = demes.Builder(time_units="generations", generation_time=1)
    b.add_deme("ANC", epochs=[{"start_size": N_ANC, "end_time": T}])
    b.add_deme("N1", ancestors=["ANC"], epochs=[{"start_size": N1}])
    b.add_deme("N2", ancestors=["ANC"], epochs=[{"start_size": N2}])
    if m12 > 0:
        b.add_migration(source="N1", dest="N2", rate=m12)
    if m21 > 0:
        b.add_migration(source="N2", dest="N1", rate=m21)
    return b.resolve()

def demes_object(N_ANC, N1, N2, T, m12, m21):
    return _make_demes_graph(N_ANC, N1, N2, T, m12, m21)

def msprime_model(N_ANC, N1, N2, T, m12, m21):
    g = _make_demes_graph(N_ANC, N1, N2, T, m12, m21)
    return msprime.Demography.from_demes(g)

# --------------------------------------------------------------------------------------
# Expected LD (σD²) from parameters (log10 space) using same Demes graph
# --------------------------------------------------------------------------------------
def expected_ld_sigmaD2_log10(
    log10_params,
    r_bins,
    sampled_demes=("N1", "N2"),
    normalization=0,
):
    N_ANC, N1, N2, T, m12, m21 = 10 ** np.asarray(log10_params, float)
    g = demes_object(N_ANC, N1, N2, T, m12, m21)

    rho_edges = 4.0 * N_ANC * np.asarray(r_bins, float)
    y_edges = moments.Demes.LD(g, sampled_demes=list(sampled_demes), rho=rho_edges)

    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    y_mids  = moments.Demes.LD(g, sampled_demes=list(sampled_demes), rho=rho_mids)

    y_bins = [ (y_edges[i] + y_edges[i+1] + 4*y_mids[i]) / 6.0 for i in range(len(rho_mids)) ]
    y_bins.append(y_edges[-1])

    y = moments.LD.LDstats(y_bins, num_pops=y_edges.num_pops, pop_ids=y_edges.pop_ids)
    y = moments.LD.Inference.sigmaD2(y, normalization=normalization)
    return y

def _remove_normalized_from_model(y_sigmaD2, normalization=0, keep_het=False):
    y = moments.LD.LDstats(copy.deepcopy(y_sigmaD2[:]),
                           num_pops=y_sigmaD2.num_pops,
                           pop_ids=y_sigmaD2.pop_ids)
    y = moments.LD.Inference.remove_normalized_lds(y, normalization=normalization)
    return y if keep_het else y[:-1]

def _prepare_empirical_means_vcs(mv, normalization=0, num_pops=2, keep_het=False, statistics=None):
    ms = copy.deepcopy(mv["means"])
    vcs = copy.deepcopy(mv["varcovs"])
    if statistics is None:
        ms, vcs = moments.LD.Inference.remove_normalized_data(
            ms, vcs, normalization=normalization, num_pops=num_pops
        )
    if not keep_het:
        ms = ms[:-1]
        vcs = vcs[:-1]
    return ms, vcs

def _ld_loglik_given_model(means_list, varcovs_list, y_model_list, jitter=1e-12):
    if not (len(means_list) == len(varcovs_list) == len(y_model_list)):
        raise ValueError("Lengths of means/varcovs/model per bin must match")
    ll = 0.0
    for x, Sigma, mu in zip(means_list, varcovs_list, y_model_list):
        x = np.asarray(x, float)
        mu = np.asarray(mu, float)
        if x.size == 0:
            continue
        S = np.array(Sigma, float)
        if S.ndim == 2 and S.size > 1:
            if jitter and jitter > 0:
                S = S + np.eye(S.shape[0]) * jitter
            Si = np.linalg.inv(S)
            d = x - mu
            ll += -0.5 * d @ Si @ d
        else:
            d = x - mu
            ll += -0.5 * float(d @ d)
    return ll

def ld_loglikelihood_log10(
    log10_params,
    r_bins,
    mv,
    sampled_demes=("N1","N2"),
    normalization=0,
    keep_het=False,
):
    y = expected_ld_sigmaD2_log10(log10_params, r_bins, sampled_demes, normalization)
    y_vecs = _remove_normalized_from_model(y, normalization=normalization, keep_het=keep_het)
    ms, vcs = _prepare_empirical_means_vcs(mv, normalization=normalization, num_pops=y.num_pops, keep_het=keep_het)
    y_arrays = [np.asarray(y_vecs[i], float) for i in range(len(ms))]
    return _ld_loglik_given_model(ms, vcs, y_arrays)

def optimize_ld_lbfgs(
    start_values,
    lower_bounds,
    upper_bounds,
    r_bins,
    mv,
    sampled_demes=("N1","N2"),
    normalization=0,
    verbose=False,
    rtol=1e-8,
):
    start_values = np.asarray(start_values, float)
    lb = np.log10(np.asarray(lower_bounds, float))
    ub = np.log10(np.asarray(upper_bounds, float))

    def loglik(log10_params):
        return ld_loglikelihood_log10(
            log10_params, r_bins, mv,
            sampled_demes=sampled_demes,
            normalization=normalization,
            keep_het=False,
        )

    grad_fn = nd.Gradient(loglik, step=1e-4)

    def objective(log10_params, grad):
        ll = loglik(log10_params)
        if grad.size > 0:
            grad[:] = grad_fn(log10_params)
        if verbose:
            print(f"[LL={ll:.6g}] log10_params={np.array2string(log10_params, precision=4)}")
        return ll

    opt = nlopt.opt(nlopt.LD_LBFGS, start_values.size)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_max_objective(objective)
    opt.set_ftol_rel(rtol)

    x0 = np.log10(start_values)
    x_hat = opt.optimize(x0)
    ll_max = opt.last_optimize_result()
    return 10 ** x_hat, ll_max

def ld_parameter_grid_loglik(
    param_values,
    which_index,
    grid_values,
    r_bins,
    mv,
    sampled_demes=("N1","N2"),
    normalization=0,
):
    param_values = np.asarray(param_values, float)
    ll_vals = []
    for v in grid_values:
        test = param_values.copy()
        test[which_index] = float(v)
        ll = ld_loglikelihood_log10(
            np.log10(test), r_bins, mv,
            sampled_demes=sampled_demes,
            normalization=normalization,
            keep_het=False,
        )
        ll_vals.append(ll)
    return np.array(ll_vals)

# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------
def write_samples_and_rec_map(L, r, n):
    with open(f"{DATA_DIR}/samples.txt", "w+") as fout:
        fout.write("sample\tpop\n")
        for jj in range(2):
            for ii in range(n):
                fout.write(f"tsk_{jj * n + ii}\tN{jj+1}\n")
    with open(f"{DATA_DIR}/flat_map.txt", "w+") as fout:
        fout.write("pos\tMap(cM)\n")
        fout.write("0\t0\n")
        fout.write(f"{int(L)}\t{r * int(L) * 100}\n")

# --------------------------------------------------------------------------------------
# LD parsing
# --------------------------------------------------------------------------------------
def get_LD_stats(rep_ii, r_bins):
    vcf_file = f"{DATA_DIR}/split_mig.{rep_ii}.vcf.gz"
    t1 = time.time()
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file=f"{DATA_DIR}/flat_map.txt",
        pop_file=f"{DATA_DIR}/samples.txt",
        pops=["N1", "N2"],
        r_bins=r_bins,
        report=False,
    )
    print("  finished rep", rep_ii, "in", int(time.time() - t1), "seconds")
    return ld_stats

@ray.remote(num_cpus=1)
def get_LD_stats_remote(rep_ii, r_bins):
    return rep_ii, get_LD_stats(rep_ii, r_bins)

# --------------------------------------------------------------------------------------
# msprime replicate simulation (parallel) — Demes → msprime
# --------------------------------------------------------------------------------------
@ray.remote(num_cpus=1)
def simulate_one_rep(rep_i, true_pars, seqlen, mu, r, n):
    import msprime as _ms
    import demes as _demes
    import os as _os

    N_ANC, N1, N2, T, m12, m21 = true_pars
    b = _demes.Builder(time_units="generations", generation_time=1)
    b.add_deme("ANC", epochs=[{"start_size": N_ANC, "end_time": T}])
    b.add_deme("N1", ancestors=["ANC"], epochs=[{"start_size": N1}])
    b.add_deme("N2", ancestors=["ANC"], epochs=[{"start_size": N2}])
    if m12 > 0:
        b.add_migration(source="N1", dest="N2", rate=m12)
    if m21 > 0:
        b.add_migration(source="N2", dest="N1", rate=m21)
    g = b.resolve()
    demog = _ms.Demography.from_demes(g)

    ts = _ms.sim_ancestry(
        {"N1": n, "N2": n},
        demography=demog,
        sequence_length=int(seqlen),
        recombination_rate=r,
        random_seed=42 + rep_i,
    )
    ts = _ms.sim_mutations(ts, rate=mu, random_seed=1_000_000 + rep_i)

    vcf_name = f"{DATA_DIR}/split_mig.{rep_i}.vcf"
    with open(vcf_name, "w+") as fout:
        ts.write_vcf(fout, allow_position_zero=True)
    _os.system(f"gzip -f {vcf_name}")
    return vcf_name + ".gz"

def run_msprime_replicates_parallel(true_pars, num_reps, seqlen, mu, r, n, max_parallel=4):
    vcf_paths = []
    pending = []
    for rep_i in range(num_reps):
        pending.append(simulate_one_rep.remote(rep_i, list(true_pars), seqlen, mu, r, n))
        if len(pending) >= max_parallel:
            ready, pending = ray.wait(pending, num_returns=1)
            for obj in ready:
                vcf_paths.append(ray.get(obj))
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for obj in ready:
            vcf_paths.append(ray.get(obj))
    return vcf_paths

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- core parameters
    n = 10
    num_reps = 100
    true_pars = np.array([1e4, 1e3, 2e4, 2e4, 1e-6, 1e-5])  # [N_ANC, N1, N2, T, m12, m21]
    seqlen = int(1e6)
    mu = 1.5e-8
    r = 1.5e-8

    # r-bin edges (raw r), log-spaced
    r_bins = np.concatenate(([0], np.logspace(-6, -3, 16)))

    # cache paths for empirical LD summaries
    mv_path = f"{DATA_DIR}/means.varcovs.split_mig.{num_reps}_reps.bp"
    boot_path = f"{DATA_DIR}/bootstrap_sets.split_mig.{num_reps}_reps.bp"

    # --- load or compute empirical MV
    try:
        print("loading data if pre-computed")
        import pickle
        with open(mv_path, "rb") as fin:
            mv = pickle.load(fin)
        with open(boot_path, "rb") as fin:
            all_boot = pickle.load(fin)
    except (IOError, FileNotFoundError):
        MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "4"))
        if not ray.is_initialized():
            ray.init(num_cpus=MAX_PARALLEL, ignore_reinit_error=True, include_dashboard=False)

        print(f"running msprime and writing vcfs in parallel (MAX_PARALLEL={MAX_PARALLEL})")
        _ = run_msprime_replicates_parallel(
            true_pars, num_reps=num_reps, seqlen=seqlen, mu=mu, r=r, n=n, max_parallel=MAX_PARALLEL
        )

        print("writing samples and recombination map")
        write_samples_and_rec_map(L=seqlen, r=r, n=n)

        print("parsing LD statistics in parallel")
        ld_stats = {}
        pending = []
        for ii in range(num_reps):
            pending.append(get_LD_stats_remote.remote(ii, r_bins))
            if len(pending) >= MAX_PARALLEL:
                ready, pending = ray.wait(pending, num_returns=1)
                for obj in ready:
                    k, v = ray.get(obj)
                    ld_stats[k] = v
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for obj in ready:
                k, v = ray.get(obj)
                ld_stats[k] = v

        print("computing mean and varcov matrix from LD statistics sums")
        mv = moments.LD.Parsing.bootstrap_data(ld_stats)

    # --- expected (at true pars) for visual check (not saved)
    print("computing expectations under the model (σD²)")
    y = expected_ld_sigmaD2_log10(np.log10(true_pars), r_bins, sampled_demes=("N1","N2"), normalization=0)

    # --- plot expected vs empirical curves
    _ = moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=r_bins,
        stats_to_plot=[
            ["DD_0_0"],
            ["DD_0_1"],
            ["DD_1_1"],
            ["Dz_0_0_0"],
            ["Dz_0_1_1"],
            ["Dz_1_1_1"],
            ["pi2_0_0_1_1"],
            ["pi2_0_1_0_1"],
            ["pi2_1_1_1_1"],
        ],
        labels=[
            [r"$D_0^2$"],
            [r"$D_0 D_1$"],
            [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$Dz_{0,1,1}$"],
            [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"],
            [r"$\pi_{2;0,1,0,1}$"],
            [r"$\pi_{2;1,1,1,1}$"],
        ],
        rows=3,
        plot_vcs=True,
        show=False,
        fig_size=(6, 4),
        output="split_mig_comparison.pdf",
    )
    print("==== saved split_mig_comparison.pdf ====")

    # --- LD-only optimization (log10 space)
    print("running LD-only optimization (L-BFGS, log10 space)")
    lb = np.array([5e2, 2e2, 2e3, 1e3, 1e-8, 1e-8])  # lower bounds
    ub = np.array([5e5, 5e5, 5e5, 1e6, 1e-2, 1e-2])  # upper bounds
    start = np.sqrt(lb * ub)

    fitted_pars, llmax = optimize_ld_lbfgs(
        start_values=start,
        lower_bounds=lb,
        upper_bounds=ub,
        r_bins=r_bins,
        mv=mv,
        sampled_demes=("N1","N2"),
        normalization=0,
        verbose=True,
        rtol=1e-8,
    )

    print("\nTrue parameters     :", true_pars)
    print("Fitted parameters   :", fitted_pars)
    print("Max composite LL    :", llmax)

    # --- 1D log-likelihood surfaces for ALL parameters
    param_names = ["N_ANC", "N1", "N2", "T", "m12", "m21"]
    grids = []

    # Build sensible log-spaced grids around the MLE (clipped to bounds).
    # Wider range for sizes & time (×/÷50), a bit narrower for migration (×/÷100).
    for i, name in enumerate(param_names):
        center = fitted_pars[i]
        lo = lb[i]
        hi = ub[i]
        if i <= 3:  # sizes + time
            gmin = max(lo, center / 50.0)
            gmax = min(hi, center * 50.0)
        else:       # migration rates
            gmin = max(lo, center / 100.0)
            gmax = min(hi, center * 100.0)
        # guard against degenerate ranges
        if gmin <= 0:
            gmin = max(lo, 1e-12)
        if gmax <= gmin * 1.0001:
            gmax = min(hi, gmin * 10.0)
        grids.append(np.logspace(np.log10(gmin), np.log10(gmax), 41))

    # Compute and save each profile plot
    for i, (name, grid) in enumerate(zip(param_names, grids)):
        ll_curve = ld_parameter_grid_loglik(
            param_values=fitted_pars,
            which_index=i,
            grid_values=grid,
            r_bins=r_bins,
            mv=mv,
            sampled_demes=("N1","N2"),
            normalization=0,
        )
        plt.figure(figsize=(6,4), dpi=150)
        plt.plot(grid, ll_curve, "-o", linewidth=1.0, markersize=3, label="profile LL")
        # verticals at true value and MLE
        plt.axvline(x=true_pars[i], linestyle="--", label="true")
        plt.axvline(x=fitted_pars[i], linestyle="-.", label="MLE")
        plt.xscale("log")
        plt.ylabel("Composite log-likelihood")
        plt.xlabel(name)
        plt.legend(frameon=False)
        plt.tight_layout()
        outpng = f"data/ld_loglik_surface_{name}.png"
        plt.savefig(outpng)
        plt.close()
        print(f"==== saved {outpng} ====")

    # Info prints
    g = demes_object(*true_pars)
    print(f"\nDemes Object:\n{g}")
    print("\nMsprime Demography:")
    print(msprime_model(*true_pars))

"""
Example of using MomentsLD to fit a simple isolation-with-migration model
with Ray-parallelized msprime replicates and Ray-parallelized LD parsing.
"""

import os
import time
import pickle
import numpy as np
import msprime
import demes
import moments
import nlopt
import numdifftools as nd  # kept since you imported it
import ray

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
os.system("rm -f ./data/*.vcf.gz ./data/*.h5")  # quiet if nothing to remove

# --------------------------------------------------------------------------------------
# Models (Demes is the single source of truth)
# --------------------------------------------------------------------------------------
def _make_demes_graph(N_ANC, N1, N2, T, m12, m21):
    """
    Build the Demes graph once; use it everywhere else.
    """
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
    # keep your function name; now powered by the single builder
    return _make_demes_graph(N_ANC, N1, N2, T, m12, m21)

def msprime_model(N_ANC, N1, N2, T, m12, m21):
    # guarantee alignment by converting the Demes graph to msprime
    g = _make_demes_graph(N_ANC, N1, N2, T, m12, m21)
    return msprime.Demography.from_demes(g)

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
# msprime replicate simulation (parallel) — build Demes inside the worker, convert to msprime
# --------------------------------------------------------------------------------------
@ray.remote(num_cpus=1)
def simulate_one_rep(rep_i, true_pars, seqlen, mu, r, n):
    """
    Simulate ONE replicate and write ./data/split_mig.{rep_i}.vcf.gz.
    Returns the written path.
    """
    # Import inside the worker to avoid serializing big modules
    import msprime as _ms
    import demes as _demes

    N_ANC, N1, N2, T, m12, m21 = true_pars

    # Build Demes graph and convert to msprime Demography (guaranteed match)
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
    os.system(f"gzip -f {vcf_name}")
    return vcf_name + ".gz"

def run_msprime_replicates_parallel(true_pars, num_reps, seqlen,
                                    mu, r, n, max_parallel=4):
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
    # parameters
    n = 20
    num_reps = 100
    true_pars = np.array([1e4, 1e3, 2e4, 2e4, 1e-6, 1e-5])  # [N_ANC, N1, N2, T, m12, m21]
    seqlen = int(1e6)
    mu = 1.5e-8
    r = 1.5e-8

    # r-bin edges
    r_bins = np.concatenate(([0], np.logspace(-6, -3, 16)))

    # cache paths
    mv_path = f"{DATA_DIR}/means.varcovs.split_mig.{num_reps}_reps.bp"
    boot_path = f"{DATA_DIR}/bootstrap_sets.split_mig.{num_reps}_reps.bp"

    try:
        print("loading data if pre-computed")
        with open(mv_path, "rb") as fin:
            mv = pickle.load(fin)
        with open(boot_path, "rb") as fin:
            all_boot = pickle.load(fin)
    except (IOError, FileNotFoundError):
        # Init Ray once; reuse for both sim and LD
        MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "4"))
        if not ray.is_initialized():
            ray.init(num_cpus=MAX_PARALLEL, ignore_reinit_error=True, include_dashboard=False)

        print(f"running msprime and writing vcfs in parallel (MAX_PARALLEL={MAX_PARALLEL})")
        _vcfs = run_msprime_replicates_parallel(
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
        with open(mv_path, "wb+") as fout:
            pickle.dump(mv, fout)
        with open(boot_path, "wb+") as fout:
            pickle.dump(ld_stats, fout)

    print("computing expectations under the model")
    g = demes_object(*true_pars)  # same graph the sim used
    rho = 4 * true_pars[0] * r_bins  # N_ref = N_ANC here
    y = moments.Demes.LD(g, sampled_demes=["N1", "N2"], rho=rho)
    y = moments.LD.LDstats(
        [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    fig = moments.LD.Plotting.plot_ld_curves_comp(
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
    print("==== done plotting ====")
    print(f"Demes Object:\n{g}")
    print("Msprime Demography:")
    print(msprime_model(*true_pars))

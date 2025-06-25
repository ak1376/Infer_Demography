import demes
import msprime
import moments

def bottleneck_model(sampled_params):

    N0, nuB, nuF, t_bottleneck_start, t_bottleneck_end = (
        sampled_params["N0"],
        sampled_params["Nb"],
        sampled_params["N_recover"],
        sampled_params["t_bottleneck_start"],
        sampled_params["t_bottleneck_end"],
    )
    b = demes.Builder()
    b.add_deme(
        "N0",
        epochs=[
            dict(start_size=N0, end_time=t_bottleneck_start),
            dict(start_size=nuB, end_time=t_bottleneck_end),
            dict(start_size=nuF, end_time=0),
        ],
    )
    g = b.resolve()

    return g

def split_isolation_model(sampled_params):

    # Unpack the sampled parameters
    Na, N1, N2, m, t_split = (
        sampled_params["Na"],  # Effective population size of the ancestral population
        sampled_params["N1"],  # Size of population 1 after split
        sampled_params["N2"],  # Size of population 2 after split
        sampled_params["m"],   # Migration rate between populations
        sampled_params["t_split"],  # Time of the population split (in generations)
    )

    b = demes.Builder()
    b.add_deme("Na", epochs=[dict(start_size=Na, end_time=t_split)])
    b.add_deme("N1", ancestors=["Na"], epochs=[dict(start_size=N1)])
    b.add_deme("N2", ancestors=["Na"], epochs=[dict(start_size=N2)])
    b.add_migration(demes=["N1", "N2"], rate=m)
    g = b.resolve()
    return g


def split_migration_model(sampled_params):
    # Unpack the sampled parameters
    N0, N1, N2, m12, m21, t_split = (
        sampled_params["N0"],  # Effective population size of the ancestral population
        sampled_params["N1"],  # Size of population 1 after split
        sampled_params["N2"],  # Size of population 2 after split
        sampled_params["m12"], # Migration rate from N1 to N2
        sampled_params["m21"], # Migration rate from N2 to N1
        sampled_params["t_split"],  # Time of the population split (in generations)
    )

    # Define the demographic model using demes
    b = demes.Builder()
    # Ancestral population
    b.add_deme("N0", epochs=[dict(start_size=N0, end_time=t_split)])
    # Derived populations after split
    b.add_deme("N1", ancestors=["N0"], epochs=[dict(start_size=N1)])
    b.add_deme("N2", ancestors=["N0"], epochs=[dict(start_size=N2)])
    # Asymmetric migration: Different migration rates for each direction
    b.add_migration(source="N1", dest="N2", rate=m12)  # Migration from N1 to N2
    b.add_migration(source="N2", dest="N1", rate=m21)  # Migration from N2 to N1
    # Resolve and return the demography graph
    g = b.resolve()
    return g

def simulation(sampled_params, model_type, experiment_config):
    """
    Simulates a demographic model based on the provided parameters and model type.

    Parameters:
    - sampled_params (dict): Dictionary containing parameters for the demographic model.
    - model_type (str): Type of demographic model to simulate ('bottleneck', 'split_isolation', 'split_migration').

    Returns:
    - demes.Graph: A demes graph representing the simulated demographic model.
    """
    
    if model_type == "bottleneck":
        g = bottleneck_model(sampled_params)
    elif model_type == "split_isolation":
        g = split_isolation_model(sampled_params)
    elif model_type == "split_migration":
        g = split_migration_model(sampled_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
     
    # Make sure you are calling the right demes names 
    samples = {pop_name: num_samples for pop_name, num_samples in experiment_config['num_samples'].items()}

    demog = msprime.Demography.from_demes(g)

    # Simulate ancestry for two populations (joint simulation)
    ts = msprime.sim_ancestry(
        samples=samples,  # Two populations
        demography=demog,
        sequence_length=experiment_config['genome_length'],
        recombination_rate=experiment_config['recombination_rate'],
        random_seed=experiment_config['seed'],
    )
    
    # Simulate mutations over the ancestry tree sequence
    ts = msprime.sim_mutations(ts, rate=experiment_config['mutation_rate'], random_seed=experiment_config['seed'])

    return ts, g


def create_SFS(ts):
    """
    Generate the site frequency spectrum (SFS) using the simulated TreeSequence (ts).

    Parameters:
    - ts: TreeSequence object containing the simulated data.
    - num_samples: Dictionary with deme names as keys and the number of samples as values.

    Returns:
    - sfs: The moments Spectrum object for the given demographic data.
    """
    
    # Define sample sets dynamically for the SFS
    sample_sets = [
        ts.samples(population=pop.id) 
        for pop in ts.populations() 
        if len(ts.samples(population=pop.id)) > 0  # Exclude populations with no samples
    ]
                
    sfs = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False  # <-- crucial
    )

    # Convert to 1D or 2D moments Spectrum
    sfs = moments.Spectrum(sfs)

    pop_names = ["N1", "N2"]

    sfs.pop_ids = pop_names
    
    return sfs
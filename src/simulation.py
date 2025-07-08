# simulation.py

import demes
import msprime
import moments
import stdpopsim

def bottleneck_model(sampled_params):

    N0, N_bottleneck, N_recover, t_bottleneck_start, t_bottleneck_end = (
        sampled_params["N0"],  # Effective population size of the ancestral population
        sampled_params["N_bottleneck"],
        sampled_params["N_recover"],
        sampled_params["t_bottleneck_start"],
        sampled_params["t_bottleneck_end"],
    )
    b = demes.Builder()
    b.add_deme(
        "N0",
        epochs=[
            dict(start_size=N0, end_time=t_bottleneck_start),
            dict(start_size=N_bottleneck, end_time=t_bottleneck_end),
            dict(start_size=N_recover, end_time=0),
        ],
    )
    g = b.resolve()

    return g

def split_isolation_model(sampled_params):

    # Unpack the sampled parameters
    Na, N1, N2, m, t_split = (
        sampled_params["N0"],  # Effective population size of the ancestral population
        sampled_params["N1"],  # Size of population 1 after split
        sampled_params["N2"],  # Size of population 2 after split
        sampled_params["m"],   # Migration rate between populations
        sampled_params["t_split"],  # Time of the population split (in generations)
    )

    b = demes.Builder()
    b.add_deme("N0", epochs=[dict(start_size=Na, end_time=t_split)])
    b.add_deme("N1", ancestors=["N0"], epochs=[dict(start_size=N1)])
    b.add_deme("N2", ancestors=["N0"], epochs=[dict(start_size=N2)])
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

def drosophila_three_epoch(sampled_params):
    """
    Simulates a demographic model for Drosophila melanogaster based on the Out of Africa model.

    Parameters:
    - sampled_params (dict): Dictionary containing parameters for the demographic model.

    Returns:
    - demes.Graph: A demes graph representing the simulated demographic model.
    """

    species = stdpopsim.get_species("DroMel")
    model = species.get_demographic_model("OutOfAfrica_2L06")

    # Unpack the sampled parameters
    N0 = sampled_params["N0"]  # Ancestral population size
    AFR_recover = sampled_params["AFR"]  # Post expansion African population size
    EUR_bottleneck = sampled_params["EUR_bottleneck"]  # European bottleneck pop size
    EUR_recover = sampled_params["EUR_recover"]  # Modern European population size after recovery
    T_AFR_expansion = sampled_params["T_AFR_expansion"]  # Expansion of population in Africa
    T_AFR_EUR_split = sampled_params["T_AFR_EUR_split"]  # African-European Divergence
    T_EUR_expansion = sampled_params["T_EUR_expansion"]  # European

    # print(model.model.events[2].initial_size) # Ancestral population size
    # print(model.model.populations[0].initial_size) # Post expansion African population size
    # print(model.model.events[0].initial_size) # European bottleneck pop size
    # print(model.model.populations[1].initial_size) # Modern European population size
    # print(model.model.events[2].time) # Expansion of population in Africa
    # print(model.model.events[1].time) # African-European Divergence
    # print(model.model.events[0].time) # European population expansion

    # Set the demographic model parameters
    model.model.events[2].initial_size = N0  # Set ancestral population size
    model.model.populations[0].initial_size = AFR_recover  # Set post expansion African
    model.model.events[0].initial_size = EUR_bottleneck  # Set European bottleneck population size
    model.model.populations[1].initial_size = EUR_recover  # Set modern European population size
    model.model.events[2].time = T_AFR_expansion  # Set expansion of population in Africa
    model.model.events[1].time = T_AFR_EUR_split  # Set African-European divergence
    model.model.events[0].time = T_EUR_expansion  # Set European population expansion

    
    # Run the bottleneck model simulation
    g = model.model.to_demes()

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
    elif model_type == "drosophila_three_epoch":
        g = drosophila_three_epoch(sampled_params)
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

    # Get the population names from the TreeSequence
    pop_names = [
        pop.metadata.get("name", f"pop{pop.id}")
        for pop in ts.populations()           # iterate, no arguments
    ]

    # I don't want the ancestral size in the pop names
    if len(pop_names) > 1 and pop_names[0] == "N0":
        pop_names = pop_names[1:]  # Remove the ancestral population name

    # if ts.num_populations == 1:
    #     pop_names = [ts.populations(0).metadata.get("name", "N0")]
    # else:
    #     pop_names = ["N1", "N2"]

    sfs.pop_ids = pop_names
    
    return sfs
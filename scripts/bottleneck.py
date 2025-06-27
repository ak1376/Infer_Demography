#!/usr/bin/env python3
import argparse
import json
import pickle
import matplotlib.pyplot as plt
import demesdraw
import sys
from pathlib import Path
# Add src directory to Python path
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from simulation import bottleneck_model, simulation, create_SFS
from moments_inference import fit_model
from dadi_inference import fit_model as dadi_fit_model
import moments


def main(experiment_config, sampled_params):
    """
    Run the split migration model simulation and save visual + data outputs.

    Parameters:
    - experiment_config: Dictionary of experiment settings.
    - sampled_params: Dictionary of demographic model parameters.
    """

    # Save the sampled parameters for reference
    with open("./data/bottleneck_sampled_params.pkl", "wb") as f:
        pickle.dump(sampled_params, f)


    # Run the split migration model simulation
    g = bottleneck_model(sampled_params)

    # Draw and save the demography graph
    ax = demesdraw.tubes(g)
    ax.set_title("Bottleneck Model")
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population Size")
    plt.savefig("./data/demes_bottleneck_model.png", dpi=300, bbox_inches='tight')

    # Simulate and generate the tree sequence and SFS
    ts, g = simulation(sampled_params, model_type="bottleneck", experiment_config=experiment_config)
    SFS = create_SFS(ts)

    # Save outputs
    with open("./data/bottleneck_SFS.pkl", "wb") as f:
        pickle.dump(SFS, f)
    ts.dump("./data/bottleneck_tree_sequence.trees")

    start = [sampled_params['N0'], sampled_params['N_bottleneck'], sampled_params["N_recover"],
             sampled_params["t_bottleneck_start"], sampled_params["t_bottleneck_end"]]
    
    start = moments.Misc.perturb_params(start, fold=0.1)

    param_names = [
        "N0", "N_bottleneck", "N_recover", "t_bottleneck_start", "t_bottleneck_end"
    ]
    
    # run the optimisations exactly as you already do
    fit      = fit_model(SFS, start=start, g=g, experiment_config=experiment_config, sampled_params=sampled_params)
    dadi_fit = dadi_fit_model(SFS, start=start, g=g, experiment_config=experiment_config, sampled_params=sampled_params)

    # convert each array → dict
    fit      = [dict(zip(param_names, p.tolist())) for p in fit]
    dadi_fit = [dict(zip(param_names, p.tolist())) for p in dadi_fit]

    moments_file_name = f"./inferences/moments/{experiment_config['demographic_model']}_fit_params.pkl"
    dadi_file_name = f"./inferences/dadi/{experiment_config['demographic_model']}_fit_params.pkl"

    with open(moments_file_name, "wb") as f:
        pickle.dump(fit, f)

    # Save the best fit parameters for dadi inference
    with open(dadi_file_name, "wb") as f:
        pickle.dump(dadi_fit, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run split isolation model simulation and generate SFS")

    parser.add_argument("--experiment_config", type=str, required=True,
                        help="Path to experiment config JSON file")

    # Optional CLI override for each parameter
    parser.add_argument("--N0", type=int, default=10000, help="Effective population size of the ancestral population")
    parser.add_argument("--N_bottleneck", type=int, default=2000, help="Size of population after bottleneck")
    parser.add_argument("--N_recover", type=int, default=5000, help="Size of population 2 after split")
    parser.add_argument("--t_bottleneck_start", type=float, default=300, help="Migration rate ")
    parser.add_argument("--t_bottleneck_end", type=float, default=100, help="Time of split (generations)")

    args = parser.parse_args()

    # Load experiment config
    with open(args.experiment_config, "r") as f:
        experiment_config = json.load(f)

    # Build sampled parameter dictionary
    sampled_params = {
        "N0": args.N0,
        "N_bottleneck": args.N_bottleneck,
        "N_recover": args.N_recover,
        "t_bottleneck_start": args.t_bottleneck_start,
        "t_bottleneck_end": args.t_bottleneck_end
    }

    main(experiment_config, sampled_params)

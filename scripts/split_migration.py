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

from simulation import split_migration_model, simulation, create_SFS
from moments_inference import fit_model
import moments


def main(experiment_config, sampled_params):
    """
    Run the split migration model simulation and save visual + data outputs.

    Parameters:
    - experiment_config: Dictionary of experiment settings.
    - sampled_params: Dictionary of demographic model parameters.
    """

    # Save the sampled parameters for reference
    with open("./data/split_migration_sampled_params.pkl", "wb") as f:
        pickle.dump(sampled_params, f)


    # Run the split migration model simulation
    g = split_migration_model(sampled_params)

    # Draw and save the demography graph
    ax = demesdraw.tubes(g)
    ax.set_title("Split Migration Model")
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population Size")
    plt.savefig("./data/demes_split_migration_model.png", dpi=300, bbox_inches='tight')

    # Simulate and generate the tree sequence and SFS
    ts, g = simulation(sampled_params, model_type="split_migration", experiment_config=experiment_config)
    SFS = create_SFS(ts)

    # Save outputs
    with open("./data/split_migration_SFS.pkl", "wb") as f:
        pickle.dump(SFS, f)
    ts.dump("./data/split_migration_tree_sequence.trees")

    # Fit the model to the SFS

    start = [sampled_params["N0"], sampled_params["N1"], sampled_params["N2"],
             sampled_params["m12"], sampled_params["m21"], sampled_params["t_split"]]
    
    start = moments.Misc.perturb_params(start, fold=0.1)
    
    fit = fit_model(SFS, start=start, g = g, experiment_config = experiment_config)
    
    opt_params = {
        "N0": fit[0],
        "N1": fit[1],
        "N2": fit[2],
        "m12": fit[3],
        "m21": fit[4],
        "t_split": fit[5]
    }

    with open("./inferences/moments/split_migration_fit_params.pkl", "wb") as f:
        pickle.dump(opt_params, f)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run split migration model simulation and generate SFS")

    parser.add_argument("--experiment_config", type=str, required=True,
                        help="Path to experiment config JSON file")

    # Optional CLI override for each parameter
    parser.add_argument("--N0", type=int, default=10000, help="Ancestral population size")
    parser.add_argument("--N1", type=int, default=5000, help="Size of population 1 after split")
    parser.add_argument("--N2", type=int, default=5000, help="Size of population 2 after split")
    parser.add_argument("--m12", type=float, default=0.001, help="Migration rate from N1 to N2")
    parser.add_argument("--m21", type=float, default=0.001, help="Migration rate from N2 to N1")
    parser.add_argument("--t_split", type=float, default=10000, help="Time of split (generations)")

    args = parser.parse_args()

    # Load experiment config
    with open(args.experiment_config, "r") as f:
        experiment_config = json.load(f)

    # Build sampled parameter dictionary
    sampled_params = {
        "N0": args.N0,
        "N1": args.N1,
        "N2": args.N2,
        "m12": args.m12,
        "m21": args.m21,
        "t_split": args.t_split
    }

    main(experiment_config, sampled_params)

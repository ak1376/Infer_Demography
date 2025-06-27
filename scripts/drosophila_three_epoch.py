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

from simulation import drosophila_three_epoch, simulation, create_SFS
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

    # Create the data and inferences directories if they don't exist
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path("./inferences/moments").mkdir(parents=True, exist_ok=True)
    Path("./inferences/dadi").mkdir(parents=True, exist_ok=True)


    # Save the sampled parameters for reference
    with open("./data/drosophila_three_epoch_sampled_params.pkl", "wb") as f:
        pickle.dump(sampled_params, f)


    # Run the split migration model simulation
    g = drosophila_three_epoch(sampled_params)

    # Draw and save the demography graph
    ax = demesdraw.tubes(g)
    ax.set_title("Drosophila Three Epoch Model")
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Population Size")
    plt.savefig("./data/demes_drosophila_three_epoch.png", dpi=300, bbox_inches='tight')

    # Simulate and generate the tree sequence and SFS
    ts, g = simulation(sampled_params, model_type="drosophila_three_epoch", experiment_config=experiment_config)
    SFS = create_SFS(ts)

    # Save outputs
    with open("./data/drosophila_three_epoch_SFS.pkl", "wb") as f:
        pickle.dump(SFS, f)
    ts.dump("./data/drosophila_three_epoch_tree_sequence.trees")

    N0 = sampled_params["N0"]  # Ancestral population size
    AFR_recover = sampled_params["AFR"]  # Post expansion African population size
    EUR_bottleneck = sampled_params["EUR_bottleneck"]  # European bottleneck pop size
    EUR_recover = sampled_params["EUR_recover"]  # Modern European population size after recovery
    T_AFR_expansion = sampled_params["T_AFR_expansion"]  # Expansion of population in Africa
    T_AFR_EUR_split = sampled_params["T_AFR_EUR_split"]  # African-European Divergence
    T_EUR_expansion = sampled_params["T_EUR_expansion"]  # European

    start = [
        N0,          
        AFR_recover, 
        EUR_bottleneck,
        EUR_recover,
        T_AFR_expansion, 
        T_AFR_EUR_split,
        T_EUR_expansion
    ]
    
    start = moments.Misc.perturb_params(start, fold=0.1)

    # names in the exact order the optimisation routines output them
    param_names = [
        "N0",
        "AFR",
        "EUR_bottleneck",
        "EUR_recover",
        "T_AFR_expansion",
        "T_AFR_EUR_split",
        "T_EUR_expansion",
    ]

    # run the optimisations exactly as you already do
    fit      = fit_model(SFS, start=start, g=g, experiment_config=experiment_config)
    dadi_fit = dadi_fit_model(SFS, start=start, g=g, experiment_config=experiment_config)

    # convert each array → dict
    fit      = [dict(zip(param_names, p.tolist())) for p in fit]
    dadi_fit = [dict(zip(param_names, p.tolist())) for p in dadi_fit]

    with open("./inferences/moments/drosophila_three_epoch_fit_params.pkl", "wb") as f:
        pickle.dump(fit, f)

    # Save the best fit parameters for dadi inference
    with open("./inferences/dadi/drosophila_three_epoch_fit_params.pkl", "wb") as f:
        pickle.dump(dadi_fit, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drosophila three epoch model simulation and generate SFS")

    parser.add_argument("--experiment_config", type=str, required=True,
                        help="Path to experiment config JSON file")

    # Optional CLI override for each parameter
    parser.add_argument("--N0", type=int, default=10000, help="Ancestral population size")
    parser.add_argument("--AFR", type=int, default=10000, help="Post expansion African population size")
    parser.add_argument("--EUR_bottleneck", type=int, default=5000, help="European bottleneck population size")
    parser.add_argument("--EUR_recover", type=int, default=5000, help="Modern European population size after recovery")
    parser.add_argument("--T_AFR_expansion", type=float, default=10000, help="Time of expansion of population in Africa (generations)")
    parser.add_argument("--T_AFR_EUR_split", type=float, default=5000, help="Time of African-European divergence (generations)")
    parser.add_argument("--T_EUR_expansion", type=float, default=2000, help="Time of European population expansion (generations)")

    args = parser.parse_args()

    # Load experiment config
    with open(args.experiment_config, "r") as f:
        experiment_config = json.load(f)

    # Build sampled parameter dictionary
    sampled_params = {
        "N0": args.N0,
        "AFR": args.AFR,
        "EUR_bottleneck": args.EUR_bottleneck,
        "EUR_recover": args.EUR_recover,
        "T_AFR_expansion": args.T_AFR_expansion,
        "T_AFR_EUR_split": args.T_AFR_EUR_split,
        "T_EUR_expansion": args.T_EUR_expansion
    }

    main(experiment_config, sampled_params)

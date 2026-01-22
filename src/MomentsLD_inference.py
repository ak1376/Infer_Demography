"""
MomentsLD demographic parameter inference using linkage disequilibrium.

This module provides functions to:
1. Load and aggregate LD statistics from multiple simulation windows
2. Compute expected LD under demographic models using Demes graphs
3. Optimize demographic parameters via composite Gaussian likelihood
4. Generate comparison plots between empirical and theoretical LD
"""

from __future__ import annotations

import json
import logging
import pickle
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import moments
import nlopt
import numdifftools as nd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_R_BINS = np.array(
    [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
)
JITTER = 1e-12  # numerical stability for matrix inversion
CONVERGENCE_TOL = 1e-8  # default convergence tolerance for optimization


# =============================================================================
# Data Loading and Preparation
# =============================================================================


def load_sampled_params(sim_dir: Path, required: bool = True) -> Optional[Dict]:
    """Load sampled parameters from simulation directory.

    Parameters
    ----------
    sim_dir
        Directory containing 'sampled_params.pkl'.
    required
        If True, raise if file is missing; otherwise return None.

    Returns
    -------
    dict or None
    """
    pkl_file = sim_dir / "sampled_params.pkl"
    if not pkl_file.exists():
        if required:
            raise FileNotFoundError(f"sampled_params.pkl missing in {pkl_file.parent}")
        logging.info(
            "sampled_params.pkl not found in %s; continuing without true parameters",
            pkl_file.parent,
        )
        return None

    with pkl_file.open("rb") as f:
        return pickle.load(f)


def load_config(config_path: Path) -> Dict:
    """Load configuration from JSON file."""
    with config_path.open("r") as f:
        return json.load(f)


def aggregate_ld_statistics(ld_root: Path) -> Dict[str, List[np.ndarray]]:
    """
    Aggregate LD statistics from multiple windows into means and covariances.

    Parameters
    ----------
    ld_root
        Path to MomentsLD directory containing LD_stats/ subdirectory.

    Returns
    -------
    dict
        Dictionary with 'means' and 'varcovs' for empirical LD statistics.

    Side effects
    ------------
    Creates:
        - means.varcovs.pkl  (aggregated statistics)
        - bootstrap_sets.pkl (bootstrap data for variance estimation)
    """
    means_file = ld_root / "means.varcovs.pkl"
    boots_file = ld_root / "bootstrap_sets.pkl"

    # Return cached results if available
    if means_file.exists() and boots_file.exists():
        with means_file.open("rb") as f:
            return pickle.load(f)

    # Load individual LD statistics files
    ld_stats_dir = ld_root / "LD_stats"
    ld_files = list(ld_stats_dir.glob("LD_stats_window_*.pkl"))

    if not ld_files:
        raise RuntimeError(f"No LD statistics files found in {ld_stats_dir}")

    ld_stats = {}
    for pkl_file in ld_files:
        window_id = int(pkl_file.stem.split("_")[-1])
        with pkl_file.open("rb") as f:
            ld_stats[window_id] = pickle.load(f)

    # Aggregate using moments.LD
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    bootstrap_sets = moments.LD.Parsing.get_bootstrap_sets(ld_stats)

    # Save results
    with means_file.open("wb") as f:
        pickle.dump(mv, f)
    with boots_file.open("wb") as f:
        pickle.dump(bootstrap_sets, f)

    logging.info("Aggregated %d LD windows → %s", len(ld_stats), ld_root)
    return mv


# =============================================================================
# Theoretical LD Computation
# =============================================================================


def compute_theoretical_ld(
    params: List[float],
    param_names: List[str],
    demographic_model,
    r_bins: np.ndarray,
    populations: List[str],
):
    """
    Compute expected LD statistics under a demographic model.

    Parameters
    ----------
    params
        Parameter values in log10 space.
    param_names
        Parameter names, same order as `params`.
    demographic_model
        Function that creates a Demes graph from a parameter dict.
    r_bins
        Recombination rate bin edges (same as in empirical LD).
    populations
        List of population names to sample from.

    Returns
    -------
    moments.LD.LDstats
        Expected σD² statistics.
    """
    # Convert to absolute scale and create parameter dictionary
    param_values = 10 ** np.array(params)
    param_dict = dict(zip(param_names, param_values))

    # Create demographic graph
    graph = demographic_model(param_dict)

    # Reference population size for scaling
    ref_size = (
        param_dict.get("N0")
        or param_dict.get("N_ANC")
        or next((v for k, v in param_dict.items() if k.startswith("N")), 1.0)
    )

    # Compute LD using Simpson-like integration across r-bins
    rho_edges = 4.0 * ref_size * np.array(r_bins)
    ld_edges = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_edges)

    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_mids)

    # Simpson's rule weighted average within each bin
    ld_bins = [
        (ld_edges[i] + ld_edges[i + 1] + 4 * ld_mids[i]) / 6.0
        for i in range(len(rho_mids))
    ]
    ld_bins.append(ld_edges[-1])  # Add final edge value

    # Convert to σD² format
    ld_stats = moments.LD.LDstats(
        ld_bins, num_pops=ld_edges.num_pops, pop_ids=ld_edges.pop_ids
    )

    # Debug: Log theoretical LD properties
    logging.info(
        f"[DEBUG] Theoretical LD - num_pops: {ld_stats.num_pops}, pop_ids: {ld_stats.pop_ids}"
    )
    logging.info(f"[DEBUG] Theoretical LD - shape: {len(ld_bins)} bins")
    logging.info(f"[DEBUG] Populations parameter passed: {populations}")
    logging.info(
        f"[DEBUG] Expected stats for {ld_stats.num_pops} pops: {moments.LD.Util.moment_names(ld_stats.num_pops)}"
    )

    try:
        result = moments.LD.Inference.sigmaD2(ld_stats)
        logging.info(
            f"[DEBUG] Theoretical sigmaD2 - shape: {len(result)}, num_pops: {result.num_pops}"
        )
        return result
    except Exception as e:
        logging.error(f"[DEBUG] Error in sigmaD2 conversion: {e}")
        raise


def prepare_data_for_comparison(
    theoretical_ld,
    empirical_data: Dict[str, List[np.ndarray]],
    normalization: int = 0,
):
    """
    Prepare theoretical and empirical data for likelihood comparison.

    Parameters
    ----------
    theoretical_ld
        Output from compute_theoretical_ld().
    empirical_data
        Dictionary with 'means' and 'varcovs' keys.
    normalization
        LD normalization scheme (0 = no normalization).

    Returns
    -------
    (theory_arrays, empirical_means, empirical_covariances)
    """
    # Process theoretical predictions
    theory_processed = moments.LD.LDstats(
        theoretical_ld[:],
        num_pops=theoretical_ld.num_pops,
        pop_ids=theoretical_ld.pop_ids,
    )
    theory_processed = moments.LD.Inference.remove_normalized_lds(
        theory_processed, normalization=normalization
    )
    # Remove heterozygosity
    theory_arrays = [np.array(pred) for pred in theory_processed[:-1]]

    # Process empirical data
    emp_means = [np.array(x) for x in empirical_data["means"]]
    emp_covars = [np.array(x) for x in empirical_data["varcovs"]]

    # Debug: Log empirical data properties
    logging.info(f"[DEBUG] Empirical data - means: {len(emp_means)} arrays")
    logging.info(
        f"[DEBUG] Empirical data - mean shapes: {[arr.shape for arr in emp_means]}"
    )
    logging.info(f"[DEBUG] Empirical data - covars: {len(emp_covars)} arrays")
    logging.info(
        f"[DEBUG] Empirical data - covar shapes: {[arr.shape for arr in emp_covars]}"
    )

    # Debug: Log detailed contents of first few empirical arrays
    for i, (mean_arr, cov_arr) in enumerate(zip(emp_means[:3], emp_covars[:3])):
        logging.info(f"[DEBUG] Bin {i} - mean values: {mean_arr}")
        logging.info(f"[DEBUG] Bin {i} - covar shape: {np.array(cov_arr).shape}")

    # Debug: Check what remove_normalized_data expects for single population
    logging.info(f"[DEBUG] About to call remove_normalized_data with:")
    logging.info(f"[DEBUG] - len(emp_means): {len(emp_means)}")
    logging.info(f"[DEBUG] - len(emp_covars): {len(emp_covars)}")
    logging.info(f"[DEBUG] - normalization: {normalization}")
    logging.info(f"[DEBUG] - num_pops: {theoretical_ld.num_pops}")

    # Remove normalized statistics
    try:
        # Let moments.LD determine the statistics automatically based on num_pops
        logging.info(
            f"[DEBUG] Calling remove_normalized_data with num_pops={theoretical_ld.num_pops}, normalization={normalization}"
        )

        emp_means, emp_covars = moments.LD.Inference.remove_normalized_data(
            emp_means,
            emp_covars,
            normalization=normalization,
            num_pops=theoretical_ld.num_pops,
        )
        logging.info(
            f"[DEBUG] After remove_normalized_data - means: {len(emp_means)}, covars: {len(emp_covars)}"
        )
    except Exception as e:
        logging.error(f"[DEBUG] Error in remove_normalized_data: {e}")
        logging.error(
            f"[DEBUG] - normalization: {normalization}, num_pops: {theoretical_ld.num_pops}"
        )

        # Debug the expected vs actual statistics
        expected_stats = moments.LD.Util.moment_names(theoretical_ld.num_pops)
        logging.error(f"[DEBUG] Expected LD statistics: {expected_stats[0]}")
        logging.error(f"[DEBUG] Expected H statistics: {expected_stats[1]}")

        # Check what the empirical data actually contains
        logging.error(f"[DEBUG] Empirical data structure:")
        for i, (mean_arr, cov_arr) in enumerate(
            zip(emp_means[:3], emp_covars[:3])
        ):  # Show first 3
            logging.error(
                f"[DEBUG] Bin {i} - mean shape: {mean_arr.shape}, values: {mean_arr[:5] if len(mean_arr) > 0 else 'empty'}"
            )
            logging.error(f"[DEBUG] Bin {i} - covar shape: {np.array(cov_arr).shape}")
        raise

    # Remove heterozygosity statistics
    emp_means = emp_means[:-1]
    emp_covars = emp_covars[:-1]

    return theory_arrays, emp_means, emp_covars


# =============================================================================
# Likelihood Computation
# =============================================================================


def compute_composite_likelihood(
    empirical_means: List[np.ndarray],
    empirical_covariances: List[np.ndarray],
    theoretical_predictions: List[np.ndarray],
) -> float:
    """
    Compute composite Gaussian log-likelihood.

    Parameters
    ----------
    empirical_means
        List of empirical mean vectors for each r-bin.
    empirical_covariances
        List of covariance matrices for each r-bin.
    theoretical_predictions
        List of model prediction vectors for each r-bin.

    Returns
    -------
    float
        Composite log-likelihood.
    """
    total_loglik = 0.0

    logging.info(f"[DEBUG] Computing likelihood for {len(empirical_means)} r-bins")
    logging.info(
        f"[DEBUG] Input shapes - means: {[arr.shape for arr in empirical_means]}"
    )
    logging.info(
        f"[DEBUG] Input shapes - covars: {[arr.shape for arr in empirical_covariances]}"
    )
    logging.info(
        f"[DEBUG] Input shapes - preds: {[arr.shape for arr in theoretical_predictions]}"
    )

    for i, (obs, cov, pred) in enumerate(
        zip(empirical_means, empirical_covariances, theoretical_predictions)
    ):
        logging.info(
            f"[DEBUG] Processing r-bin {i}: obs={obs.shape}, cov={np.array(cov).shape}, pred={pred.shape}"
        )

        if len(obs) == 0:
            logging.info(f"[DEBUG] Skipping empty bin {i}")
            continue

        residual = obs - pred
        cov_matrix = np.array(cov)

        logging.info(
            f"[DEBUG] Bin {i} - residual shape: {residual.shape}, cov_matrix shape: {cov_matrix.shape}"
        )

        if cov_matrix.ndim == 2 and cov_matrix.size > 1:
            # Add small jitter for numerical stability
            cov_matrix += np.eye(cov_matrix.shape[0]) * JITTER

            try:
                cov_inv = np.linalg.inv(cov_matrix)
                total_loglik -= 0.5 * float(residual @ cov_inv @ residual)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                cov_inv = np.linalg.pinv(cov_matrix)
                total_loglik -= 0.5 * float(residual @ cov_inv @ residual)
        else:
            # Scalar variance case
            total_loglik -= 0.5 * float(residual @ residual)

    return total_loglik


def objective_function(
    log_params: np.ndarray,
    param_names: List[str],
    demographic_model,
    r_bins: np.ndarray,
    empirical_data: Dict[str, List[np.ndarray]],
    populations: List[str],
    normalization: int = 0,
) -> float:
    """
    Objective function for optimization: composite log-likelihood.

    Parameters
    ----------
    log_params
        Parameters in log10 space.
    param_names
        List of parameter names.
    demographic_model
        Function creating Demes graph.
    r_bins
        Recombination rate bins.
    empirical_data
        Dictionary with empirical LD statistics.
    populations
        Population names to sample.
    normalization
        LD normalization scheme.

    Returns
    -------
    float
        Composite log-likelihood.
    """
    try:
        logging.info(f"[DEBUG] Objective function called with log_params: {log_params}")

        # Compute theoretical LD
        logging.info(f"[DEBUG] Computing theoretical LD...")
        theoretical_ld = compute_theoretical_ld(
            log_params, param_names, demographic_model, r_bins, populations
        )
        logging.info(f"[DEBUG] Theoretical LD computed successfully")

        # Prepare data for comparison
        logging.info(f"[DEBUG] Preparing data for comparison...")
        theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(
            theoretical_ld, empirical_data, normalization
        )
        logging.info(
            f"[DEBUG] Data preparation completed - theory: {len(theory_arrays)}, emp_means: {len(emp_means)}, emp_covars: {len(emp_covars)}"
        )

        # Compute likelihood
        logging.info(f"[DEBUG] Computing composite likelihood...")
        likelihood = compute_composite_likelihood(emp_means, emp_covars, theory_arrays)
        logging.info(f"[DEBUG] Likelihood computed: {likelihood}")
        return likelihood

    except Exception as e:
        logging.warning("Error in objective function: %s", e)
        logging.warning(f"[DEBUG] Error details - type: {type(e)}, args: {e.args}")
        import traceback

        logging.warning(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return -np.inf


# =============================================================================
# Parameter Optimization
# =============================================================================


def handle_fixed_parameters(
    config: Dict,
    sampled_params: Optional[Dict[str, float]],
    param_names: List[str],
) -> List[Optional[float]]:
    """
    Parse configuration to determine which parameters should be fixed.

    Parameters
    ----------
    config
        Configuration dictionary.
    sampled_params
        Dictionary of sampled parameter values (optional).
    param_names
        List of all parameter names.

    Returns
    -------
    list
        Each element is either None (free parameter) or a float (fixed value).
    """
    fixed_values = [None] * len(param_names)
    fixed_config = config.get("fixed_parameters", {})

    for i, param_name in enumerate(param_names):
        if param_name not in fixed_config:
            continue

        fixed_spec = fixed_config[param_name]

        if isinstance(fixed_spec, (int, float)):
            fixed_values[i] = float(fixed_spec)
        elif isinstance(fixed_spec, str) and fixed_spec.lower() in {"sampled", "true"}:
            if sampled_params is None or param_name not in sampled_params:
                logging.warning(
                    "Config requested %s be fixed to '%s', but sampled_params "
                    "are unavailable. Leaving this parameter free instead.",
                    param_name,
                    fixed_spec,
                )
                continue
            fixed_values[i] = float(sampled_params[param_name])
        else:
            raise ValueError(
                f"Invalid fixed parameter specification for {param_name}: {fixed_spec}"
            )

    return fixed_values


def create_free_parameter_vectors(
    full_params: np.ndarray,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    fixed_values: Optional[List[Optional[float]]],
):
    """
    Extract free parameters and their bounds from full parameter specifications.

    Returns
    -------
    (free_params, free_lower_bounds, free_upper_bounds, expand_function)
    """
    if fixed_values is None or all(v is None for v in fixed_values):
        # All parameters are free
        return full_params, bounds_lower, bounds_upper, lambda x: x

    free_indices = [i for i, fixed_val in enumerate(fixed_values) if fixed_val is None]
    free_params = full_params[free_indices]
    free_lower = bounds_lower[free_indices]
    free_upper = bounds_upper[free_indices]

    def expand_to_full(free_values: np.ndarray) -> np.ndarray:
        """Reconstruct full parameter vector from free parameters."""
        full = np.zeros(len(fixed_values))
        free_idx = 0
        for i, fixed_val in enumerate(fixed_values):
            if fixed_val is None:
                full[i] = free_values[free_idx]
                free_idx += 1
            else:
                full[i] = fixed_val
        return full

    return free_params, free_lower, free_upper, expand_to_full


def optimize_parameters(
    start_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    param_names: List[str],
    demographic_model,
    r_bins: np.ndarray,
    empirical_data: Dict[str, List[np.ndarray]],
    populations: List[str],
    normalization: int = 0,
    tolerance: float = CONVERGENCE_TOL,
    verbose: bool = True,
    fixed_values: Optional[List[Optional[float]]] = None,
):
    """
    Optimize demographic parameters using L-BFGS (via nlopt).

    Returns
    -------
    (optimal_parameters, max_log_likelihood, status_code)
    """
    # Convert to log10 space for optimization
    start_log10 = np.log10(np.maximum(start_values, 1e-300))
    bounds_lower_log10 = np.log10(np.maximum(lower_bounds, 1e-300))
    bounds_upper_log10 = np.log10(upper_bounds)

    # Handle fixed parameters
    fixed_log10 = (
        None
        if fixed_values is None
        else [None if v is None else np.log10(max(v, 1e-300)) for v in fixed_values]
    )

    free_start, free_lower, free_upper, expand_function = create_free_parameter_vectors(
        start_log10, bounds_lower_log10, bounds_upper_log10, fixed_log10
    )

    # Track best solution found
    best_likelihood = -np.inf
    best_params = None

    def nlopt_objective(free_params, gradient):
        nonlocal best_likelihood, best_params

        # Expand to full parameter set
        full_params_log10 = expand_function(free_params)

        # Compute likelihood
        likelihood = objective_function(
            full_params_log10,
            param_names,
            demographic_model,
            r_bins,
            empirical_data,
            populations,
            normalization,
        )

        # Track best solution
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_params = np.array(free_params)

        # Compute numerical gradient if requested
        if gradient.size > 0:
            gradient_func = nd.Gradient(
                lambda p: objective_function(
                    expand_function(p),
                    param_names,
                    demographic_model,
                    r_bins,
                    empirical_data,
                    populations,
                    normalization,
                ),
                step=1e-4,
            )
            gradient[:] = gradient_func(free_params)

        if verbose:
            param_values = 10**full_params_log10
            param_str = ", ".join(
                f"{name}={val:.3g}" for name, val in zip(param_names, param_values)
            )
            print(f"LL = {likelihood:.6f} | {param_str}")

        return likelihood

    # Set up and run optimization
    optimizer = nlopt.opt(nlopt.LD_LBFGS, len(free_start))
    optimizer.set_lower_bounds(free_lower)
    optimizer.set_upper_bounds(free_upper)
    optimizer.set_max_objective(nlopt_objective)
    optimizer.set_ftol_rel(tolerance)

    try:
        optimal_free = optimizer.optimize(free_start)
        status = optimizer.last_optimize_result()
        max_likelihood = optimizer.last_optimum_value()
    except Exception as e:
        logging.warning("Optimization failed: %s. Using best solution found.", e)
        if best_params is not None:
            optimal_free = best_params
            max_likelihood = best_likelihood
            status = -1
        else:
            optimal_free = free_start
            max_likelihood = best_likelihood
            status = -1

    # Use tracked best if better
    if best_params is not None and best_likelihood > max_likelihood:
        optimal_free = best_params
        max_likelihood = best_likelihood

    # Convert back to absolute scale
    optimal_full_log10 = expand_function(optimal_free)
    optimal_params = 10**optimal_full_log10

    return optimal_params, max_likelihood, status


# =============================================================================
# Visualization
# =============================================================================


def _load_demographic_function(config):
    """Helper to load the demographic model function from src/simulation.py."""
    # Import as part of the src package so relative imports inside simulation.py work
    demo_module = importlib.import_module("src.simulation")
    model_name = config["demographic_model"]

    if model_name == "drosophila_three_epoch":
        return getattr(demo_module, "drosophila_three_epoch")

    return getattr(demo_module, model_name + "_model")


def create_comparison_plot(
    config: Dict,
    sampled_params: Dict[str, float],
    empirical_data: Dict[str, List[np.ndarray]],
    r_bins: np.ndarray,
    output_path: Path,
):
    """
    Create comparison plot between empirical and theoretical LD curves.

    Parameters
    ----------
    config
        Configuration dictionary with demographic model info.
    sampled_params
        Dictionary of sampled parameter values.
    empirical_data
        Dictionary with empirical LD statistics.
    r_bins
        Recombination rate bins.
    output_path
        Path where to save the PDF.
    """
    if output_path.exists():
        return

    try:
        demo_function = _load_demographic_function(config)
        populations = list(config["num_samples"].keys())

        param_names = list(config["priors"].keys())
        log_params = [np.log10(sampled_params[name]) for name in param_names]

        theoretical_ld = compute_theoretical_ld(
            log_params, param_names, demo_function, r_bins, populations
        )

        # Extract empirical data (excluding heterozygosity like in the docs example)
        emp_means = empirical_data["means"][:-1]
        emp_covars = empirical_data["varcovs"][:-1]

        # Handle dimension mismatch gracefully
        r_vec_plot = r_bins
        theory_for_plot = theoretical_ld

        if len(emp_means) < len(r_bins):
            logging.warning(
                "Truncating r_bins from %d to %d for plotting",
                len(r_bins),
                len(emp_means),
            )
            r_vec_plot = r_bins[: len(emp_means)]
            theory_for_plot = compute_theoretical_ld(
                [np.log10(sampled_params[name]) for name in param_names],
                param_names,
                demo_function,
                r_vec_plot,
                populations,
            )

        # Define statistics to plot based on model type
        if config["demographic_model"] == "bottleneck":
            stats_to_plot = [["DD_0_0"], ["Dz_0_0_0"], ["pi2_0_0_0_0"]]
            labels = [[r"$D_0^2$"], [r"$Dz_{0,0,0}$"], [r"$\pi_{2;0,0,0,0}$"]]
            rows = 2
        else:
            stats_to_plot = [
                ["DD_0_0"],
                ["DD_0_1"],
                ["DD_1_1"],
                ["Dz_0_0_0"],
                ["Dz_0_1_1"],
                ["Dz_1_1_1"],
                ["pi2_0_0_1_1"],
                ["pi2_0_1_0_1"],
                ["pi2_1_1_1_1"],
            ]
            labels = [
                [r"$D_0^2$"],
                [r"$D_0 D_1$"],
                [r"$D_1^2$"],
                [r"$Dz_{0,0,0}$"],
                [r"$Dz_{0,1,1}$"],
                [r"$Dz_{1,1,1}$"],
                [r"$\pi_{2;0,0,1,1}$"],
                [r"$\pi_{2;0,1,0,1}$"],
                [r"$\pi_{2;1,1,1,1}$"],
            ]
            rows = 3

        fig = moments.LD.Plotting.plot_ld_curves_comp(
            theory_for_plot,
            emp_means,
            emp_covars,
            rs=r_vec_plot,
            stats_to_plot=stats_to_plot,
            labels=labels,
            rows=rows,
            plot_vcs=True,
            show=False,
            fig_size=(6, 4),
        )

        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        logging.info("Comparison plot saved → %s", output_path)

    except Exception as e:
        logging.warning("Plot generation failed: %s", e)
        logging.warning("Creating empty file to satisfy dependencies")
        output_path.touch()


# =============================================================================
# Main Interface
# =============================================================================


def run_momentsld_inference(
    config: Dict,
    empirical_data: Dict[str, List[np.ndarray]],
    results_dir: Path,
    r_bins: np.ndarray,
    sampled_params: Optional[Dict[str, float]] = None,
):
    """
    Main function to run MomentsLD demographic parameter inference.

    This function:
    1. Sets up the optimization problem from configuration
    2. Handles any fixed parameters
    3. Runs L-BFGS optimization to find best-fit parameters
    4. Saves results and creates comparison plots

    Parameters
    ----------
    config
        Configuration dictionary with priors, demographic model, etc.
    empirical_data
        Dictionary with aggregated LD statistics.
    results_dir
        Directory where results will be saved.
    r_bins
        Recombination rate bin edges.
    sampled_params
        Optional true parameter values for fixing parameters and plotting.
    """
    results_file = results_dir / "best_fit.pkl"
    if results_file.exists():
        logging.info("Results already exist - skipping optimization")
        return

    # Load demographic model function
    demo_function = _load_demographic_function(config)

    # Extract parameter setup from configuration
    priors = config["priors"]
    param_names = list(priors.keys())
    lower_bounds = np.array([prior[0] for prior in priors.values()])
    upper_bounds = np.array([prior[1] for prior in priors.values()])
    start_values = np.sqrt(lower_bounds * upper_bounds)  # geometric mean starting point

    # Get populations to sample
    populations = list(config.get("num_samples", {}).keys())
    if not populations:
        raise ValueError(
            "Configuration must specify 'num_samples' to determine populations"
        )

    # Parse optimization settings
    normalization = config.get("ld_normalization", 0)
    tolerance = config.get("ld_rtol", CONVERGENCE_TOL)
    verbose = config.get("ld_verbose", True)

    # Handle fixed parameters
    fixed_values = handle_fixed_parameters(config, sampled_params, param_names)

    # Run optimization
    logging.info("Starting MomentsLD parameter optimization...")
    optimal_params, max_likelihood, status = optimize_parameters(
        start_values=start_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        param_names=param_names,
        demographic_model=demo_function,
        r_bins=r_bins,
        empirical_data=empirical_data,
        populations=populations,
        normalization=normalization,
        tolerance=tolerance,
        verbose=verbose,
        fixed_values=fixed_values,
    )

    # Save results
    results = {
        "best_params": dict(zip(param_names, optimal_params)),
        "best_lls": max_likelihood,
        "status": status,
    }

    with results_file.open("wb") as f:
        pickle.dump(results, f)

    logging.info(
        "Optimization completed: LL = %.6f, status = %s",
        max_likelihood,
        status,
    )
    logging.info("Results saved → %s", results_file)

    # Create comparison plot if sampled parameters are available
    if sampled_params is not None:
        plot_file = results_dir / "empirical_vs_theoretical_comparison.pdf"
        create_comparison_plot(
            config, sampled_params, empirical_data, r_bins, plot_file
        )

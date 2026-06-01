import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import torch

from copy import copy
from tqdm.auto import tqdm
from typing import List, Union
import time

from utils.datasets.uci_covariate_shift import sample_shifted_covariates
from utils.argparser import _str2bool, _parse_n_cal
from datasets.uci_covariate_shift import get_uci_dataset
from baselines.cb.cb import run_mcmc, create_list_of_features_for_linear_model, run_CBMA


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


############################################################
# Experiment Code
############################################################
def main(dataset_name: str, data_dir: str, results_dir: str, model_results_dir: str, n_cal: int = 5, 
         fit_posterior: bool = False, only_fit_posterior: bool = False, num_trials: int = 300,
         use_grid: bool = True, adaptive_grid: bool = True, check_only_endpoints_adaptively: bool = True,
         grid_initial_shift: float = 10, grid_size: int = 100, max_grid_expansions: int = 10, 
         max_n_plot: int = 200, shifts_to_run: Union[None, List] = None,
         save_results: bool = True, exp_name: str = "",
         ):
    """
    Run CBMA experiments on UCI covariate-shift settings.

    Parameters
    ----------
    dataset_name : str
        UCI dataset identifier.
    data_dir : str
        Directory containing dataset files.
    results_dir : str
        Directory where experiment results are saved.
    model_results_dir : str
        Directory containing or storing CBMA model artifacts.
    n_cal : int or str, default=5
        Calibration-set size or ``"standard"``.
    fit_posterior : bool, default=False
        Whether to fit the Bayesian posterior before evaluation.
    only_fit_posterior : bool, default=False
        Whether to stop after fitting the posterior.
    num_trials : int, default=300
        Number of calibration/test trials per shift.
    use_grid : bool, default=True
        Whether to use grid-based conformal region computation.
    adaptive_grid : bool, default=True
        Whether to adaptively expand the grid.
    check_only_endpoints_adaptively : bool, default=True
        Whether adaptive checks use only interval endpoints.
    grid_initial_shift : float, default=10
        Initial padding used for the candidate y-grid.
    grid_size : int, default=100
        Initial number of y-grid points.
    max_grid_expansions : int, default=10
        Maximum adaptive grid expansions.
    max_n_plot : int, default=200
        Maximum grid points used by downstream plotting/evaluation code.
    shifts_to_run : list[int] or None, optional
        Shift indices to run. If ``None``, run all shifts.
    save_results : bool, default=True
        Whether to save outputs to disk.
    exp_name : str, default=""
        Prefix added to saved result filenames.

    Returns
    -------
    None
        Runs the experiment and optionally writes result files.
    """
    assert n_cal in [5, 10, 25, 50, "standard"], "Invalid calibration set size. Calibration set size must be one of [5, 10, 25, 50, 'standard']"
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Get dataset
    X, y, _, shift_vectors = get_uci_dataset(dataset_name, data_dir)
    n = X.shape[0]
    d = X.shape[1]
    print(f"Dataset {dataset_name} has size n:", n, " dimension d:", d)

    # Data args
    test_frac = 0.2   
    n_test = int(test_frac * n)
    n_train_max = n - n_test
    n_train = min(int(n_train_max//2), 5000)
    if n_cal == "standard":
        n_cal = n_train
    alpha = max(0.1, 1/(n_cal + 1))  # ensure alpha is not too small
    print(f"Using alpha={alpha} for n_cal={n_cal}")    
    print("Max training set size:", n_train_max)
    print("Training set size:", n_train)    

    # Modelling args
    n_models = 4
    features_list = create_list_of_features_for_linear_model(X.shape[1], n_models)
    n_chains = 4   
    n_samples_per_chain = B = 100    
    cb_model = None  # unused

    # Experiment args
    methods = ["cbma"]
    results = np.empty((num_trials, len(shift_vectors), len(methods), 2))
    times = []
    model_name = "linear"
    model_save_suffix = "cbma"


    ####################################
    # Experiment loop
    ###################################
    for j, shift in enumerate(shift_vectors):    
        if shifts_to_run is not None and j not in shifts_to_run:
            print(f"Skipping shift {j} as it is not in shifts_to_run.")
            continue        
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)            
        extra_suffix = f"{model_save_suffix}_shift_{j}"

        # Get remaining data to be used for training, test and calibration data
        X_remaining, y_remaining = copy(X), copy(y)
        remaining_indices = np.arange(len(y))        

        # Standardize
        X_mean = np.mean(X_remaining, axis=0)
        X_std = np.std(X_remaining, axis=0) + 1e-6
        X_remaining_standardized = (X_remaining - X_mean) / X_std  # we use the standardized version of the data to sample shifted covariates

        # Sample shift
        if shift is not None:
            shift_idx = sample_shifted_covariates(X_remaining_standardized, shift, n_train)
            X_train_ = X_remaining[shift_idx]
            y_train_ = y_remaining[shift_idx]
            remaining_indices_after_train = np.setdiff1d(remaining_indices, remaining_indices[shift_idx])
        else:
            X_train_ = X_remaining[:n_train]
            y_train_ = y_remaining[:n_train]
            remaining_indices_after_train = remaining_indices[n_train:]

        # Find statistics for datasets 
        X_mean = np.mean(X_train_, axis=0)
        X_std = np.std(X_train_, axis=0) + 1e-6
        y_mean = np.mean(y_train_)
        y_std = np.std(y_train_) + 1e-6

        X_train_ = (X_train_ - X_mean) / X_std
        y_train_ = (y_train_ - y_mean) / y_std        

        # Fit the posterior on the training data and save the results
        run_mcmc(X_train_, y_train_, dataset_name, results_dir=model_results_dir,
                features_of_models=features_list, B=n_samples_per_chain,
                n_chains=n_chains, n_training_size=n_train, extra_suffix=extra_suffix, 
                fit_posterior=fit_posterior)
        print("Trained fixed model for new shift.")
        if only_fit_posterior:
            print("Only fitting posterior. Skipping calculation of conformal intervals.")
            continue

        t0 = time.time()
        for i in tqdm(range(num_trials)):
            # Sample calibration data
            cal_indices = np.random.choice(remaining_indices_after_train, size=n_cal, replace=False)
            X_cal_raw, y_cal_raw = X[cal_indices], y[cal_indices]
            X_cal = (X_cal_raw - X_mean) / X_std
            y_cal = (y_cal_raw - y_mean) / y_std
            y_plot = np.linspace(np.min(y_cal) - grid_initial_shift, np.max(y_cal) + grid_initial_shift, grid_size)  # default is 100 points, 2 difference
            remaining_indices_after_cal = np.setdiff1d(remaining_indices_after_train, cal_indices)

            # Sample test
            test_indices = np.random.choice(remaining_indices_after_cal, size=n_test, replace=False)
            X_test_raw, y_test_raw = X[test_indices], y[test_indices]
            X_test = (X_test_raw - X_mean) / X_std
            y_test = (y_test_raw - y_mean) / y_std                                    

            # Get CBMA results (Conformal Bayes with Model Averaging)
            cbma_results = run_CBMA(use_grid, X_cal, y_cal, X_test, y_test, y_plot, 
                                    dataset_name, results_dir=model_results_dir,
                                    alpha=alpha, 
                                    features_of_models=features_list,
                                    extra_suffix=extra_suffix,
                                    n_training_size=n_train,
                                    adaptive_grid=adaptive_grid,
                                    check_only_endpoints_adaptively=check_only_endpoints_adaptively,
                                    max_grid_expansions=max_grid_expansions,
                                    max_n_plot=max_n_plot)

            # Add CBMA results
            coverage_cbma = cbma_results["coverage_cbma"]
            length_cbma = cbma_results["length_cbma"]
            avg_coverage_cbma = np.mean(coverage_cbma)  # average over test points
            avg_length_cbma = np.mean(length_cbma)
            results[i, j, methods.index("cbma"), 0] = (avg_length_cbma)
            results[i, j, methods.index("cbma"), 1] = (avg_coverage_cbma)

    t1 = time.time()
    times.append(t1 - t0)                                                    
            
            


    ####################################
    # Save results
    ###################################
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_name = f"uci_{dataset_name}_methods_{'_'.join(methods)}_trials{num_trials}_alpha{alpha:.2f}_ncal{n_cal}_ntrain{n_train}_model_{model_name}_samples{n_samples_per_chain}"
    print("Exp. name is:\n", save_name)
    if save_results:
        config = {"methods": methods,
                "alpha": alpha,
                "shift_vectors": shift_vectors,
                "num_trials": num_trials,
                "n_test": n_test,
                "n_train_max": n_train_max,
                "n_train": n_train,
                "n_cal": n_cal,
                "num_trials": num_trials,
                "features_list": features_list,
                "n_chains": n_chains,
                "n_samples_per_chain": n_samples_per_chain,
                "n_models": n_models,
                "cb_model": cb_model,
                "model_results_dir": model_results_dir,
                "time_taken_for_all_trials": times
                }

        # Save the the results
        np.savez_compressed(
                os.path.join(results_dir, f"{exp_name}_{save_name}_results.npz"),
                results=results,
                rmses=None,
                config=config
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for CBMA experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run UCI CBMA experiments.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_results_dir", type=str, required=True)
    parser.add_argument("--n_cal", type=_parse_n_cal, default=5)
    parser.add_argument("--fit_posterior", type=_str2bool, default=False)
    parser.add_argument("--only_fit_posterior", type=_str2bool, default=False)
    parser.add_argument("--num_trials", type=int, default=300)
    parser.add_argument("--use_grid", type=_str2bool, default=True)
    parser.add_argument("--adaptive_grid", type=_str2bool, default=True)
    parser.add_argument("--check_only_endpoints_adaptively", type=_str2bool, default=True)
    parser.add_argument("--grid_initial_shift", type=float, default=10)
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--max_grid_expansions", type=int, default=10)
    parser.add_argument("--max_n_plot", type=int, default=200)
    parser.add_argument("--shifts_to_run", nargs="+", type=int, default=None, help="List of shift indices to run. If not provided, runs all shifts.")
    parser.add_argument("--save_results", type=_str2bool, default=True)
    parser.add_argument("--exp_name", type=str, default="")
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    main(
        dataset_name=_args.dataset_name,
        data_dir=_args.data_dir,
        results_dir=_args.results_dir,
        model_results_dir=_args.model_results_dir,
        n_cal=_args.n_cal,
        fit_posterior=_args.fit_posterior,
        only_fit_posterior=_args.only_fit_posterior,
        num_trials=_args.num_trials,
        use_grid=_args.use_grid,
        adaptive_grid=_args.adaptive_grid,
        check_only_endpoints_adaptively=_args.check_only_endpoints_adaptively,
        grid_initial_shift=_args.grid_initial_shift,
        grid_size=_args.grid_size,
        max_grid_expansions=_args.max_grid_expansions,
        max_n_plot=_args.max_n_plot,
        shifts_to_run=_args.shifts_to_run,
        save_results=_args.save_results,
        exp_name=_args.exp_name
    )

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Union, List
import time

from baselines.cb.cb import run_mcmc, create_list_of_features_for_linear_model, run_CB
from utils.datasets.image import get_image_dataset
from utils.argparser import _str2bool, _parse_n_cal

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(dataset_name: str, embeddings_path: str, results_dir: str, model_results_dir: str, n_cal: int = 5, 
        num_trials: int = 300, fit_posterior: bool = False, only_fit_posterior: bool = False,
         use_grid: bool = True, adaptive_grid: bool = True, check_only_endpoints_adaptively: bool = True,
         grid_initial_shift: float = 10, grid_size: int = 100, max_grid_expansions: int = 10, 
         max_n_plot: int = 200, sets_to_run: Union[None, List] = None,
         save_results: bool = True, exp_name: str = "",
        ):
    """
    Run CB experiments on image datasets.

    Parameters
    ----------
    dataset_name : str
        Image dataset identifier.
    embeddings_path : str
        Path to embedding features used for experiments.
    results_dir : str
        Directory where experiment results are saved.
    model_results_dir : str
        Directory containing or storing CB model artifacts.
    n_cal : int or str, default=5
        Calibration-set size or ``"standard"``.
    num_trials : int, default=300
        Number of calibration/test trials per set.
    fit_posterior : bool, default=False
        Whether to fit the Bayesian posterior before evaluation.
    only_fit_posterior : bool, default=False
        Whether to stop after fitting the posterior.
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
        Maximum grid points used by downstream evaluation code.
    sets_to_run : list[int] or None, optional
        Set indices to run. If ``None``, run all sets.
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
    X_train, y_train, remaining_sets_for_cal, remaining_sets_for_cal_names, n, n_train = get_image_dataset(dataset_name, embeddings_path)

    # Data args
    test_frac = 0.2
    methods = ["cb"]
    alpha = max(0.1, 1/(n_cal + 1))  # ensure alpha is not too small
    print(f"Using alpha={alpha} for n_cal={n_cal}")

    # Experiment args
    results = np.empty((num_trials, len(remaining_sets_for_cal), len(methods), 2))   # variable to store results
    model_name_suffix = "cb"   # "" for CB, cbma_for CBMA
    times = []
                        
    # Model args
    n_models = 1  # use all of the features
    features_list = create_list_of_features_for_linear_model(X_train.shape[1], n_models)
    n_chains = 4   
    n_samples_per_chain = B = 100
    cb_model = 0   

    # Rename training set
    X_train_ = X_train  
    y_train_ = y_train  
    n_train = X_train_.shape[0]

    # Fit the posterior on the training data and save the results
    extra_suffix = f"{model_name_suffix}_subset_id"
    run_mcmc(X_train_, y_train_, dataset_name, results_dir=model_results_dir,
                features_of_models=features_list, B=n_samples_per_chain,
                n_chains=n_chains, n_training_size=n_train, 
                extra_suffix=extra_suffix, fit_posterior=fit_posterior)
    print("Trained fixed model for in distribution data.")
    if only_fit_posterior:
        return

    ################################
    # Experiment loop
    ################################
    # Running experiment for the different wds and different trials
    for j, hyp in enumerate(remaining_sets_for_cal):
        if sets_to_run is not None and j not in sets_to_run:
            print(f"Skipping set {j} ({remaining_sets_for_cal_names[j]}) as it is not in sets_to_run.")
            continue
        # Get the test data for this set (ID or OOD)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)        
        X_test_raw, y_test_raw = remaining_sets_for_cal[j]["all_test_data"]   # this is standardized
        n_test = int(test_frac * X_test_raw.shape[0])

        t0 = time.time()
        for i in tqdm(range(num_trials)):
            test_inds = np.random.choice(np.arange(X_test_raw.shape[0]), n_test)
            remaining_indices = np.setdiff1d(np.arange(X_test_raw.shape[0]), test_inds)
            X_test, y_test = X_test_raw[test_inds], y_test_raw[test_inds]
                
            # Sample calibration data
            if n_cal == "standard":
                n_cal = len(remaining_indices)
                alpha = max(0.1, 1/(n_cal + 1))  # ensure alpha is not too small            
            cal_indices = np.random.choice(remaining_indices, size=n_cal, replace=False)
            X_cal, y_cal = X_test_raw[cal_indices], y_test_raw[cal_indices]    # note that we do not need to standardize as we have already standardized
            y_plot = np.linspace(np.min(y_cal) - grid_initial_shift, np.max(y_cal) + grid_initial_shift, grid_size)  # default is 100 points, 2 difference

            # Get CB results (Conformal Bayes as in Fong and Holmes)
            cb_results = run_CB(use_grid, X_cal, y_cal, X_test, y_test, y_plot, 
                                    dataset_name, results_dir=model_results_dir,
                                    alpha=alpha, features_of_models=features_list, n_training_size=n_train,
                                    extra_suffix=extra_suffix, adaptive_grid=adaptive_grid,
                                    max_grid_expansions=max_grid_expansions, max_n_plot=max_n_plot)


            # Add CB results
            coverage_cb = cb_results["coverage_cb"][:, :, cb_model]
            length_cb = cb_results["length_cb"][:, :, cb_model]
            avg_coverage_cb = np.mean(coverage_cb)  # average over test points
            avg_length_cb = np.mean(length_cb)
            results[i, j, methods.index("cb"), 0] = avg_length_cb
            results[i, j, methods.index("cb"), 1] = avg_coverage_cb

        t1 = time.time()
        times.append(t1 - t0)


    ####################################
    # Save results
    ###################################
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_name = f"images_{dataset_name}_methods_{'_'.join(methods)}_trials{num_trials}_alpha{alpha:.2f}_ncal{n_cal}_model_linear_nsamples_{n_samples_per_chain}"
    if save_results:
        config = {"methods": methods,
                "alpha": alpha,
                "num_trials": num_trials,
                "n_test": n_test,
                "n_train": n_train,
                "n_cal": n_cal,
                "features_list": features_list,
                "n_chains": n_chains,
                "n_samples_per_chain": n_samples_per_chain,
                "n_models": n_models,
                "cb_model": cb_model,
                "model_results_dir": model_results_dir,
                "remaining_sets_for_cal_names": remaining_sets_for_cal_names,
                "time_taken_for_all_trials": times,
                }

        # Save the the results
        np.savez_compressed(
                os.path.join(results_dir, f"{exp_name}_{save_name}_results.npz"),
                results=results,
                config=config
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for image CB experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run image CB experiments.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_results_dir", type=str, required=True)
    parser.add_argument("--n_cal", type=_parse_n_cal, default=5)
    parser.add_argument("--num_trials", type=int, default=300)
    parser.add_argument("--fit_posterior", type=_str2bool, default=False)
    parser.add_argument("--only_fit_posterior", type=_str2bool, default=False)
    parser.add_argument("--use_grid", type=_str2bool, default=True)
    parser.add_argument("--adaptive_grid", type=_str2bool, default=True)
    parser.add_argument("--check_only_endpoints_adaptively", type=_str2bool, default=True)
    parser.add_argument("--grid_initial_shift", type=float, default=10)
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--max_grid_expansions", type=int, default=10)
    parser.add_argument("--max_n_plot", type=int, default=200)
    parser.add_argument("--sets_to_run", nargs="+", type=int, default=None, help="List of set indices to run. If not provided, runs all sets.")
    parser.add_argument("--save_results", type=_str2bool, default=True)
    parser.add_argument("--exp_name", type=str, default="")
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    main(
        dataset_name=_args.dataset_name,
        embeddings_path=_args.embeddings_path,
        results_dir=_args.results_dir,
        model_results_dir=_args.model_results_dir,
        n_cal=_args.n_cal,
        num_trials=_args.num_trials,
        fit_posterior=_args.fit_posterior,
        only_fit_posterior=_args.only_fit_posterior,
        use_grid=_args.use_grid,
        adaptive_grid=_args.adaptive_grid,
        check_only_endpoints_adaptively=_args.check_only_endpoints_adaptively,
        grid_initial_shift=_args.grid_initial_shift,
        grid_size=_args.grid_size,
        max_grid_expansions=_args.max_grid_expansions,
        max_n_plot=_args.max_n_plot,
        sets_to_run=_args.sets_to_run,
        save_results=_args.save_results,
        exp_name=_args.exp_name,
    )

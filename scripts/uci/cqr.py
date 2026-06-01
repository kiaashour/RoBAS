import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import os
from tqdm.auto import tqdm
from typing import Union, List
import time
from copy import copy

from utils.datasets.uci_covariate_shift import sample_shifted_covariates
from datasets.uci_covariate_shift import get_uci_dataset
from utils.argparser import _str2bool, _parse_n_cal

from baselines.cqr.nonconformist.nc import RegressorNc
from baselines.cqr.nonconformist.nc import QuantileRegErrFunc
from baselines.cqr.cqr import helper

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Defining model hyperparameters
RF_ARGS = {"n_estimators": 100, "max_depth": None, "random_state": 42, "n_jobs": -1, 
           "min_samples_leaf": 1, "coverage_factor": 0.85, "test_ratio": 0.05,
           "range_vals": 30/100, "CV": True, "num_vals": 10, 
          }


KRR_ARGS = {"alpha_reg": 1, "gamma": None, "n_jobs": -1}
LR_ARGS = {"fit_intercept": True, "n_jobs": -1}
MODEL_ARGS_DICT = {"rf": RF_ARGS, "krr": KRR_ARGS, "linear": LR_ARGS}


############################################################
# Experiment Code
############################################################
def main(dataset_name, data_dir, results_dir, model_name: str = "rf", 
        n_cal: int = 5, num_trials: int = 300, shifts_to_run: Union[None, List] = None,
        save_results: bool = True,
        exp_name: str = "",
        ):
    """
    Run CQR experiments on UCI covariate-shift settings.

    Parameters
    ----------
    dataset_name : str
        UCI dataset identifier.
    data_dir : str
        Directory containing dataset files.
    results_dir : str
        Directory where experiment results are saved.
    model_name : str, default="rf"
        Model backend for CQR.
    n_cal : int or str, default=5
        Calibration-set size or ``"standard"``.
    num_trials : int, default=300
        Number of calibration/test trials per shift.
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
    model_args = MODEL_ARGS_DICT[model_name]
    if model_name == "rf":
        model_args["max_features"] = d

    # Experiment args
    methods = ["cqr"]
    quantiles = [alpha / 2, 1 - (alpha / 2)] 
    results = np.empty((num_trials, len(shift_vectors), len(methods), 2))
    times = []
    print("Quantile range:", quantiles[1] - quantiles[0])


    ####################################
    # Experiment loop
    ###################################
    for j, shift in enumerate(shift_vectors):    
        # Decide whether to skip experiment
        if shifts_to_run is not None and j not in shifts_to_run:
            print(f"Skipping shift {j} as it is not in shifts_to_run.")
            continue        
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)            

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

        # Train model 
        if model_name == "rf":
            model = helper.QuantileForestRegressorAdapter(
                model=None,
                fit_params=None,
                quantiles=quantiles,
                params=model_args,
            )
            nc = RegressorNc(model, QuantileRegErrFunc())
            trained_icp = helper.train_model_for_icp(nc, X_train_, y_train_)
        else:
            raise NotImplementedError("Only RF is currently implemented for CQR. Please set model_name to 'rf'.")
        print("Trained fixed model for new shift.")


        t0 = time.time()
        for i in tqdm(range(num_trials)):
            cal_indices = np.random.choice(remaining_indices_after_train, size=n_cal, replace=False)
            X_cal_raw, y_cal_raw = X[cal_indices], y[cal_indices]
            X_cal = (X_cal_raw - X_mean) / X_std
            y_cal = (y_cal_raw - y_mean) / y_std
            remaining_indices_after_cal = np.setdiff1d(remaining_indices_after_train, cal_indices)

            # Sample test
            test_indices = np.random.choice(remaining_indices_after_cal, size=n_test, replace=False)
            X_test_raw, y_test_raw = X[test_indices], y[test_indices]
            X_test = (X_test_raw - X_mean) / X_std
            y_test = (y_test_raw - y_mean) / y_std          

            # Run experiment
            y_l, y_u = helper.run_icp_with_fitted_icp(trained_icp, X_test, y_test, X_cal, y_cal, alpha)            
            coverage_local, length_local = helper.compute_coverage(y_test, y_l, y_u, alpha)            
            results[i, j, :, 0] = length_local
            results[i, j, :, 1] = coverage_local

        t1 = time.time()
        times.append(t1 - t0)

    ####################################
    # Save results
    ###################################
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_name = f"uci_{dataset_name}_methods_{'_'.join(methods)}_trials{num_trials}_alpha{alpha:.2f}_ncal{n_cal}_ntrain{n_train}_model_{model_name}"
    print("Experiment name is:")
    print(save_name)
    if save_results:
        config = {"methods": methods,
                "alpha": alpha,
                "shift_vectors": shift_vectors,
                "num_trials": num_trials,
                "n_test": n_test,
                "n_train_max": n_train_max,
                "n_train": n_train,
                "n_cal": n_cal,
                "model_name": model_name,
                "model_args": model_args,
                "time_taken_for_all_trials": times
                }                                                    

        # Save the the results
        np.savez_compressed(
                os.path.join(results_dir, f"{exp_name}_{save_name}_results.npz"),
                results=results,
                config=config            
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for CQR experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run UCI CQR experiments.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="rf")
    parser.add_argument("--n_cal", type=_parse_n_cal, default=5)
    parser.add_argument("--num_trials", type=int, default=300)
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
        model_name=_args.model_name,
        n_cal=_args.n_cal,
        num_trials=_args.num_trials,
        shifts_to_run=_args.shifts_to_run,
        save_results=_args.save_results,
        exp_name=_args.exp_name
    )

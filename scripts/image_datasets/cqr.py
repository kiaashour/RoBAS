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

from baselines.cqr.nonconformist.nc import RegressorNc
from baselines.cqr.nonconformist.nc import QuantileRegErrFunc
from baselines.cqr.cqr import helper
from utils.datasets.image import get_image_dataset
from utils.argparser import _str2bool, _parse_n_cal

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
def main(dataset_name: str, model_name: str, embeddings_path: str, results_dir: str, 
        save_results: bool = True,
        n_cal: int = 5,  num_trials: int = 300, sets_to_run: Union[None, List] = None,
        exp_name: str = ""
        ):
    """
    Run CQR experiments on image datasets.

    Parameters
    ----------
    dataset_name : str
        Image dataset identifier.
    model_name : str
        Predictive model backend for CQR.
    embeddings_path : str
        Path to embedding features used for experiments.
    results_dir : str
        Directory where experiment results are saved.
    save_results : bool, default=True
        Whether to save outputs to disk.
    n_cal : int or str, default=5
        Calibration-set size or ``"standard"``.
    num_trials : int, default=300
        Number of calibration/test trials per set.
    sets_to_run : list[int] or None, optional
        Set indices to run. If ``None``, run all sets.
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

    ################################
    # Getting data
    ################################
    # Load the data
    X_train, y_train, remaining_sets_for_cal, remaining_sets_for_cal_names, n, n_train = get_image_dataset(dataset_name, embeddings_path)

    ################################                        
    # Experiment args
    ################################
    # Data args
    test_frac = 0.2
    n_train = X_train.shape[0]
    alpha = max(0.1, 1/(n_cal + 1))  # ensure alpha is not too small
    print(f"Using alpha={alpha} for n_cal={n_cal}")

    # Experiment args
    methods = ["cqr"]
    quantiles = [alpha / 2, 1 - (alpha / 2)] 
    results = np.empty((num_trials, len(remaining_sets_for_cal), len(methods), 2))
    times = []

    # Modelling arguments
    model_args = MODEL_ARGS_DICT[model_name]
    if model_name == "rf":
        model_args["max_features"] = X_train.shape[1]    

    # Rename training set
    X_train_ = X_train
    y_train_ = y_train

    # Train fixed model
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
    print("Trained model.")    

    ####################################
    # Experiment loop
    ###################################
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
            # Sample test data
            test_inds = np.random.choice(np.arange(X_test_raw.shape[0]), n_test)
            remaining_indices = np.setdiff1d(np.arange(X_test_raw.shape[0]), test_inds)
            X_test, y_test = X_test_raw[test_inds], y_test_raw[test_inds]

            # Sample calibration data
            if n_cal == "standard":
                n_cal = len(remaining_indices)
                alpha = max(0.1, 1/(n_cal + 1))  # ensure alpha is not too small            
            cal_indices = np.random.choice(remaining_indices, size=n_cal, replace=False)
            X_cal, y_cal = X_test_raw[cal_indices], y_test_raw[cal_indices]    # note that we do not need to standardize as we have already standardized

            # Run experiment:
            y_l, y_u = helper.run_icp_with_fitted_icp(trained_icp, X_test, y_test, X_cal, y_cal, alpha)            
            coverage, length = helper.compute_coverage(y_test, y_l, y_u, alpha)                
            results[i, j, :, 0] = length
            results[i, j, :, 1] = coverage

        t1 = time.time()
        times.append(t1 - t0)



    ####################################
    # Save results
    ###################################
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_name = f"images_{dataset_name}_methods_{'_'.join(methods)}_trials{num_trials}_alpha{alpha:.2f}_ncal{n_cal}_model_{model_name}"       
    if save_results:
        config = {"methods": methods,
                "alpha": alpha,
                "num_trials": num_trials,
                "n_test": n_test,
                "n_train": n_train,
                "n_cal": n_cal,
                "model_name": model_name,
                "model_args": model_args,
                "remaining_sets_for_cal_names": remaining_sets_for_cal_names,                
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
    Build the command-line parser for image CQR experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run image CQR experiments.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--save_results", type=_str2bool, default=True)
    parser.add_argument("--n_cal", type=_parse_n_cal, default=5)
    parser.add_argument("--num_trials", type=int, default=300)
    parser.add_argument("--sets_to_run", nargs="+", type=int, default=None, help="List of set indices to run. If not provided, runs all sets.")
    parser.add_argument("--exp_name", type=str, default="")
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    main(
        dataset_name=_args.dataset_name,
        model_name=_args.model_name,
        embeddings_path=_args.embeddings_path,
        results_dir=_args.results_dir,
        save_results=_args.save_results,
        n_cal=_args.n_cal,
        num_trials=_args.num_trials,
        sets_to_run=_args.sets_to_run,
        exp_name=_args.exp_name,
    )

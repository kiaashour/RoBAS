import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np

from copy import copy
from typing import List, Union
import time

from utils.datasets.uci_covariate_shift import sample_shifted_covariates
from utils.argparser import _str2bool, _parse_n_cal
from datasets.uci_covariate_shift import get_uci_dataset
from baselines.cb.cb import run_mcmc, create_list_of_features_for_linear_model, run_CB
from scripts.uci_parallel._parallel import BASE_SEED, build_trial_splits, run_trial_tasks, set_global_random_seed


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

_TRIAL_CONTEXT = {}


def _init_cb_trial_context(context):
    """
    Store shared trial context for CB worker execution.

    Parameters
    ----------
    context : dict
        Shared objects and settings used by each trial worker.

    Returns
    -------
    None
        Updates module-level trial context in place.
    """
    global _TRIAL_CONTEXT
    _TRIAL_CONTEXT = context


def _run_cb_trial(task):
    """
    Run one CB trial and return aggregate metrics.

    Parameters
    ----------
    task : tuple[int, np.ndarray, np.ndarray]
        Trial tuple ``(trial_idx, cal_indices, test_indices)``.

    Returns
    -------
    tuple[int, float, float]
        Trial index, mean interval length, and mean coverage.
    """
    trial_idx, cal_indices, test_indices = task
    context = _TRIAL_CONTEXT

    # Seed per trial so execution is reproducible regardless of worker scheduling.
    set_global_random_seed(context["trial_seed_base"] + context["shift_index"] * context["num_trials"] + trial_idx)

    X = context["X"]
    y = context["y"]
    X_mean = context["X_mean"]
    X_std = context["X_std"]
    y_mean = context["y_mean"]
    y_std = context["y_std"]

    X_cal_raw, y_cal_raw = X[cal_indices], y[cal_indices]
    X_cal = (X_cal_raw - X_mean) / X_std
    y_cal = (y_cal_raw - y_mean) / y_std
    y_plot = np.linspace(
        np.min(y_cal) - context["grid_initial_shift"],
        np.max(y_cal) + context["grid_initial_shift"],
        context["grid_size"],
    )

    X_test_raw, y_test_raw = X[test_indices], y[test_indices]
    X_test = (X_test_raw - X_mean) / X_std
    y_test = (y_test_raw - y_mean) / y_std

    cb_results = run_CB(
        context["use_grid"],
        X_cal,
        y_cal,
        X_test,
        y_test,
        y_plot,
        context["dataset_name"],
        results_dir=context["model_results_dir"],
        alpha=context["alpha"],
        features_of_models=context["features_list"],
        n_training_size=context["n_train"],
        extra_suffix=context["extra_suffix"],
        adaptive_grid=context["adaptive_grid"],
        check_only_endpoints_adaptively=context["check_only_endpoints_adaptively"],
        max_grid_expansions=context["max_grid_expansions"],
        max_n_plot=context["max_n_plot"],
        grid_eval_batch_size=context["grid_eval_batch_size"],
    )

    coverage_cb = cb_results["coverage_cb"][:, :, context["cb_model"]]
    length_cb = cb_results["length_cb"][:, :, context["cb_model"]]
    avg_coverage_cb = float(np.mean(coverage_cb))
    avg_length_cb = float(np.mean(length_cb))
    return trial_idx, avg_length_cb, avg_coverage_cb


############################################################
# Experiment Code
############################################################
def main(dataset_name, data_dir, results_dir, model_results_dir, n_cal: int = 5, fit_posterior: bool = False,
         only_fit_posterior: bool = False, num_trials: int = 300,
         use_grid: bool = True, adaptive_grid: bool = True, check_only_endpoints_adaptively: bool = True,
         grid_initial_shift: float = 10, grid_size: int = 100, max_grid_expansions: int = 10, 
         max_n_plot: int = 200, grid_eval_batch_size: int = 256, shifts_to_run: Union[None, List] = None,
         num_workers: int = 1, split_seed: int = BASE_SEED, trial_seed_base: int = BASE_SEED,
         save_results: bool = True, exp_name: str = "",
         ):
    """
    Run CB experiments on UCI covariate-shift settings.

    Parameters
    ----------
    dataset_name : str
        UCI dataset identifier.
    data_dir : str
        Directory containing dataset files.
    results_dir : str
        Directory where experiment results are saved.
    model_results_dir : str
        Directory containing or storing CB model artifacts.
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
    grid_eval_batch_size : int, default=256
        Batch size for grid evaluation.
    shifts_to_run : list[int] or None, optional
        Shift indices to run. If ``None``, run all shifts.
    num_workers : int, default=1
        Number of parallel trial workers.
    split_seed : int, default=BASE_SEED
        Seed for sampling trial splits.
    trial_seed_base : int, default=BASE_SEED
        Base seed for per-trial worker seeding.
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
    assert num_workers >= 1, "num_workers must be >= 1."
    set_global_random_seed(BASE_SEED)

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
    n_models = 1  # use all of the features
    features_list = create_list_of_features_for_linear_model(X.shape[1], n_models)
    n_chains = 4   
    n_samples_per_chain = B = 100    
    cb_model = 0   

    # Experiment args
    methods = ["cb"]
    results = np.empty((num_trials, len(shift_vectors), len(methods), 2))
    times = []
    model_save_suffix = "cb"

    ####################################
    # Experiment loop
    ###################################
    for j, shift in enumerate(shift_vectors):
        # Decide whether to skip experiment
        if shifts_to_run is not None and j not in shifts_to_run:
            print(f"Skipping shift {j} as it is not in shifts_to_run.")
            continue        
        # Keep the legacy behavior: each shift starts from the same seed.
        set_global_random_seed(BASE_SEED)
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
            unique_train_count = int(np.unique(shift_idx).size)
            X_train_ = X_remaining[shift_idx]
            y_train_ = y_remaining[shift_idx]
            remaining_indices_after_train = np.setdiff1d(remaining_indices, remaining_indices[shift_idx])
        else:
            unique_train_count = n_train
            X_train_ = X_remaining[:n_train]
            y_train_ = y_remaining[:n_train]
            remaining_indices_after_train = remaining_indices[n_train:]

        # Find statistics for datasets 
        X_mean = np.mean(X_train_, axis=0)
        X_std = np.std(X_train_, axis=0) + 1e-6
        y_mean = np.mean(y_train_)
        y_std = np.std(y_train_) + 1e-6
        if unique_train_count < n_train:
            eff_frac = unique_train_count / n_train
            print(
                f"Shift {j}: sampled {n_train} training rows with replacement, "
                f"unique rows={unique_train_count} ({eff_frac:.2%})."
            )
        if y_std < 1e-3:
            print(
                f"Shift {j}: warning - very small y_std={y_std:.3e}. "
                "Normalized calibration/test targets can become extreme and trigger larger adaptive grids."
            )

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

        trial_tasks = build_trial_splits(
            remaining_indices_after_train=remaining_indices_after_train,
            n_cal=n_cal,
            n_test=n_test,
            num_trials=num_trials,
            split_seed=split_seed,
        )
        trial_context = {
            "X": X,
            "y": y,
            "X_mean": X_mean,
            "X_std": X_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "use_grid": use_grid,
            "dataset_name": dataset_name,
            "model_results_dir": model_results_dir,
            "alpha": alpha,
            "features_list": features_list,
            "n_train": n_train,
            "extra_suffix": extra_suffix,
            "adaptive_grid": adaptive_grid,
            "check_only_endpoints_adaptively": check_only_endpoints_adaptively,
            "max_grid_expansions": max_grid_expansions,
            "max_n_plot": max_n_plot,
            "grid_eval_batch_size": grid_eval_batch_size,
            "grid_initial_shift": grid_initial_shift,
            "grid_size": grid_size,
            "cb_model": cb_model,
            "shift_index": j,
            "num_trials": num_trials,
            "trial_seed_base": trial_seed_base,
        }
        _init_cb_trial_context(trial_context)

        t0 = time.time()
        trial_outputs = run_trial_tasks(
            _run_cb_trial,
            trial_tasks,
            num_workers=num_workers,
            desc=f"shift {j} trials",
            initializer=_init_cb_trial_context,
            initargs=(trial_context,),
        )
        for trial_idx, avg_length_cb, avg_coverage_cb in trial_outputs:
            results[trial_idx, j, methods.index("cb"), 0] = avg_length_cb
            results[trial_idx, j, methods.index("cb"), 1] = avg_coverage_cb

        t1 = time.time()
        times.append(t1 - t0)


    ####################################
    # Save results
    ###################################
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_name = f"uci_{dataset_name}_methods_{'_'.join(methods)}_trials{num_trials}_alpha{alpha:.2f}_ncal{n_cal}_ntrain{n_train}_model_linear_samples{n_samples_per_chain}"
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
                "num_workers": num_workers,
                "grid_eval_batch_size": grid_eval_batch_size,
                "split_seed": split_seed,
                "trial_seed_base": trial_seed_base,
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
    Build the command-line parser for CB experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run UCI CB experiments.")
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
    parser.add_argument(
        "--grid_eval_batch_size",
        type=int,
        default=256,
        help="Batch size over test points for JAX y-grid evaluations. Lower values reduce peak memory.",
    )
    parser.add_argument("--shifts_to_run", nargs="+", type=int, default=None, help="List of shift indices to run. If not provided, runs all shifts.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of trial workers. Use >1 to parallelize.")
    parser.add_argument("--split_seed", type=int, default=BASE_SEED, help="Seed used to sample trial calibration/test splits.")
    parser.add_argument("--trial_seed_base", type=int, default=BASE_SEED, help="Base seed used to seed each trial worker call.")
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
        grid_eval_batch_size=_args.grid_eval_batch_size,
        shifts_to_run=_args.shifts_to_run,
        num_workers=_args.num_workers,
        split_seed=_args.split_seed,
        trial_seed_base=_args.trial_seed_base,
        save_results=_args.save_results,
        exp_name=_args.exp_name
    )

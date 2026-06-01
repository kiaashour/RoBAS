import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import os
import time
from typing import Union, List
from copy import copy

from baselines.cb.cb import create_list_of_features_for_linear_model, predict_from_saved_bayesian_linear_model
from utils.datasets.uci_covariate_shift import sample_shifted_covariates
from datasets.uci_covariate_shift import get_uci_dataset
from utils.experiments.uci import experiment
from utils.models import make_model
from utils.argparser import _str2bool, _parse_n_cal
from scripts.uci_parallel._parallel import BASE_SEED, build_trial_splits, run_trial_tasks, set_global_random_seed

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Defining model hyperparameters
RF_ARGS = {"n_estimators": 100, "max_depth": None, "random_state": 42, "n_jobs": -1}
KRR_ARGS = {"alpha_reg": 1, "gamma": None, "n_jobs": -1}
LR_ARGS = {"fit_intercept": True, "n_jobs": -1}
LR_ARGS_CQR = {"n_neighbors": 11}
KRR_ARGS_CQR = {"n_neighbors": 11}
MODEL_ARGS_DICT = {"rf": RF_ARGS, "krr": KRR_ARGS, "linear": LR_ARGS,
                   "krr_cqr": KRR_ARGS_CQR, "linear_cqr": LR_ARGS_CQR}
AVAILABLE_METHODS = ["dto", "dta", "hoff", "hs_plugin", "hs_eb"]

_TRIAL_CONTEXT = {}


def _init_main_methods_trial_context(context):
    """
    Store shared trial context for main-method worker execution.

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


def _run_main_methods_trial(task):
    """
    Run one trial for the selected conformal methods.

    Parameters
    ----------
    task : tuple[int, np.ndarray, np.ndarray]
        Trial tuple ``(trial_idx, cal_indices, test_indices)``.

    Returns
    -------
    tuple[int, np.ndarray, np.ndarray]
        Trial index, interval widths per method, and coverages per method.
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
    X_train = context["X_train"]
    y_train = context["y_train"]
    methods = context["methods"]
    model_name = context["model_name"]
    model_args = context["model_args"]

    X_cal_raw, y_cal_raw = X[cal_indices], y[cal_indices]
    X_cal = (X_cal_raw - X_mean) / X_std
    y_cal = (y_cal_raw - y_mean) / y_std

    X_test_raw, y_test_raw = X[test_indices], y[test_indices]
    X_test = (X_test_raw - X_mean) / X_std
    y_test = (y_test_raw - y_mean) / y_std

    data_dict = {
        "train": (X_train, y_train),
        "cal": (X_cal, y_cal),
        "val": (np.zeros((10, X.shape[1])), np.zeros((10,))),  # dummy validation set
        "test": (X_test, y_test),
    }

    if model_name == "blr_preds":
        y_cal_hat = predict_from_saved_bayesian_linear_model(
            X=X_cal,
            dataset_name=context["dataset_name"],
            results_dir=context["blr_model_results_dir"],
            features_of_models=context["features_list"],
            model_index=1,
            n_training_size=context["n_train"],
            extra_suffix=context["extra_suffix"],
        )
        y_test_hat = predict_from_saved_bayesian_linear_model(
            X=X_test,
            dataset_name=context["dataset_name"],
            results_dir=context["blr_model_results_dir"],
            features_of_models=context["features_list"],
            model_index=1,
            n_training_size=context["n_train"],
            extra_suffix=context["extra_suffix"],
        )
        y_train_hat = predict_from_saved_bayesian_linear_model(
            X=X_train,
            dataset_name=context["dataset_name"],
            results_dir=context["blr_model_results_dir"],
            features_of_models=context["features_list"],
            model_index=1,
            n_training_size=context["n_train"],
            extra_suffix=context["extra_suffix"],
        )
        pred_dict = {"train": y_train_hat, "cal": y_cal_hat, "val": np.zeros(10,), "test": y_test_hat}
        resolved_model_name = "precomputed_predictions"
    else:
        pred_dict = None
        resolved_model_name = model_name

    widths, coverages, _ = experiment(
        data_dict=data_dict,
        methods=methods,
        verbose=False,
        score_kwargs=context["score_kwargs"],
        model_name=resolved_model_name,
        alpha=context["alpha"],
        use_grid=context["use_grid"],
        grid_args=context["grid_args"],
        pretrained_model=context["pretrained_model"],
        pred_dict=pred_dict,
        **model_args,
    )
    return trial_idx, np.asarray(widths), np.asarray(coverages)


############################################################
# Experiment Code
############################################################
def main(dataset_name, data_dir, results_dir, 
        methods: List = ["dto", "dta", "hoff_fixed", "hs_eb", "hs_plugin"],
        model_name: str = "rf", blr_model_results_dir: Union[None, str] = None,
        n_cal: int = 5, num_trials: int = 300, use_grid: bool = False, shifts_to_run: Union[None, List] = None,
        num_workers: int = 1, split_seed: int = BASE_SEED, trial_seed_base: int = BASE_SEED,
        exp_name: str = "", save_results: bool = True,
        ):
    """
    Run main conformal-method experiments on UCI covariate-shift settings.

    Parameters
    ----------
    dataset_name : str
        UCI dataset identifier.
    data_dir : str
        Directory containing dataset files.
    results_dir : str
        Directory where experiment results are saved.
    methods : list, default=["dto", "dta", "hoff_fixed", "hs_eb", "hs_plugin"]
        Conformal methods evaluated in each trial.
    model_name : str, default="rf"
        Predictive model backend.
    blr_model_results_dir : str or None, optional
        Directory containing saved BLR predictions when ``model_name="blr_preds"``.
    n_cal : int or str, default=5
        Calibration-set size or ``"standard"``.
    num_trials : int, default=300
        Number of calibration/test trials per shift.
    use_grid : bool, default=False
        Whether to compute conformal regions with grid search.
    shifts_to_run : list[int] or None, optional
        Shift indices to run. If ``None``, run all shifts.
    num_workers : int, default=1
        Number of parallel trial workers.
    split_seed : int, default=BASE_SEED
        Seed for sampling trial splits.
    trial_seed_base : int, default=BASE_SEED
        Base seed for per-trial worker seeding.
    exp_name : str, default=""
        Prefix added to saved result filenames.
    save_results : bool, default=True
        Whether to save outputs to disk.

    Returns
    -------
    None
        Runs the experiment and optionally writes result files.
    """
    valid_model_names = list(MODEL_ARGS_DICT.keys()) + ["blr_preds"]
    assert model_name in valid_model_names, f"Invalid model_name. Must be one of {valid_model_names}"
    assert set(methods).issubset(AVAILABLE_METHODS), f"Unknown methods: {set(methods) - set(AVAILABLE_METHODS)}"
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
    if model_name == "blr_preds" and blr_model_results_dir is None:
        raise ValueError("blr_model_results_dir must be provided when model_name == 'blr_preds'.")

    model_args = MODEL_ARGS_DICT[model_name] if model_name != "blr_preds" else {}

    # Args for blr preds model, i.e. we are using saved blr predictions
    n_models = 1 
    features_list = create_list_of_features_for_linear_model(X.shape[1], n_models)
    model_save_suffix = "cb"

    # Args for finding prediction intervals
    grid_args = {
                "grid_radius": 1,
                "grid_size": 50000,
                "max_grid_size": 50000,
                "max_refinements": 50000,
                "return_interval": True
    }    

    # Experiment args    
    score_kwargs = {"hoff": {"tau2": 1/n_cal}}
    results = np.empty((num_trials, len(shift_vectors), len(methods), 2))
    times = []


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

        # Train model if not using saved BLR predictions
        if model_name != "blr_preds":
            pretrained_model = make_model(X_train_, y_train_, verbose=False, model_name=model_name, **model_args)
            assert pretrained_model is not None, "Failed to train model with hyperparameters " + str(shift)
            print("Trained fixed model for new shift.")
        else:
            pretrained_model = None

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
            "X_train": X_train_,
            "y_train": y_train_,
            "methods": methods,
            "score_kwargs": score_kwargs,
            "model_name": model_name,
            "model_args": model_args,
            "alpha": alpha,
            "use_grid": use_grid,
            "grid_args": grid_args,
            "pretrained_model": pretrained_model,
            "blr_model_results_dir": blr_model_results_dir,
            "features_list": features_list,
            "n_train": n_train,
            "dataset_name": dataset_name,
            "extra_suffix": extra_suffix,
            "shift_index": j,
            "num_trials": num_trials,
            "trial_seed_base": trial_seed_base,
        }
        _init_main_methods_trial_context(trial_context)

        t0 = time.time()
        trial_outputs = run_trial_tasks(
            _run_main_methods_trial,
            trial_tasks,
            num_workers=num_workers,
            desc=f"shift {j} trials",
            initializer=_init_main_methods_trial_context,
            initargs=(trial_context,),
        )
        for trial_idx, widths, coverages in trial_outputs:
            results[trial_idx, j, :, 0] = widths
            results[trial_idx, j, :, 1] = coverages

        t1 = time.time()
        times.append(t1 - t0)


    ####################################
    # Save results
    ###################################
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_name = f"uci_{dataset_name}_methods_{'_'.join(methods)}_trials{num_trials}_alpha{alpha:.2f}_ncal{n_cal}_ntrain{n_train}_usegrid{use_grid}_model_{model_name}"
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
                "num_trials": num_trials,
                "score_kwargs": score_kwargs,
                "use_grid": use_grid,
                "grid_args": grid_args,
                "model_name": model_name,
                "model_args": model_args,
                "num_workers": num_workers,
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
    Build the command-line parser for main-method experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run UCI main-method experiments.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="rf")
    parser.add_argument("--blr_model_results_dir", type=str, default=None)
    parser.add_argument("--n_cal", type=_parse_n_cal, default=5)
    parser.add_argument("--num_trials", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of trial workers. Use >1 to parallelize.")
    parser.add_argument("--split_seed", type=int, default=BASE_SEED, help="Seed used to sample trial calibration/test splits.")
    parser.add_argument("--trial_seed_base", type=int, default=BASE_SEED, help="Base seed used to seed each trial worker call.")
    parser.add_argument("--use_grid", type=_str2bool, default=False)
    parser.add_argument("--save_results", type=_str2bool, default=True)
    parser.add_argument("--shifts_to_run", nargs="+", type=int, default=None, help="List of shift indices to run. If not provided, runs all shifts.")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--methods", nargs="+", default=["dto"])
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    main(
        dataset_name=_args.dataset_name,
        data_dir=_args.data_dir,
        results_dir=_args.results_dir,
        model_name=_args.model_name,
        blr_model_results_dir=_args.blr_model_results_dir,
        n_cal=_args.n_cal,
        num_trials=_args.num_trials,
        num_workers=_args.num_workers,
        split_seed=_args.split_seed,
        trial_seed_base=_args.trial_seed_base,
        use_grid=_args.use_grid,
        save_results=_args.save_results,
        shifts_to_run=_args.shifts_to_run,
        exp_name=_args.exp_name,
        methods=_args.methods,
    )

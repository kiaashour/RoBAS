import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import time
from typing import List, Union

import numpy as np

from baselines.cqr.cqr import helper
from baselines.cqr.nonconformist.nc import QuantileRegErrFunc
from baselines.cqr.nonconformist.nc import RegressorNc
from scripts.image_datasets_parallel._parallel import (
    BASE_SEED,
    build_image_trial_splits,
    run_trial_tasks,
    set_global_random_seed,
)
from utils.argparser import _parse_n_cal, _str2bool
from utils.datasets.image import get_image_dataset

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Defining model hyperparameters
RF_ARGS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
    "min_samples_leaf": 1,
    "coverage_factor": 0.85,
    "test_ratio": 0.05,
    "range_vals": 30 / 100,
    "CV": True,
    "num_vals": 10,
}
KRR_ARGS = {"alpha_reg": 1, "gamma": None, "n_jobs": -1}
LR_ARGS = {"fit_intercept": True, "n_jobs": -1}
MODEL_ARGS_DICT = {"rf": RF_ARGS, "krr": KRR_ARGS, "linear": LR_ARGS}

_TRIAL_CONTEXT = {}


def _init_cqr_trial_context(context):
    """
    Store shared trial context for CQR worker execution.

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


def _run_cqr_trial(task):
    """
    Run one CQR trial and return aggregate metrics.

    Parameters
    ----------
    task : tuple[int, np.ndarray, np.ndarray]
        Trial tuple ``(trial_idx, test_indices, cal_indices)``.

    Returns
    -------
    tuple[int, float, float]
        Trial index, mean interval length, and mean coverage.
    """
    trial_idx, test_indices, cal_indices = task
    context = _TRIAL_CONTEXT

    # Seed per trial so execution is reproducible regardless of worker scheduling.
    set_global_random_seed(
        context["trial_seed_base"] + context["set_index"] * context["num_trials"] + trial_idx
    )

    X_test_raw = context["X_test_raw"]
    y_test_raw = context["y_test_raw"]

    X_test = X_test_raw[test_indices]
    y_test = y_test_raw[test_indices]

    X_cal = X_test_raw[cal_indices]
    y_cal = y_test_raw[cal_indices]

    y_l, y_u = helper.run_icp_with_fitted_icp(
        context["trained_icp"], X_test, y_test, X_cal, y_cal, context["alpha"]
    )
    coverage, length = helper.compute_coverage(y_test, y_l, y_u, context["alpha"])
    return trial_idx, float(length), float(coverage)


############################################################
# Experiment Code
############################################################
def main(
    dataset_name: str,
    model_name: str,
    embeddings_path: str,
    results_dir: str,
    save_results: bool = True,
    n_cal: int = 5,
    num_trials: int = 300,
    sets_to_run: Union[None, List] = None,
    num_workers: int = 1,
    split_seed: int = BASE_SEED,
    trial_seed_base: int = BASE_SEED,
    exp_name: str = "",
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
    num_workers : int, default=1
        Number of parallel trial workers.
    split_seed : int, default=BASE_SEED
        Seed for sampling trial splits.
    trial_seed_base : int, default=BASE_SEED
        Base seed for per-trial worker seeding.
    exp_name : str, default=""
        Prefix added to saved result filenames.

    Returns
    -------
    None
        Runs the experiment and optionally writes result files.
    """
    assert n_cal in [5, 10, 25, 50, "standard"], (
        "Invalid calibration set size. Calibration set size must be one of [5, 10, 25, 50, 'standard']"
    )
    assert num_workers >= 1, "num_workers must be >= 1."
    set_global_random_seed(BASE_SEED)

    ################################
    # Getting data
    ################################
    X_train, y_train, remaining_sets_for_cal, remaining_sets_for_cal_names, _, _ = get_image_dataset(
        dataset_name, embeddings_path
    )

    ################################
    # Experiment args
    ################################
    test_frac = 0.2
    n_train = X_train.shape[0]

    requested_n_cal = n_cal
    resolved_n_cal = n_cal
    alpha = None
    if requested_n_cal != "standard":
        resolved_n_cal = int(requested_n_cal)
        alpha = max(0.1, 1 / (resolved_n_cal + 1))
        print(f"Using alpha={alpha} for n_cal={requested_n_cal}")
    else:
        # Quantile model is trained once; use the minimum allowed alpha for the
        # base quantile band, and resolve calibration size per set in the loop.
        alpha = 0.1
        print("Using per-set standard calibration size: n_cal = |remaining_indices|.")
        print(f"Using alpha={alpha} for training the shared CQR base model.")

    methods = ["cqr"]
    results = np.empty((num_trials, len(remaining_sets_for_cal), len(methods), 2))
    times = []
    alpha_per_set: list[float | None] = [None] * len(remaining_sets_for_cal)
    n_cal_per_set: list[int | None] = [None] * len(remaining_sets_for_cal)

    model_args = MODEL_ARGS_DICT[model_name].copy()
    if model_name == "rf":
        model_args["max_features"] = X_train.shape[1]

    quantiles = [alpha / 2, 1 - (alpha / 2)]

    # Train fixed model
    if model_name == "rf":
        model = helper.QuantileForestRegressorAdapter(
            model=None,
            fit_params=None,
            quantiles=quantiles,
            params=model_args,
        )
        nc = RegressorNc(model, QuantileRegErrFunc())
        trained_icp = helper.train_model_for_icp(nc, X_train, y_train)
    else:
        raise NotImplementedError("Only RF is currently implemented for CQR. Please set model_name to 'rf'.")
    print("Trained model.")

    ####################################
    # Experiment loop
    ###################################
    for j, _ in enumerate(remaining_sets_for_cal):
        if sets_to_run is not None and j not in sets_to_run:
            print(f"Skipping set {j} ({remaining_sets_for_cal_names[j]}) as it is not in sets_to_run.")
            continue

        # Keep legacy behavior: each set starts from the same global seed.
        set_global_random_seed(BASE_SEED)

        X_test_raw, y_test_raw = remaining_sets_for_cal[j]["all_test_data"]
        n_test = int(test_frac * X_test_raw.shape[0])

        # When n_cal="standard", resolve n_cal from this set's candidate pool
        # size so each set is internally feasible even when pools differ.
        if requested_n_cal == "standard":
            trial_tasks, resolved_n_cal_set = build_image_trial_splits(
                pool_size=X_test_raw.shape[0],
                n_cal="standard",
                n_test=n_test,
                num_trials=num_trials,
                split_seed=split_seed,
            )
            alpha_set = max(0.1, 1 / (resolved_n_cal_set + 1))
            print(
                f"Set {j} ({remaining_sets_for_cal_names[j]}): "
                f"resolved standard n_cal={resolved_n_cal_set}, alpha={alpha_set}"
            )
        else:
            resolved_n_cal_set = resolved_n_cal
            alpha_set = alpha
            trial_tasks, _ = build_image_trial_splits(
                pool_size=X_test_raw.shape[0],
                n_cal=resolved_n_cal_set,
                n_test=n_test,
                num_trials=num_trials,
                split_seed=split_seed,
            )

        n_cal_per_set[j] = int(resolved_n_cal_set)
        alpha_per_set[j] = float(alpha_set)

        trial_context = {
            "X_test_raw": X_test_raw,
            "y_test_raw": y_test_raw,
            "trained_icp": trained_icp,
            "alpha": alpha_set,
            "trial_seed_base": trial_seed_base,
            "set_index": j,
            "num_trials": num_trials,
        }
        _init_cqr_trial_context(trial_context)

        t0 = time.time()
        trial_outputs = run_trial_tasks(
            _run_cqr_trial,
            trial_tasks,
            num_workers=num_workers,
            desc=f"set {j} trials",
            initializer=_init_cqr_trial_context,
            initargs=(trial_context,),
        )
        for trial_idx, length, coverage in trial_outputs:
            results[trial_idx, j, :, 0] = length
            results[trial_idx, j, :, 1] = coverage

        t1 = time.time()
        times.append(t1 - t0)

    ####################################
    # Save results
    ###################################
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    executed_set_indices = [
        idx
        for idx in range(len(remaining_sets_for_cal))
        if sets_to_run is None or idx in sets_to_run
    ]
    if not executed_set_indices:
        raise ValueError("No valid sets_to_run entries were provided.")

    first_executed_set_idx = executed_set_indices[0]
    alpha_for_name = (
        alpha if requested_n_cal != "standard" else alpha_per_set[first_executed_set_idx]
    )
    n_cal_for_name = (
        resolved_n_cal if requested_n_cal != "standard" else n_cal_per_set[first_executed_set_idx]
    )
    save_name = (
        f"images_{dataset_name}_methods_{'_'.join(methods)}_trials{num_trials}_"
        f"alpha{alpha_for_name:.2f}_ncal{n_cal_for_name}_model_{model_name}"
    )
    if save_results:
        config_alpha = alpha if requested_n_cal != "standard" else alpha_per_set[first_executed_set_idx]
        config_n_cal = resolved_n_cal if requested_n_cal != "standard" else n_cal_per_set[first_executed_set_idx]
        config = {
            "methods": methods,
            "alpha": config_alpha,
            "num_trials": num_trials,
            "n_test": n_test,
            "n_train": n_train,
            "n_cal": config_n_cal,
            "requested_n_cal": requested_n_cal,
            "model_name": model_name,
            "model_args": model_args,
            "remaining_sets_for_cal_names": remaining_sets_for_cal_names,
            "num_workers": num_workers,
            "split_seed": split_seed,
            "trial_seed_base": trial_seed_base,
            "time_taken_for_all_trials": times,
        }
        if requested_n_cal == "standard":
            config["n_cal_per_set"] = n_cal_per_set
            config["alpha_per_set"] = alpha_per_set

        np.savez_compressed(
            os.path.join(results_dir, f"{exp_name}_{save_name}_results.npz"),
            results=results,
            config=config,
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of trial workers. Use >1 to parallelize.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=BASE_SEED,
        help="Seed used to sample trial calibration/test splits.",
    )
    parser.add_argument(
        "--trial_seed_base",
        type=int,
        default=BASE_SEED,
        help="Base seed used to seed each trial worker call.",
    )
    parser.add_argument(
        "--sets_to_run",
        nargs="+",
        type=int,
        default=None,
        help="List of set indices to run. If not provided, runs all sets.",
    )
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
        num_workers=_args.num_workers,
        split_seed=_args.split_seed,
        trial_seed_base=_args.trial_seed_base,
        exp_name=_args.exp_name,
    )

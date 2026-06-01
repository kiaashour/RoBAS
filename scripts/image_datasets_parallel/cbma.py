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

from baselines.cb.cb import create_list_of_features_for_linear_model
from baselines.cb.cb import run_CBMA
from baselines.cb.cb import run_mcmc
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

_TRIAL_CONTEXT = {}


def _init_cbma_trial_context(context):
    """
    Store shared trial context for CBMA worker execution.

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


def _run_cbma_trial(task):
    """
    Run one CBMA trial and return aggregate metrics.

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
    y_plot = np.linspace(
        np.min(y_cal) - context["grid_initial_shift"],
        np.max(y_cal) + context["grid_initial_shift"],
        context["grid_size"],
    )

    cbma_results = run_CBMA(
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
        max_grid_expansions=context["max_grid_expansions"],
        max_n_plot=context["max_n_plot"],
    )

    coverage_cbma = cbma_results["coverage_cbma"]
    length_cbma = cbma_results["length_cbma"]
    avg_coverage_cbma = float(np.mean(coverage_cbma))
    avg_length_cbma = float(np.mean(length_cbma))
    return trial_idx, avg_length_cbma, avg_coverage_cbma


def main(
    dataset_name: str,
    embeddings_path: str,
    results_dir: str,
    model_results_dir: str,
    n_cal: int = 5,
    num_trials: int = 300,
    fit_posterior: bool = False,
    only_fit_posterior: bool = False,
    use_grid: bool = True,
    adaptive_grid: bool = True,
    check_only_endpoints_adaptively: bool = True,
    grid_initial_shift: float = 10,
    grid_size: int = 100,
    max_grid_expansions: int = 10,
    max_n_plot: int = 200,
    sets_to_run: Union[None, List] = None,
    num_workers: int = 1,
    split_seed: int = BASE_SEED,
    trial_seed_base: int = BASE_SEED,
    save_results: bool = True,
    exp_name: str = "",
):
    """
    Run CBMA experiments on image datasets.

    Parameters
    ----------
    dataset_name : str
        Image dataset identifier.
    embeddings_path : str
        Path to embedding features used for experiments.
    results_dir : str
        Directory where experiment results are saved.
    model_results_dir : str
        Directory containing or storing CBMA model artifacts.
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
        Compatibility flag preserved for CLI usage.
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
    # Kept for CLI compatibility with existing experiment commands. We do not
    # pass this flag into `run_CBMA` to preserve legacy script behavior.
    _ = check_only_endpoints_adaptively
    assert n_cal in [5, 10, 25, 50, "standard"], (
        "Invalid calibration set size. Calibration set size must be one of [5, 10, 25, 50, 'standard']"
    )
    assert num_workers >= 1, "num_workers must be >= 1."
    set_global_random_seed(BASE_SEED)

    X_train, y_train, remaining_sets_for_cal, remaining_sets_for_cal_names, _, _ = get_image_dataset(
        dataset_name, embeddings_path
    )

    methods = ["cbma"]
    test_frac = 0.2

    requested_n_cal = n_cal
    resolved_n_cal = n_cal
    alpha = None
    if requested_n_cal != "standard":
        resolved_n_cal = int(requested_n_cal)
        alpha = max(0.1, 1 / (resolved_n_cal + 1))
        print(f"Using alpha={alpha} for n_cal={requested_n_cal}")
    else:
        print("Using per-set standard calibration size: n_cal = |remaining_indices|.")

    results = np.empty((num_trials, len(remaining_sets_for_cal), len(methods), 2))
    model_name_suffix = "cbma"
    times = []
    alpha_per_set: list[float | None] = [None] * len(remaining_sets_for_cal)
    n_cal_per_set: list[int | None] = [None] * len(remaining_sets_for_cal)

    n_models = 4
    features_list = create_list_of_features_for_linear_model(X_train.shape[1], n_models)
    n_chains = 4
    n_samples_per_chain = 100

    n_train = X_train.shape[0]

    extra_suffix = f"{model_name_suffix}_subset_id"
    run_mcmc(
        X_train,
        y_train,
        dataset_name,
        results_dir=model_results_dir,
        features_of_models=features_list,
        B=n_samples_per_chain,
        n_chains=n_chains,
        n_training_size=n_train,
        extra_suffix=extra_suffix,
        fit_posterior=fit_posterior,
    )
    print("Trained fixed model for in distribution data.")
    if only_fit_posterior:
        return

    ################################
    # Experiment loop
    ################################
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
            "use_grid": use_grid,
            "dataset_name": dataset_name,
            "model_results_dir": model_results_dir,
            "alpha": alpha_set,
            "features_list": features_list,
            "extra_suffix": extra_suffix,
            "n_train": n_train,
            "adaptive_grid": adaptive_grid,
            "max_grid_expansions": max_grid_expansions,
            "max_n_plot": max_n_plot,
            "grid_initial_shift": grid_initial_shift,
            "grid_size": grid_size,
            "trial_seed_base": trial_seed_base,
            "set_index": j,
            "num_trials": num_trials,
        }
        _init_cbma_trial_context(trial_context)

        t0 = time.time()
        trial_outputs = run_trial_tasks(
            _run_cbma_trial,
            trial_tasks,
            num_workers=num_workers,
            desc=f"set {j} trials",
            initializer=_init_cbma_trial_context,
            initargs=(trial_context,),
        )
        for trial_idx, avg_length_cbma, avg_coverage_cbma in trial_outputs:
            results[trial_idx, j, methods.index("cbma"), 0] = avg_length_cbma
            results[trial_idx, j, methods.index("cbma"), 1] = avg_coverage_cbma

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
        f"alpha{alpha_for_name:.2f}_ncal{n_cal_for_name}_model_linear_nsamples_{n_samples_per_chain}"
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
            "features_list": features_list,
            "n_chains": n_chains,
            "n_samples_per_chain": n_samples_per_chain,
            "n_models": n_models,
            "model_results_dir": model_results_dir,
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
    Build the command-line parser for image CBMA experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run image CBMA experiments.")
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
        num_workers=_args.num_workers,
        split_seed=_args.split_seed,
        trial_seed_base=_args.trial_seed_base,
        save_results=_args.save_results,
        exp_name=_args.exp_name,
    )

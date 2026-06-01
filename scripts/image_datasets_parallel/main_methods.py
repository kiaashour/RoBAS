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

from baselines.cb.cb import (
    create_list_of_features_for_linear_model,
    predict_from_saved_bayesian_linear_model,
)
from scripts.image_datasets_parallel._parallel import (
    BASE_SEED,
    build_image_trial_splits,
    run_trial_tasks,
    set_global_random_seed,
)
from utils.argparser import _parse_n_cal, _str2bool
from utils.datasets.image import get_image_dataset
from utils.experiments.uci import experiment
from utils.models import make_model

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
MODEL_ARGS_DICT = {
    "rf": RF_ARGS,
    "krr": KRR_ARGS,
    "linear": LR_ARGS,
    "krr_cqr": KRR_ARGS_CQR,
    "linear_cqr": LR_ARGS_CQR,
}
AVAILABLE_METHODS = ["dto", "dta", "hoff", "hoff_fixed", "hs_plugin", "hs_eb"]

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
        Trial tuple ``(trial_idx, test_indices, cal_indices)``.

    Returns
    -------
    tuple[int, np.ndarray, np.ndarray]
        Trial index, interval widths per method, and coverages per method.
    """
    trial_idx, test_indices, cal_indices = task
    context = _TRIAL_CONTEXT

    # Seed per trial so execution is reproducible regardless of worker scheduling.
    set_global_random_seed(
        context["trial_seed_base"] + context["set_index"] * context["num_trials"] + trial_idx
    )

    X_test_raw = context["X_test_raw"]
    y_test_raw = context["y_test_raw"]
    yhat_test_raw = context["yhat_test_raw"]

    X_test = X_test_raw[test_indices]
    y_test = y_test_raw[test_indices]
    yhat_test = yhat_test_raw[test_indices]

    X_cal = X_test_raw[cal_indices]
    y_cal = y_test_raw[cal_indices]
    yhat_cal = yhat_test_raw[cal_indices]

    pred_dict = {
        "train": context["yhat_train"],
        "cal": yhat_cal,
        "val": np.zeros(3),
        "test": yhat_test,
    }

    data_dict = {
        "train": (context["X_train"], context["y_train"]),
        "cal": (X_cal, y_cal),
        "val": (np.zeros((3, context["X_train"].shape[1])), np.zeros((3,))),
        "test": (X_test, y_test),
    }

    widths, coverages, _ = experiment(
        data_dict=data_dict,
        methods=context["methods"],
        verbose=False,
        score_kwargs=context["score_kwargs"],
        model_name="precomputed_predictions",
        alpha=context["alpha"],
        use_grid=context["use_grid"],
        grid_args=context["grid_args"],
        pretrained_model=None,
        pred_dict=pred_dict,
        **context["model_args"],
    )
    return trial_idx, np.asarray(widths), np.asarray(coverages)


############################################################
# Experiment Code
############################################################
def main(
    dataset_name: str,
    model_name: str,
    embeddings_path: str,
    results_dir: str,
    blr_model_results_dir: Union[None, str] = None,
    n_cal: int = 5,
    num_trials: int = 300,
    methods: List = ["dto", "dta", "hoff_fixed", "hs_eb", "hs_plugin"],
    use_grid: bool = False,
    sets_to_run: Union[None, List] = None,
    num_workers: int = 1,
    split_seed: int = BASE_SEED,
    trial_seed_base: int = BASE_SEED,
    save_results: bool = True,
    exp_name: str = "",
):
    """
    Run main conformal-method experiments on image datasets.

    Parameters
    ----------
    dataset_name : str
        Image dataset identifier.
    model_name : str
        Predictive model backend.
    embeddings_path : str
        Path to embedding features used for experiments.
    results_dir : str
        Directory where experiment results are saved.
    blr_model_results_dir : str or None, optional
        Directory containing saved BLR predictions when ``model_name="blr_preds"``.
    n_cal : int or str, default=5
        Calibration-set size or ``"standard"``.
    num_trials : int, default=300
        Number of calibration/test trials per set.
    methods : list, default=["dto", "dta", "hoff_fixed", "hs_eb", "hs_plugin"]
        Conformal methods evaluated in each trial.
    use_grid : bool, default=False
        Whether to compute conformal regions with grid search.
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
    valid_model_names = list(MODEL_ARGS_DICT.keys()) + ["blr_preds"]
    assert model_name in valid_model_names, f"Invalid model_name. Must be one of {valid_model_names}"
    assert n_cal in [5, 10, 25, 50, "standard"], (
        "Invalid calibration set size. Calibration set size must be one of [5, 10, 25, 50, 'standard']"
    )
    assert set(methods).issubset(AVAILABLE_METHODS), f"Unknown methods: {set(methods) - set(AVAILABLE_METHODS)}"
    assert num_workers >= 1, "num_workers must be >= 1."
    set_global_random_seed(BASE_SEED)

    ################################
    # Getting data
    ################################
    X_train, y_train, remaining_sets_for_cal, remaining_sets_for_cal_names, _, n_train = get_image_dataset(
        dataset_name, embeddings_path
    )

    model_args = MODEL_ARGS_DICT[model_name].copy() if model_name != "blr_preds" else {}

    # BLR saved-posterior prediction args.
    n_models = 1
    features_list = create_list_of_features_for_linear_model(X_train.shape[1], n_models)
    model_save_suffix = "cb_subset_id"
    if model_name == "blr_preds" and blr_model_results_dir is None:
        raise ValueError("blr_model_results_dir must be provided when model_name == 'blr_preds'.")

    if model_name != "blr_preds":
        pretrained_model = make_model(X_train, y_train, verbose=False, model_name=model_name, **model_args)
        assert pretrained_model is not None, "Failed to train model."
        print("Trained fixed model for new shift.")
    else:
        pretrained_model = None

    # For precomputed-prediction experiments, train predictions are only used by
    # downstream metrics.
    if model_name == "blr_preds":
        yhat_train = predict_from_saved_bayesian_linear_model(
            X=X_train,
            dataset_name=dataset_name,
            results_dir=blr_model_results_dir,
            features_of_models=features_list,
            model_index=1,
            n_training_size=n_train,
            extra_suffix=model_save_suffix,
        )
    else:
        yhat_train = np.zeros_like(y_train)

    # Precompute candidate-pool predictions once for each evaluation split.
    for i, remaining_set in enumerate(remaining_sets_for_cal):
        X_pool, _ = remaining_set["all_test_data"]
        if model_name == "blr_preds":
            yhat_test = predict_from_saved_bayesian_linear_model(
                X=X_pool,
                dataset_name=dataset_name,
                results_dir=blr_model_results_dir,
                features_of_models=features_list,
                model_index=1,
                n_training_size=n_train,
                extra_suffix=model_save_suffix,
            )
        else:
            yhat_test = np.asarray(pretrained_model.predict(X_pool)).squeeze()
        remaining_sets_for_cal[i]["all_test_preds"] = yhat_test

    ################################
    # Experiment args
    ################################
    test_frac = 0.2
    n_train = X_train.shape[0]

    requested_n_cal = n_cal
    resolved_n_cal = n_cal
    alpha = None
    score_kwargs = None

    if requested_n_cal != "standard":
        resolved_n_cal = int(requested_n_cal)
        alpha = max(0.1, 1 / (resolved_n_cal + 1))
        print(f"Using alpha={alpha} for n_cal={requested_n_cal}")
        score_kwargs = {"hoff": {"tau2": 1 / resolved_n_cal}}
    else:
        print("Using per-set standard calibration size: n_cal = |remaining_indices|.")
    results = np.empty((num_trials, len(remaining_sets_for_cal), len(methods), 2))
    times = []
    alpha_per_set: list[float | None] = [None] * len(remaining_sets_for_cal)
    n_cal_per_set: list[int | None] = [None] * len(remaining_sets_for_cal)
    tau2_per_set: list[float | None] = [None] * len(remaining_sets_for_cal)

    grid_args = {
        "grid_radius": 1,
        "grid_size": 50000,
        "max_grid_size": 50000,
        "max_refinements": 50000,
        "return_interval": True,
    }

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
        yhat_test_raw = remaining_sets_for_cal[j]["all_test_preds"]
        n_test = int(test_frac * X_test_raw.shape[0])

        if requested_n_cal == "standard":
            trial_tasks, resolved_n_cal_set = build_image_trial_splits(
                pool_size=X_test_raw.shape[0],
                n_cal="standard",
                n_test=n_test,
                num_trials=num_trials,
                split_seed=split_seed,
            )
            alpha_set = max(0.1, 1 / (resolved_n_cal_set + 1))
            score_kwargs_set = {"hoff": {"tau2": 1 / resolved_n_cal_set}}
            print(
                f"Set {j} ({remaining_sets_for_cal_names[j]}): "
                f"resolved standard n_cal={resolved_n_cal_set}, alpha={alpha_set}"
            )
        else:
            resolved_n_cal_set = resolved_n_cal
            alpha_set = alpha
            score_kwargs_set = score_kwargs
            trial_tasks, _ = build_image_trial_splits(
                pool_size=X_test_raw.shape[0],
                n_cal=resolved_n_cal_set,
                n_test=n_test,
                num_trials=num_trials,
                split_seed=split_seed,
            )

        n_cal_per_set[j] = int(resolved_n_cal_set)
        alpha_per_set[j] = float(alpha_set)
        tau2_per_set[j] = float(score_kwargs_set["hoff"]["tau2"])

        trial_context = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test_raw": X_test_raw,
            "y_test_raw": y_test_raw,
            "yhat_test_raw": yhat_test_raw,
            "yhat_train": yhat_train,
            "methods": methods,
            "score_kwargs": score_kwargs_set,
            "model_args": model_args,
            "alpha": alpha_set,
            "use_grid": use_grid,
            "grid_args": grid_args,
            "trial_seed_base": trial_seed_base,
            "set_index": j,
            "num_trials": num_trials,
        }
        _init_main_methods_trial_context(trial_context)

        t0 = time.time()
        trial_outputs = run_trial_tasks(
            _run_main_methods_trial,
            trial_tasks,
            num_workers=num_workers,
            desc=f"set {j} trials",
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
        config_score_kwargs = (
            score_kwargs
            if requested_n_cal != "standard"
            else {"hoff": {"tau2": tau2_per_set[first_executed_set_idx]}}
        )
        config = {
            "methods": methods,
            "alpha": config_alpha,
            "num_trials": num_trials,
            "n_test": n_test,
            "n_train": n_train,
            "n_cal": config_n_cal,
            "requested_n_cal": requested_n_cal,
            "score_kwargs": config_score_kwargs,
            "use_grid": use_grid,
            "grid_args": grid_args,
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
            config["tau2_per_set"] = tau2_per_set
        if model_name == "blr_preds":
            config["blr_model_results_dir"] = blr_model_results_dir
            config["features_list"] = features_list
            config["model_save_suffix"] = model_save_suffix

        np.savez_compressed(
            os.path.join(results_dir, f"{exp_name}_{save_name}_results.npz"),
            results=results,
            config=config,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for image main-method experiments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run image main-method experiments.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--blr_model_results_dir", type=str, default=None)
    parser.add_argument("--use_grid", type=_str2bool, default=False)
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
    parser.add_argument("--methods", nargs="+", default=["dto"])
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    main(
        dataset_name=_args.dataset_name,
        model_name=_args.model_name,
        embeddings_path=_args.embeddings_path,
        results_dir=_args.results_dir,
        blr_model_results_dir=_args.blr_model_results_dir,
        use_grid=_args.use_grid,
        save_results=_args.save_results,
        n_cal=_args.n_cal,
        num_trials=_args.num_trials,
        num_workers=_args.num_workers,
        split_seed=_args.split_seed,
        trial_seed_base=_args.trial_seed_base,
        exp_name=_args.exp_name,
        methods=_args.methods,
        sets_to_run=_args.sets_to_run,
    )

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import os
import numpy as np
from tqdm.auto import tqdm
import time

from typing import Union, List

from baselines.cb.cb import create_list_of_features_for_linear_model, predict_from_saved_bayesian_linear_model
from utils.experiments.uci import experiment
from utils.models import make_model
from utils.datasets.image import get_image_dataset
from utils.argparser import _str2bool, _parse_n_cal    

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


############################################################
# Experiment Code
############################################################
def main(dataset_name: str, model_name: str, 
        embeddings_path: str, results_dir: str, blr_model_results_dir: Union[None, str] = None,
        n_cal: int = 5,  num_trials: int = 300, methods: List = ["dto", "dta", "hoff_fixed", "hs_eb", "hs_plugin"],
        use_grid: bool = False, sets_to_run: Union[None, List] = None,
        save_results: bool = True, exp_name: str = "",
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
    assert n_cal in [5, 10, 25, 50, "standard"], "Invalid calibration set size. Calibration set size must be one of [5, 10, 25, 50, 'standard']"
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    ################################
    # Getting data
    ################################
    # Load the data
    X_train, y_train, remaining_sets_for_cal, remaining_sets_for_cal_names, n, n_train = get_image_dataset(dataset_name, embeddings_path)

    # Modelling args
    model_args = MODEL_ARGS_DICT[model_name].copy() if model_name != "blr_preds" else {}

    # BLR saved-posterior prediction args
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
    # Data args
    test_frac = 0.2
    n_train = X_train.shape[0]
    if n_cal == "standard":
        n_cal = n_train
    alpha = max(0.1, 1/(n_cal + 1))  # ensure alpha is not too small
    print(f"Using alpha={alpha} for n_cal={n_cal}")

    # Experiment args
    score_kwargs = {"hoff": {"tau2": 1/n_cal}}
    results = np.empty((num_trials, len(remaining_sets_for_cal), len(methods), 2))
    times = []

    # Args for finding prediction intervals
    grid_args = {
                "grid_radius": 1,
                "grid_size": 50000,
                "max_grid_size": 50000,
                "max_refinements": 50000,
                "return_interval": True
    }        

    # Rename training set
    X_train_ = X_train
    y_train_ = y_train

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
            yhat_test_raw = remaining_sets_for_cal[j]["all_test_preds"]
            yhat_test = yhat_test_raw[test_inds]

            # Sample calibration data
            if n_cal == "standard":
                n_cal = len(remaining_indices)
                alpha = max(0.1, 1/(n_cal + 1))  # ensure alpha is not too small            
            cal_indices = np.random.choice(remaining_indices, size=n_cal, replace=False)
            X_cal, y_cal = X_test_raw[cal_indices], y_test_raw[cal_indices]    # note that we do not need to standardize as we have already standardized
            yhat_cal = yhat_test_raw[cal_indices]

            # Create prediction dict, where for train and val we use dummy
            pred_dict = {
                "train": yhat_train,
                "cal": yhat_cal,
                "val": np.zeros(3),   # dummy validation set
                "test": yhat_test
            }

            # Create data dict
            data_dict = {
                "train": (X_train_, y_train_),
                "cal": (X_cal, y_cal),
                "val": (np.zeros((3, X_train.shape[1])), np.zeros((3,))),   # dummy validation set
                "test": (X_test, y_test),
            }

            # Run experiment (using precomputed prediction)
            w, c, _ = experiment(data_dict=data_dict,
                            methods=methods,
                            verbose=False,
                            score_kwargs=score_kwargs,
                            model_name="precomputed_predictions",
                            alpha=alpha,
                            use_grid=use_grid,
                            grid_args=grid_args,            
                            pretrained_model=None,  # we are using precomputed predictions
                            pred_dict=pred_dict,
                            **model_args)
            results[i, j, :, 0] = w
            results[i, j, :, 1] = c

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
                "score_kwargs": score_kwargs,
                "use_grid": use_grid,
                "grid_args": grid_args,
                "model_name": model_name,
                "model_args": model_args,
                "remaining_sets_for_cal_names": remaining_sets_for_cal_names,
                "time_taken_for_all_trials": times
                }
        if model_name == "blr_preds":
            config["blr_model_results_dir"] = blr_model_results_dir
            config["features_list"] = features_list
            config["model_save_suffix"] = model_save_suffix

        # Save the the results
        np.savez_compressed(
                os.path.join(results_dir, f"{exp_name}_{save_name}_results.npz"),
                results=results,
                config=config
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
    parser.add_argument("--sets_to_run", nargs="+", type=int, default=None, help="List of set indices to run. If not provided, runs all sets.")
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
        exp_name=_args.exp_name,
        methods=_args.methods,        
        sets_to_run=_args.sets_to_run,
    )

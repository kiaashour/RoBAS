import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets.synthetic import get_normal_data, DATA_ARGS, DATA_ARGS_TO_EXP_NAME
from utils.experiments.synthetic import synthetic_experiment
from utils.argparser import _str2bool

import os
import numpy as np
from tqdm.auto import tqdm
from typing import List

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Default experiment parameters
N_SAMPLES = 50000
N_TRAIN = 100
N_TEST = 1000
GRID_ARGS = {  # default grid arguments
    "grid_radius": 1,
    "grid_size": 50000,
    "max_grid_size": 50000,
    "max_refinements": 50000,
    "return_interval": True,
}
AVAILABLE_METHODS = ["dto", "dta", "hoff", "hs_plugin", "hs_eb"]
DEFAULT_RESULTS_PATH = "/data/localhost/not-backed-up/ashouritaklimi/conformal/results/synthetic"


def main(
    data_var: float = 1,
    n_cal: int = 10,
    verbose: bool = False,
    use_grid: bool = False,
    num_trials: int = 300,
    methods: List[str] = ["dto"],
    save_results: bool = False,
    results_dir: str = DEFAULT_RESULTS_PATH,
    exp_name: str = "",
):
    """ Main function for carrying out synthetic experiments.

    Parameters:
    -----------
    data_var : float
        Variance of the normal distribution to sample from.
    n_cal : int
        Number of calibration samples.
    verbose : bool
        If True, print progress and warnings.
    use_grid : bool
        If True, use grid for finding prediction intervals.
    n_trials : int
        Number of trials to run for each configuration.
    methods : List[str]
        List of methods to use for the experiments.
    save_results : bool
        If True, save the results to disk.
    results_dir : str
        Directory to save the results.
    exp_name : str
        Name of the experiment (used for saving results).

    Returns
    -------
    results : np.ndarray
        The results of the experiments.
    """
    # Check for correct specifications
    assert (data_var, n_cal) in DATA_ARGS, f"Unknown data configuration for (data_var={data_var}, n_cal={n_cal})"
    assert set(methods).issubset(AVAILABLE_METHODS), f"Unknown methods: {set(methods) - set(AVAILABLE_METHODS)}"

    # Specifying experiment
    data_config_name = DATA_ARGS_TO_EXP_NAME.get((data_var, n_cal), None)
    std = np.sqrt(data_var)
    theta_vals = DATA_ARGS[(data_var, n_cal)]
    theta_val_dicts = [{"theta": a} for a in theta_vals]

    alpha = max(0.1, 1 / (n_cal + 1))
    print(f"Using alpha={alpha} for n_cal={n_cal}")

    # Method specific args
    hoff_args = {"tau2": 1 / n_cal}
    score_kwargs = {"hoff": hoff_args, "hs": {}}

    # Results arrays
    results = np.empty((num_trials, len(theta_val_dicts), len(methods), 2))

    #######################################################
    # Run experiments
    #######################################################
    for j, args in enumerate(theta_val_dicts):
        for i in tqdm(range(num_trials)):
            #######################################################
            # Get the data
            #######################################################
            if args.get("a", 0) != 0:
                sng_a = np.sign(args.get("theta", 0))  # for better plotting
            else:
                sng_a = 1
            y = sng_a * get_normal_data(
                n=N_SAMPLES, mean=np.abs(args.get("theta", None)), std=std, rng=np.random.default_rng(42)
            )
            n = len(y)

            n_test_ = N_TEST
            rng_i = np.random.default_rng(i)
            idx = rng_i.permutation(n)

            # Indices
            train_end = n - n_test_
            test_start = n - n_test_

            # Create the training set
            y_train = y[idx[:train_end]]
            y_test = y[idx[test_start:]]

            # Convert data to float64 for stability
            y_train = y_train.astype(np.float64)
            y_test = y_test.astype(np.float64)

            # Squeeze y arrays (to get the right shape)
            y_test = y_test.squeeze()
            y_train = y_train.squeeze()

            #######################################################
            # Splits
            #######################################################
            # Create training and calibration set
            idx = rng_i.permutation(len(y_train))
            train_indices = idx[:N_TRAIN]
            cal_indices = idx[N_TRAIN : N_TRAIN + n_cal]

            y_train_ = y_train[train_indices]
            y_cal = y_train[cal_indices]
            data_dict = {
                "train": (y_train_),
                "cal": (y_cal),
                "val": (np.zeros((1,))),  # Dummy unused validation set
                "test": (y_test),
            }

            w, c, _ = synthetic_experiment(
                data_dict=data_dict,
                methods=methods,
                verbose=verbose,
                score_kwargs=score_kwargs,
                alpha=alpha,
                use_grid=use_grid,
                grid_args=GRID_ARGS,
                **args,
            )

            # Store the width and coverage results
            results[i, j, :, 0] = w
            results[i, j, :, 1] = c

    #######################################################
    # Save results
    #######################################################
    # Extend experiment name and create results folders
    save_name = f"{exp_name}_synthetic_methods_{'_'.join(methods)}_trials{num_trials}_alpha{alpha}_ncal{n_cal}_std{std}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if save_results:
        config = {
            "methods": methods,
            "alpha": alpha,
            "theta_vals": theta_vals,
            "num_trials": num_trials,
            "n_cal": n_cal,
            "std": std,
            "num_trials": num_trials,
            "score_kwargs": score_kwargs,
            "use_grid": use_grid,
            "grid_args": GRID_ARGS,
        }

        # Save the the results
        np.savez_compressed(
            os.path.join(results_dir, f"{save_name}_results.npz"), results=results, config=config
        )

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic conformal experiments.")
    parser.add_argument("--data_var", type=float, default=1)
    parser.add_argument("--n_cal", type=int, default=10)
    parser.add_argument("--verbose", type=_str2bool, default=False)
    parser.add_argument("--use_grid", type=_str2bool, default=False)
    parser.add_argument("--num_trials", type=int, default=300)
    parser.add_argument("--methods", nargs="+", default=["dto"])
    parser.add_argument("--save_results", type=_str2bool, default=False)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--exp_name", type=str, default="")
    return parser


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    main(
        data_var=_args.data_var,
        n_cal=_args.n_cal,
        verbose=_args.verbose,
        use_grid=_args.use_grid,
        num_trials=_args.num_trials,
        methods=_args.methods,
        save_results=_args.save_results,
        results_dir=_args.results_dir,
        exp_name=_args.exp_name,
    )

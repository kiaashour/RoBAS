import numpy as np
from copy import deepcopy
from robas.regression import conformal_region_full


def synthetic_experiment(
    data_dict: dict,
    verbose: bool,
    alpha: float,
    methods: list,
    score_kwargs: dict = None,
    use_grid: bool = False,
    grid_args: dict = None,
    return_opt_success: bool = False,
    **model_args,
):
    """
    Run a conformal prediction experiment using our mixture model.

    Parameters
    ----------
    data_dict : dict
        A dictionary containing the following keys:
        - 'train': (X_train, y_train)
        - 'cal': (X_cal, y_cal)
        - 'val': (X_val, y_val)
        - 'test': (X_test, y_test)
    verbose : bool
        If True, print progress and warnings.
    alpha : float
        Miscoverage level.
    methods : list of str
        List of conformal scoring methods to evaluate.
    score_kwargs : dict or None
        Extra kwargs forwarded to `conformal_region_full(..., score_kwargs=score_kwargs)`.
    use_grid : bool
        If True, use grid-based optimization for conformal region.
    grid_args : dict or None
        Extra kwargs forwarded to `conformal_region_full(..., grid_args=grid_args)`.
    return_opt_success : bool
        If True, return optimization success flag (only if `use_grid=False`).
    **model_args :
        Extra kwargs forwarded to the model constructor.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - widths: List of average widths of the conformal regions.
        - coverages: List of coverage probabilities of the conformal regions.
        - rmse: Root mean squared error of the model predictions.
        - opt_success: Optimization success flag (only if `use_grid=False`).

    """
    # Create copies for score kwargs
    score_kwargs_eb = deepcopy(score_kwargs)
    score_kwargs_eb["hoff"]["tau2"] = None

    # Unpack data
    y_train_ = data_dict["train"]
    y_cal = data_dict["cal"]
    y_val = data_dict["val"]
    y_test = data_dict["test"]
    n = y_train_.shape

    # Fix shapes
    y_cal = y_cal.squeeze()
    y_val = y_val.squeeze()

    #####################################################
    # Fit model
    #####################################################
    y_mean = np.mean(y_train_)
    eps_cal = y_cal
    eps_val = y_val
    eps_test = y_test

    #####################################################
    # Get RMSE For model
    #####################################################
    rmse = np.sqrt(np.mean((y_mean - y_test) ** 2))
    if verbose:
        print(f"n: {n}\tRMSE: {rmse:.3f}")

    #####################################################
    # Evaluate methods
    #####################################################
    widths = []
    coverages = []

    # Loop over methods
    for method in methods:
        if method == "hoff":
            assert (
                score_kwargs["hoff"].get("tau2") is not None
            ), "For 'hoff', must provide fixed tau2 in score_kwargs['hoff']['tau2']"
            sc_kwargs = score_kwargs
        else:
            sc_kwargs = score_kwargs_eb

        # Get conformal prediction region
        res = conformal_region_full(
            eps_cal,
            eps_val,
            score_method=method,
            alpha=alpha,
            score_kwargs=sc_kwargs,
            use_grid=use_grid,
            return_opt_success=return_opt_success,
            **grid_args,
        )
        if return_opt_success and not use_grid:
            l, u, opt_success = res
        else:
            l, u = res

        # Evaluate region in terms of width and coverage
        width = (u - l).mean()
        coverage = np.mean(np.logical_and(eps_test <= u, eps_test >= l))

        # Store results
        widths.append(width)
        coverages.append(coverage)

        # Print results
        if verbose:
            print(f"n: {n}\tMethod: {method.upper()}\t" f"Coverage: {coverage:.3f}\tWidth: {width:.3f}")
    if return_opt_success and not use_grid:
        return widths, coverages, rmse, opt_success
    return widths, coverages, rmse

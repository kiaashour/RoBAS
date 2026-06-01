import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from baselines.cqr.nonconformist.nc import NcFactory
from baselines.cqr.cqr import helper

from copy import copy


def make_model(X_train, y_train, verbose, model_name, **model_args):
    """
    Train a model on the training data. This is used when we want to keep our training
    data fixed across multiple experiments.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs.
    y_train : np.ndarray
        Training targets.
    verbose : bool
        If True, print progress and warnings.
    model_name : str
        One of {"rf", "random_forest", "randomforest"} for Random Forest,
        {"linear", "linear_regression", "linreg"} for Linear Regression,
        {"krr", "kernel_ridge_regression", "kernelridgeregression"} for Kernel Ridge Regression,
        {"precomputed_predictions"} for precomputed predictions.
    model_args : dict
        Extra kwargs forwarded to the model constructor.

    Returns
    -------
    model
        The trained model.
    """
    n, d = X_train.shape
    try:
        model_key = (model_name or "").lower().strip()
        if model_key in {"rf", "random_forest", "randomforest"}:
            #####################################################
            # Fit Random Forest model
            #####################################################
            # Force non-standardized residuals (RF has no predictive variance)
            model = RandomForestRegressor(**model_args)
            model.fit(X_train, y_train)

        elif model_key in ["linear", "linear_regression", "linreg"]:
            #####################################################
            # Fit Linear Regression model
            #####################################################
            from sklearn.linear_model import LinearRegression

            # Using default hyperparameters
            model = LinearRegression(**model_args)
            model.fit(X_train, y_train)

        elif model_key in ["linear_cqr"]:
            ###################################################################
            # Fit Linear Regression model following implementation in CQR
            ###################################################################
            nc = NcFactory.create_nc(
                linear_model.LinearRegression(),    # default hyperparameters
                normalizer_model=KNeighborsRegressor(n_neighbors=model_args["n_neighbors"])
            )        
            trained_icp = helper.train_model_for_icp(nc, X_train, y_train)
            model = copy(trained_icp.nc_function.model)                     

        elif model_key in ["krr", "kernel_ridge_regression", "kernelridgeregression"]:
            #####################################################
            # Fit Kernel Ridge Regression model
            #####################################################
            # Using default hyperparameters
            model = KernelRidge(alpha=model_args["alpha_reg"], gamma=model_args["gamma"], kernel="rbf")
            model.fit(X_train, y_train)

        elif model_key in ["krr_cqr"]:
            ###################################################################
            # Fit Kernel Ridge Regression model following implementation in CQR
            ###################################################################
            # Using default hyperparameters
            nc = NcFactory.create_nc(linear_model.Ridge(),   
                normalizer_model=KNeighborsRegressor(n_neighbors=model_args["n_neighbors"])
            )
            trained_icp = helper.train_model_for_icp(nc, X_train, y_train)
            model = copy(trained_icp.nc_function.model)                

        else:
            raise ValueError(f"Unknown model_name '{model_name}'.")

    except (np.linalg.LinAlgError, FloatingPointError, RuntimeError, ValueError) as e:
        if verbose:
            print(f"[WARN] Trial failed for n={n}: {e}. Using dummy NaN results.")
        return np.nan

    return model

import torch

import numpy as np
from copy import deepcopy
from robas.regression import conformal_region_full

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

try:
    import tensorflow as tf
    TF_INVALID_ARGUMENT_ERRORS = (tf.errors.InvalidArgumentError,)
except Exception:
    tf = None
    TF_INVALID_ARGUMENT_ERRORS = ()


def experiment(
    data_dict,
    verbose,
    model_name,
    methods,
    score_kwargs=None,
    return_residuals_and_metrics_only=False,
    metric="rmse",
    pretrained_model=None,
    batch_mode=False,
    pred_dict=None,
    use_grid=False,
    grid_args={},
    alpha=0.1,
    **model_args,
):
    """
    Run a conformal prediction experiment using either a GPflow GPR model or an XGBoost regressor.

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
    methods : list of str
        List of conformal scoring methods to evaluate.
    model_name : str
        One of {"rf", "random_forest", "randomforest"} for Random Forest,
        {"linear", "linear_regression", "linreg", "linear_ridge"} for Linear Regression,
        {"krr", "kernel_ridge_regression", "kernelridgeregression"} for Kernel Ridge Regression,
        {"sklearn_mlp"} for Scikit-learn MLP Regressor,
        {"torch_nn"} for PyTorch Neural Network,
        {"torch_nn_cqr"} for PyTorch Neural Network with Conformal Quantile Regression,
        {"precomputed_predictions"} for precomputed predictions.
    grid_args : dict
        Extra kwargs forwarded to `conformal_region_full(..., **grid_args)`.
    use_grid : bool, default=False
        If True, use grid-based conformal region computation.
    alpha : float, default=0.1
        Miscoverage level for conformal regions.
    pretrained_model : object or None, default=None
        If provided, use this pretrained model instead of fitting a new one.
    score_kwargs : dict or None
        Extra kwargs forwarded to `conformal_region_full(..., score_kwargs=score_kwargs)`.
    metric : str, default="rmse"
        Which metric to compute. "rmse" is the RMSE on the test set and
        "residuals" is the mean of the residuals on the  calibration set.
    return_residuals_and_metrics_only : bool, default=False
        If True, only return residuals and metrics without computing conformal regions.
    model_args : dict
        Extra kwargs forwarded to the model constructor.

    Returns
    -------
    model
        The trained model.
    """
    # Create copies for score kwargs
    score_kwargs_eb = deepcopy(score_kwargs)
    score_kwargs_eb["hoff"]["tau2"] = None

    # Unpack data
    X_train_, y_train_ = data_dict["train"]
    X_cal, y_cal = data_dict["cal"]
    X_val, y_val = data_dict["val"]
    X_test, y_test = data_dict["test"]
    n, d = X_train_.shape

    # Fix shapes
    if not batch_mode:
        y_cal = y_cal.squeeze()
        y_val = y_val.squeeze()

    # Unpack prediction and fix shape (if provided)
    if pred_dict is not None:
        yhat_train = pred_dict["train"].squeeze() if not batch_mode else pred_dict["train"]
        yhat_cal = pred_dict["cal"].squeeze() if not batch_mode else pred_dict["cal"]
        yhat_val = pred_dict["val"].squeeze() if not batch_mode else pred_dict["val"]
        yhat_test = pred_dict["test"].squeeze() if not batch_mode else pred_dict["test"]

    # Get device if used by model args
    if "device" in model_args:
        device = torch.device(model_args["device"] if torch.cuda.is_available() else "cpu")

    # Set model to pretrained model
    if pretrained_model is not None:
        model = pretrained_model

    def _dummy_results():
        if return_residuals_and_metrics_only:
            return [np.nan] * len(methods), [np.nan] * len(methods), 0.0
        else:
            dummy_res_dict = {method: {"interval": (np.array([np.nan]), np.array([np.nan]))} for method in methods}
            dummy_widths = [np.nan] * len(methods)
            dummy_coverages = [np.nan] * len(methods)
            return dummy_widths, dummy_coverages, dummy_res_dict

    try:
        model_key = (model_name or "").lower().strip()
        if model_key in {"rf", "random_forest", "randomforest"}:
            #####################################################
            # Fit Random Forest model
            #####################################################
            if pretrained_model is None:
                model = RandomForestRegressor(**model_args)
                model.fit(X_train_, y_train_)

            # Training prediction/residuals
            yhat_train = np.asarray(model.predict(X_train_)).squeeze()
            if not np.isfinite(yhat_train).all():
                raise FloatingPointError("NaNs in training predictions")
            eps_train = y_train_ - yhat_train  

            # Calibration predictions/residuals
            yhat_cal = np.asarray(model.predict(X_cal)).squeeze()
            if not np.isfinite(yhat_cal).all():
                raise FloatingPointError("NaNs in calibration predictions")
            eps_cal = y_cal - yhat_cal  

            # Validation predictions/residuals
            yhat_val = np.asarray(model.predict(X_val)).squeeze()
            if not np.isfinite(yhat_val).all():
                raise FloatingPointError("NaNs in validation predictions")
            eps_val = y_val - yhat_val  

            # Test predictions/residuals
            yhat_test = np.asarray(model.predict(X_test)).squeeze()
            if not np.isfinite(yhat_test).all():
                raise FloatingPointError("NaNs in test predictions")
            eps_test = y_test - yhat_test  

        elif model_key in ["linear", "linear_regression", "linreg", "linear_ridge"]:
            #####################################################
            # Fit Linear Regression model
            #####################################################
            from sklearn.linear_model import LinearRegression
            from sklearn.linear_model import Ridge

            # Using default hyperparameters
            if pretrained_model is None:
                if model_key in ["linear", "linear_regression", "linreg"]:
                    model = LinearRegression(**model_args)
                    model.fit(X_train_, y_train_)
                elif model_key in ["linear_ridge"]:
                    model = Ridge(**model_args)
                    model.fit(X_train_, y_train_)
                else:
                    raise ValueError(f"Unknown model_key '{model_key}' for linear model.")

            # Training predictions/residuals
            yhat_train = np.asarray(model.predict(X_train_)).squeeze()
            if not np.isfinite(yhat_train).all():
                raise FloatingPointError("NaNs in training predictions")
            eps_train = y_train_ - yhat_train  

            # Calibration predictions/residuals
            yhat_cal = np.asarray(model.predict(X_cal)).squeeze()
            if not np.isfinite(yhat_cal).all():
                raise FloatingPointError("NaNs in calibration predictions")
            eps_cal = y_cal - yhat_cal  

            # Validation predictions/residuals
            yhat_val = np.asarray(model.predict(X_val)).squeeze()
            if not np.isfinite(yhat_val).all():
                raise FloatingPointError("NaNs in validation predictions")
            eps_val = y_val - yhat_val  

            # Test predictions/residuals
            yhat_test = np.asarray(model.predict(X_test)).squeeze()
            if not np.isfinite(yhat_test).all():
                raise FloatingPointError("NaNs in test predictions")
            eps_test = y_test - yhat_test  

        elif model_key in ["krr", "kernel_ridge_regression", "kernelridgeregression"]:
            #####################################################
            # Fit Kernel Ridge Regression model
            #####################################################
            # Using default hyperparameters
            if pretrained_model is None:
                print("Training KRR model....")
                model = KernelRidge(alpha=model_args["alpha_reg"], gamma=model_args["gamma"], kernel="rbf")
                model.fit(X_train_, y_train_)

            # Training predictions/residuals
            yhat_train = np.asarray(model.predict(X_train_)).squeeze()
            if not np.isfinite(yhat_train).all():
                raise FloatingPointError("NaNs in training predictions")
            eps_train = y_train_ - yhat_train  

            # Calibration predictions/residuals
            yhat_cal = np.asarray(model.predict(X_cal)).squeeze()
            if not np.isfinite(yhat_cal).all():
                raise FloatingPointError("NaNs in calibration predictions")
            eps_cal = y_cal - yhat_cal

            # Validation predictions/residuals
            yhat_val = np.asarray(model.predict(X_val)).squeeze()
            if not np.isfinite(yhat_val).all():
                raise FloatingPointError("NaNs in validation predictions")
            eps_val = y_val - yhat_val  

            # Test predictions/residuals
            yhat_test = np.asarray(model.predict(X_test)).squeeze()
            if not np.isfinite(yhat_test).all():
                raise FloatingPointError("NaNs in test predictions")
            eps_test = y_test - yhat_test  

        elif model_key in ["sklearn_mlp"]:
            #####################################################
            # Fit sklearn MLP Regressor model
            #####################################################
            from sklearn.neural_network import MLPRegressor

            # Using default hyperparameters
            if pretrained_model is None:
                model = MLPRegressor(
                    hidden_layer_sizes=model_args.get("hidden_layer_sizes", (100,)),
                    activation=model_args.get("activation", "relu"),
                    solver=model_args.get("solver", "adam"),
                    learning_rate_init=model_args.get("learning_rate_init", 0.001),
                    max_iter=model_args.get("max_iter", 200),
                    random_state=model_args.get("random_state", 42),
                    alpha=model_args.get("alpha", 0.0),
                )
            model.fit(X_train_, y_train_)

            # Calibration predictions/residuals
            yhat_cal = np.asarray(model.predict(X_cal)).squeeze()
            if not np.isfinite(yhat_cal).all():
                raise FloatingPointError("NaNs in calibration predictions")
            eps_cal = y_cal - yhat_cal  

            # Validation predictions/residuals
            yhat_val = np.asarray(model.predict(X_val)).squeeze()
            if not np.isfinite(yhat_val).all():
                raise FloatingPointError("NaNs in validation predictions")
            eps_val = y_val - yhat_val  

            # Test predictions/residuals
            yhat_test = np.asarray(model.predict(X_test)).squeeze()
            if not np.isfinite(yhat_test).all():
                raise FloatingPointError("NaNs in test predictions")
            eps_test = y_test - yhat_test  

            # Get training residuals
            yhat_train = model(torch.from_numpy(X_train_).float().to(device)).detach().cpu().numpy().squeeze()
            if not np.isfinite(yhat_train).all():
                raise FloatingPointError("NaNs in training predictions")
            eps_train = y_train_ - yhat_train  

            # Calibration predictions/residuals
            yhat_cal = model(torch.from_numpy(X_cal).float().to(device)).detach().cpu().numpy().squeeze()
            if not np.isfinite(yhat_cal).all():
                raise FloatingPointError("NaNs in calibration predictions")
            eps_cal = y_cal - yhat_cal  

            # Validation predictions/residuals
            yhat_val = model(torch.from_numpy(X_val).float().to(device)).detach().cpu().numpy().squeeze()
            if not np.isfinite(yhat_val).all():
                raise FloatingPointError("NaNs in validation predictions")
            eps_val = y_val - yhat_val  

            # Test predictions/residuals
            yhat_test = model(torch.from_numpy(X_test).float().to(device)).detach().cpu().numpy().squeeze()
            if not np.isfinite(yhat_test).all():
                raise FloatingPointError("NaNs in test predictions")
            eps_test = y_test - yhat_test  

        elif model_name in ["torch_nn_cqr"]:  # neural network as in the CQR code base
            #####################################################
            # PyTorch Neural Network model is trained using CQR codebase
            #####################################################
            assert pretrained_model is not None, "For 'torch_nn_cqr', must provide pretrained_model"

            # Get training residuals
            yhat_train = model.predict(X_train_).squeeze()
            if not np.isfinite(yhat_train).all():
                raise FloatingPointError("NaNs in training predictions")
            eps_train = y_train_ - yhat_train  

            # Calibration predictions/residuals
            yhat_cal = model.predict(X_cal).squeeze()
            if not np.isfinite(yhat_cal).all():
                raise FloatingPointError("NaNs in calibration predictions")
            eps_cal = y_cal - yhat_cal  

            # Validation predictions/residuals
            yhat_val = model.predict(X_val).squeeze()
            if not np.isfinite(yhat_val).all():
                raise FloatingPointError("NaNs in validation predictions")
            eps_val = y_val - yhat_val  

            # Test predictions/residuals
            yhat_test = model.predict(X_test).squeeze()
            if not np.isfinite(yhat_test).all():
                raise FloatingPointError("NaNs in test predictions")
            eps_test = y_test - yhat_test

        elif model_name in ["precomputed_predictions"]:
            #####################################################
            # Use precomputed predictions
            #####################################################
            if pred_dict is None:
                raise ValueError("pred_dict must be provided when model_name is 'precomputed_predictions'.")

            # Calibration residuals
            eps_cal = y_cal - yhat_cal  

            # Validation residuals
            eps_val = y_val - yhat_val  

            # Test residuals
            eps_test = y_test - yhat_test  

        else:
            raise ValueError(f"Unknown model_name '{model_name}'.")

    except TF_INVALID_ARGUMENT_ERRORS + (np.linalg.LinAlgError, FloatingPointError, RuntimeError, ValueError) as e:
        if verbose:
            print(f"[WARN] Trial failed for n={n}: {e}. Using dummy NaN results.")
        return _dummy_results()

    #################################################
    # Get RMSE For model
    #####################################################
    rmse_cal, rmse_train, rmse_test = np.nan, np.nan, np.nan
    if metric == "rmse":
        rmse_test = np.sqrt(np.mean((yhat_test - y_test) ** 2))
        rmse_train = np.sqrt(np.mean((yhat_train - y_train_) ** 2))
    elif metric == "residuals":
        rmse_cal = np.mean(eps_cal)
    if verbose:
        print(f"n: {n}\tModel: {model_name.upper()}\tRMSE: {rmse_test:.3f}")
        print(f"n: {n}\tModel: {model_name.upper()}\tTrain RMSE: {rmse_train:.3f}")
    rmse_dict = {"test": rmse_test, "train": rmse_train, "cal": rmse_cal}

    #################################################
    # Evaluate methods
    #####################################################
    widths = []
    coverages = []

    if return_residuals_and_metrics_only:
        res_dict = {
            "cal": eps_cal,
            "val": eps_val,
            "test": eps_test,
        }
        return res_dict, rmse_dict

    # Loop over methods
    for method in methods:
        if method == "hoff":
            assert (
                score_kwargs["hoff"].get("tau2") is not None
            ), "For 'hoff_fixed', must provide fixed tau2 in score_kwargs['hoff']['tau2']"
            sc_kwargs = score_kwargs
        else:
            sc_kwargs = score_kwargs_eb

        # Get conformal prediction region
        res = conformal_region_full(
            eps_cal,
            eps_val,
            score_method=method,
            alpha=alpha,
            use_grid=use_grid,
            score_kwargs=sc_kwargs,
            **grid_args,
        )

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
    return widths, coverages, rmse_dict

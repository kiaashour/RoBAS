import os

import math
import numpy as np
import pymc as pm
from typing import Optional, List
import scipy.optimize.elementwise as oe

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp


################################################################################
# Model helpers
################################################################################
def create_list_of_features_for_linear_model(num_total_features: int, n_models: int):
    """
    Create a list of feature indices for linear models, where each model uses an
    increasing number of features.

    Example:
        num_total_features=8, n_models=4 -> [[0,1],[0,1,2,3],[0,1,2,3,4,5],[0..7]]

    Parameters
    ----------
    num_total_features : int
        Total number of features available (d).
    n_models : int
        Number of models (K).

    Returns
    -------
    list_of_feature_indices : list[list[int]]
        list_of_feature_indices[k] is the list of feature indices used by model k.
    """
    d_block = num_total_features // n_models
    list_of_feature_indices = []
    for k in range(n_models):
        if k < n_models - 1:
            feature_indices = list(range(0, (k + 1) * d_block))
        else:
            feature_indices = list(range(0, num_total_features))
        list_of_feature_indices.append(feature_indices)
    return list_of_feature_indices


################################################################################
# Bayesian linear regression (PyMC) + approximate log marginal likelihood
################################################################################
def fit_mcmc_normal(y: np.ndarray, x: np.ndarray, B: int, n_chains: int = 4, seed: int = 100):
    """
    Fit a Bayesian linear regression in PyMC:

        beta ~ Normal(0, 5)            shape (p,)
        intercept ~ Normal(0, 5)       scalar
        sigma ~ HalfNormal(1)          scalar
        y_i ~ Normal(x_i^T beta + intercept, sigma)

    Parameters
    ----------
    y : (n,) np.ndarray
    x : (n, p) np.ndarray
    B : int
        Draws per chain.
    n_chains : int
        Number of MCMC chains.
    seed : int

    Returns
    -------
    beta_post : (M, p) np.ndarray
        Posterior draws of beta, M = n_chains * B.
    intercept_post : (M, 1) np.ndarray
    sigma_post : (M, 1) np.ndarray
    logml : float
        Approximate log marginal likelihood using a harmonic-mean-like estimator.
    """
    with pm.Model() as model:
        p = np.shape(x)[1]
        a = 5  # default in paper is 5

        beta = pm.Normal("beta", mu=0.0, sigma=a, shape=p)
        intercept = pm.Normal("intercept", mu=0.0, sigma=a)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        _ = pm.Normal("obs", mu=pm.math.dot(x, beta) + intercept, sigma=sigma, observed=y)

        trace = pm.sample(B, random_seed=seed, chains=n_chains, return_inferencedata=True)

    beta_post = trace.posterior["beta"].values.reshape(-1, p)  # (M, p)
    intercept_post = trace.posterior["intercept"].values.reshape(-1, 1)  # (M, 1)
    sigma_post = trace.posterior["sigma"].values.reshape(-1, 1)  # (M, 1)

    # Approximate log marginal likelihood from log posterior density samples.
    # (harmonic mean estimator; may be unstable)
    logp = trace["sample_stats"].lp
    logml = -logsumexp(-logp.values) - np.log(len(logp))  # note the negative sign

    return beta_post, intercept_post, sigma_post, float(logml)


def run_mcmc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    results_dir: str,
    features_of_models=None,
    B: int = 2000,
    n_chains: int = 4,
    n_training_size: int = 0,
    seed: int = 100,
    extra_suffix="",
    fit_posterior=True,
):
    """
    Fit K Bayesian linear models on the training set only and
     posterior draws.

    Parameters
    ----------
    X_train : (n_train, d) np.ndarray
    y_train : (n_train,) np.ndarray
    dataset_name : str
    results_dir : str
    features_of_models : list[list[int]]
        Length K. features_of_models[k] are feature indices used by model k.
    B : int
        Draws per chain.
    n_chains : int
        Number of chains.
    n_training_size : int
        Label used in filenames (often equals n_train).
    seed : int
        Base random seed.

    Saves (to results_dir/samples):
    - beta_post{dataset_name}_model_{k}_{n_training_size}_{n_training_size}.npy
        shape (rep=1, M, p_k)
    - intercept_post..., sigma_post... similarly
    - logml_j{dataset_name}_{n_training_size}.npy
        shape (rep=1, K)
    extra_suffix : str
        This is to help with avoiding refitting of the model
    """
    if features_of_models is None:
        raise ValueError("features_of_models must be provided.")

    K = len(features_of_models)  # number of models
    rep = 1  # this script uses one repetition by default

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/samples", exist_ok=True)
    logml_j = np.zeros((rep, K), dtype=float)  # (1, K)

    for k in range(1, K + 1):
        suffix = f"{dataset_name}_model_{k}_{n_training_size}"
        # Check if we have already fit the posteriors
        if (
            os.path.exists(f"{results_dir}/samples/beta_post{suffix}_{n_training_size}_{extra_suffix}.npy")
            and not fit_posterior
        ):
            print("Skipping MCMC sampling because posterior has alrady been fit")
            return
        if (
            os.path.exists(f"{results_dir}/samples/intercept_post{suffix}_{n_training_size}_{extra_suffix}.npy")
            and not fit_posterior
        ):
            print("Skipping MCMC sampling because posterior has alrady been fit")
            return
        if (
            os.path.exists(f"{results_dir}/samples/sigma_post{suffix}_{n_training_size}_{extra_suffix}.npy")
            and not fit_posterior
        ):
            print("Skipping MCMC sampling because posterior has alrady been fit")
            return
        if (
            os.path.exists(f"{results_dir}/samples/logml_j{dataset_name}_{n_training_size}_{extra_suffix}.npy")
            and not fit_posterior
        ):
            print("Skipping MCMC sampling because posterior has alrady been fit")
            return

        Xk = X_train[:, features_of_models[k - 1]]  # (n_train, p_k)
        beta_post_k, intercept_post_k, sigma_post_k, logml = fit_mcmc_normal(
            y_train, Xk, B=B, n_chains=n_chains, seed=seed + k
        )

        # Add rep dimension to match existing file conventions: (1, M, p_k)
        beta_post_k = beta_post_k[None, :, :]  # shape (1, M, p_k)
        intercept_post_k = intercept_post_k[None, :, :]  # shape (1, M, 1)
        sigma_post_k = sigma_post_k[None, :, :]  # shape (1, M, 1)

        logml_j[0, k - 1] = logml

        np.save(f"{results_dir}/samples/beta_post{suffix}_{n_training_size}_{extra_suffix}.npy", beta_post_k)
        np.save(f"{results_dir}/samples/intercept_post{suffix}_{n_training_size}_{extra_suffix}.npy", intercept_post_k)
        np.save(f"{results_dir}/samples/sigma_post{suffix}_{n_training_size}_{extra_suffix}.npy", sigma_post_k)
    np.save(f"{results_dir}/samples/logml_j{dataset_name}_{n_training_size}_{extra_suffix}.npy", logml_j)


################################################################################
# JAX utilities: posterior predictive log density for Gaussian linear model
################################################################################
# @jit
def _logmeanexp(log_vals: jnp.ndarray, axis: int):
    """
    Compute log(mean(exp(log_vals))) in a numerically stable way.

    Parameters
    ----------
    log_vals : jnp.ndarray
    axis : int

    Returns
    -------
    jnp.ndarray
        log-mean-exp reduced along axis.
    """
    return logsumexp(log_vals, axis=axis) - jnp.log(log_vals.shape[axis])


@jit
def model_log_pred_density_point(
    beta_post: jnp.ndarray, intercept_post: jnp.ndarray, sigma_post: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
):
    """
    Posterior predictive log density at a single point (x, y) for ONE model.

    Shapes
    ------
    beta_post      : (M, p)
    intercept_post : (M, 1)
    sigma_post     : (M, 1)
    x              : (p,), single input
    y              : () scalar, single input

    Returns
    -------
    log p(y | x, D_train, model)  : () scalar
    """
    # mu_m = x^T beta_m + intercept_m
    mu = jnp.dot(beta_post, x) + intercept_post[:, 0]  # (M,)
    logpdf_m = norm.logpdf(y, loc=mu, scale=sigma_post[:, 0])  # (M,)
    return _logmeanexp(logpdf_m, axis=0)  # scalar, mean log posterior predictive


# Vectorize across calibration points: (n_points,)
model_log_pred_density_points = jit(vmap(model_log_pred_density_point, in_axes=(None, None, None, 0, 0), out_axes=0))


@jit
def model_log_pred_density_grid_for_one_x(
    beta_post: jnp.ndarray, intercept_post: jnp.ndarray, sigma_post: jnp.ndarray, x: jnp.ndarray, y_grid: jnp.ndarray
):
    """
    Log posterior predictive density on a y-grid for ONE fixed covariate x.

    Shapes
    ------
    beta_post      : (M, p)
    intercept_post : (M, 1)
    sigma_post     : (M, 1)
    x              : (p,)
    y_grid         : (n_plot,)

    Returns
    -------
    log p(y_grid[l] | x, D_train, model) : (n_plot,)
    """
    mu = jnp.dot(beta_post, x) + intercept_post[:, 0]  # (M,)
    # Broadcast: y_grid[:,None] with mu[None,:] -> (n_plot, M)
    logpdf = norm.logpdf(y_grid[:, None], loc=mu[None, :], scale=sigma_post[:, 0][None, :])  # (n_plot, M)
    return _logmeanexp(logpdf, axis=1)  # (n_plot,)


# Vectorize across test points x_test: output (n_test, n_plot)
model_log_pred_density_grid = jit(
    vmap(model_log_pred_density_grid_for_one_x, in_axes=(None, None, None, 0, None), out_axes=0)
)


def _batched_model_log_pred_density_grid(
    beta_post: jnp.ndarray,
    intercept_post: jnp.ndarray,
    sigma_post: jnp.ndarray,
    X_test: jnp.ndarray,
    y_grid: jnp.ndarray,
    batch_size: Optional[int] = 256,
):
    """
    Compute test-grid log predictive densities in test-point batches.

    Batching bounds peak memory without changing numerical results.
    """
    if batch_size is None or batch_size <= 0:
        return model_log_pred_density_grid(beta_post, intercept_post, sigma_post, X_test, y_grid)

    n_test = int(X_test.shape[0])
    if n_test <= batch_size:
        return model_log_pred_density_grid(beta_post, intercept_post, sigma_post, X_test, y_grid)

    out = []
    for start in range(0, n_test, batch_size):
        stop = min(start + batch_size, n_test)
        out.append(model_log_pred_density_grid(beta_post, intercept_post, sigma_post, X_test[start:stop], y_grid))
    return jnp.concatenate(out, axis=0)


################################################################################
# Rank-based inclusion using conformity scores
################################################################################
def _expand_y_grid_until_region_is_bounded(
    *,
    beta: jnp.ndarray,
    intercept: jnp.ndarray,
    sigma: jnp.ndarray,
    X_test_k: jnp.ndarray,
    log_s_cal: jnp.ndarray,
    alpha: float,
    n_test: int,
    n_cal: int,
    y_plot_init: np.ndarray,
    max_expansions: int = 8,
    expand_chunk: Optional[int] = None,
    max_n_plot: int = 5000,
    check_only_endpoints: bool = False,
    grid_eval_batch_size: Optional[int] = 256,
):
    """
    Expand a 1D y-grid until it is *wide enough* to identify interval endpoints.

    We approximate the conformal set
        C_alpha(x) = { y : p(y) > alpha }
    by evaluating the acceptance rule on a discrete grid `y_plot`.

    If the grid is too narrow, it can happen that the *boundary points* of the grid
    are still accepted (i.e. `y_plot[0]` is accepted or `y_plot[-1]` is accepted).
    Then extracting bounds via min/max accepted grid points underestimates the true
    interval width (the true boundary lies outside the grid).

    This helper fixes that by *expanding the grid outward* (left and/or right)
    until we find rejected points at the extremes.

    What "bounded" means here
    ---------------------------
    We stop expanding when BOTH of these are true:
      - All test points reject the leftmost grid value (no interval extends past it).
      - All test points reject the rightmost grid value.

    This guarantees that, for every test point, the min/max accepted grid points are
    not truncated by the grid endpoints.

    Parameters
    ----------
    beta, intercept, sigma : posterior draws for a single model (JAX arrays)
    X_test_k               : (n_test, p_k) covariates for this model
    log_s_cal              : (n_cal,) calibration log-scores for this model
    alpha                  : split conformal miscoverage level used in the rank test
    n_test                 : number of test points
    n_cal                  : number of calibration points
    y_plot_init            : initial y-grid (1D, increasing)
    max_expansions         : maximum number of outward expansions to attempt
    expand_chunk           : number of points to add each expansion side.
                             Default: len(y_plot_init) (keeps spacing constant and
                             grows the range in reasonably-sized jumps).
    max_n_plot             : hard cap on total number of grid points (safety).

    Returns
    -------
    y_plot_np      : (n_plot_final,) numpy array of the expanded grid.
    region         : (n_test, n_plot_final) boolean acceptance mask.
    log_s_star_grid: (n_test, n_plot_final) log predictive scores on the final grid.
    """
    y_plot_np = np.asarray(y_plot_init, dtype=float)

    if y_plot_np.ndim != 1 or y_plot_np.size < 2:
        raise ValueError("y_plot must be a 1D array with at least 2 grid points.")

    # Ensure the grid is increasing; the extension logic assumes monotone spacing.
    if not np.all(np.diff(y_plot_np) > 0):
        raise ValueError("y_plot must be strictly increasing.")

    # Estimate (uniform) spacing. If the initial grid is not perfectly uniform,
    # we use the median spacing as a robust estimate to keep extensions stable.
    delta = float(np.median(np.diff(y_plot_np)))
    if delta <= 0:
        raise ValueError("y_plot must have positive spacing.")

    # How many points to add per expansion step.
    chunk = int(expand_chunk) if expand_chunk is not None else int(y_plot_np.size)
    if chunk <= 0:
        raise ValueError("expand_chunk must be positive.")

    # Iteratively expand until endpoints are rejected for ALL test points.
    for i in range(max_expansions):
        if check_only_endpoints:
            # ------------------------------------------------------------
            # Only evaluate acceptance at the *current* endpoints.
            # ------------------------------------------------------------
            y_left = float(y_plot_np[0])
            y_right = float(y_plot_np[-1])

            y_left_vec = jnp.full((n_test,), y_left)
            y_right_vec = jnp.full((n_test,), y_right)

            log_s_left = model_log_pred_density_points(beta, intercept, sigma, X_test_k, y_left_vec)  # (n_test,)
            log_s_right = model_log_pred_density_points(beta, intercept, sigma, X_test_k, y_right_vec)  # (n_test,)

            # Rank-based acceptance at a single candidate y:
            # include iff (#{i: s_cal_i <= s*(y)} + 1) > alpha * (n_cal + 1)
            counts_left = jnp.sum(log_s_cal[None, :] <= log_s_left[:, None], axis=1)
            counts_right = jnp.sum(log_s_cal[None, :] <= log_s_right[:, None], axis=1)
            left_accept = (counts_left + 1) > alpha * (n_cal + 1)
            right_accept = (counts_right + 1) > alpha * (n_cal + 1)

            left_any_accepted = bool(np.any(np.asarray(left_accept)))
            right_any_accepted = bool(np.any(np.asarray(right_accept)))

            if (not left_any_accepted) and (not right_any_accepted):
                bounded = True
                break
        else:
            # ------------------------------------------------------------
            # Compute the whole region each iteration.
            # ------------------------------------------------------------
            y_plot_j = jnp.asarray(y_plot_np)
            log_s_star_grid = _batched_model_log_pred_density_grid(
                beta, intercept, sigma, X_test_k, y_plot_j, batch_size=grid_eval_batch_size
            )
            region = split_conformal_region_many_tests(log_s_cal, log_s_star_grid, alpha)

            left_any_accepted = bool(np.any(np.asarray(region[:, 0])))
            right_any_accepted = bool(np.any(np.asarray(region[:, -1])))

            if (not left_any_accepted) and (not right_any_accepted):
                return y_plot_np, region, log_s_star_grid

        # Safety cap to avoid uncontrolled memory growth.
        # Apply this regardless of `check_only_endpoints`; otherwise endpoint-only
        # adaptive checks can grow the grid far past `max_n_plot`.
        n_after = y_plot_np.size + chunk * int(left_any_accepted) + chunk * int(right_any_accepted)
        if n_after > max_n_plot:
            break

        # Expand where needed, keeping the same spacing `delta`.
        if left_any_accepted:
            new_left = y_plot_np[0] - delta * np.arange(chunk, 0, -1)
            y_plot_np = np.concatenate([new_left, y_plot_np])

        if right_any_accepted:
            new_right = y_plot_np[-1] + delta * np.arange(1, chunk + 1)
            y_plot_np = np.concatenate([y_plot_np, new_right])

    # If we exit the loop, we did not fully bound the region within max_expansions.
    y_plot_j = jnp.asarray(y_plot_np)
    log_s_star_grid = _batched_model_log_pred_density_grid(
        beta, intercept, sigma, X_test_k, y_plot_j, batch_size=grid_eval_batch_size
    )
    region = split_conformal_region_many_tests(log_s_cal, log_s_star_grid, alpha)
    return y_plot_np, region, log_s_star_grid


def _split_conformal_interval_bracketed(
    *,
    log_s_cal: jnp.ndarray,  # (n_cal,)
    beta: jnp.ndarray,  # (M, p)
    intercept: jnp.ndarray,  # (M, 1)
    sigma: jnp.ndarray,  # (M, 1)
    X_test: jnp.ndarray,  # (n_test, p)
    alpha: float,
    return_opt_success: bool = False,
):
    """Compute split-conformal *interval* bounds [l, u] without a y-grid.

    Parameters
    ----------
    log_s_cal:
        Calibration log conformity scores for the *fixed* calibration set.
        Shape (n_cal,).
    beta, intercept, sigma:
        Posterior draws for one Bayesian linear model.
    X_test:
        Test covariates for this model (already sliced to the model's features).
        Shape (n_test, p).
    alpha:
        Miscoverage level. The conformal set is {y : p(y) > alpha}.
    return_opt_success:
        If True, also return a boolean indicating whether all bracketing/root
        operations reported success.

    Returns
    -------
    l, u : np.ndarray, np.ndarray
        Arrays of shape (n_test,) giving lower/upper bounds for each test point.
    success : bool (optional)
        Only if return_opt_success=True.
    """
    n_cal = int(log_s_cal.shape[0])
    n_test = int(X_test.shape[0])

    threshold_count = int(np.floor(alpha * (n_cal + 1)))
    p_threshold = (threshold_count + 1) / (n_cal + 1)
    log_threshold = float(np.log(p_threshold))

    # -----------------------------------------------------------------------
    # Initial guess for the centre y* (argmax p(y))
    # -----------------------------------------------------------------------
    beta_mean = jnp.mean(beta, axis=0)  # (p,)
    intercept_mean = jnp.mean(intercept[:, 0])  # ()
    y0 = np.asarray(jnp.dot(X_test, beta_mean) + intercept_mean)  # (n_test,)

    # `mask` selects which batch elements are being optimised by SciPy's
    # elementwise routines (here: all test points).
    mask = np.arange(n_test)

    def objective(y_test: np.ndarray, mask_idx: np.ndarray) -> np.ndarray:
        """Return log p(y_test) for each selected test point.

        SciPy's elementwise bracketing utilities pass `y_test` as a numpy array
        (one entry per batch element). We compute the candidate log conformity
        scores log_s_star(x_i, y_i) and then the rank-based p-values.
        """
        y_test = np.asarray(y_test)
        if y_test.ndim == 0:
            y_test = y_test[None]  # ensure 1D

        # Slice the test covariates to the active batch elements.
        X_sub = X_test[jnp.asarray(mask_idx)]

        # Candidate log conformity scores for each test point at its candidate y.
        # Shape: (b,)
        log_s_star = model_log_pred_density_points(beta, intercept, sigma, X_sub, jnp.asarray(y_test))

        # Rank-based p-values:
        # counts[i] = #{j: log_s_cal[j] <= log_s_star[i]}
        counts = jnp.sum(log_s_cal[None, :] <= log_s_star[:, None], axis=1)
        p_vals = (counts + 1.0) / (n_cal + 1.0)

        # Work in log-space for numerical stability (and because Algorithm 1
        # uses optimisation/root-finding on p(y)).
        log_p = jnp.log(p_vals + 1e-12)
        return np.asarray(log_p)

    # Track success flags so callers can diagnose optimisation failures.
    success_flags = []

    # -----------------------------------------------------------------------
    # find y* = argmax_y p(y)
    # -----------------------------------------------------------------------
    # We maximise objective(y) == log p(y) by *minimising* -objective(y).
    bracket_res = oe.bracket_minimum(lambda x, m: -objective(x, m), y0, args=(mask,))
    success_flags.append(getattr(bracket_res, "success", False))
    bracket = bracket_res.bracket

    minimum_res = oe.find_minimum(lambda x, m: -objective(x, m), bracket, args=(mask,))
    success_flags.append(getattr(minimum_res, "success", False))
    y_star = minimum_res.x  # (n_test,)

    # -----------------------------------------------------------------------
    # upper endpoint u
    # -----------------------------------------------------------------------
    # Find the smallest y >= y_star such that p(y) <= alpha (discrete threshold).
    bracket_u_res = oe.bracket_root(
        lambda x, m: objective(x, m) - log_threshold,
        y_star,
        xmin=y_star,  # enforce searching to the right of the mode
        args=(mask,),
    )
    success_flags.append(getattr(bracket_u_res, "success", False))

    root_u_res = oe.find_root(
        lambda x, m: objective(x, m) - log_threshold,
        bracket_u_res.bracket,
        args=(mask,),
    )
    success_flags.append(getattr(root_u_res, "success", False))
    u = root_u_res.x  # (n_test,)

    # -----------------------------------------------------------------------
    # ower endpoint l
    # -----------------------------------------------------------------------
    # Mirror trick used in alternative_code.py:
    # solve for y <= y_star by searching for a root in x >= -y_star of
    # objective(-x) - log_threshold = 0, then map back with y = -x.
    bracket_l_res = oe.bracket_root(
        lambda x, m: objective(-x, m) - log_threshold,
        -y_star,
        xmin=-y_star,
        args=(mask,),
    )
    success_flags.append(getattr(bracket_l_res, "success", False))

    root_l_res = oe.find_root(
        lambda x, m: objective(-x, m) - log_threshold,
        bracket_l_res.bracket,
        args=(mask,),
    )
    success_flags.append(getattr(root_l_res, "success", False))
    l = -root_l_res.x  # (n_test,)

    # Numerical safety: enforce l <= u elementwise.
    l, u = np.minimum(l, u), np.maximum(l, u)

    if return_opt_success:
        return l, u, all(success_flags)
    return l, u


@jit
def split_conformal_region_from_log_scores(log_s_cal: jnp.ndarray, log_s_star_grid: jnp.ndarray, alpha: float):
    """
    Compute split conformal region on y_grid for one test point using
    rank-based conformal p-values:

        rank(y) = #{i in cal : s_i <= s*(y)}
        include y if rank(y) > alpha * (n_cal + 1)

    We compare in log-space (monotone).

    Shapes
    ------
    log_s_cal       : (n_cal,), conformity scores for the calibration points.
    log_s_star_grid : (n_plot,), conformity scores for the candidate values.

    Returns
    -------
    region : (n_plot,) boolean
    """
    n_cal = log_s_cal.shape[0]
    # counts[l] = #{i: log_s_cal[i] <= log_s_star_grid[l]}
    counts = jnp.sum(log_s_cal[:, None] <= log_s_star_grid[None, :], axis=0)  # (n_plot,)
    ranks = counts + 1  # add 1, (n_plot,)
    return ranks > alpha * (n_cal + 1)


# Vectorize region computation across test points: (n_test, n_plot)
split_conformal_region_many_tests = jit(
    vmap(split_conformal_region_from_log_scores, in_axes=(None, 0, None), out_axes=0)
)


################################################################################
# Split-Conformal CB (per model) and Split-Conformal CBMA
################################################################################
def _load_posteriors_for_models(
    results_dir: str, dataset_name: str, features_of_models, n_training_size: int, extra_suffix
):
    """
    Load saved posterior draws for each model.

    Returns
    -------
    posts : list[dict]
        posts[k] has:
            beta_post      (M, p_k)
            intercept_post (M, 1)
            sigma_post     (M, 1)
    logml : (K,) jnp.ndarray
        log marginal likelihoods from training.
    """
    K = len(features_of_models)
    logml_j = jnp.load(f"{results_dir}/samples/logml_j{dataset_name}_{n_training_size}_{extra_suffix}.npy")  # (1, K)
    logml = logml_j[0]  # (K,)

    posts = []
    for k in range(1, K + 1):
        suffix = f"{dataset_name}_model_{k}_{n_training_size}"
        beta = jnp.load(f"{results_dir}/samples/beta_post{suffix}_{n_training_size}_{extra_suffix}.npy")[0]  # (M, p_k)
        intercept = jnp.load(f"{results_dir}/samples/intercept_post{suffix}_{n_training_size}_{extra_suffix}.npy")[
            0
        ]  # (M, 1)
        sigma = jnp.load(f"{results_dir}/samples/sigma_post{suffix}_{n_training_size}_{extra_suffix}.npy")[0]  # (M, 1)

        posts.append({"beta": beta, "intercept": intercept, "sigma": sigma})
    return posts, logml


@jit
def get_conformal_bounds(region_cb: jnp.ndarray, y_plot: jnp.ndarray):
    """
    Extracts the smallest and largest accepted y values for each test point.

    Args:
        region_cb : (n_test, n_plot) boolean array indicating acceptance.
        y_plot    : (n_plot,) array of grid values.

    Returns:
        region_l : (1, n_test) Smallest accepted value (or inf if none).
        region_u : (1, n_test) Largest accepted value (or -inf if none).
    """
    # 0. Pre-calculate fallbacks (Global min/max of the grid)
    global_min = jnp.min(y_plot)
    global_max = jnp.max(y_plot)

    # Lower Bound (Smallest accepted value)
    # Replace False entries with +infinity so they are ignored by min()
    # jnp.where broadcasts y_plot to match region_cb shape automatically
    candidates_l = jnp.where(region_cb, y_plot, jnp.inf)
    min_vals = jnp.min(candidates_l, axis=1)

    # Replace +inf (no accepted points) with global_min
    # jnp.isinf checks for infinity
    region_l = jnp.where(jnp.isinf(min_vals), global_min, min_vals)

    # Upper Bound (Largest accepted value)
    # Replace False entries with -infinity so they are ignored by max()
    candidates_u = jnp.where(region_cb, y_plot, -jnp.inf)
    max_vals = jnp.max(candidates_u, axis=1)

    # Replace -inf (no accepted points) with global_max
    region_u = jnp.where(jnp.isinf(max_vals), global_max, max_vals)

    return region_l.reshape(
        -1,
    ), region_u.reshape(
        -1,
    )


def run_CB(
    use_grid: bool,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_plot: np.ndarray,
    dataset_name: str,
    results_dir: str,
    alpha: float,
    features_of_models,
    n_training_size: int = 0,
    extra_suffix: str = "",
    adaptive_grid: bool = False,
    max_grid_expansions: int = 5,
    check_only_endpoints_adaptively: bool = False,
    grid_expand_chunk: Optional[int] = None,
    max_n_plot: int = 5000,
    grid_eval_batch_size: Optional[int] = 256,
):
    """
    Split conformal Conformal Bayes (CB) PER MODEL.

    Conformity score for model k:
        s_k(x,y) = p_k(y | x, D_train)  (posterior predictive density)

    Calibration scores:
        s_cal_i = s_k(X_cal[i], y_cal[i])   for i=1..n_cal

    Test grid scores:
        s_star_{t,l} = s_k(X_test[t], y_plot[l])

    Region rule (rank-based split conformal):
        rank_{t,l} = 1 + #{i: s_cal_i <= s_star_{t,l}}
        include y_plot[l] if rank_{t,l} > alpha*(n_cal+1)

    Shapes
    ------
    X_cal  : (n_cal, d)
    y_cal  : (n_cal,)
    X_test : (n_test, d)
    y_test : (n_test,)
    y_plot : (n_plot,)

    Returns
    -------
    dict with:
      region_cb   : (1, n_test, n_plot, K)  boolean (stored as float 0/1 in numpy)
      coverage_cb : (1, n_test, K)          float {0,1}
      length_cb   : (1, n_test, K)          float (approx length on y_plot grid)
    """
    K = len(features_of_models)
    n_test = X_test.shape[0]
    if use_grid:
        n_plot = y_plot.shape[0]

    posts, logml = _load_posteriors_for_models(
        results_dir, dataset_name, features_of_models, n_training_size, extra_suffix
    )

    # Allocate outputs
    if use_grid:
        region_cb = np.zeros((1, n_test, n_plot, K), dtype=bool)
    else:
        region_cb = None
    coverage_cb = np.zeros((1, n_test, K), dtype=float)
    length_cb = np.zeros((1, n_test, K), dtype=float)

    # Convert data to JAX
    if use_grid:
        y_plot_j = jnp.asarray(y_plot)
    X_cal_j = jnp.asarray(X_cal)
    y_cal_j = jnp.asarray(y_cal)
    X_test_j = jnp.asarray(X_test)
    n_cal = y_cal.shape[0]

    for k in range(K):
        feats = features_of_models[k]
        X_cal_k = X_cal_j[:, feats]  # (n_cal, p_k)
        X_test_k = X_test_j[:, feats]  # (n_test, p_k)

        beta = posts[k]["beta"]  # (M, p_k)
        intercept = posts[k]["intercept"]  # (M, 1)
        sigma = posts[k]["sigma"]  # (M, 1)

        # Calibration log scores: (n_cal,)
        # This returns the log posterior predictive density (conformity score) for the calibration data points
        log_s_cal = model_log_pred_density_points(beta, intercept, sigma, X_cal_k, y_cal_j)

        if use_grid:
            if adaptive_grid:
                # Expand the grid until we see rejected points at the extremes
                y_plot_np, region, _ = _expand_y_grid_until_region_is_bounded(
                    beta=beta,
                    intercept=intercept,
                    sigma=sigma,
                    X_test_k=X_test_k,
                    log_s_cal=log_s_cal,
                    alpha=alpha,
                    n_test=n_test,
                    n_cal=n_cal,
                    y_plot_init=y_plot,
                    max_expansions=max_grid_expansions,
                    expand_chunk=grid_expand_chunk,
                    max_n_plot=max_n_plot,
                    check_only_endpoints=check_only_endpoints_adaptively,
                    grid_eval_batch_size=grid_eval_batch_size,
                )
                y_plot_j = jnp.asarray(y_plot_np)

                # Extract interval bounds from the (possibly expanded) grid.
                region_l, region_u = get_conformal_bounds(region, y_plot_j)

            else:
                # Test grid log scores: (n_test, n_plot)
                # These are the log posterior predictive density for x_{i, test}, y_{j, candidate}, i.e. the
                # ith test point and jth candidate value
                log_s_star_grid = _batched_model_log_pred_density_grid(
                    beta, intercept, sigma, X_test_k, y_plot_j, batch_size=grid_eval_batch_size
                )

                # Split conformal region for all tests: (n_test, n_plot)
                region = split_conformal_region_many_tests(log_s_cal, log_s_star_grid, alpha)
                region_l, region_u = get_conformal_bounds(region, y_plot)  # shapes
                region_cb[0, :, :, k] = np.asarray(region)

        else:
            # Grid-free (bracketed) interval computation
            region_l, region_u = _split_conformal_interval_bracketed(
                log_s_cal=log_s_cal,
                beta=beta,
                intercept=intercept,
                sigma=sigma,
                X_test=X_test_k,
                alpha=alpha,
                return_opt_success=False,
            )

        # Coverage
        cov = (y_test >= region_l) & (y_test <= region_u)  # shape (n_test, 1)
        coverage_cb[0, :, k] = np.asarray(cov, dtype=float)

        # Use region_l, region_u to find length
        length_cb[0, :, k] = np.asarray(region_u - region_l, dtype=float)

    return {"region_cb": region_cb, "coverage_cb": coverage_cb, "length_cb": length_cb}


def run_CBMA(
    use_grid: bool,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_plot: np.ndarray,
    dataset_name: str,
    results_dir: str,
    alpha: float,
    features_of_models,
    n_training_size: int = 0,
    model_prior: np.ndarray | None = None,
    extra_suffix: str = "",
    adaptive_grid: bool = False,
    max_grid_expansions: int = 5,
    check_only_endpoints_adaptively: bool = False,
    grid_expand_chunk: Optional[int] = None,
    max_n_plot: int = 5000,
    grid_eval_batch_size: Optional[int] = 256,
):
    """
    Split conformal CBMA using a fixed BMA mixture built from training data only.

    Conformity score:
        s_CBMA(x,y) = sum_k pi_k * p_k(y | x, D_train)

    where pi_k = P(M_k | D_train) are computed from training marginal likelihoods
    (and optional model priors).

    Shapes
    ------
    X_cal  : (n_cal, d)
    y_cal  : (n_cal,)
    X_test : (n_test, d)
    y_test : (n_test,)
    y_plot : (n_plot,)

    Returns
    -------
    dict with:
      region_cbma   : (1, n_test, n_plot) boolean
      coverage_cbma : (1, n_test)         float {0,1}
      length_cbma   : (1, n_test)         float
      pi_k          : (K,)                posterior model weights used
    """
    K = len(features_of_models)
    n_test = X_test.shape[0]
    n_plot = y_plot.shape[0]
    dy = float(y_plot[1] - y_plot[0])

    posts, logml = _load_posteriors_for_models(
        results_dir, dataset_name, features_of_models, n_training_size, extra_suffix
    )

    # Model prior (optional). If None, use uniform prior.
    if model_prior is None:
        log_prior = jnp.zeros((K,))  # uniform
    else:
        model_prior = np.asarray(model_prior, dtype=float)
        model_prior = model_prior / model_prior.sum()
        log_prior = jnp.log(jnp.asarray(model_prior))

    # Posterior model weights from TRAINING data:
    #   log_pi_k ∝ log_prior_k + logml_k
    log_pi = log_prior + logml
    log_pi = log_pi - logsumexp(log_pi)
    pi = jnp.exp(log_pi)  # (K,)

    # Allocate outputs
    if use_grid:
        region_cbma = np.zeros((1, n_test, n_plot), dtype=bool)
    else:
        region_cbma = None
    coverage_cbma = np.zeros((1, n_test), dtype=float)
    length_cbma = np.zeros((1, n_test), dtype=float)

    # Convert data to JAX
    y_plot_j = jnp.asarray(y_plot)
    X_cal_j = jnp.asarray(X_cal)
    y_cal_j = jnp.asarray(y_cal)
    X_test_j = jnp.asarray(X_test)
    n_cal = y_cal.shape[0]

    # Compute per-model calibration log predictive densities: (K, n_cal)
    log_pred_cal_models = []
    for k in range(K):
        feats = features_of_models[k]
        X_cal_k = X_cal_j[:, feats]  # (n_cal, p_k)

        beta = posts[k]["beta"]
        intercept = posts[k]["intercept"]
        sigma = posts[k]["sigma"]

        # Get the conformity scores on the calibration set
        log_pred_cal_k = model_log_pred_density_points(beta, intercept, sigma, X_cal_k, y_cal_j)  # (n_cal,)
        log_pred_cal_models.append(log_pred_cal_k)

    log_pred_cal_models = jnp.stack(log_pred_cal_models, axis=0)  # (K, n_cal)

    # Mixture calibration log-scores: log s_cal = log sum_k pi_k * exp(log_pred_cal_k)
    log_s_cal = logsumexp(log_pi[:, None] + log_pred_cal_models, axis=0)  # (n_cal,)

    if use_grid:
        if not adaptive_grid:
            # Mixture test-grid log-scores: (n_test, n_plot)
            log_s_star_grid = None
            for k in range(K):
                feats = features_of_models[k]
                X_test_k = X_test_j[:, feats]  # (n_test, p_k)

                beta = posts[k]["beta"]
                intercept = posts[k]["intercept"]
                sigma = posts[k]["sigma"]

                log_pred_test_k = _batched_model_log_pred_density_grid(
                    beta, intercept, sigma, X_test_k, y_plot_j, batch_size=grid_eval_batch_size
                )  # (n_test, n_plot)
                weighted_log_pred = log_pi[k] + log_pred_test_k
                if log_s_star_grid is None:
                    log_s_star_grid = weighted_log_pred
                else:
                    log_s_star_grid = jnp.logaddexp(log_s_star_grid, weighted_log_pred)

            # Split conformal region for all test points: (n_test, n_plot)
            region = split_conformal_region_many_tests(log_s_cal, log_s_star_grid, alpha)
            region_l, region_u = get_conformal_bounds(region, y_plot)  # shapes
        else:
            # Keep legacy behavior: adaptive expansion relies on the final model.
            beta = posts[-1]["beta"]
            intercept = posts[-1]["intercept"]
            sigma = posts[-1]["sigma"]

            # Expand the grid until we see rejected points at the extremes
            y_plot_np, region, _ = _expand_y_grid_until_region_is_bounded(
                beta=beta,
                intercept=intercept,
                sigma=sigma,
                X_test_k=X_test_j,
                log_s_cal=log_s_cal,
                alpha=alpha,
                n_test=n_test,
                n_cal=n_cal,
                y_plot_init=y_plot,
                max_expansions=max_grid_expansions,
                expand_chunk=grid_expand_chunk,
                max_n_plot=max_n_plot,
                check_only_endpoints=check_only_endpoints_adaptively,
                grid_eval_batch_size=grid_eval_batch_size,
            )
            y_plot_j = jnp.asarray(y_plot_np)                

            # Extract interval bounds from the (possibly expanded) grid.
            region_l, region_u = get_conformal_bounds(region, y_plot_j)                                       

    else:
        # Grid-free (bracketed) interval computation
        region_l, region_u = _split_conformal_interval_bracketed(
            log_s_cal=log_s_cal,
            beta=beta,
            intercept=intercept,
            sigma=sigma,
            X_test=X_test_j,
            alpha=alpha,
            return_opt_success=False,
        )        

    # Get coverage and length
    cov = (y_test >= region_l) & (y_test <= region_u)  # shape (n_test, 1)
    coverage_cbma[0, :] = np.asarray(cov, dtype=float)
    length_cbma[0, :] = np.asarray(region_u - region_l, dtype=float)    

    return {
        "region_cbma": region_cbma,
        "coverage_cbma": coverage_cbma,
        "length_cbma": length_cbma,
        "pi_k": np.asarray(pi),
    }


def predict_from_saved_bayesian_linear_model(
    X: np.ndarray,
    dataset_name: str,
    results_dir: str,
    features_of_models: List,
    model_index: int,
    n_training_size: int = 0,
    extra_suffix: str = "",
):
    """
    Predict E[y | x, D_train] from one saved Bayesian linear model.

    Parameters
    ----------
    X : (n, d) np.ndarray
        Input features to predict on.
    dataset_name : str
    results_dir : str
    features_of_models : list[list[int]]
        Feature indices for each saved model.
    model_index : int
        1-based model index (same indexing convention used in saved filenames).
    n_training_size : int
        Training-size label used in saved posterior filenames.
    extra_suffix : str
        Extra suffix used in saved posterior filenames.

    Returns
    -------
    y_pred_mean : (n,) np.ndarray
        Posterior mean prediction for each row of X.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (n, d). Got shape={X.shape}.")

    K = len(features_of_models)
    if model_index < 1 or model_index > K:
        raise ValueError(f"model_index must be in [1, {K}], got model_index={model_index}.")

    posts, _ = _load_posteriors_for_models(
        results_dir=results_dir,
        dataset_name=dataset_name,
        features_of_models=features_of_models,
        n_training_size=n_training_size,
        extra_suffix=extra_suffix,
    )

    feats = features_of_models[model_index - 1]
    if len(feats) == 0:
        raise ValueError("Selected model has no features.")

    if int(np.max(feats)) >= X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} columns, but model {model_index} expects " f"feature index {int(np.max(feats))}."
        )

    X_k = jnp.asarray(X[:, feats])  # (n, p_k)
    beta = posts[model_index - 1]["beta"]  # (M, p_k)
    intercept = posts[model_index - 1]["intercept"][:, 0]  # (M,)

    # Posterior draws of the regression mean for each x:
    # shape (n, M), where M is number of posterior samples.
    mu_draws = X_k @ beta.T + intercept[None, :]
    y_pred_mean = jnp.mean(mu_draws, axis=1)  # (n,)

    return np.asarray(y_pred_mean)

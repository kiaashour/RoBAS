import warnings
from typing import Any, Optional, Tuple, Callable

import jax.numpy as jnp
import numpy as np
import scipy.optimize.elementwise as oe
import scipy.stats as stats
import numpy as np

from .nonconformity_measures import (
    dta_full,
    dto_full,
    hs_full,
    hs_eb_full,
    hs_full_plugin_s2,
)


def conformal_region_full(
    y_train: np.ndarray,  # [n] | [b, n]
    y_val: np.ndarray,  # [m] | [b, m]
    *,
    score_method: str = "dta",
    alpha: float = 0.1,
    use_grid: bool = False,
    score_kwargs: Optional[dict] = None,
    return_opt_success: bool = False,
    return_grid_and_accepted_for_conformal_full_grid: bool = False,
    **grid_kwargs: Any,
):
    """
    Compute a conformal prediction region for 1D targets.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets with shape ``(n,)`` or batched shape ``(b, n)``.
    y_val : np.ndarray
        Validation targets with shape ``(m,)`` or ``(b, m)``.
    score_method : str, default="dta"
        Nonconformity score method (for example ``"dta"``, ``"dto"``, ``"hs"``, ``"hs_eb"``, ``"hs_plugin"``, or ``"hoff"``).
    alpha : float, default=0.1
        Miscoverage level.
    use_grid : bool, default=False
        If ``True``, use grid search; otherwise use interval optimization.
    score_kwargs : dict or None, optional
        Extra options for the selected score method.
    return_opt_success : bool, default=False
        If ``True`` and ``use_grid`` is ``False``, also return optimization success.
    return_grid_and_accepted_for_conformal_full_grid : bool, default=False
        If ``True`` with grid mode, also return the acceptance mask and full grid.
    **grid_kwargs : Any
        Extra keyword arguments forwarded to grid mode.

    Returns
    -------
    Any
        Conformal region output from grid mode or interval mode, optionally with success metadata.
    """
    if y_train.ndim != 1 and use_grid:
        raise ValueError("Batched y_train only supported with interval method.")

    score_args = tuple()

    # Select conformity score function
    if score_method == "dta":
        score_fn = dta_full
    elif score_method == "dto":
        score_fn = dto_full
    elif score_method == "hs_plugin":
        score_fn = hs_full_plugin_s2
    elif score_method == "hs_eb":
        score_fn = hs_eb_full        
    elif score_method == "hs":
        # s2_hat = np.var(y_val, axis=-1, ddof=1)  # [] | [b]
        if score_kwargs.get(score_method, None) is None:
            raise ValueError(
                "Score method HS must have score_kwargs variance hyperparameter specified"
            )
        else:
            s2_hat = score_kwargs[score_method]["variance"]
        score_fn = hs_full
        score_args = (s2_hat,)
    elif score_method == "hoff":
        mu_shape = (1,) if y_train.ndim == 1 else (y_train.shape[0],)
        mu = np.zeros(mu_shape)  # [b]
        return _hoff_exact_interval(
            y_train,
            mu,
            alpha=alpha,
            **score_kwargs["hoff"],
        )
    else:
        raise ValueError(f"Unknown score method: {score_method}")

    if use_grid:
        # Get conformal region using grid search
        return _conformal_region_full_grid(
            y_train,
            alpha=alpha,
            score_fn=score_fn,
            score_args=score_args,
            return_grid_and_accepted_for_conformal_full_grid=return_grid_and_accepted_for_conformal_full_grid,
            **grid_kwargs,
        )
    # Get conformal region using interval search
    return _conformal_region_full_interval(
        y_train,
        alpha=alpha,
        score_fn=score_fn,
        score_args=score_args,
        return_opt_success=return_opt_success,
    )


def _conformal_region_full_grid(
    y_train: np.ndarray,  # [n]
    *,
    alpha: float = 0.1,
    score_fn: callable,
    score_args: Tuple = (),
    grid_guess: Optional[Tuple[float, float]] = None,
    grid_radius: float = 1.0,
    grid_size: int = 1000,
    max_grid_size: int = 10000,
    max_refinements: int = 10,
    return_interval: bool = False,
    return_grid_and_accepted_for_conformal_full_grid: bool = False,
):
    """
    Compute a conformal region using a 1D adaptive grid search.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets with shape ``(n,)``.
    alpha : float, default=0.1
        Miscoverage level.
    score_fn : callable
        Score function that returns scores for each test grid point.
    score_args : tuple, default=()
        Extra positional arguments forwarded to ``score_fn``.
    grid_guess : tuple[float, float] or None, default=None
        Initial grid bounds. If ``None``, bounds are inferred from ``y_train``.
    grid_radius : float, default=1.0
        Radius factor used when expanding the search bounds.
    grid_size : int, default=1000
        Initial number of grid points.
    max_grid_size : int, default=10000
        Maximum allowed grid size after refinements.
    max_refinements : int, default=10
        Maximum bound-expansion iterations per side.
    return_interval : bool, default=False
        If ``True``, return interval bounds instead of accepted grid points.
    return_grid_and_accepted_for_conformal_full_grid : bool, default=False
        If ``True``, return accepted points, acceptance mask, and full grid.

    Returns
    -------
    Any
        Accepted grid points, interval bounds, or grid diagnostics depending on flags.
    """
    n = len(y_train)

    def conformal_region(y_test):
        """
        Evaluate conformal acceptance for candidate targets.

        Parameters
        ----------
        y_test : np.ndarray
            Candidate targets with shape ``(m,)``.

        Returns
        -------
        np.ndarray
            Boolean acceptance mask with shape ``(m,)``.
        """
        # Compute scores with nonconformity score
        scores = np.asarray(
            score_fn(
                jnp.asarray(y_train),
                jnp.asarray(y_test),
                *score_args,
            ).block_until_ready()
        )  # [grid_size, n + 1]

        # Compute threshold for acceptance
        threshold = np.floor(alpha * (n + 1))

        # Check if in conformal region
        return np.sum(
            scores[:, :-1] >= scores[:, [-1]],
            axis=-1,
        ) > (
            threshold
            - 1  # Note: We are using -1 instead of the original threshold, and > instead of the original >=; these are equivalent
        )  # [grid_size]

    # Defining initial step size for the grid search
    l = grid_radius * min(np.std(y_train), 1.0) / np.sqrt(n) * stats.norm.ppf(alpha / 2)
    u = -l

    # Defining initial bounds for the grid
    if grid_guess is None:
        grid_l_init = np.min(y_train)
        grid_u_init = np.max(y_train)
    else:
        grid_l_init = grid_guess[0]
        grid_u_init = grid_guess[1]

    grid_l = grid_l_init + l
    num_refinements_l = 0

    # Refine the grid until the bounds are outside the conformal region
    # (this assumes the conformal region is an interval)
    while conformal_region(
        grid_l[None]
    ):  # check if lower bound is in the conformal region
        if num_refinements_l >= max_refinements:
            warnings.warn(
                "Maximum number of refinements reached. "
                "Consider increasing the `max_refinements` parameter."
            )
            break

        # Expand the lower bound
        l *= 2
        grid_l = grid_l_init + l
        num_refinements_l += 1

    grid_u = grid_u_init - l
    num_refinements_u = 0

    # Refine the grid until the bounds are outside the conformal region
    # (this assumes the conformal region is an interval)
    while conformal_region(
        grid_u[None]
    ):  # check if upper bound is in the conformal region
        if num_refinements_u >= max_refinements:
            warnings.warn(
                "Maximum number of refinements reached. "
                "Consider increasing the `max_refinements` parameter."
            )
            break

        # Expand the upper bound
        u *= 2
        grid_u = grid_u_init + u
        num_refinements_u += 1

    # Adjust the grid size in a way that is proportional to the average number of refinements
    grid_size = np.clip(
        grid_size * 2 ** ((num_refinements_l + num_refinements_u) // 2),
        None,
        max_grid_size,
    )

    # Create test grid
    y_test_grid = np.linspace(
        grid_l,
        grid_u,
        grid_size,
    )

    # Get boolean array corresponding to points in the conformal region
    in_region = conformal_region(y_test_grid)  # [grid_size]
    if return_grid_and_accepted_for_conformal_full_grid:
        return y_test_grid[in_region], in_region, y_test_grid
    if return_interval:
        l_idx = np.argmax(in_region) - 1
        r_idx = grid_size - np.argmax(in_region[::-1])
        return y_test_grid[l_idx], y_test_grid[r_idx]
    return y_test_grid[in_region]


def _conformal_region_full_interval(
    y_train: np.ndarray,  # [n] | [b, n]
    *,
    alpha: float = 0.1,
    score_fn: callable,
    score_args: Tuple = (),
    return_opt_success=False,
):
    """
    Compute conformal interval bounds via numerical optimization.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets with shape ``(n,)`` or batched shape ``(b, n)``.
    alpha : float, default=0.1
        Miscoverage level.
    score_fn : callable
        Score function evaluated inside the optimization loop.
    score_args : tuple, default=()
        Extra positional arguments forwarded to ``score_fn``.
    return_opt_success : bool, default=False
        If ``True``, also return an aggregate optimization-success flag.

    Returns
    -------
    Any
        Lower and upper bounds (and optionally a success flag).
    """
    if y_train.ndim == 1:
        y_train = y_train[None]  # [1, n]
        # score_args = tuple(args[None] for args in score_args)
    b, n = y_train.shape  # b: batch size, n: number of training points

    def objective(
        y_test: float,  # [b]
        mask: int,  # [b]
    ):
        """
        Compute log conformal p-values for candidate targets.

        Parameters
        ----------
        y_test : np.ndarray
            Candidate targets for selected batch items.
        mask : np.ndarray
            Batch indices used to select rows from ``y_train`` and ``score_args``.

        Returns
        -------
        np.ndarray
            Log p-values with shape ``(len(mask),)``.
        """
        args = tuple(arg[mask] for arg in score_args)

        # Get the conformity score for a single point in the augmented set
        # of training and test points
        scores = np.asarray(
            score_fn(
                jnp.asarray(y_train[mask]),  # [b, n]
                jnp.asarray(y_test[:, None]),  # [b, 1]
                *args,
            ).block_until_ready()
        )  # [b, 1, n + 1]
        # Get the proportion of scores that are greater than or equal to the score for each
        # test point
        res = np.mean(
            scores[..., :-1] >= scores[..., [-1]],
            axis=-1,
        )[
            :, 0
        ]  # [b]
        return np.log(res + 1e-10)

    mask = np.arange(b)
    threshold = np.log(
        np.floor(alpha * (n + 1)) / (n + 1)
    )  # get threshold for acceptance
    y_mean = np.mean(y_train, axis=-1)  # [b]

    # Track success flags
    success_flags = []

    # Minimum (centre)
    bracket_res = oe.bracket_minimum(
        lambda x, y: -objective(x, y), y_mean, args=(mask,)
    )
    success_flags.append(getattr(bracket_res, "success", False))
    bracket = bracket_res.bracket
    minimum_res = oe.find_minimum(lambda x, y: -objective(x, y), bracket, args=(mask,))
    success_flags.append(getattr(minimum_res, "success", False))
    grid_center = minimum_res.x  # [b]

    # Upper root
    bracket_u_res = oe.bracket_root(
        lambda x, y: objective(x, y) - threshold,
        grid_center,
        xmin=grid_center,
        args=(mask,),
    )
    success_flags.append(getattr(bracket_u_res, "success", False))
    bracket_u = bracket_u_res.bracket
    root_u_res = oe.find_root(
        lambda x, y: objective(x, y) - threshold,
        bracket_u,
        args=(mask,),
    )
    success_flags.append(getattr(root_u_res, "success", False))
    u = root_u_res.x  # [b]

    # Lower root (mirror)
    bracket_l_res = oe.bracket_root(
        lambda x, y: objective(-x, y) - threshold,
        -grid_center,
        xmin=-grid_center,
        args=(mask,),
    )
    success_flags.append(getattr(bracket_l_res, "success", False))
    bracket_l = bracket_l_res.bracket
    root_l_res = oe.find_root(
        lambda x, y: objective(-x, y) - threshold,
        bracket_l,
        args=(mask,),
    )
    success_flags.append(getattr(root_l_res, "success", False))
    l = -root_l_res.x  # [b]

    if return_opt_success:
        return l, u, all(success_flags)
    return l, u


def _hoff_exact_interval(
    y_train: np.ndarray,  # [n] | [b, n]
    mu: np.ndarray,  # [] | [b]
    *,
    alpha: float = 0.1,
    tau2: float = 1.0,
):
    """
    Compute Hoff exact conformal interval bounds.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets with shape ``(n,)`` or ``(b, n)``.
    mu : np.ndarray
        Mean parameter with shape ``()`` or ``(b,)``.
    alpha : float, default=0.1
        Miscoverage level.
    tau2 : float, default=1.0
        Hoff variance hyperparameter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper interval bounds, each with shape ``(b,)``.
    """
    if y_train.ndim == 1:
        y_train = y_train[None]  # [1, n]
        mu = mu[None]  # [1 ,b]

    b, n = y_train.shape  # b: batch size, n: number of training points
    k = int(np.floor(alpha * (n + 1)))  # conformity threshold

    sum_y = np.sum(y_train, axis=-1)  # [b]
    term1_numerator = 2 * ((mu / tau2) + sum_y)  # [b]
    inv_tau_sq_plus_n_plus_1 = (1.0 / tau2) + n + 1  # [b]
    term1 = term1_numerator / inv_tau_sq_plus_n_plus_1  # [b]

    denominator = (
        1 - (2 / inv_tau_sq_plus_n_plus_1) + 1e-10
    )  # Adding a small constant to avoid division by zero [b]

    def g(y: np.ndarray):  # [b, n]
        """
        Apply the Hoff affine transform to candidate values.

        Parameters
        ----------
        y : np.ndarray
            Input array with shape ``(b, n)``.

        Returns
        -------
        np.ndarray
            Transformed values with shape ``(b, n)``.
        """
        return (term1 - y) / denominator  # [b, n]

    g_vals = g(y_train)  # [b, n]
    v = np.concatenate([y_train, g_vals], axis=-1)  # [b, 2n]

    # Acquire the bounds of the prediction region, the kth and (2n − k + 1)th order statistics of v.
    v_sorted = np.sort(v, axis=-1)  # [b, 2n]
    l = v_sorted[:, k - 1]  # [b]
    u = v_sorted[:, 2 * n - k]  # [b]
    return l, u


def hoff_eb_tau_sq(y: np.ndarray) -> Tuple[float, float]:
    """
    Estimate ``tau^2`` with an empirical-Bayes plug-in formula.

    Parameters
    ----------
    y : np.ndarray
        Observations with shape ``(n,)``.

    Returns
    -------
    float
        Estimated ``tau^2``.
    """
    if len(y.shape) == 1:
        n = y.shape[0]
    else:
        raise ValueError("Batched input not supported for hoff_eb_tau_sq.")
    ybar = y.mean(axis=-1)  # [b]

    # Centered sum of squares S_c = sum (y_i - ybar)^2
    Sc = np.sum((y - ybar) ** 2, axis=-1) + 1e-10  # [b]

    # Interior candidate for τ^2
    tau2_candidate = ((n - 1) * (ybar**2) / Sc) - (1.0 / n)  # [b]
    tau2_hat = np.maximum(0.0, tau2_candidate)  # [b]
    return tau2_hat

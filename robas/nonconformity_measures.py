import jax
import jax.numpy as jnp
import jax.scipy.special as jsc
import numpy as np

from utils.dawson import dawson_1f1
from typing import Any, Optional, Tuple, Callable

HS_S2_EPS = 1e-12


@jax.jit
def dta_full(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # [m]
) -> jnp.ndarray:  # [m, n + 1]
    """
    Compute per-test, per-index scores by vectorizing a lower-level measure over all
    test targets and all insertion indices. Distance to average (DTA) nonconformity
    score.

    Given a 1D training target array y_train of length n and a 1D test target array
    y_test of length m, this function evaluates the underlying scalar measure for
    each test value and for every index k in [0, n], returning an array of shape
    (m, n + 1) with entries out[i, k] = measure(y_train, y_test[i], k).

    Parameters
    ----------
    y_train : jnp.ndarray
        1D array of length n containing training targets.
    y_test : jnp.ndarray
        1D array of length m containing test targets.

    Returns
    -------
    jnp.ndarray
        2D array of shape (m, n + 1) where row i corresponds to y_test[i] and
        column k corresponds to the evaluation at insertion/index k (0-based).

    Notes
    -----
    - Implemented with JAX jit and vmap to batch over test points and all indices
    k ∈ {0, ..., n} without Python loops.
    - y_train is treated as fixed when sweeping over test points and indices.
    """
    n = y_train.shape[-1]

    # Note that vectorize here is for conciseness
    return jnp.vectorize(
        jax.vmap(
            jax.vmap(_dta, in_axes=(None, None, -1)),  # output of shape [m, n, n+1]
            in_axes=(None, -1, None),  # output of shape [m, n+1]
        ),
        excluded=(2,),  # we want the indices from 0 to n to be treated as static and not looped over
        signature="(n),(m)->(m,n_p_1)",
    )(
        y_train,
        y_test,
        jnp.arange(n + 1),
    )


def _dta(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # []
    i: int,
) -> jnp.ndarray:  # []
    """
    Compute the DTA (Distance to Average) absolute nonconformity score.

    Given a vector of training responses y_train with length n, a candidate test
    response y_test, and an index i, this returns a scalar JAX array representing
    the absolute deviation-based score used in conformal prediction:

    - If i == n (interpreted as the test point):
        | y_test - mean(y_train) |

    - If 0 <= i < n (a training point):
        | (1 + 1/n) * y_train[i] - (sum(y_train) + y_test) / n |

    The implementation uses jax.lax.cond to preserve JIT- and vmap-friendly control
    flow.

    Parameters
    ----------
    y_train : jnp.ndarray, shape (n,)
        One-dimensional array of training responses.
    y_test : jnp.ndarray, shape ()
        Scalar JAX array with the candidate test response value.
    i : int
        Index specifying which point’s score to compute. Use i == n for the test
        point; otherwise 0 <= i < n should reference a training point.

    Returns
    -------
    jnp.ndarray, shape ()
        Scalar absolute nonconformity score (same dtype as y_train/y_test).
    """
    n = y_train.shape[0]

    def case_i_eq_n(_):
        return jnp.abs(y_test - jnp.mean(y_train))  # nonconformity score in the case where i == n (our test point)

    def case_i_lt_n(_):
        return jnp.abs(
            (1 + 1 / n) * y_train[i] - (jnp.sum(y_train) + y_test) / n
        )  # nonconformity score in the case where i < n

    return jax.lax.cond(i == n, case_i_eq_n, case_i_lt_n, operand=None)


@jax.jit
def dto_full(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # [m]
) -> jnp.ndarray:  # [m, n + 1]
    """
    Compute per-test, per-index scores by vectorizing a lower-level measure over all
    test targets and all insertion indices. Distance to origin (DTO) nonconformity
    score.

    Given a 1D training target array y_train of length n and a 1D test target array
    y_test of length m, this function evaluates the underlying scalar measure for
    each test value and for every index k in [0, n], returning an array of shape
    (m, n + 1) with entries out[i, k] = measure(y_train, y_test[i], k).

    Parameters
    ----------
    y_train : jnp.ndarray
        1D array of length n containing training targets.
    y_test : jnp.ndarray
        1D array of length m containing test targets.

    Returns
    -------
    jnp.ndarray
        2D array of shape (m, n + 1) where row i corresponds to y_test[i] and
        column k corresponds to the evaluation at insertion/index k (0-based).

    Notes
    -----
    - Implemented with JAX jit and vmap to batch over test points and all indices
    k ∈ {0, ..., n} without Python loops.
    - y_train is treated as fixed when sweeping over test points and indices.
    """
    n = y_train.shape[-1]

    return jnp.vectorize(
        jax.vmap(
            jax.vmap(_dto, in_axes=(None, None, -1)),  # output of shape [m, n, n+1]
            in_axes=(None, -1, None),  # output of shape [m, n+1]
        ),
        excluded=(2,),  # we want the indices from 0 to n to be treated as static and not looped over
        signature="(n),(m)->(m,n_p_1)",
    )(
        y_train,
        y_test,
        jnp.arange(n + 1),
    )


def _dto(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # []
    i: int,
) -> jnp.ndarray:  # []
    """
    Compute a distance-to-origin (DTO) nonconformity score for scalar labels.

    Given training labels y_train of length n and a test label y_test, this returns:
    - abs(y_test) when i == n (interpreting i == n as the test point), or
    - abs(y_train[i]) when 0 <= i < n (a calibration/training point).

    The branching is implemented with jax.lax.cond to remain JIT/VMAP-friendly
    without evaluating the non-selected branch.

    Parameters
    ----------
    y_train: jnp.ndarray
        Training labels with shape (n,).
    y_test: jnp.ndarray
        Scalar test label with shape ().
    i:  (int):
        Index of the point whose nonconformity score is computed.
        Must satisfy 0 <= i <= n, where n = y_train.shape[0].
        The value i == n refers to the test point.

    Returns:
    ---------
    jnp.ndarray
        Scalar array with shape () containing the DTO nonconformity score.

    Notes:
        - This implements an absolute-value (L1) nonconformity measure for 1D outputs.
        - If i is not in [0, n], the behavior is invalid (out-of-bounds indexing for y_train).
    """
    n = y_train.shape[0]

    def case_i_eq_n(_):
        return jnp.abs(y_test)  # nonconformity score in the case where i == n (our test point)

    def case_i_lt_n(_):
        return jnp.abs(y_train[i])  # conformity score in the case where i < n

    return jax.lax.cond(i == n, case_i_eq_n, case_i_lt_n, operand=None)


@jax.jit
def hs_full(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # [m]
    s2: float,  # []
) -> jnp.ndarray:  # [m, n + 1]
    """
    Compute per-test, per-index scores by vectorizing a lower-level measure over all
    test targets and all insertion indices. Horseshoe (HS) nonconformity score.

    Given a 1D training target array y_train of length n and a 1D test target array
    y_test of length m, this function evaluates the underlying scalar measure for
    each test value and for every index k in [0, n], returning an array of shape
    (m, n + 1) with entries out[i, k] = measure(y_train, y_test[i], k).


    Parameters
    ----------
    y_train : jnp.ndarray
        1D array of length n containing training targets.
    y_test : jnp.ndarray
        1D array of length m containing test targets.
    s2 : float
        Scalar variance parameter for the horseshoe nonconformity measure.


    Returns
    -------
    jnp.ndarray
        2D array of shape (m, n + 1) where row i corresponds to y_test[i] and
        column k corresponds to the evaluation at insertion/index k (0
    """
    n = y_train.shape[-1]

    return jnp.vectorize(
        jax.vmap(
            jax.vmap(_hs, in_axes=(None, None, -1, None)),  # output of shape [m, n, n+1]
            in_axes=(None, -1, None, None),  # output of shape [m, n+1]
        ),
        excluded=(2,),  # we want the indices from 0 to n to be treated as static and not looped over
        signature="(n),(m),()->(m,n_p_1)",
    )(
        y_train,
        y_test,
        jnp.arange(n + 1),
        s2,
    )


def _hs(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # []
    i: int,
    s2: float,
) -> jnp.ndarray:  # []
    """
    Compute Horseshoe nonconformity score for scalar labels.

    Given training labels y_train of length n and a test label y_test, this returns:
    - abs(y_test) when i == n (interpreting i == n as the test point), or
    - abs(y_train[i]) when 0 <= i < n (a calibration/training point).

    The branching is implemented with jax.lax.cond to remain JIT/VMAP-friendly
    without evaluating the non-selected branch.

    Parameters
    ----------
    y_train (jnp.ndarray): Training labels with shape (n,).
    y_test (jnp.ndarray): Scalar test label with shape ().
    i (int): Index of the point whose nonconformity score is computed.
        Must satisfy 0 <= i <= n, where n = y_train.shape[0].
        The value i == n refers to the test point.

    Returns:
    --------
        jnp.ndarray: Scalar array with shape () containing the Horseshoe nonconformity score.
    """
    n = y_train.shape[0]

    def case_i_eq_n(_):
        n_y_mean_sq = jnp.square(jnp.sum(y_train)) / n
        n_y_var = jnp.sum(jnp.square(y_train)) - n_y_mean_sq
        return n_y_mean_sq, n_y_var

    def case_i_lt_n(_):
        y_i = y_train[i]
        n_y_mean_sq = jnp.square(jnp.sum(y_train) - y_i + y_test) / n
        n_y_var = jnp.sum(jnp.square(y_train)) - jnp.square(y_i) + jnp.square(y_test) - n_y_mean_sq
        return n_y_mean_sq, n_y_var

    n_y_mean_sq, n_y_var = jax.lax.cond(i == n, case_i_eq_n, case_i_lt_n, operand=None)

    # use Kummer's transformation for stability:
    # _1F_1(a, b, -x) = _1F_1(b - a, b, x) * exp(-x)
    res = -n_y_var / (2 * s2) + jnp.log(jsc.hyp1f1(1.5 - 1.0, 1.5, n_y_mean_sq / (2 * s2))) - n_y_mean_sq / (2 * s2)
    return res


@jax.jit
def hs_eb_full(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # [m]
) -> jnp.ndarray:  # [m, n + 1]
    """
    Compute per-test, per-index scores by vectorizing a lower-level measure over all
    test targets and all insertion indices. Horseshoe with empirical Bayes (HS-EB) nonconformity score.

    Given a 1D training target array y_train of length n and a 1D test target array
    y_test of length m, this function evaluates the underlying scalar measure for
    each test value and for every index k in [0, n], returning an array of shape
    (m, n + 1) with entries out[i, k] = measure(y_train, y_test[i], k).

    Parameters
    ----------
    y_train : jnp.ndarray
        1D array of length n containing training targets.
    y_test : jnp.ndarray
        1D array of length m containing test targets.

    Returns
    -------
    jnp.ndarray
        2D array of shape (m, n + 1) where row i corresponds to y_test[i] and
        column k corresponds to the evaluation at insertion/index k (0-based).
    """
    n = y_train.shape[-1]

    return jnp.vectorize(
        jax.vmap(
            jax.vmap(_hs_eb, in_axes=(None, None, -1)),  # output of shape [m, n, n+1]
            in_axes=(None, -1, None),  # output of shape [m, n+1]
        ),
        excluded=(2,),  # we want the indices from 0 to n to be treated as static and not looped over
        signature="(n),(m)->(m,n_p_1)",
    )(
        y_train,
        y_test,
        jnp.arange(n + 1),
    )


def _hs_eb(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # []
    i: int,
) -> jnp.ndarray:  # []
    """
    Compute Horseshoe with empirical Bayes (HS-EB) nonconformity score for scalar labels.
    Given training labels y_train of length n and a test label y_test, this returns:
    - a score based on y_test when i == n (interpreting i == n as the test point), or
    - a score based on y_train[i] when 0 <= i < n (a calibration/training point).

    The branching is implemented with jax.lax.cond to remain JIT/VMAP-friendly
    without evaluating the non-selected branch.

    Parameters
    ----------
    y_train (jnp.ndarray): Training labels with shape (n,).
    y_test (jnp.ndarray): Scalar test label with shape ().
    i (int): Index of the point whose nonconformity score is computed.
        Must satisfy 0 <= i <= n, where n = y_train.shape[0].
        The value i == n refers to the test point.

    Returns:
    --------
        jnp.ndarray: Scalar array with shape () containing the Horseshoe nonconformity score
    """
    n = y_train.shape[0]

    def case_i_eq_n(_):
        y_i = y_test
        y_mean = jnp.mean(y_train)
        y_var = (jnp.sum(jnp.square(y_train)) - n * jnp.square(y_mean)) / n
        return y_i, y_mean, y_var

    def case_i_lt_n(_):
        y_i = y_train[i]
        y_mean = (jnp.sum(y_train) - y_i + y_test) / n
        y_var = (jnp.sum(jnp.square(y_train)) - jnp.square(y_i) + jnp.square(y_test) - n * jnp.square(y_mean)) / n
        return y_i, y_mean, y_var

    y_i, y_mean, y_var = jax.lax.cond(i == n, case_i_eq_n, case_i_lt_n, operand=None)
    t2 = jnp.clip(
        jnp.square(y_mean) - y_var / n,
        min=0.0,
    )
    return jnp.square(y_i - t2 / (t2 + y_var / (n + 1)) * y_mean)


@jax.jit
def hs_full_plugin_s2(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # [m]
) -> jnp.ndarray:  # [m, n + 1]
    """
    Compute per-test, per-index Horseshoe (HS) nonconformity scores.

    The variance term s2 is estimated from the leave-one-out data used for each score:
      - for i == n (test-point score): variance of y_train
      - for i < n  (calibration-point score): variance of the LOO-replaced sample
        (y_train with y_i replaced by y_test)

    Parameters
    ----------
    y_train : jnp.ndarray
        1D array of length n containing training targets.
    y_test : jnp.ndarray
        1D array of length m containing test targets.
    s2_eps : float, default=1e-12
        Numerical floor applied to the estimated variance to avoid division by zero.

    Returns
    -------
    jnp.ndarray
        Array of shape (m, n + 1) with HS scores.
    """
    n = y_train.shape[-1]

    return jnp.vectorize(
        jax.vmap(
            jax.vmap(
                _hs_plugin_s2,
                in_axes=(None, None, -1),
            ),
            in_axes=(None, -1, None),
        ),
        excluded=(2,),  # indices are treated as static inputs, not vectorized over by jnp.vectorize
        signature="(n),(m)->(m,n_p_1)",
    )(
        y_train,
        y_test,
        jnp.arange(n + 1),
    )


def _hs_plugin_s2(
    y_train: jnp.ndarray,  # [n]
    y_test: jnp.ndarray,  # []
    i: int,
) -> jnp.ndarray:  # []
    """
    Compute Horseshoe nonconformity score for a scalar label using an LOO variance estimate.

    For each score, s2 is estimated from the same leave-one-out data used in the score:
      - i == n: use variance of y_train
      - i < n : use variance of y_train with y_i replaced by y_test

    The variance estimator is the unbiased sample variance:
        s2_hat = [sum_j (z_j - z_bar)^2] / (n - 1)
    where n = len(y_train), with a numerical floor `s2_eps`.
    """
    n = y_train.shape[0]

    def case_i_eq_n(_):
        # LOO data for the test-point score is y_train itself
        n_y_mean_sq = jnp.square(jnp.sum(y_train)) / n
        n_y_var = jnp.sum(jnp.square(y_train)) - n_y_mean_sq  # = sum_j (z_j - z_bar)^2
        return n_y_mean_sq, n_y_var

    def case_i_lt_n(_):
        # LOO data for calibration point i is y_train with y_i replaced by y_test
        y_i = y_train[i]
        n_y_mean_sq = jnp.square(jnp.sum(y_train) - y_i + y_test) / n
        n_y_var = (
            jnp.sum(jnp.square(y_train)) - jnp.square(y_i) + jnp.square(y_test) - n_y_mean_sq
        )  # = sum_j (z_j - z_bar)^2 on the LOO-replaced sample
        return n_y_mean_sq, n_y_var

    n_y_mean_sq, n_y_var = jax.lax.cond(i == n, case_i_eq_n, case_i_lt_n, operand=None)

    # Plugin s2 (based on augmented set) terms
    # (r1,...,rn,r_{n+1}), where r_{n+1} is the candidate y_test.
    z_aug = jnp.concatenate([y_train, y_test[None]], axis=0)  # (n+1,)
    z_mean = jnp.mean(z_aug)
    ssq = jnp.sum(jnp.square(z_aug - z_mean))
    denom = jnp.asarray(n if (n + 1) > 1 else 1, dtype=n_y_var.dtype)  # (n+1-1)
    s2_hat = ssq / denom
    s2_eff = s2_hat + jnp.asarray(HS_S2_EPS, dtype=n_y_var.dtype)

    # Using Dawson function for stability instead of Kummer's transformation
    res = -n_y_var / (2 * s2_eff) + jnp.log(dawson_1f1(-n_y_mean_sq / (2 * s2_eff)))


    return res

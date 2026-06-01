import numpy as np


def generate_shift_vector(n, nonzero_pos, nonzero_vals):
    """
    For a vector of length n, generates a shift vector with non-zero values at specified positions.

    Parameters:
    ------------
    n : int
        Length of the shift vector.
    nonzero_pos : List[int]
        Indices where the shift vector should have non-zero values.
    nonzero_vals : List[float]
        Values to place at the specified indices.

    Returns
    -------
    np.ndarray
        The generated shift vector.
    """
    shift_vector = np.zeros(n)
    shift_vector[nonzero_pos] = nonzero_vals
    return shift_vector


def sample_shifted_covariates(X, shift_vector, n_samples):
    """
    Samples indices from X shifted exponentially by shift_vector.

    Parameters:
    ------------
    X : np.ndarray
        Covariate matrix of shape (n, d).
    shift_vector : np.ndarray
        Shift vector of shape (d,).
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    np.ndarray
        Indices of the sampled covariates.
    """
    n, d = X.shape
    scores = X @ shift_vector.reshape(-1, 1)  # (n, 1)
    w = np.exp(scores)

    # Raise errors if there are any NaNs or infs in weights
    if np.any(np.isnan(w)) or np.any(np.isinf(w)):
        raise ValueError("Sampling weights contain NaNs or infinite values.")
    p = w / w.sum()  # (n, 1)
    idx = np.random.choice(n, size=n_samples, replace=True, p=p.flatten())
    return idx

import numpy as np

# Default data parameters
# fmt: off
DATA_ARGS = {
            (1,5): [-5, -3, -2, -1.5, -1, -0.75, -0.25, 0.0, 0.25, 0.75, 1, 1.5, 2, 3, 5],
             
            (1,10): [-3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],

            (1,25): [-2, -1.75, -1.5, -1.25, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
                     -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2],

            (1,50): [-1.75, -1.5, -1.25, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
                           -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                           0.9, 1, 1.25, 1.5, 1.75],

            (0.1, 10): [-1.25, -1, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.25],

            (0.5, 10): [-2.25, -2, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25],

            (2, 10): [-4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],

            (5,10): [-7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, -0.25, 0,
                        0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

            }


DATA_ARGS_TO_EXP_NAME = {
    (0.1, 10): "var0_1_cal10",
    (0.5, 10): "var0_5_cal10",
    (1, 5): "var1_cal5",
    (1, 10): "var1_cal10",
    (1, 25): "var1_cal25",
    (1, 50): "var1_cal50",
    (2, 10): "var2_cal10",
    (5, 10): "var5_cal10"
}
# fmt: on


def get_normal_data(n: int, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic data from a normal distribution.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the normal distribution.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n, 1) containing the generated samples.
    """
    Y = rng.normal(loc=mean, scale=std, size=(n, 1))
    return Y

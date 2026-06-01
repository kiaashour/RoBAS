import numpy as np
import pandas as pd
import torch

from utils.datasets.uci_covariate_shift import generate_shift_vector, sample_shifted_covariates


# Dataset args
SHIFT_POSITIONS = {"airfoil": [0, 4], "concrete": [0, 6], "facebook_1": [3, 33]}
SHIFT_VALS = {"airfoil": [None, [-1, 1],  [-5, 5], [-10, 10]],
              "concrete": [None, [-1, 1], [-5, 5], [-10, 10]],
              "facebook_1": [None, [-1, 1],  [-5, 5],  [-10, 10]]
            }

def get_uci_dataset(dataset_name: str, data_dir: str, test_frac: float = 0.2):
    """ Get UCI dataset.

    Parameters
    ----------
    dataset_name: str
        Name of dataset - "airfoil", "concrete" or "facebook_1".
    data_dir: str
        Directory containing the dataset files
    test_frac: float
        Fraction of dataset to be used for testing

    Returns
    -------
    X: np.ndarray
        Feature matrix
    y: np.ndarray
        Target vector
    n: int
        Number of samples
    shift_vectors: list
        List of shift vectors for each shift value
    """
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Get datasets
    if dataset_name == "airfoil":
        df = pd.read_csv(f"{data_dir}/airfoil_self_noise.dat", sep="\t", header=None)
        response = 5
        y = df[response].values
        X = df.drop(columns=[response]).values
    elif dataset_name == "concrete":
        df = np.loadtxt(open(f"{data_dir}/" + 'Concrete_Data.csv', "rb"), delimiter=",", skiprows=1)
        X = df[:, :-1]
        y = df[:, -1:]
    elif dataset_name == "facebook_1":
        df = pd.read_csv(f"{data_dir}/" + 'Features_Variant_1.csv')        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values       
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    n = len(y)

    # Convert X, y to floats
    X = X.astype(np.float32)
    y = y.astype(np.float32).squeeze()

    # Create shifts
    shift_vals = SHIFT_VALS[dataset_name]
    shift_pos = SHIFT_POSITIONS[dataset_name]
    shift_vectors = [None if val is None else generate_shift_vector(X.shape[1], nonzero_pos=shift_pos, nonzero_vals=val) for val in shift_vals]
    return X, y, n, shift_vectors

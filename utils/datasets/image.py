import pandas as pd
import numpy as np


def get_image_dataset(dataset_name: str, embeddings_path: str):
    """ Function to get image dataset and create the required splits.

    Parameters:
    -----------
    dataset_name : str
        The name of the dataset to load (e.g., "vvolume", "utkfaces").
    embeddings_path : str
        The path to the directory containing the embeddings.

    Returns:
    --------
    dict
        A dictionary containing the train, validation, and test splits.
    """
    # Load the data
    if dataset_name == "vvolume":
        print("Loading vvolume dataset...")
        embeddings_dict = pd.read_pickle(embeddings_path)
        X_train, y_train = embeddings_dict['train']["images"], embeddings_dict['train']["labels"]
        X_test_in, y_test_in = embeddings_dict['val']["images"], embeddings_dict['val']["labels"]
        X_test_out, y_test_out = embeddings_dict['test']["images"], embeddings_dict['test']["labels"]

        X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-12
        y_mean, y_std = y_train.mean(0), y_train.std(0) + 1e-12

        # Standardize datasets
        X_train = (X_train - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        X_test_in = (X_test_in - X_mean) / X_std
        y_test_in = (y_test_in - y_mean) / y_std
        X_test_out = (X_test_out - X_mean) / X_std
        y_test_out = (y_test_out - y_mean) / y_std
        n = X_train.shape[0]
        n_train = min(n, 5000)   # cap for computational efficiency on our machines; we do not observe any qualitative differences

        # Sample training points from X_train, y_train
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx[:n_train]]
        y_train = y_train[idx[:n_train]]    

        # Get remaining data to be used for sampling calibration and test data
        remaining_sets_for_cal_names = ["in", "out"]
        remaining_sets_for_cal = [
                            {"all_test_data": (X_test_in, y_test_in)},
                            {"all_test_data": (X_test_out, y_test_out)}
                            ]
        
    elif dataset_name == "utkfaces":
        print("Loading utkfaces dataset...")
        train_id = np.load(f"{embeddings_path}/train_embeddings.npz")
        X_train, y_train = train_id["emb"], train_id["age"]

        test_id = np.load(f"{embeddings_path}/id_test_embeddings.npz")
        X_test_in, y_test_in = test_id["emb"], test_id["age"]

        ood_test_1 = np.load(f"{embeddings_path}/ood1_embeddings.npz")
        X_test_out_1, y_test_out_1 = ood_test_1["emb"], ood_test_1["age"]

        ood_test_2 = np.load(f"{embeddings_path}/ood2_embeddings.npz")
        X_test_out_2, y_test_out_2 = ood_test_2["emb"], ood_test_2["age"]

        # Print out statistics
        print("Min and max age of ID train set:", y_train.min(), y_train.max())
        print("Min and max age of ID test set:", y_test_in.min(), y_test_in.max())
        print("Min and max age of OOD test set 1:", y_test_out_1.min(), y_test_out_1.max())
        print("Min and max age of OOD test set 2:", y_test_out_2.min(), y_test_out_2.max())           

        # Get dataset statistics and standardize
        X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-12
        y_mean, y_std = y_train.mean(0), y_train.std(0) + 1e-12

        X_train = (X_train - X_mean) / X_std
        y_train = (y_train - y_mean) / y_std
        X_test_in = (X_test_in - X_mean) / X_std
        y_test_in = (y_test_in - y_mean) / y_std
        X_test_out_1 = (X_test_out_1 - X_mean) / X_std
        y_test_out_1 = (y_test_out_1 - y_mean) / y_std
        X_test_out_2 = (X_test_out_2 - X_mean) / X_std
        y_test_out_2 = (y_test_out_2 - y_mean) / y_std
        n = X_train.shape[0]
        n_train = min(n, 5000)

        # Sample n_train training points from X_train, y_train
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx[:n_train]]
        y_train = y_train[idx[:n_train]]

        # Get remaining data to be used for sampling calibration and test data
        remaining_sets_for_cal_names = ["in", "out1",  "out2", ]
        remaining_sets_for_cal = [
                    {"all_test_data": (X_test_in, y_test_in)},
                    {"all_test_data": (X_test_out_1, y_test_out_1)},
                    {"all_test_data": (X_test_out_2, y_test_out_2)},
                    ]          

    else:
        raise ValueError("Invalid dataset choice. dataset_name must be one of: vvolume, utkfaces.")    

    return X_train, y_train, remaining_sets_for_cal, remaining_sets_for_cal_names, n, n_train
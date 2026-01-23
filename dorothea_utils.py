import os

import numpy as np
import scipy.sparse as sp

# download from https://archive.ics.uci.edu/static/public/169/dorothea.zip to data_cache/dorothea


def load_dorothea_data(
    data_dir="./data_cache/dorothea", subset="train", valid=True, feature_dim=100000
):
    """
    Loads Dorothea dataset.

    Args:
        data_dir: Directory containing the .data and .labels files
        subset: 'train', 'valid', or 'test' (test has no labels usually)
        valid: If True, merges train and valid sets (if subset='train')
               (Note: typical usage in this repo merges extensive data,
                but here we might just return the requested subset or merge if requested)

    Returns:
        X (csr_matrix): Shape (n_samples, feature_dim)
        y (array): Shape (n_samples,)
    """

    # If valid=True and subset='train', we load both and concatenate
    if subset == "train" and valid:
        X_train, y_train = load_dorothea_file(data_dir, "train", feature_dim)
        try:
            X_valid, y_valid = load_dorothea_file(data_dir, "valid", feature_dim)
            X = sp.vstack([X_train, X_valid])
            y = np.concatenate([y_train, y_valid])
        except FileNotFoundError:
            print("Warning: Validation set not found, returning only train.")
            X, y = X_train, y_train

        return X, y
    else:
        return load_dorothea_file(data_dir, subset, feature_dim)


def load_dorothea_file(data_dir, subset, feature_dim):
    data_path = os.path.join(data_dir, f"dorothea_{subset}.data")
    labels_path = os.path.join(data_dir, f"dorothea_{subset}.labels")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Read Data (Sparse format: indices of non-zero features)
    row_ind = []
    col_ind = []
    data_val = []

    with open(data_path, "r") as f:
        for r, line in enumerate(f):
            indices = [int(x) for x in line.strip().split()]
            for c in indices:
                row_ind.append(r)
                col_ind.append(c - 1)
                data_val.append(1)

    # Determine num_samples
    num_samples = r + 1

    X = sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(num_samples, feature_dim))

    # Read Labels
    if os.path.exists(labels_path):
        y = np.loadtxt(labels_path, dtype=int)
    else:
        # Test set might not have labels
        y = np.zeros(num_samples)

    return X, y

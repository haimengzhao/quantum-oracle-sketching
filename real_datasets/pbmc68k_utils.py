import numpy as np
import scipy.sparse as sp
import scvelo as scv
from sklearn.preprocessing import LabelEncoder


def load_pbmc68k_data(min_samples=1, normalize=True, binary=True):
    """
    Load PBMC68k dataset using scvelo.

    Args:
        min_samples: Minimum number of samples a gene must appear in (non-zero).
                     Analogous to min_df in text vectorization.
        normalize: If True, apply log1p normalization
        binary: If True, keep only the two largest cell type classes

    Returns:
        X (csr_matrix): Shape (n_cells, n_genes)
        y (array): Shape (n_cells,) - encoded cell type labels
        label_names (list): List of cell type names
    """
    # Load dataset
    adata = scv.datasets.pbmc68k(file_path="./data_cache/pbmc68k.h5ad")

    # Get expression matrix (already sparse CSR)
    X = adata.X

    # Get labels
    labels = adata.obs["celltype"].values

    # Filter to binary classification if requested
    if binary:
        # Get class counts
        unique, counts = np.unique(labels, return_counts=True)
        # Get top 2 classes by count
        top2_idx = np.argsort(counts)[-2:]
        top2_classes = unique[top2_idx]

        # Filter samples
        mask = np.isin(labels, top2_classes)
        X = X[mask]
        labels = labels[mask]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    label_names = list(le.classes_)

    # Normalize if requested
    if normalize:
        if sp.issparse(X):
            X = X.toarray()
        X = np.log1p(X)
        X = sp.csr_matrix(X)

    # Filter genes by min_samples (number of cells with non-zero expression)
    if min_samples > 1:
        X, gene_indices = filter_genes_by_frequency(X, min_samples)

    return X, y, label_names


def filter_genes_by_frequency(X, min_samples):
    """
    Filter genes that appear in at least min_samples samples.
    Analogous to min_df filtering in text vectorization.

    Args:
        X: sparse matrix (n_samples, n_genes)
        min_samples: minimum number of samples a gene must be non-zero in

    Returns:
        X_filtered: matrix with only genes meeting the threshold
        gene_indices: indices of kept genes
    """
    # Count non-zero samples per gene (column)
    if sp.issparse(X):
        gene_counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        gene_counts = np.sum(X != 0, axis=0)
        if hasattr(gene_counts, "ravel"):
            gene_counts = gene_counts.ravel()

    # Keep genes that appear in at least min_samples
    gene_indices = np.where(gene_counts >= min_samples)[0]
    X_filtered = X[:, gene_indices]

    return X_filtered, gene_indices


def get_min_samples_sweep():
    """
    Returns a list of min_samples thresholds to sweep over,
    similar to min_df sweep in text datasets.
    Logarithmically spaced from 1 to a significant fraction of samples.
    """
    # For ~36k samples (binary), sweep from 1 to ~20k
    min_samples = np.unique(np.logspace(np.log10(1), np.log10(20000), 30).astype(int))
    return list(min_samples)


import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# DNA nucleotides
NUCLEOTIDES = "ACGT"

def load_splice_data(min_samples=1, binary=True):
    """
    Load UCI Splice Junction dataset.
    
    Args:
        min_samples: Minimum number of samples a feature must appear in
        binary: If True, keep only EI and IE classes (exclude N)
    
    Returns:
        X (csr_matrix): k-mer frequency features
        y (array): encoded labels
        label_names (list): class names
    """
    from ucimlrepo import fetch_ucirepo
    
    # Fetch dataset
    dataset = fetch_ucirepo(id=69)
    X_df = dataset.data.features
    y_df = dataset.data.targets
    
    # Get sequences (combine all 60 positions)
    sequences = []
    for idx, row in X_df.iterrows():
        seq = ''.join(str(v).upper() for v in row.values)
        sequences.append(seq)
    
    # Get labels
    labels = y_df['class'].values
    
    # Filter to binary classification if requested
    if binary:
        # Keep only EI and IE (exclude N/Neither)
        mask = (labels == 'EI') | (labels == 'IE')
        sequences = [s for s, m in zip(sequences, mask) if m]
        labels = labels[mask]
    
    # Compute k-mer features
    X, kmer_to_idx = compute_kmer_features(sequences, k=6)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    label_names = list(le.classes_)
    
    # Filter k-mers by min_samples
    if min_samples > 1:
        X, _ = filter_features_by_frequency(X, min_samples)
    
    return X, y, label_names


def compute_kmer_features(sequences, k=6):
    """
    Compute k-mer frequency features for DNA sequences.
    """
    kmer_to_idx = {}
    idx = 0
    
    # First pass: collect all k-mers
    for seq in sequences:
        seq_clean = ''.join(c for c in seq if c in NUCLEOTIDES)
        for i in range(len(seq_clean) - k + 1):
            kmer = seq_clean[i:i+k]
            if kmer not in kmer_to_idx:
                kmer_to_idx[kmer] = idx
                idx += 1
    
    n_features = len(kmer_to_idx)
    n_samples = len(sequences)
    
    # Build sparse matrix
    row_ind = []
    col_ind = []
    data_val = []
    
    for r, seq in enumerate(sequences):
        seq_clean = ''.join(c for c in seq if c in NUCLEOTIDES)
        kmer_counts = Counter()
        for i in range(len(seq_clean) - k + 1):
            kmer = seq_clean[i:i+k]
            if kmer in kmer_to_idx:
                kmer_counts[kmer] += 1
        
        # Normalize by sequence length
        total = sum(kmer_counts.values())
        if total > 0:
            for kmer, count in kmer_counts.items():
                row_ind.append(r)
                col_ind.append(kmer_to_idx[kmer])
                data_val.append(count / total)
    
    X = sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(n_samples, n_features))
    return X, kmer_to_idx


def filter_features_by_frequency(X, min_samples):
    """
    Filter features that appear in at least min_samples samples.
    """
    if sp.issparse(X):
        feature_counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        feature_counts = np.sum(X != 0, axis=0)
        if hasattr(feature_counts, 'ravel'):
            feature_counts = feature_counts.ravel()
    
    feature_indices = np.where(feature_counts >= min_samples)[0]
    X_filtered = X[:, feature_indices]
    
    return X_filtered, feature_indices


def get_min_samples_sweep():
    """
    Returns a list of min_samples thresholds to sweep over.
    """
    # For ~1500 binary samples, sweep from 1 to ~1000
    min_samples = np.unique(np.logspace(np.log10(1), np.log10(1000), 55).astype(int))
    return list(min_samples)

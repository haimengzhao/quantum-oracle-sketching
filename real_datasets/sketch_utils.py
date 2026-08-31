"""Shared interface for the classical oblivious sketches used in the sweeps.

An oblivious sketch compresses the feature dimension with a data-independent
linear map P: the sweep scripts train on X_sketch = X P and, for PCA, lift the
top direction back to the original feature space as v = P v_sketch. Each
method is registered in OBLIVIOUS_SKETCHES as a pair of functions:

    sketch(X, n_features, seed) -> (X_sketch, info)
        Apply the projection at the requested sketch dimension. Must return
        (X, None) when n_features >= X.shape[1], so the full-dimension
        endpoint of every sweep is exactly the original data.

    lift(v_sketch, info) -> v
        Apply P to a vector from sketch space, returning a vector in the
        original feature space. info is the value returned by sketch;
        info is None means the identity. The callers normalize the lifted
        vector, so P does not need orthonormal columns.

To add a new oblivious sketch, implement such a pair and add an
ObliviousSketch entry to OBLIVIOUS_SKETCHES. Every *_svm.py / *_pca.py sweep
script then accepts the new mode key through its --mode flag, and the JSON
provenance written by sketch_metadata() is derived from the registry.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp


DEFAULT_SKETCH_SEED_SAMPLE_SEED = 42
SPARSE_JL_TRANSFORM = "balanced_signed_bucket"


def sample_sketch_seeds(n_seeds=5, sample_seed=DEFAULT_SKETCH_SEED_SAMPLE_SEED):
    """Sample sketch seeds reproducibly from a fixed RNG seed."""
    rng = np.random.default_rng(sample_seed)
    return [int(seed) for seed in rng.integers(0, 2**32 - 1, size=n_seeds)]


def mean_and_sem(values):
    """Mean and SEM over all scalar values in a nested list/array."""
    vals = np.array(values, dtype=float).reshape(-1)
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(vals.size))


def _balanced_bucket_assignment(n_features, n_buckets, rng):
    """Assign features to n_buckets random buckets of near-equal size.

    Bucket sizes differ by at most one. Consumes exactly one permutation
    from rng, so callers can draw further variates deterministically.
    """
    perm = rng.permutation(n_features)
    bucket_of_feature = np.empty(n_features, dtype=np.int64)
    bucket_of_feature[perm] = np.arange(n_features) * n_buckets // n_features
    bucket_sizes = np.bincount(bucket_of_feature, minlength=n_buckets).astype(
        np.float64
    )
    return bucket_of_feature, bucket_sizes


def _balanced_one_sparse_sketch(X, n_buckets, seed, signed):
    """One-sparse projection onto balanced random buckets, optionally signed.

    Each original feature is assigned to one bucket; the bucketed feature is
    the 1/sqrt(bucket size)-normalized (and, if signed, sign-flipped) sum of
    its features, i.e. X_sketch = X P with one nonzero per row of P.
    """
    X = sp.csr_matrix(X)
    n_features = X.shape[1]
    n_buckets = min(int(n_buckets), n_features)

    if n_buckets >= n_features:
        return X, None

    rng = np.random.default_rng(seed)
    bucket_of_feature, bucket_sizes = _balanced_bucket_assignment(
        n_features, n_buckets, rng
    )

    weights = 1.0 / np.sqrt(bucket_sizes[bucket_of_feature])
    if signed:
        signs = rng.choice([-1.0, 1.0], size=n_features)
        weights = signs * weights

    projection = sp.csr_matrix(
        (weights, (np.arange(n_features), bucket_of_feature)),
        shape=(n_features, n_buckets),
    )

    X_sketch = (X @ projection).tocsr()
    X_sketch.eliminate_zeros()
    if signed:
        return X_sketch, (bucket_of_feature, bucket_sizes, signs)
    return X_sketch, (bucket_of_feature, bucket_sizes)


def bucket_features(X, n_buckets, seed=42):
    """Merge D original features into n_buckets random balanced buckets.

    The bucketed feature is the normalized sum of features in that bucket:

        z_b = (1 / sqrt(|B_b|)) * sum_{j in B_b} x_j.

    Equivalently, X_bucket = X P, where P has one nonzero entry per row.
    With the 1/sqrt(|B_b|) normalization, the columns of P are orthonormal
    (P.T @ P = I). For PCA, lift_bucket_vector(v, info) returns P v.

    If n_buckets >= D, no approximation is made and the original matrix is
    returned. This makes the full-dimension endpoint exactly the original data.
    """
    return _balanced_one_sparse_sketch(X, n_buckets, seed, signed=False)


def sparse_jl_features(X, n_buckets, seed=42):
    """Balanced signed one-sparse JL projection.

    This is the signed version of bucket_features: each original feature is
    assigned to one balanced bucket and multiplied by an independent random
    sign before the bucket sum. The 1/sqrt(bucket size) normalization keeps
    the projection columns orthonormal. If k >= D, the original matrix is
    returned exactly, so the full-dimension endpoint is unchanged.
    """
    return _balanced_one_sparse_sketch(X, n_buckets, seed, signed=True)


def lift_bucket_vector(v_bucket, bucket_info):
    """Lift a vector from bucket space back to the original feature space."""
    if bucket_info is None:
        return np.asarray(v_bucket)

    bucket_of_feature, bucket_sizes = bucket_info
    return np.asarray(v_bucket)[bucket_of_feature] / np.sqrt(
        bucket_sizes[bucket_of_feature]
    )


def lift_sparse_jl_vector(v_jl, sparse_jl_info):
    """Lift a balanced signed sparse JL vector to original feature space."""
    if sparse_jl_info is None:
        return np.asarray(v_jl)

    bucket_of_feature, bucket_sizes, signs = sparse_jl_info
    return (
        signs
        * np.asarray(v_jl)[bucket_of_feature]
        / np.sqrt(bucket_sizes[bucket_of_feature])
    )


@dataclass(frozen=True)
class ObliviousSketch:
    """One classical oblivious sketching method, as used by the sweep scripts.

    name: short human-readable name, used in dataset annotations and logs.
    label: label of the classical streaming curve in plots.
    description: one-line description shown in the --mode CLI help.
    transform: stable identifier recorded in JSONs for cache validation.
    sketch, lift: the function pair described in the module docstring.
    """

    name: str
    label: str
    description: str
    transform: str
    sketch: Callable
    lift: Callable


OBLIVIOUS_SKETCHES = {
    "bucket": ObliviousSketch(
        name="bucket",
        label="Classical feature hashing",
        description="random balanced feature buckets (feature hashing)",
        transform="balanced_bucket",
        sketch=bucket_features,
        lift=lift_bucket_vector,
    ),
    "jl": ObliviousSketch(
        name="sparse JL",
        label="Classical sparse JL",
        description="balanced signed sparse JL projection",
        transform=SPARSE_JL_TRANSFORM,
        sketch=sparse_jl_features,
        lift=lift_sparse_jl_vector,
    ),
}


def get_sketch(mode):
    """Look up an oblivious sketch by its mode key."""
    try:
        return OBLIVIOUS_SKETCHES[mode]
    except KeyError:
        raise ValueError(
            f"unknown sketch mode {mode!r}; "
            f"available: {', '.join(OBLIVIOUS_SKETCHES)}"
        ) from None


def sketch_features(X, n_features, mode, seed=42):
    """Apply the oblivious sketch registered under mode to X."""
    return get_sketch(mode).sketch(X, n_features, seed=seed)


def lift_sketched_vector(v_sketch, sketch_info, mode):
    """Lift a sketched vector back to the original feature space."""
    return get_sketch(mode).lift(v_sketch, sketch_info)


def streaming_label(mode, default="Classical streaming"):
    """Plot label of the classical streaming curve for the given mode."""
    if mode in OBLIVIOUS_SKETCHES:
        return OBLIVIOUS_SKETCHES[mode].label
    return default


def sketch_metadata(mode, seeds, n_features_list):
    """Provenance block recorded in the JSON written by a sketch sweep."""
    sketch = get_sketch(mode)
    return {
        "truncation_mode": mode,
        f"{mode}_seeds": [int(seed) for seed in seeds],
        f"{mode}_n_features": [int(k) for k in n_features_list],
        f"{mode}_seed_sample_seed": DEFAULT_SKETCH_SEED_SAMPLE_SEED,
        f"{mode}_transform": sketch.transform,
    }


def lifted_vector_norm(v_lifted):
    """Norm used when evaluating a lifted PCA direction.

    All current lifts live directly in the original feature space, where the
    variance-recovery formula needs a unit direction.
    """
    return np.linalg.norm(v_lifted)


def validate_truncation_data(data, mode, path="loaded JSON"):
    """Reject stale or mismatched cached numerics before plotting."""
    feature_mode = mode in OBLIVIOUS_SKETCHES
    has_feature_sweep = "raw_data_by_n_features" in data
    if has_feature_sweep != feature_mode:
        raise ValueError(f"{path} does not match --mode {mode}")

    if not feature_mode:
        return

    if data.get("truncation_mode") != mode:
        raise ValueError(f"{path} has truncation_mode={data.get('truncation_mode')!r}")
    expected = get_sketch(mode).transform
    recorded = data.get(f"{mode}_transform")
    if recorded != expected:
        raise ValueError(
            f"{path} records {mode}_transform={recorded!r}, expected {expected!r}"
        )


def matrix_nnz(X):
    """Number of stored/nonzero entries for space accounting."""
    if sp.issparse(X):
        return int(X.getnnz())
    return int(np.size(X))


def svds_input(X):
    """Return a float matrix accepted by scipy.sparse.linalg.svds."""
    if sp.issparse(X):
        return X.asfptype()
    X = np.asarray(X)
    if np.issubdtype(X.dtype, np.floating):
        return X
    return X.astype(np.float32)


def max_sparsity(X):
    """Maximum row/column sparsity used by the existing space formulas."""
    if not sp.issparse(X):
        return max(X.shape) if np.size(X) else 0

    X = sp.csr_matrix(X)
    row_sparsity = int(X.getnnz(axis=1).max()) if X.shape[0] else 0
    col_sparsity = int(X.getnnz(axis=0).max()) if X.shape[1] else 0
    return max(row_sparsity, col_sparsity)

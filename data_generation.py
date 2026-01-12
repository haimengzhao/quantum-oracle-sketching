import jax.numpy as jnp
from jax import random

from utils import complex_dtype, int_dtype, real_dtype


class matrix_data:
    """
    Class to simulate access to a real matrix via sampling its non-zero elements or rows uniformly randomly.

    Attributes:
        matrix: array of shape (dim1, dim2), the matrix to sample from.
        shape: tuple, the shape of the matrix.
        num_generated_samples: int, the number of samples generated so far.

    Methods:
        get_matrix_element_data(key, num_samples, return_values=True):
            sample num_samples non-zero matrix elements.
        get_row_data(key, num_samples):
            sample num_samples rows of the matrix.
    """

    def __init__(self, matrix):
        self.matrix = matrix.astype(real_dtype)
        self.shape = matrix.shape
        self.num_generated_samples = jnp.int64(0)
        self._nz_rows, self._nz_cols = jnp.nonzero(self.matrix)
        self._nz_rows = self._nz_rows.astype(int_dtype)
        self._nz_cols = self._nz_cols.astype(int_dtype)
        self._nnz = self._nz_rows.shape[0]

    def get_matrix_element_data(self, key, num_samples, return_values=True):
        """
        Get uniform random samples of non-zero matrix elements.

        Args:
            key: jax.random.PRNGKey, the random key.
            num_samples: int, number of samples to generate.
            return_values: bool, whether to return the values of the sampled elements.

        Returns:
            sampled_rows: array of shape (num_samples,), the row indices of the sampled elements.
            sampled_cols: array of shape (num_samples,), the column indices of the sampled elements.
            sampled_values (if return_values=True): array of shape (num_samples,), the values of the sampled elements.
        """

        self.num_generated_samples += num_samples

        sampled_indices = random.randint(
            key, shape=(num_samples,), minval=0, maxval=self._nnz, dtype=int_dtype
        )
        sampled_rows = self._nz_rows[sampled_indices]
        sampled_cols = self._nz_cols[sampled_indices]

        if return_values:
            sampled_values = self.matrix[sampled_rows, sampled_cols]
            return sampled_rows, sampled_cols, sampled_values

        return sampled_rows, sampled_cols

    def get_row_data(self, key, num_samples):
        """
        Get uniform random samples of rows of the matrix.

        Args:
            key: jax.random.PRNGKey, the random key.
            num_samples: int, number of samples to generate.

        Returns:
            sampled_rows: array of shape (num_samples,), the row indices of the sampled rows.
            sampled_row_vectors: array of shape (num_samples, self.shape[1]), the sampled row vectors.
        """
        num_rows = self.shape[0]
        sampled_rows = random.choice(
            key,
            jnp.arange(num_rows, dtype=int_dtype),
            shape=(num_samples,),
            replace=True,
        )
        sampled_row_vectors = self.matrix[sampled_rows]

        self.num_generated_samples += num_samples

        return sampled_rows, sampled_row_vectors


class vector_data:
    """
    Class to simulate access to a real vector via sampling its components uniformly randomly.

    Attributes:
        vector: array of shape (dim,), the vector to sample from.
        length: int, the length of the vector.
        num_generated_samples: int, the number of samples generated so far.

    Methods:
        get_data(key, num_samples):
            sample num_samples components of the vector.
    """

    def __init__(self, vector):
        self.vector = vector.astype(real_dtype)
        self.length = vector.shape[0]
        self.num_generated_samples = jnp.int64(0)

    def get_data(self, key, num_samples):
        """
        Get uniform random samples of components of the vector.

        Args:
            key: jax.random.PRNGKey, the random key.
            num_samples: int, number of samples to generate.

        Returns:
            sampled_indices: array of shape (num_samples,), the indices of the sampled components.
            sampled_values: array of shape (num_samples,), the values of the sampled components.
        """

        sampled_indices = random.choice(
            key,
            jnp.arange(self.length, dtype=int_dtype),
            shape=(num_samples,),
            replace=True,
        ).astype(int_dtype)
        sampled_values = self.vector[sampled_indices]

        self.num_generated_samples += num_samples

        return sampled_indices, sampled_values


class boolean_data:
    """
    Class to simulate access to a boolean function via sampling uniformly random queries.

    Attributes:
        truth_table: array of shape (N,), the truth table of the boolean function.
        length: int, the size of the support of the boolean function.
        num_generated_samples: int, the number of samples generated so far.

    Methods:
        get_data(key, num_samples):
            sample num_samples queries to the boolean function.
    """

    def __init__(self, truth_table):
        self.truth_table = truth_table.astype(int_dtype)
        self.length = truth_table.shape[0]
        self.num_generated_samples = jnp.int64(0)

    def get_data(self, key, num_samples):
        """
        Get uniform random samples of queries to the boolean function.

        Args:
            key: jax.random.PRNGKey, the random key.
            num_samples: int, number of samples to generate.

        Returns:
            sampled_indices: array of shape (num_samples,), the indices of the sampled queries.
            sampled_values: array of shape (num_samples,), the values of the sampled queries.
        """
        sampled_indices = random.choice(
            key,
            jnp.arange(self.length, dtype=int_dtype),
            shape=(num_samples,),
            replace=True,
        ).astype(int_dtype)
        sampled_values = self.truth_table[sampled_indices]

        self.num_generated_samples += num_samples

        return sampled_indices, sampled_values

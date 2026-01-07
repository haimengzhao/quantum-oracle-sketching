import jax.numpy as jnp
from jax import random


class matrix_data:
    """
    Class to simulate access to a matrix via sampling its non-zero elements or rows uniformly randomly.

    Attributes:
        matrix: array of shape (dim1, dim2), the matrix to sample from.
        shape: tuple, the shape of the matrix.
        num_generated_samples: int, the number of samples generated so far.

    Methods:
        get_matrix_element_data(key, num_samples):
            sample num_samples non-zero matrix elements.
        get_row_data(key, num_samples):
            sample num_samples rows of the matrix.
    """

    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape
        self.num_generated_samples = 0

    def get_matrix_element_data(self, key, num_samples):
        # uniform random samples of non-zero matrix elements
        rows, cols = jnp.nonzero(self.matrix)
        indices = jnp.arange(len(rows))
        sampled_indices = random.choice(
            key, indices, shape=(num_samples,), replace=True
        )
        sampled_rows = rows[sampled_indices]
        sampled_cols = cols[sampled_indices]
        sampled_values = self.matrix[sampled_rows, sampled_cols]

        self.num_generated_samples += num_samples

        return sampled_rows, sampled_cols, sampled_values

    def get_row_data(self, key, num_samples):
        # uniform random samples of rows
        num_rows = self.shape[0]
        sampled_rows = random.choice(
            key, jnp.arange(num_rows), shape=(num_samples,), replace=True
        )
        sampled_values = self.matrix[sampled_rows]

        self.num_generated_samples += num_samples

        return sampled_rows, sampled_values


class vector_data:
    """
    Class to simulate access to a vector via sampling its components uniformly randomly.

    Attributes:
        vector: array of shape (dim,), the vector to sample from.
        length: int, the length of the vector.
        num_generated_samples: int, the number of samples generated so far.

    Methods:
        get_data(key, num_samples):
            sample num_samples components of the vector.
    """

    def __init__(self, vector):
        self.vector = vector
        self.length = vector.shape[0]
        self.num_generated_samples = 0

    def get_data(self, key, num_samples):
        # uniform random components of the vector
        sampled_indices = random.choice(
            key, jnp.arange(self.length), shape=(num_samples,), replace=True
        )
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
        self.truth_table = truth_table
        self.length = truth_table.shape[0]
        self.num_generated_samples = 0

    def get_data(self, key, num_samples):
        # uniform random queries
        sampled_indices = random.choice(
            key, jnp.arange(self.length), shape=(num_samples,), replace=True
        )
        sampled_values = self.truth_table[sampled_indices]

        self.num_generated_samples += num_samples

        return sampled_indices, sampled_values

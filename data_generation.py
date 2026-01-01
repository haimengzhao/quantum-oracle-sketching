import jax.numpy as jnp
from jax import random


class matrix_data:
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape

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
        return sampled_rows, sampled_cols, sampled_values

    def get_random_row_data(self, key, num_samples):
        # uniform random samples of rows
        num_rows = self.shape[0]
        sampled_rows = random.choice(
            key, jnp.arange(num_rows), shape=(num_samples,), replace=True
        )
        sampled_values = self.matrix[sampled_rows]
        return sampled_rows, sampled_values


class vector_data:
    def __init__(self, vector):
        self.vector = vector
        self.length = vector.shape[0]

    def get_data(self, key, num_samples):
        # uniform random components of the vector
        sampled_indices = random.choice(
            key, jnp.arange(self.length), shape=(num_samples,), replace=True
        )
        sampled_values = self.vector[sampled_indices]
        return sampled_indices, sampled_values

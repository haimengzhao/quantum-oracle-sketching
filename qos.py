import jax
import jax.numpy as jnp
from jax import random

import qsvt
import utils
from utils import complex_dtype, int_dtype, real_dtype

"""
This module implements quantum state and oracle sketching methods
with expected unitary. That means we instantiate each oracle query
with the expected unitary. This is significantly more efficient than
the active sampling method in qos_sampling.py. According to the 
mixing lemma, the error of the simulation here upper bounds the 
error of the real-world scenario where we have random channels. 
In other words, this is a pessimistic simulation of the real-world
performance. The real performance would be better than what we simulate here.
"""


def q_state_sketch_flat(vector, num_samples):
    """
    Construct the quantum state sketch of a flat vector.

    Use 1 ancilla qubit.

    Assume that the vector is flat: all components are plus or minus one.

    Args:
        vector: array of shape (dim,), the input vector
        num_samples: int, number of samples to use in the sketch

    Returns:
        quantum state sketch as an array of shape (dim,)
    """
    dim = vector.shape[0]
    prob = jnp.ones_like(vector, dtype=real_dtype) / dim
    t = jnp.pi * dim

    # expected single gate
    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / num_samples * (1 - vector) / 2))

    # concatenate all gates
    log_diag = num_samples * log_diag
    diag = jnp.exp(log_diag)

    # apply the gate to the all plus state
    state = diag / jnp.sqrt(dim)

    return state


"""
Tests
"""


def _test_q_state_sketch_flat(key):
    print("-" * 10)
    print("Testing quantum state sketching for flat vectors...")
    # random \pm 1 vector
    N = 1000
    num_samples = int(1e7)

    print(f"Testing vector with dimension N = {N:.2e}")

    x = random.randint(key, (N,), minval=0, maxval=2, dtype=int_dtype) * 2 - 1

    print(f"Sample size: {num_samples:.2e}")

    state = q_state_sketch_flat(x, num_samples)

    # test norm
    print(f"State norm squared: 1 + {jnp.linalg.norm(state) ** 2 - 1:.3e}")

    # test reconstruction
    recon_x = state
    error = jnp.linalg.norm(recon_x - x / jnp.sqrt(N))
    print(f"State reconstruction error in Euclidean distance: {error:.3e}")
    assert jnp.allclose(error, 0, atol=1e-2)


if __name__ == "__main__":
    key = random.PRNGKey(0)

    _test_q_state_sketch_flat(key)

    print("-" * 10)
    print("All tests passed.")

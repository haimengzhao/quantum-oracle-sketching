import jax.numpy as jnp
from jax import random

from data_generation import matrix_data, vector_data


def q_state_sketch_flat(data, N):
    """
    Construct quantum state sketch from vector data samples.
    Assume that the vector is flat: all components are plus or minus one.

    Args:
        data: tuple of (sampled_indices, sampled_values)
        N: int, dimension of the target quantum state

    Returns:
        quantum state sketch as an array of shape (N,)
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * N
    phase = jnp.zeros(N, dtype=jnp.complex64)
    phase = phase.at[sampled_indices].add((1 - sampled_values) / 2)
    phase = phase * t / num_samples
    state = jnp.exp(1j * phase) / jnp.sqrt(N)

    return state


# test q_state_sketch_flat

if __name__ == "__main__":
    print("Testing q_state_sketch_flat...")

    # random \pm 1 vector
    N = 100
    key = random.PRNGKey(0)
    x = random.randint(key, (N,), minval=0, maxval=2) * 2 - 1

    vec_data = vector_data(x)

    key, subkey = random.split(key)
    num_samples = int(1e5)
    data = vec_data.get_data(subkey, num_samples=num_samples)

    state = q_state_sketch_flat(data, N)
    # test norm
    print("State norm squared:", jnp.linalg.norm(state) ** 2)
    assert jnp.isclose(jnp.linalg.norm(state) ** 2, 1.0, atol=1e-2)

    # test reconstruction
    recon_x = jnp.real(state) * jnp.sqrt(N)
    assert jnp.allclose(recon_x, x, atol=1e-1)
    print("Reconstruction error:", jnp.linalg.norm(recon_x - x))

    print("All tests passed.")

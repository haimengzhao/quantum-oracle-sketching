import jax.numpy as jnp
from jax import random

from data_generation import boolean_data, matrix_data, vector_data


def q_state_sketch_flat_unitary(data, N):
    """
    Construct quantum state sketch preparation unitary from vector data samples.
    Assume that the vector is flat: all components are plus or minus one.

    Args:
        data: tuple of (sampled_indices, sampled_values in {+1,-1})
        N: int, dimension of the target quantum state

    Returns:
        diagonal of the state preparation unitary as an array of shape (N,)
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * N
    phase = jnp.zeros(N, dtype=jnp.complex64)
    phase = phase.at[sampled_indices].add((1 - sampled_values) / 2)
    phase = phase * t / num_samples
    diag = jnp.exp(1j * phase)

    return diag


def q_state_sketch_flat(data, N):
    """
    Construct quantum state sketch from vector data samples.
    Assume that the vector is flat: all components are plus or minus one.

    Args:
        data: tuple of (sampled_indices, sampled_values in {+1,-1})
        N: int, dimension of the target quantum state

    Returns:
        quantum state sketch as an array of shape (N,)
    """
    diag = q_state_sketch_flat_unitary(data, N)
    # shortcut for diagonal unitary applied to |+> state
    state = diag / jnp.sqrt(N)
    return state


def q_oracle_sketch_boolean(data, N):
    """
    Construct quantum oracle sketch from boolean data samples.

    Args:
        data: tuple of (sampled_indices, sampled_values in {0,1})
        N: int, support size of the target boolean function
    Returns:
        diagonal of the phase oracle sketch as an array of shape (N,)
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * N
    phase = jnp.zeros(N, dtype=jnp.complex64)
    phase = phase.at[sampled_indices].add(sampled_values)
    phase = phase * t / num_samples
    diag = jnp.exp(1j * phase)

    return diag


def _test_q_state_sketch_flat(key):
    # random \pm 1 vector
    N = 100
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


def _test_q_oracle_sketch_boolean(key):
    # random boolean function
    N = 100
    f = random.randint(key, (N,), minval=0, maxval=2)

    bool_data = boolean_data(f)

    key, subkey = random.split(key)
    num_samples = int(1e5)
    data = bool_data.get_data(subkey, num_samples=num_samples)

    diag = q_oracle_sketch_boolean(data, N)

    # test unitarity
    print("Oracle unitarity check:", jnp.allclose(jnp.abs(diag), 1.0))
    assert jnp.allclose(jnp.abs(diag), 1.0)

    # test reconstruction
    recon_f = (1 - jnp.real(diag)) / 2
    assert jnp.allclose(recon_f, f, atol=1e-1)
    print("Reconstruction error:", jnp.max(jnp.abs(recon_f - f)))


if __name__ == "__main__":
    key = random.PRNGKey(0)

    print("Testing q_state_sketch_flat...")
    _test_q_state_sketch_flat(key)

    print("Testing q_oracle_sketch_boolean...")
    _test_q_oracle_sketch_boolean(key)

    print("All tests passed.")

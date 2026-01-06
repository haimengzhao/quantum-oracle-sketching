import jax
import jax.numpy as jnp
from jax import random

import qsvt
import utils

complex_dtype = jnp.complex128
real_dtype = jnp.float64


def amplitude_amplification(
    unnormalized_state, degree, target_norm=0.9, threshold=None
):
    """Perform amplitude amplification on an unnormalized quantum state.

    Implicitly increasing num_ancilla by 1.

    Note that we can safely ingnore the garbage blocks from the block encoding
    that gives us the unnormalized state, since they never enter the relevant
    subspace for amplitude amplification.
    We will populate the garbage blocks with the canonical Halmos dilation,
    which is automatically a Hermitian unitary block encoding.

    Args:
        unnormalized_state: array of shape (dim,), the unnormalized quantum state vector.
        degree: odd int, the degree of the QSVT polynomial for amplitude amplification.
        target_norm: float, the target norm for amplitude amplification scaling.
        threshold: float or None, the threshold for amplitude amplification.
            If None, defaults to the norm of the unnormalized state divided by 2.

    Returns:
        transformed_state: array of shape (dim,), the state vector after amplitude amplification.
    """
    dim = unnormalized_state.shape[0]
    norm = jnp.linalg.norm(unnormalized_state)
    if norm == 0:
        raise ValueError("The input state has zero norm and cannot be amplified.")

    if threshold is None:
        threshold = norm / 2

    # Get QSVT angles for the sign function
    angle_set, scale = qsvt.get_qsvt_angles_sign(
        degree=degree, threshold=threshold, rescale=target_norm
    )

    print(f"Amplitude amplification scaling factor: {scale}")
    print(
        f"Angle set length: {len(angle_set)}, degree: {degree}, threshold: {threshold}"
    )

    # Construct the Halmos dilation block encoding
    # the block encoded matrix is
    # A = [
    #   [0, unnormalized_state],
    #   [unnormalized_state^†, 0]
    # ]
    hermitian_embed = jnp.block(
        [
            [jnp.zeros((dim, dim)), unnormalized_state[:, None]],
            [unnormalized_state[None, :].conj(), jnp.zeros((1, 1))],
        ]
    )
    halmos_block_encoding = utils.halmos_dilation(hermitian_embed)

    # Apply QSVT to the Halmos block encoding
    transformed_block_encoding = qsvt.apply_qsvt(
        halmos_block_encoding, num_ancilla=1, angle_set=angle_set
    )

    # Extract the block encoded matrix
    transformed_matrix = utils.get_block_encoded(
        transformed_block_encoding, num_ancilla=1
    )

    # Use LCU to extract the real part
    transformed_matrix = (transformed_matrix + transformed_matrix.conj().T) / 2

    # Extract the transformed state
    transformed_state = transformed_matrix[:-1, -1]

    return transformed_state


"""
Tests
"""


def _test_amplitude_amplification(key):
    dim = 1000
    initial_norm = 0.2
    target_norm = 0.99
    degree = 51

    # random vector
    key, subkey = random.split(key)
    v = random.normal(subkey, (dim,))
    norm = jnp.linalg.norm(v)
    if norm > 1:
        v = v / norm * initial_norm
    norm = jnp.linalg.norm(v)
    print(f"Initial norm of the vector: {norm:.6f}")

    state_aa = amplitude_amplification(v, degree=degree, target_norm=target_norm)
    error_qsvt = jnp.linalg.norm(v / jnp.linalg.norm(v) - state_aa / target_norm)
    final_norm = jnp.linalg.norm(state_aa)

    print(f"Final norm after amplitude amplification: {final_norm:.6f}")
    print(f"Target norm: {target_norm:.6f}")
    print(f"QSVT error: {error_qsvt:.6e}")

    assert jnp.isclose(error_qsvt, 0, atol=1e-2)


if __name__ == "__main__":
    key = random.PRNGKey(42)

    print("-" * 10)
    print("Testing amplitude amplification...")
    _test_amplitude_amplification(key)

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

    Note that we ingnore the garbage blocks from the block encoding
    that gives us the unnormalized state, since they never enter the relevant
    subspace for amplitude amplification in quantum oracle sketching + QSVT.
    This is because quantum oracle sketching and QSVT have a fixed circuit
    structure and only data are random.
    This means that the garbage blocks are always equal to the canonical Halmos dilation
    up to a fixed unitary transformation that does not depend on data, which does
    not affect the relevant block.
    Hence, we will populate the garbage blocks with the canonical Halmos dilation,
    which is automatically a Hermitian unitary block encoding.

    Args:
        unnormalized_state: array of shape (dim,) or (degree, dim),
            the unnormalized quantum state vector, or its imperfect instantiations.
        degree: odd int, the degree of the QSVT polynomial for amplitude amplification.
        target_norm: float, the target norm for amplitude amplification scaling.
        threshold: float or None, the threshold for amplitude amplification.
            If None, defaults to the norm of the unnormalized state divided by 2.
            Minimum value is 1e-2 for numerical stability.

    Returns:
        transformed_state: array of shape (dim,), the state vector after amplitude amplification.
    """
    imperfect = len(unnormalized_state.shape) == 2

    norm = jnp.min(jnp.sqrt(jnp.sum(jnp.abs(unnormalized_state) ** 2, axis=-1)))
    if norm == 0:
        raise ValueError("The input state has zero norm and cannot be amplified.")

    if threshold is None:
        threshold = max(norm * 0.5, 1e-2)

    # Get QSVT angles for the sign function
    angle_set, scale = qsvt.get_qsvt_angles_sign(
        degree=degree, threshold=float(threshold), rescale=target_norm
    )

    def _embed(state):
        # Construct the Halmos dilation block encoding
        # the block encoded matrix is
        # A = [
        #   [0, unnormalized_state],
        #   [unnormalized_state^†, 0]
        # ]
        dim = state.shape[-1]
        hermitian_embed = jnp.block(
            [
                [jnp.zeros((dim, dim)), state[:, None]],
                [state[None, :].conj(), jnp.zeros((1, 1))],
            ]
        )
        halmos_block_encoding = utils.halmos_dilation(hermitian_embed)
        return halmos_block_encoding

    # Get the Halmos block encoding
    halmos_block_encoding = (
        jax.vmap(_embed)(unnormalized_state)
        if imperfect
        else _embed(unnormalized_state)
    )

    # Apply QSVT to the Halmos block encoding
    if imperfect:
        transformed_block_encoding = qsvt.apply_qsvt_imperfect(
            halmos_block_encoding, num_ancilla=1, angle_set=angle_set
        )
    else:
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


def _test_amplitude_amplification_imperfect(key, noise_level=0.01):
    dim = 100
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

    # create imperfect instantiations
    key, subkey = random.split(key)
    noise = noise_level * random.normal(subkey, (degree, dim)) * jnp.linalg.norm(v)
    v_imperfect = jnp.tile(v, (degree, 1)) + noise

    state_aa = amplitude_amplification(
        v_imperfect, degree=degree, target_norm=target_norm
    )
    error_qsvt = jnp.linalg.norm(v / jnp.linalg.norm(v) - state_aa / target_norm)
    final_norm = jnp.linalg.norm(state_aa)

    print(f"Final norm after amplitude amplification: {final_norm:.6f}")
    print(f"Target norm: {target_norm:.6f}")
    print(f"QSVT error: {error_qsvt:.6e}")
    print(f"QSVT relative error: {error_qsvt / noise_level:.6e}")

    assert jnp.isclose(error_qsvt / noise_level, 0, atol=1e1)


if __name__ == "__main__":
    key = random.PRNGKey(42)

    print("-" * 10)
    print("Testing amplitude amplification...")
    _test_amplitude_amplification(key)

    print("-" * 10)
    print("Testing amplitude amplification with imperfect instantiations...")
    _test_amplitude_amplification_imperfect(key)

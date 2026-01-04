import jax
import jax.numpy as jnp
from jax import random

import qsvt
import utils
from data_generation import boolean_data, matrix_data, vector_data

complex_dtype = jnp.complex128
real_dtype = jnp.float64


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
    phase = jnp.zeros(N, dtype=real_dtype)
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
    Construct quantum oracle sketch from boolean function data samples.

    Args:
        data: tuple of (sampled_indices, sampled_values in {0,1})
        N: int, support size of the target boolean function
    Returns:
        diagonal of the phase oracle sketch as an array of shape (N,)
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * N
    phase = jnp.zeros(N, dtype=real_dtype)
    phase = phase.at[sampled_indices].add(sampled_values)
    phase = phase * t / num_samples
    diag = jnp.exp(1j * phase)

    return diag


def q_oracle_sketch_matrix_element(data, N):
    """
    Construct a nearly Hermitian block encoding of
    the sparse element oracle |i>|j> -> A_{ij} |i>|j> from matrix data samples.

    Assume that the matrix elements are in [-1,1].

    Strict Hermiticity condition is not guaranteed
    because the construction uses two separate batches of random data
    for the two unitaries in the LCU construction.

    Args:
        data: tuple of (sampled_row_indices, sampled_col_indices, sampled_values)
        N: int, dimension of the target matrix
    Returns:
        a (nearly) Hermitian block encoding of the sparse element oracle as an array of shape (2N, 2N)
    """

    # divide data into 2 batches
    # one for cU, one for cU^\dagger in the LCU construction

    cut = data[0].shape[0] // 2
    batched_data = [
        (data[0][:cut], data[1][:cut], data[2][:cut]),
        (data[0][cut:], data[1][cut:], data[2][cut:]),
    ]

    # estimate the number of non-zero elements in the samples

    nnz = jnp.unique(data[0] * N + data[1]).shape[0]

    t = nnz

    phase_seq = jnp.zeros((2, N**2), dtype=real_dtype)

    # 1. construct the unitary U: |i>|j> -> exp( i * B_{ij} ) |i>|j>
    # where B_{ij} = arcsin( A_{ij} )

    for i, batch in enumerate(batched_data):
        sampled_row_indices = batch[0]
        sampled_col_indices = batch[1]
        sampled_values = batch[2]
        num_samples = sampled_row_indices.shape[0]

        phase = jnp.zeros((N, N), dtype=real_dtype)
        phase = phase.at[sampled_row_indices, sampled_col_indices].add(
            jnp.arcsin(sampled_values)
        )
        phase = phase * t / num_samples

        phase = phase.reshape(N**2)

        phase_seq = phase_seq.at[i].set(phase)

    # 2. construct the block encoding of |i>|j> -> sin(B_{ij}) |i>|j> using LCU
    # we use the circuit S_a X_a H_a (c_a U_2^\dagger) X_a (c_a U_1) H_a T_a
    # where a is the ancilla and S = [[-i, 0], [0, 1]], T = [[1, 0], [0, -i]]
    # analytic calculation shows that the resulting block encoding is
    # [[ (U_1-U_2^\dagger)/(2i) , (U_1 + U_2^\dagger)/2  ],
    #  [ (U_1 + U_2^\dagger)/2, -(U_1 - U_2^\dagger)/(2i) ]]
    # which is a Hermitian block encoding of sin(B) with one ancilla qubit

    U1_diag = jnp.exp(1j * phase_seq[0])
    U2_dagger_diag = jnp.exp(-1j * phase_seq[1])

    sin = jnp.diag((U1_diag - U2_dagger_diag) / (2j))
    cos = jnp.diag((U1_diag + U2_dagger_diag) / 2)

    block_encoding = jnp.block(
        [
            [sin, cos],
            [cos, -sin],
        ]
    )

    return block_encoding


"""
Tests
"""


def _test_q_state_sketch_flat(key):
    # random \pm 1 vector
    N = 1000
    x = random.randint(key, (N,), minval=0, maxval=2) * 2 - 1

    vec_data = vector_data(x)

    key, subkey = random.split(key)
    num_samples = int(1e6)
    data = vec_data.get_data(subkey, num_samples=num_samples)

    state = q_state_sketch_flat(data, N)

    # test norm
    print("State norm squared:", jnp.linalg.norm(state) ** 2)
    assert jnp.isclose(jnp.linalg.norm(state) ** 2, 1.0, atol=1e-2)

    # test reconstruction
    recon_x = state
    print(
        "State reconstruction error in trace distance:",
        jnp.sqrt(1 - jnp.abs(jnp.vdot(recon_x, x / jnp.sqrt(N))) ** 2),
    )
    assert jnp.allclose(
        jnp.sqrt(1 - jnp.abs(jnp.vdot(recon_x, x / jnp.sqrt(N))) ** 2), 0, atol=1e-1
    )


def _test_q_oracle_sketch_boolean(key):
    # random boolean function
    N = 1000
    f = random.randint(key, (N,), minval=0, maxval=2)

    bool_data = boolean_data(f)

    key, subkey = random.split(key)
    num_samples = int(1e6)
    data = bool_data.get_data(subkey, num_samples=num_samples)

    diag = q_oracle_sketch_boolean(data, N)

    # test unitarity
    print("Oracle unitarity check:", jnp.allclose(jnp.abs(diag), 1.0))
    assert jnp.allclose(jnp.abs(diag), 1.0)

    # test reconstruction
    recon_f = (1 - jnp.real(diag)) / 2
    print("Boolean function reconstruction error:", jnp.max(jnp.abs(recon_f - f)))
    assert jnp.allclose(jnp.max(jnp.abs(recon_f - f)), 0, atol=1e-1)


def _test_q_oracle_sketch_matrix_element(key):
    # random matrix
    N = 10

    key, subkey = random.split(key)
    A = random.normal(subkey, (N, N))
    A = A / jnp.linalg.norm(A)

    data_gen = matrix_data(A)
    num_samples = int(1e7)
    key, subkey = random.split(key)
    data = data_gen.get_matrix_element_data(subkey, num_samples=num_samples)

    oracle = q_oracle_sketch_matrix_element(data, N)

    print(
        "Oracle block encoding unitarity check:",
        jnp.isclose(
            jnp.linalg.norm(oracle @ oracle.conj().T - jnp.eye(2 * N**2)), 0, atol=1e-2
        ),
    )
    assert jnp.isclose(
        jnp.linalg.norm(oracle @ oracle.conj().T - jnp.eye(2 * N**2)), 0, atol=1e-2
    )

    print(
        "Oracle block encoding Hermiticity check:",
        jnp.linalg.norm(oracle - oracle.conj().T),
    )

    assert jnp.isclose(jnp.linalg.norm(oracle - oracle.conj().T), 0, atol=1e-2)

    A_reconst = jnp.diag(utils.get_block_encoded(oracle, num_ancilla=1)).reshape(N, N)

    error = jnp.linalg.norm(A - A_reconst, ord=2)
    print("Matrix reconstruction error in operator norm:", error)

    assert jnp.isclose(error, 0, atol=1e-1)


if __name__ == "__main__":
    key = random.PRNGKey(0)

    print("-" * 10)
    print("Testing quantum state sketching for flat vectors...")
    _test_q_state_sketch_flat(key)

    print("-" * 10)
    print("Testing quantum oracle sketching for boolean functions...")
    _test_q_oracle_sketch_boolean(key)

    print("-" * 10)
    print("Testing quantum oracle sketching for matrix sparse element oracle...")
    _test_q_oracle_sketch_matrix_element(key)

    print("-" * 10)
    print("All tests passed.")

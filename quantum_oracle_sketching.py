import time

import jax
import jax.numpy as jnp
from jax import random

import qsvt
import utils
from data_generation import boolean_data, matrix_data, vector_data

complex_dtype = jnp.complex128
real_dtype = jnp.float64


def q_state_sketch_flat_unitary(data, dim):
    """
    Construct quantum state sketch preparation unitary from vector data samples.

    Assume that the vector is flat: all components are plus or minus one.

    Args:
        data: tuple of (sampled_indices, sampled_values in {+1,-1})
        dim: int, dimension of the target quantum state

    Returns:
        diagonal of the state preparation unitary as an array of shape (dim,)
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * dim
    phase = jnp.zeros(dim, dtype=real_dtype)
    phase = phase.at[sampled_indices].add((1 - sampled_values) / 2)
    phase = phase * t / num_samples
    diag = jnp.exp(1j * phase)

    return diag


def q_state_sketch_flat(data, dim):
    """
    Construct quantum state sketch from vector data samples.

    Assume that the vector is flat: all components are plus or minus one.

    Args:
        data: tuple of (sampled_indices, sampled_values in {+1,-1})
        dim: int, dimension of the target quantum state

    Returns:
        quantum state sketch as an array of shape (dim,)
    """
    diag = q_state_sketch_flat_unitary(data, dim)
    # shortcut for diagonal unitary applied to |+> state
    state = diag / jnp.sqrt(dim)
    return state


def q_state_sketch(data, dim, norm, key, degree=10):
    """
    Construct quantum state sketch from vector data samples.

    Args:
        data: tuple of (sampled_indices, sampled_values)
        dim: int, dimension of the target quantum state, must be a power of 2
        norm: float, l2 norm of the target vector
        key: jax.random.PRNGKey, random key for generating random signs
        degree: even int, degree of the polynomial approximation for arcsin(x),
            default 10, data size should be a multiple of 2 * degree
    Returns:
        quantum state sketch as an array of shape (dim,)
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    sampled_values = sampled_values / norm  # normalize the vector

    # random sign O_h
    key, subkey = random.split(key)
    random_signs = random.choice(subkey, jnp.array([1.0, -1.0]), shape=(dim,))

    # 1. Construct the target basis states u = 0, ..., 2^n - 1
    # Shape: (dim,)
    u_vec = jnp.arange(dim, dtype=sampled_indices.dtype)

    # 2. Compute the bitwise interaction j_l \cdot u (mod 2)
    # We use broadcasting to create the interaction matrix of shape (num_samples, dim).
    # sampled_indices[:, None] is (num_samples, 1)
    # u_vec[None, :] is (1, dim)
    bitwise_intersection = sampled_indices[:, None] & u_vec[None, :]

    # jax.lax.population_count counts the number of set bits (Hamming weight).
    # The inner product over GF(2) is the parity of the intersection.
    parity = jax.lax.population_count(bitwise_intersection) % 2

    # 3. Convert parity to sign factors (-1)^(j_l \cdot u)
    # Shape: (num_samples, dim)
    # 0 -> +1, 1 -> -1
    interaction_signs = 1.0 - 2.0 * parity.astype(jnp.float32)

    # 4. Contribution of each sample
    # b_l * (-1)^(j_l \cdot u) * t
    # sampled_values[:, None] broadcasts to (num_samples, 1) to multiply against (num_samples, dim)
    t = dim / (norm * 5)
    contribution = (
        (random_signs[sampled_indices] * sampled_values)[:, None]
        * interaction_signs
        * t
    )

    # 5. Split into two batches of data for LCU
    # we use the circuit S_a X_a H_a (c_a U_2^\dagger) X_a (c_a U_1) H_a T_a
    # where a is the ancilla and S = [[-i, 0], [0, 1]], T = [[1, 0], [0, -i]]
    # analytic calculation shows that the resulting block encoding is
    # [[ (U_1 - U_2^\dagger)/(2i),  (U_1 + U_2^\dagger)/2     ],
    #  [ (U_1 + U_2^\dagger)/2   , -(U_1 - U_2^\dagger)/(2i) ]]
    # which is a Hermitian block encoding of sin(B) with one ancilla qubit

    # each batch contribution is further split into degree sub-batches
    # to approximate arcsin(x) by a degree-d polynomial using QSVT

    contribution = contribution.reshape(
        2, degree, num_samples // (2 * degree), dim
    )  # shape (2, degree, num_samples/(2*degree), dim)
    contribution = jnp.average(contribution, axis=2)  # shape (2, degree, dim)

    U1_diag = jnp.exp(1j * contribution[0])  # shape (degree, dim)
    U2_dagger_diag = jnp.exp(-1j * contribution[1])  # shape (degree, dim)

    vectorized_diag = jax.vmap(jnp.diag)

    sin = vectorized_diag(U1_diag - U2_dagger_diag) / (2j)  # shape (degree, dim, dim)
    cos = vectorized_diag(U1_diag + U2_dagger_diag) / 2  # shape (degree, dim, dim)

    block_encoding = jnp.concatenate(
        [
            jnp.concatenate([sin, cos], axis=1),
            jnp.concatenate([cos, -sin], axis=1),
        ],
        axis=2,
    )  # shape (degree, 2*dim, 2*dim)

    # 6. Apply QSVT to approximate arcsin(x)
    def func(x):
        return jnp.arcsin(x) / jnp.arcsin(1.0)

    angle_set = qsvt.get_qsvt_angles(
        func=func,
        degree=degree,
        rescale=1.0,
        cheb_domain=(-jnp.sin(1), jnp.sin(1)),
    )

    block_encoding = qsvt.apply_qsvt_imperfect(
        block_encoding[:-1], num_ancilla=1, angle_set=angle_set
    )  # shape (2*dim, 2*dim)

    # 6. Prepare the state
    # apply the block encoding to the all plus state
    # note that the qsvt applies arcsin as the real part
    state = jnp.sum(
        (block_encoding[:dim, :dim] + block_encoding[:dim, :dim].conj().T) / 2, axis=1
    ) / jnp.sqrt(dim)

    # apply inverse randomized Hadamard transform to restore the original vector
    hadamard = utils.unnormalized_hadamard_transform(int(jnp.round(jnp.log2(dim))))
    state = hadamard @ state / jnp.sqrt(dim)
    state = random_signs * state

    return state


def q_oracle_sketch_boolean(data, dim):
    """
    Construct quantum oracle sketch from boolean function data samples.

    Args:
        data: tuple of (sampled_indices, sampled_values in {0,1})
        dim: int, support size of the target boolean function
    Returns:
        diagonal of the phase oracle sketch as an array of shape (dim,)
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * dim
    phase = jnp.zeros(dim, dtype=real_dtype)
    phase = phase.at[sampled_indices].add(sampled_values)
    phase = phase * t / num_samples
    diag = jnp.exp(1j * phase)

    return diag


def q_oracle_sketch_matrix_element(data, dim, nnz):
    """
    Construct a nearly Hermitian block encoding of
    the sparse element oracle |i>|j> -> A_{ij} |i>|j> from matrix data samples.

    Assume that the matrix elements are in [-1,1].

    Strict Hermiticity condition is not guaranteed
    because the construction uses two separate batches of random data
    for the two unitaries in the LCU construction.

    Args:
        data: tuple of (sampled_row_indices, sampled_col_indices, sampled_values)
        dim: int, dimension of the target matrix
        nnz: int, number of non-zero elements in the matrix.
    Returns:
        a (nearly) Hermitian block encoding of the sparse element oracle as an array of shape (2, 2, dim**2)
        where the first two dimensions are the ancilla qubit space and the last dimension is the diagonal.
    """

    # divide data into 2 batches
    # one for cU, one for cU^\dagger in the LCU construction

    cut = data[0].shape[0] // 2
    batched_data = [
        (data[0][:cut], data[1][:cut], data[2][:cut]),
        (data[0][cut:], data[1][cut:], data[2][cut:]),
    ]

    t = nnz

    phase_seq = jnp.zeros((2, dim**2), dtype=real_dtype)

    # 1. construct the unitary U: |i>|j> -> exp( i * B_{ij} ) |i>|j>
    # where B_{ij} = arcsin( A_{ij} )

    for i, batch in enumerate(batched_data):
        sampled_row_indices = batch[0]
        sampled_col_indices = batch[1]
        sampled_values = batch[2]
        num_samples = sampled_row_indices.shape[0]

        phase = jnp.zeros((dim, dim), dtype=real_dtype)
        phase = phase.at[sampled_row_indices, sampled_col_indices].add(
            jnp.arcsin(sampled_values)
        )
        phase = phase * t / num_samples

        phase = phase.reshape(dim**2)

        phase_seq = phase_seq.at[i].set(phase)

    # 2. construct the block encoding of |i>|j> -> sin(B_{ij}) |i>|j> using LCU
    # we use the circuit S_a X_a H_a (c_a U_2^\dagger) X_a (c_a U_1) H_a T_a
    # where a is the ancilla and S = [[-i, 0], [0, 1]], T = [[1, 0], [0, -i]]
    # analytic calculation shows that the resulting block encoding is
    # [[ (U_1 - U_2^\dagger)/(2i),  (U_1 + U_2^\dagger)/2     ],
    #  [ (U_1 + U_2^\dagger)/2   , -(U_1 - U_2^\dagger)/(2i) ]]
    # which is a Hermitian block encoding of sin(B) with one ancilla qubit

    U1_diag = jnp.exp(1j * phase_seq[0])
    U2_dagger_diag = jnp.exp(-1j * phase_seq[1])

    sin = (U1_diag - U2_dagger_diag) / (2j)
    cos = (U1_diag + U2_dagger_diag) / 2

    block_encoding = jnp.stack([sin, cos, cos, -sin], axis=0).reshape(2, 2, dim**2)

    return block_encoding


"""
Tests
"""


def _test_q_state_sketch_flat(key):
    # random \pm 1 vector
    N = 1000

    print(f"Testing vector with dimension N = {N:.2e}")

    x = random.randint(key, (N,), minval=0, maxval=2) * 2 - 1

    vec_data = vector_data(x)

    key, subkey = random.split(key)
    num_samples = int(1e6)
    data = vec_data.get_data(subkey, num_samples=num_samples)

    print(f"Sample size: {num_samples:.2e}")

    state = q_state_sketch_flat(data, N)

    # test norm
    print(f"State norm squared: 1 + {jnp.linalg.norm(state) ** 2 - 1:.3e}")
    assert jnp.isclose(jnp.linalg.norm(state) ** 2, 1.0, atol=1e-2)

    # test reconstruction
    recon_x = state
    print(
        f"State reconstruction error in trace distance: {jnp.sqrt(1 - jnp.abs(jnp.vdot(recon_x, x / jnp.sqrt(N))) ** 2):.3e}"
    )
    assert jnp.allclose(
        jnp.sqrt(1 - jnp.abs(jnp.vdot(recon_x, x / jnp.sqrt(N))) ** 2), 0, atol=1e-1
    )


def _test_q_state_sketch(key):
    N = 16
    num_samples = 10000000

    print(f"Testing general vector with dimension N = {N}")

    key, subkey = random.split(key)
    v = random.normal(subkey, (N,))
    v = v / jnp.linalg.norm(v)

    data_gen = vector_data(v)
    key, subkey = random.split(key)
    data = data_gen.get_data(subkey, num_samples)

    print(f"Sample size: {num_samples:.2e}")

    qstate = q_state_sketch(data, N, jnp.linalg.norm(v), key, degree=10)

    error = jnp.linalg.norm(v - qstate / jnp.linalg.norm(qstate))
    print(f"State reconstruction error in l2 norm: {error:.3e}")
    assert jnp.isclose(error, 0, atol=1e-1)


def _test_q_oracle_sketch_boolean(key):
    # random boolean function
    N = 1000

    print(f"Testing boolean function with dimension N = {N:.2e}")

    f = random.randint(key, (N,), minval=0, maxval=2)

    bool_data = boolean_data(f)

    key, subkey = random.split(key)
    num_samples = int(1e6)
    data = bool_data.get_data(subkey, num_samples=num_samples)

    print(f"Sample size: {num_samples:.2e}")

    diag = q_oracle_sketch_boolean(data, N)

    # test unitarity
    print(f"Oracle unitarity check: {jnp.allclose(jnp.abs(diag), 1.0)}")
    assert jnp.allclose(jnp.abs(diag), 1.0)

    # test reconstruction
    recon_f = (1 - jnp.real(diag)) / 2
    print(f"Boolean function reconstruction error: {jnp.max(jnp.abs(recon_f - f)):.3e}")
    assert jnp.allclose(jnp.max(jnp.abs(recon_f - f)), 0, atol=1e-1)


def _test_q_oracle_sketch_matrix_element(key):
    # random matrix
    N = 10000

    print(f"Testing matrix with dimension N = {N:.2e}")
    print(f"Note that the oracle has dimension N^2 x N^2 = {N**2:.2e} x {N**2:.2e}")

    key, subkey = random.split(key)
    A = random.normal(subkey, (N, N))
    A = A / jnp.linalg.norm(A)

    data_gen = matrix_data(A)
    num_samples = int(1e8)
    key, subkey = random.split(key)
    data = data_gen.get_matrix_element_data(subkey, num_samples=num_samples)

    print(f"Sample size: {num_samples:.2e}")

    nnz = jnp.count_nonzero(A)

    start_time = time.time()

    oracle_diag = q_oracle_sketch_matrix_element(data, N, nnz=nnz)

    end_time = time.time()

    print(f"Oracle construction time: {end_time - start_time:.3e} seconds")

    if N <= 10:
        # only reconstruct the full dense matrix when N is small
        oracle = jnp.einsum("ijk,kl->iljk", oracle_diag, jnp.eye(N**2)).reshape(
            2 * N**2, 2 * N**2
        )

        print(
            "Oracle block encoding unitarity check:",
            jnp.isclose(
                jnp.linalg.norm(oracle @ oracle.conj().T - jnp.eye(2 * N**2)),
                0,
                atol=1e-2,
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
    else:
        print("Skipping unitarity and Hermiticity checks for large N.")

    A_reconst = oracle_diag[0, 0]

    error = jnp.max(jnp.abs(A_reconst - A.reshape(N**2)))
    print(f"Matrix reconstruction error in operator norm: {error:.3e}")

    assert jnp.isclose(error, 0, atol=1e-1)


if __name__ == "__main__":
    key = random.PRNGKey(0)

    print("-" * 10)
    print("Testing quantum state sketching for flat vectors...")
    _test_q_state_sketch_flat(key)

    print("-" * 10)
    print("Testing quantum state sketching for general vectors...")
    _test_q_state_sketch(key)

    print("-" * 10)
    print("Testing quantum oracle sketching for boolean functions...")
    _test_q_oracle_sketch_boolean(key)

    print("-" * 10)
    print("Testing quantum oracle sketching for matrix sparse element oracle...")
    _test_q_oracle_sketch_matrix_element(key)

    print("-" * 10)
    print("All tests passed.")

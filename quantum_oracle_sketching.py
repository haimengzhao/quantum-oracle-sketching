import time

import jax
import jax.numpy as jnp
from jax import random

import primitives
import qsvt
import utils
from data_generation import boolean_data, matrix_data, vector_data

complex_dtype = jnp.complex128
real_dtype = jnp.float64

vectorized_diag = jax.vmap(jnp.diag)


def q_state_sketch_flat_unitary(data, dim):
    """
    Construct quantum state sketch preparation unitary from vector data samples.

    Use 1 ancilla qubit.

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

    Use 1 ancilla qubit.

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


def q_state_sketch(data, dim, norm, key, degree=4):
    """
    Construct quantum state sketch from vector data samples.

    Use 2 ancilla qubit.
    One for the first LCU and QSVT, one for the second LCU to extract real part.

    Args:
        data: tuple of (sampled_indices, sampled_values)
        dim: int, dimension of the target quantum state, must be a power of 2
        norm: float, l2 norm of the target vector
        key: jax.random.PRNGKey, random key for generating random signs
        degree: even int, degree of the polynomial approximation for arcsin(x),
            default 4, data size should be a multiple of degree
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

    # 2. Compute the bitwise interaction j \cdot u (mod 2)
    # Instead of computing for each sample, which is costly for large num_samples,
    # we compute for all possible j and u.
    j_vec = jnp.arange(dim, dtype=sampled_indices.dtype)  # shape (dim,)
    bitwise_and = jnp.bitwise_and(j_vec[:, None], u_vec[None, :])  # shape (dim, dim)

    # jax.lax.population_count counts the number of set bits (Hamming weight).
    # The inner product over GF(2) is the parity of the and.
    bit_inner_product = jax.lax.population_count(bitwise_and) % 2  # shape (dim, dim)

    # 3. Convert inner products to sign factors (-1)^(j \cdot u)
    # 0 -> +1, 1 -> -1
    inner_prod_signs = 1 - 2 * bit_inner_product  # shape (dim, dim)

    # 4. Aggregate samples according to the sampled indices
    # each sampled index has a contribution to all basis states u
    # this pre-aggregation helps get rid of dependency on num_samples in later steps
    aggregated_sampled_values = jnp.zeros((degree, dim), dtype=real_dtype)

    # first divide samples into degree groups
    sampled_indices = sampled_indices.reshape(
        degree, -1
    )  # shape (degree, num_samples / degree)
    sampled_values = sampled_values.reshape(
        degree, -1
    )  # shape (degree, num_samples / degree)

    # aggregate
    # no more dependency on num_samples from here
    aggregated_sampled_values = aggregated_sampled_values.at[
        jnp.arange(degree)[:, None], sampled_indices
    ].add(sampled_values)

    # average
    aggregated_sampled_values = aggregated_sampled_values / (num_samples / degree)

    # apply random signs
    aggregated_sampled_values = aggregated_sampled_values * random_signs[None, :]
    # shape (degree, dim), corresponding to (degree, j)

    # 4. Contribution of each sample
    # sum_l (b_l (-1)^h(j_l)) * (-1)^(j_l \cdot u) * t / M
    # = sum_l (b_l / M * (-1)^h(j_l)) sum_j 1[j_l = j] * (-1)^(j \cdot u) * t
    # = sum_j ( sum_l b_l / M * (-1)^h(j_l) 1[j_l = j] ) * (-1)^(j \cdot u) * t
    # = sum_j ( aggregated_sampled_values[j] ) * (inner_prod_signs[j, u]) *  t
    t = dim / norm / 3
    contribution = (
        aggregated_sampled_values @ inner_prod_signs
    ) * t  # shape (degree, dim)

    # 5. LCU converts phase to sine
    # we use the circuit S_a X_a H_a (c_a^1 U^\dagger) (c_a^0 U) X_a H_a T_a
    # where a is the ancilla and S = [[-i, 0], [0, 1]], T = [[1, 0], [0, -i]]
    # analytic calculation shows that the resulting block encoding is
    # [[ (U - U^\dagger)/(2i),  (U + U^\dagger)/2     ],
    #  [ (U + U^\dagger)/2   , -(U - U^\dagger)/(2i) ]]
    # which is a Hermitian block encoding of sin(B) with one ancilla qubit
    # Note again that this is compatible with quantum oracle sketching,
    # since we can implement the c^0U and c^1U† unitaries simultaneously
    # using the same samples.

    def func(x):
        return jnp.arcsin(x) / jnp.arcsin(1)

    angle_set = qsvt.get_qsvt_angles(
        func=func,
        degree=degree,
        rescale=1.0,
        cheb_domain=(-jnp.sin(1), jnp.sin(1)),
        ensure_bounded=False,
    )

    U_diag = jnp.exp(1j * contribution)  # shape (degree, dim)

    sin = vectorized_diag(U_diag - U_diag.conj()) / 2j  # shape (degree, dim, dim)
    cos = vectorized_diag(U_diag + U_diag.conj()) / 2  # shape (degree, dim, dim)

    block_encoding = jnp.concatenate(
        [
            jnp.concatenate([sin, cos], axis=1),
            jnp.concatenate([cos, -sin], axis=1),
        ],
        axis=2,
    )  # shape (degree, 2*dim, 2*dim)

    # 6. Apply QSVT to approximate arcsin(x)
    block_encoding = qsvt.apply_qsvt_imperfect(
        block_encoding[:-1], num_ancilla=1, angle_set=angle_set
    )  # shape (2*dim, 2*dim)

    # 7. Prepare the state
    # apply the block encoding to the all plus state
    # note that the qsvt applies arcsin as the real part
    # we need a second LCU to get the real part only
    # Note again that this is compatible with quantum oracle sketching,
    # since we can implement the c^0U and c^1U† unitaries simultaneously
    # using the same samples.

    state = jnp.sum(
        (block_encoding[:dim, :dim] + block_encoding[:dim, :dim].conj().T) / 2,
        axis=1,
    ) / jnp.sqrt(dim)

    # 8. Apply inverse randomized Hadamard transform to restore the original vector
    hadamard = utils.unnormalized_hadamard_transform(int(jnp.round(jnp.log2(dim))))
    state = hadamard @ state / jnp.sqrt(dim)
    state = random_signs * state

    return state


def q_oracle_sketch_boolean(data, dim):
    """
    Construct quantum oracle sketch from boolean function data samples.

    Use 0 ancilla qubit.

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
    Construct a Hermitian block encoding of
    the sparse element oracle |i>|j> -> A_{ij} |i>|j> from matrix data samples.

    Use 1 ancilla qubit.

    Assume that the matrix elements are in [-1,1].

    Args:
        data: tuple of (sampled_row_indices, sampled_col_indices, sampled_values)
        dim: int, dimension of the target matrix
        nnz: int, number of non-zero elements in the matrix.
    Returns:
        a Hermitian block encoding of the sparse element oracle as an array of shape (2, 2, dim**2)
        where the first two dimensions are the ancilla qubit space and the last dimension is the diagonal.
    """

    t = nnz

    # 1. construct the unitary U: |i>|j> -> exp( i * B_{ij} ) |i>|j>
    # where B_{ij} = arcsin( A_{ij} )

    sampled_row_indices = data[0]
    sampled_col_indices = data[1]
    sampled_values = data[2]
    num_samples = sampled_row_indices.shape[0]

    phase = jnp.zeros((dim, dim), dtype=real_dtype)
    phase = phase.at[sampled_row_indices, sampled_col_indices].add(
        jnp.arcsin(sampled_values)
    )
    phase = phase * t / num_samples

    phase = phase.reshape(dim**2)

    # 2. construct the block encoding of |i>|j> -> sin(B_{ij}) |i>|j> using LCU
    # we use the circuit S_a X_a H_a (c_a^1 U^\dagger) (c_a^0 U) X_a H_a T_a
    # where a is the ancilla and S = [[-i, 0], [0, 1]], T = [[1, 0], [0, -i]]
    # analytic calculation shows that the resulting block encoding is
    # [[ (U - U^\dagger)/(2i),  (U + U^\dagger)/2     ],
    #  [ (U + U^\dagger)/2   , -(U - U^\dagger)/(2i) ]]
    # which is a Hermitian block encoding of sin(B) with one ancilla qubit
    # Note again that this is compatible with quantum oracle sketching,
    # since we can implement the c^0U and c^1U† unitaries simultaneously
    # using the same samples.

    U_diag = jnp.exp(1j * phase)

    sin = (U_diag - U_diag.conj()) / (2j)
    cos = (U_diag + U_diag.conj()) / 2

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
    N = 128
    num_samples = int(1e7)

    print(f"Testing general vector with dimension N = {N}")

    key, subkey = random.split(key)
    v = random.normal(subkey, (N,))
    v = v / jnp.linalg.norm(v)

    data_gen = vector_data(v)
    key, subkey = random.split(key)
    data = data_gen.get_data(subkey, num_samples)

    print(f"Sample size: {num_samples:.2e}")

    qstate = q_state_sketch(data, N, jnp.linalg.norm(v), key)

    prob = jnp.linalg.norm(qstate) ** 2
    print(f"Success probability: {prob:.3f}")

    error = jnp.linalg.norm(v - qstate / jnp.linalg.norm(qstate))
    print(f"State reconstruction error in l2 norm: {error:.3e}")
    assert jnp.isclose(error, 0, atol=1e-1)

    # amplitude amplification to boost success probability
    print("Testing amplitude amplification to boost success probability...")

    # get many copies of imperfect qstate
    degree = 51
    qstate_imperfect = jnp.zeros((degree, N), dtype=complex_dtype)
    for i in range(degree):
        key, subkey = random.split(key)
        data = data_gen.get_data(subkey, num_samples)

        key, subkey = random.split(key)
        qstate_imperfect = qstate_imperfect.at[i].set(
            q_state_sketch(data, N, jnp.linalg.norm(v), subkey)
        )
    qstate_aa = primitives.amplitude_amplification(
        qstate_imperfect, degree=degree, target_norm=0.99
    )

    prob_aa = jnp.linalg.norm(qstate_aa) ** 2
    print(
        f"Post-amplification success probability: {prob_aa:.3f}, was {prob:.3f} before"
    )
    error_aa = jnp.linalg.norm(v - qstate_aa / jnp.linalg.norm(qstate_aa))
    print(
        f"Post-amplification state reconstruction error in l2 norm: {error_aa:.3e}, was {error:.3e} before"
    )
    assert jnp.isclose(error_aa, 0, atol=1e-1)


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

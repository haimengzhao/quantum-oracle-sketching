import time

import jax
import jax.numpy as jnp
from jax import random

import primitives
import qsvt
import utils
from data_generation import boolean_data, matrix_data, vector_data
from utils import complex_dtype, real_dtype


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

    # U_diag = jnp.exp(1j * contribution)  # shape (degree, dim)
    # sin = vectorized_diag(U_diag - U_diag.conj()) / 2j  # shape (degree, dim, dim)
    # cos = vectorized_diag(U_diag + U_diag.conj()) / 2  # shape (degree, dim, dim)
    # use shortcut instead:
    sin = jnp.sin(contribution)  # shape (degree, dim)
    cos = jnp.cos(contribution)  # shape (degree, dim)

    block_encoding = jnp.stack([sin, cos, cos, -sin], axis=0).reshape(
        2, 2, degree, dim
    )  # shape (2, 2, degree, dim)
    block_encoding = block_encoding.transpose(2, 0, 1, 3)  # shape (degree, 2, 2, dim)

    # 6. Apply QSVT to approximate arcsin(x)
    block_encoding = qsvt.apply_qsvt_imperfect_diag(
        block_encoding[:-1], num_ancilla=1, angle_set=angle_set
    )  # shape (2, 2, dim)

    # 7. Prepare the state
    # apply the block encoding to the all plus state
    # note that the qsvt applies arcsin as the real part
    # we need a second LCU to get the real part only
    # Note again that this is compatible with quantum oracle sketching,
    # since we can implement the c^0U and c^1U† unitaries simultaneously
    # using the same samples.

    state = jnp.real(block_encoding[0, 0]) / jnp.sqrt(dim)

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


def q_oracle_sketch_matrix_element(data, dims, nnz):
    """
    Construct a Hermitian block encoding of
    the sparse element oracle |i>|j> -> A_{ij} |i>|j> from matrix data samples.

    Use 1 ancilla qubit.

    Assume that the matrix elements are in [-1,1].

    Args:
        data: tuple of (sampled_row_indices, sampled_col_indices, sampled_values)
        dims: tuple of (int, int), dimensions of the target matrix.
        nnz: int, number of non-zero elements in the matrix.

    Returns:
        the diagonal of the relevant block of the block encoding as an array of shape (dims[0]*dims[1],)
    """

    t = nnz

    # 1. construct the unitary U: |i>|j> -> exp( i * B_{ij} ) |i>|j>
    # where B_{ij} = arcsin( A_{ij} )

    sampled_row_indices = data[0]
    sampled_col_indices = data[1]
    sampled_values = data[2]
    num_samples = sampled_row_indices.shape[0]

    phase = jnp.zeros((dims[0], dims[1]), dtype=real_dtype)
    phase = phase.at[sampled_row_indices, sampled_col_indices].add(
        jnp.arcsin(sampled_values)
    )
    phase = phase * t / num_samples

    phase = phase.reshape(dims[0] * dims[1])

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

    # U_diag = jnp.exp(1j * phase)
    # sin = (U_diag - U_diag.conj()) / (2j)
    # use shortcut instead:
    sin = jnp.sin(phase)

    # return only the diagonal of the relevant block
    return sin


def q_oracle_sketch_matrix_index(data, dims, axis, sparsity, nnz):
    """
    Construct a block encoding of
    the sparse row or column index oracle.

    For example, the row index oracle is |i>|k>|0> -> |i>|k>|j(i, k)>,
    where j(i, k) is the column index of the k-th non-zero element in row i.

    Similarly, the column index oracle is |j>|k>|0> -> |j>|k>|i(j, k)>,
    where i(j, k) is the row index of the k-th non-zero element in column j.

    Use 2 ancilla qubits.

    The index register is padded to the next power of 2 internally to support binary search.

    Args:
        data: tuple of (sampled_row_indices, sampled_col_indices, sampled_values)
        dims: tuple of (int, int), dimensions of the target matrix
        axis: int, 0 for row index oracle, 1 for column index oracle
        sparsity: int, maximum number of non-zero elements per row (axis=0) or column (axis=1).
        nnz: int, number of non-zero elements in the matrix.

    Returns:
        the oracle as an array of shape (dims[axis], sparsity, dims[1 - axis])
    """

    # if axis == 1, swap row and column indices
    sampled_row_indices = data[axis]
    sampled_col_indices = data[1 - axis]
    num_samples = sampled_row_indices.shape[0]

    num_rows = dims[axis]
    bitlength_col = int(jnp.ceil(jnp.log2(dims[1 - axis])))
    num_cols = 2**bitlength_col  # pad index register to power of 2

    # 1. Construct cumulative counter unitary
    # |i>|k>|l> -> exp( i * theta(i,k,l) ) |i>|k>|l>
    # theta(i, k, l) = pi / (2s+1) * (|{j: A_{ij} \neq 0, j < l}| - k + 1/2)
    # using random gates of the form
    # |i>|k>|l> -> exp( i * pi * nnz / (2s+1) / num_samples * \sum_t 1[i_t=i, j_t<l] ) |i>|k>|l>
    # note that the -k + 1/2 part can be done separately
    # and the remaining terms does not depend on k

    phase = jnp.zeros((num_rows, num_cols), dtype=real_dtype)
    # calculate \sum_t 1[i_t=i, j_t=l]
    phase = phase.at[sampled_row_indices, sampled_col_indices].add(1.0)
    counts = phase
    # calculate \sum_t 1[i_t=i, j_t <= l]
    phase = phase.cumsum(axis=1)  # cumulative count along each row (inclusive)
    # convert to \sum_t 1[i_t=i, j_t < l]
    phase = phase - counts

    t = jnp.pi * nnz / (2 * sparsity + 1)
    phase = phase * t / num_samples

    # expand to include sparsity index k
    phase = jnp.repeat(
        phase[:, None, :], sparsity, axis=1
    )  # shape (num_rows, sparsity, num_cols)

    # - k + 1/2
    # Note that k starts from 1 to sparsity
    k_indices = jnp.arange(sparsity, dtype=real_dtype) + 1  # shape (sparsity,)
    phase = phase - (k_indices[None, :, None] - 0.5) * jnp.pi / (
        2 * sparsity + 1
    )  # shape (num_rows, sparsity, num_cols)

    phase = phase.reshape(
        num_rows * sparsity * num_cols
    )  # shape (num_rows * sparsity * num_cols,)
    # 2. Use LCU to get sin(theta(i,k,l))
    sin = jnp.sin(phase)  # shape (num_rows * sparsity * num_cols,)
    cos = jnp.cos(phase)  # shape (num_rows * sparsity * num_cols,)

    # 3. Apply the sign function using QSVT
    # to get the phase cumulative counter oracle |i>|k>|l> -> (-1)^{ 1[C(i, l) < k] } |i>|k>|l>
    threshold = jnp.pi / (4 * sparsity + 2)
    degree = 51
    angle_set, scale = qsvt.get_qsvt_angles_sign(
        degree=degree, threshold=threshold, rescale=0.99
    )

    print("Minimal signal to be digitized:", jnp.min(jnp.abs(sin)))
    print("QSVT sign function threshold:", threshold)

    block_encoding = jnp.stack([sin, cos, cos, -sin], axis=0).reshape(
        2, 2, num_rows * sparsity * num_cols
    )  # shape (2, 2, num_rows * sparsity * num_cols)
    block_encoding = qsvt.apply_qsvt_diag(
        block_encoding, num_ancilla=1, angle_set=angle_set
    )  # shape (2, 2, num_rows * sparsity * num_cols)
    # obtain the phase oracle |i>|k>|l> -> (-1)^{ 1[C(i, l) < k] } |i>|k>|l>
    block_encoding = jnp.real(
        block_encoding[0, 0]
    )  # shape (num_rows * sparsity * num_cols,)
    block_encoding = block_encoding.reshape(
        num_rows, sparsity, num_cols
    )  # shape (num_rows, sparsity, num_cols)

    # 4. Construct the XOR oracle |i>|k>|l>|0> -> |i>|k>|l>|1[C(i, l) < k]>
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    # controlled phase oracle: |0> apply identity, |1> apply phase oracle
    cont_block_encoding = jnp.stack(
        [jnp.ones_like(block_encoding), block_encoding], axis=-1
    )  # shape (num_rows, sparsity, num_cols, 2)
    xor_oracle = jnp.einsum(
        "am,ijkm,mn->ijkan",
        hadamard,
        cont_block_encoding,
        hadamard,
    )  # shape (num_rows, sparsity, num_cols, 2, 2)

    # 5. Construct the index oracle
    # initial state on |l>|o> = |0^n>|0>
    state_lo = jnp.zeros((num_cols, 2), dtype=real_dtype)
    state_lo = state_lo.at[0, 0].set(1.0)  # shape (num_cols, 2)
    state_lo = state_lo[None, None, :, :]  # shape (1, 1, num_cols, 2)
    state_lo = jnp.tile(
        state_lo, (num_rows, sparsity, 1, 1)
    )  # shape (num_rows, sparsity, num_cols, 2)

    for bit in range(bitlength_col - 1, -1, -1):  # MSB-first (bit is LSB index)
        # SWAP_{l_t, o} X_{l_t} O X_{l_t}

        high = 1 << (bitlength_col - bit - 1)
        low = 1 << bit

        # a. apply X_{l_t}
        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo[:, :, :, ::-1, :, :]
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

        # b. apply O: |i>|k>|l>|0> -> |i>|k>|l>|1[C(i, l) < k]>
        state_lo = jnp.matvec(xor_oracle, state_lo)

        # c. apply X_{l_t}
        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo[:, :, :, ::-1, :, :]
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

        # d. apply SWAP_{l_t, o}
        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo.transpose(0, 1, 2, 5, 4, 3)
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

    # final state
    # truncate the index register back to original size
    state_lo = state_lo[:, :, : dims[1], 0]  # shape (num_rows, sparsity, dims[1])

    return state_lo


def q_oracle_sketch_matrix_row_index(data, dims, sparsity):
    """
    Construct a block encoding of
    the sparse row index oracle |i>|k>|0> -> |i>|k>|j(i, k)>,
    where j(i, k) is the column index of the k-th non-zero element in row i,
    when the data are random rows of the matrix.

    Use 2 ancilla qubits.

    The index register is padded to the next power of 2 internally to support binary representation.

    Args:
        data: tuple of (sampled_row_indices, sampled_row_vectors)
        dims: tuple of (int, int), dimensions of the target matrix
        sparsity: int, maximum number of non-zero elements per row.

    Returns:
        the oracle as an array of shape (dims[0], sparsity, dims[1])
    """

    sampled_row_indices, sampled_row_vectors = data
    num_samples = sampled_row_indices.shape[0]
    bitlength_col = int(jnp.ceil(jnp.log2(dims[1])))

    def _row_nonzero_indices(row_vector):
        # get the column indices of non-zero elements in the row
        return jnp.nonzero(
            row_vector != 0,
            size=sparsity,
            fill_value=2**bitlength_col - 1,
            # pad with the last index if this row has less than sparsity non-zero elements
        )[0]

    sampled_nonzero_col_indices = jax.vmap(_row_nonzero_indices)(
        sampled_row_vectors
    )  # shape (num_samples, sparsity)

    bit_positions = jnp.arange(bitlength_col - 1, -1, -1)
    sampled_bits = (sampled_nonzero_col_indices[..., None] >> bit_positions) & 1

    row_bits = jnp.zeros((dims[0], sparsity, bitlength_col))
    row_bits = row_bits.at[sampled_row_indices].add(
        sampled_bits
    )  # shape (dims[0], sparsity, bitlength_col)

    # 1. Construct the controlled phase oracle
    # |i>|k>|1>_m -> (-1)^{j(i, k)_m} |i>|k>|1>_m
    # for each bit m of the column index j(i, k)
    t = jnp.pi * dims[0]
    phase = row_bits * t / num_samples  # shape (dims[0], sparsity, bitlength_col)
    diag = jnp.exp(1j * phase)  # shape (dims[0], sparsity, bitlength_col)
    controlled_diag = jnp.stack(
        [jnp.ones_like(diag), diag], axis=-1
    )  # shape (dims[0], sparsity, bitlength_col, 2)

    # 2. Convert to XOR oracle
    # |i>|k>|0>_m -> |i>|k>|j(i, k)_m>_m
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    xor_oracle = jnp.einsum(
        "am,ijkm,mn->ijkan",
        hadamard,
        controlled_diag,
        hadamard,
    )  # shape (dims[0], sparsity, bitlength_col, 2, 2)

    # 3. Construct the index oracle by applying the XOR oracle to |0^n>_{m's}
    # tensor product over all bits to get the full index state
    state = xor_oracle[:, :, 0, :, 0]  # shape (dims[0], sparsity, 2)
    for bit in range(1, bitlength_col):
        state = jnp.einsum(
            "ija,ijb->ijab",
            state,
            xor_oracle[:, :, bit, :, 0],
        )  # shape (dims[0], sparsity, 2^(bit-1), 2)
        state = state.reshape(dims[0], sparsity, -1)  # shape (dims[0], sparsity, 2^bit)
    state = state.reshape(dims[0], sparsity, -1)  # shape (dims[0], sparsity, num_cols)

    state = state[:, :, : dims[1]]  # truncate to original size

    return state


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
    target_norm = 0.98
    qstate_imperfect = jnp.zeros((degree, N), dtype=complex_dtype)
    with utils.suppress_stdout_stderr():
        for i in range(degree):
            key, subkey = random.split(key)
            data = data_gen.get_data(subkey, num_samples)

            key, subkey = random.split(key)
            qstate_imperfect = qstate_imperfect.at[i].set(
                q_state_sketch(data, N, jnp.linalg.norm(v), subkey)
            )

    # apply amplitude amplification
    qstate_aa = primitives.amplitude_amplification(
        qstate_imperfect, degree=degree, target_norm=target_norm
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

    print(f"Total number of samples used: {data_gen.num_generated_samples:.3e}")


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
    N1 = 1000
    N2 = 10000

    print(f"Testing matrix with dimension N1 = {N1:.2e}, N2 = {N2:.2e}")
    print(
        f"Note that the oracle has dimension (N1*N2) x (N1*N2) = {(N1 * N2):.2e} x {(N1 * N2):.2e}"
    )

    key, subkey = random.split(key)
    A = random.normal(subkey, (N1, N2))
    A = A / jnp.linalg.norm(A)

    data_gen = matrix_data(A)
    num_samples = int(1e8)
    key, subkey = random.split(key)
    data = data_gen.get_matrix_element_data(subkey, num_samples=num_samples)

    print(f"Sample size: {num_samples:.2e}")

    nnz = jnp.count_nonzero(A)

    start_time = time.time()

    oracle_diag = q_oracle_sketch_matrix_element(data, (N1, N2), nnz=nnz)

    end_time = time.time()

    print(f"Oracle construction time: {end_time - start_time:.3e} seconds")

    if max(N1, N2) <= 10:
        # only reconstruct the full dense matrix when N is small
        oracle = jnp.diag(oracle_diag)
        oracle = utils.halmos_dilation(oracle)

        print(
            "Oracle block encoding unitarity check:",
            jnp.isclose(
                jnp.linalg.norm(oracle @ oracle.conj().T - jnp.eye(2 * (N1 * N2))),
                0,
                atol=1e-2,
            ),
        )
        assert jnp.isclose(
            jnp.linalg.norm(oracle @ oracle.conj().T - jnp.eye(2 * (N1 * N2))),
            0,
            atol=1e-2,
        )

        print(
            "Oracle block encoding Hermiticity check:",
            jnp.isclose(jnp.linalg.norm(oracle - oracle.conj().T), 0, atol=1e-2),
        )

        assert jnp.isclose(jnp.linalg.norm(oracle - oracle.conj().T), 0, atol=1e-2)
    else:
        print("Skipping unitarity and Hermiticity checks for large N.")

    A_reconst = oracle_diag

    error = jnp.max(jnp.abs(A_reconst - A.reshape(N1 * N2)))
    print(f"Matrix reconstruction error in operator norm: {error:.3e}")

    assert jnp.isclose(error, 0, atol=1e-1)


def _test_q_oracle_sketch_matrix_row_index(key):
    # random sparse matrix
    dim1 = 100
    dim2 = 10
    nnz = dim1 * 3
    num_samples = int(1e7)

    print(f"Testing sparse matrix with dimension {dim1} x {dim2}, nnz = {nnz}")

    key, subkey = random.split(key)
    row_indices = random.randint(subkey, (nnz,), 0, dim1)
    key, subkey = random.split(key)
    col_indices = random.randint(subkey, (nnz,), 0, dim2)
    key, subkey = random.split(key)
    values = random.normal(subkey, (nnz,))

    A = jnp.zeros((dim1, dim2)).at[row_indices, col_indices].set(values)
    nnz = jnp.count_nonzero(A)
    row_counts = jnp.sum(A != 0, axis=1)
    row_sparsity = int(jnp.max(row_counts))

    print(f"Matrix row sparsity: {row_sparsity}")
    print("Matrix row sparsity distribution:", jnp.bincount(row_counts))

    data_gen = matrix_data(A)
    key, subkey = random.split(key)
    data = data_gen.get_row_data(subkey, num_samples)

    print(f"Sample size: {num_samples:.2e}")

    start_time = time.time()

    index_oracle = q_oracle_sketch_matrix_row_index(
        data, dims=A.shape, sparsity=row_sparsity
    )

    end_time = time.time()
    print(f"Index oracle construction time: {end_time - start_time:.3e} seconds")

    col_mask = A != 0
    col_indices = jnp.arange(dim2)
    expected_cols = jnp.where(col_mask, col_indices, dim2)
    expected_cols = jnp.sort(expected_cols, axis=1)[:, :row_sparsity]

    pred = jnp.argmax(jnp.abs(index_oracle), axis=-1)
    valid = jnp.arange(row_sparsity)[None, :] < row_counts[:, None]

    assert jnp.all((pred == expected_cols) | ~valid)
    print("Index reconstruction correct.")

    pred_value = jnp.take_along_axis(index_oracle, pred[..., None], axis=-1)[..., 0]
    error = jnp.max(jnp.where(valid, jnp.abs(1.0 - pred_value), 0.0))
    print(f"Matrix row index oracle reconstruction error: {error:.3e}")


def _test_q_oracle_sketch_matrix_index(key):
    # random sparse matrix
    dim1 = 1000
    dim2 = 100
    nnz = dim1 * 3
    num_samples = int(1e7)

    print(f"Testing sparse matrix with dimension {dim1} x {dim2}, nnz = {nnz}")

    key, subkey = random.split(key)
    row_indices = random.randint(subkey, (nnz,), 0, dim1)
    key, subkey = random.split(key)
    col_indices = random.randint(subkey, (nnz,), 0, dim2)
    key, subkey = random.split(key)
    values = random.normal(subkey, (nnz,))

    A = jnp.zeros((dim1, dim2)).at[row_indices, col_indices].set(values)
    nnz = jnp.count_nonzero(A)
    row_counts = jnp.sum(A != 0, axis=1)
    row_sparsity = int(jnp.max(row_counts))

    print(f"Matrix row sparsity: {row_sparsity}")

    data_gen = matrix_data(A)
    key, subkey = random.split(key)
    data = data_gen.get_matrix_element_data(subkey, num_samples)

    print(f"Sample size: {num_samples:.2e}")

    start_time = time.time()

    index_oracle = q_oracle_sketch_matrix_index(
        data, dims=A.shape, axis=0, sparsity=row_sparsity, nnz=nnz
    )

    end_time = time.time()
    print(f"Index oracle construction time: {end_time - start_time:.3e} seconds")

    col_mask = A != 0
    col_indices = jnp.arange(dim2)
    expected_cols = jnp.where(col_mask, col_indices, dim2)
    expected_cols = jnp.sort(expected_cols, axis=1)[:, :row_sparsity]

    pred = jnp.argmax(jnp.abs(index_oracle), axis=-1)
    valid = jnp.arange(row_sparsity)[None, :] < row_counts[:, None]

    assert jnp.all((pred == expected_cols) | ~valid)
    print("Index reconstruction correct.")

    pred_value = jnp.take_along_axis(index_oracle, pred[..., None], axis=-1)[..., 0]
    error = jnp.max(jnp.where(valid, jnp.abs(1.0 - pred_value), 0.0))
    print(f"Matrix index oracle reconstruction error: {error:.3e}")


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
    print("Testing quantum oracle sketching for matrix sparse index oracle...")
    _test_q_oracle_sketch_matrix_index(key)

    print("-" * 10)
    print(
        "Testing quantum oracle sketching for matrix sparse row index oracle with random row data..."
    )
    _test_q_oracle_sketch_matrix_row_index(key)

    print("-" * 10)
    print("All tests passed.")

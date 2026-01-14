import time

import jax
import jax.numpy as jnp
from jax import random

import primitives
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


def q_state_sketch_flat(vector, unit_num_samples):
    """
    Construct the quantum state sketch of a flat vector.

    Use 1 ancilla qubit.

    Assume that the vector is flat: all components are plus or minus one.

    Args:
        vector: array of shape (dim,), the input vector
        unit_num_samples: int, number of samples to use in each sketch

    Returns:
        (state, num_samples): quantum state sketch as an array of shape (dim,) and the number of samples used
    """
    dim = vector.shape[0]
    prob = jnp.ones_like(vector, dtype=real_dtype) / dim
    t = jnp.pi * dim

    # expected single gate
    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples * (1 - vector) / 2))

    # concatenate all gates
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)

    # apply the gate to the all plus state
    state = diag / jnp.sqrt(dim)

    return state, unit_num_samples


def q_state_sketch(vector, key, unit_num_samples, degree=4):
    """
    Construct the quantum state sketch of a general vector.

    Use 2 ancilla qubit.
    One for the first LCU and QSVT, one for the second LCU to extract real part.

    Args:
        vector: array of shape (dim,), the input vector
        key: jax.random.PRNGKey, random key for generating random signs
        unit_num_samples: int, number of samples to use in each sketch
        degree: even int, degree of the polynomial approximation for arcsin(x), default 4

    Returns:
        (state, num_samples): quantum state sketch as an array of shape (dim,) and the number of samples used

    """
    orig_dim = vector.shape[0]
    dim = int(2 ** jnp.ceil(jnp.log2(vector.shape[0])))
    vector = jnp.pad(
        vector, (0, int(dim) - orig_dim), mode="constant", constant_values=0
    )
    norm = jnp.linalg.norm(vector)
    prob = jnp.ones_like(vector, dtype=real_dtype) / dim

    # random sign O_h
    key, subkey = random.split(key)
    random_signs = random.choice(
        subkey, jnp.array([1, -1], dtype=int_dtype), shape=(dim,)
    )  # shape (dim,)

    # 1. Construct the target basis states u = 0, ..., 2^n - 1
    # Shape: (dim,)
    u_vec = jnp.arange(dim, dtype=int_dtype)

    # 2. Compute the bitwise interaction j \cdot u (mod 2)
    # Instead of computing for each sample, which is costly for large num_samples,
    # we compute for all possible j and u.
    j_vec = jnp.arange(dim, dtype=int_dtype)  # shape (dim,)
    bitwise_and = jnp.bitwise_and(j_vec[:, None], u_vec[None, :])  # shape (dim, dim)

    # jax.lax.population_count counts the number of set bits (Hamming weight).
    # The inner product over GF(2) is the parity of the and.
    bit_inner_product = jax.lax.population_count(bitwise_and) % 2  # shape (dim, dim)

    # 3. Convert inner products to sign factors (-1)^(j \cdot u)
    # 0 -> +1, 1 -> -1
    inner_prod_signs = 1 - 2 * bit_inner_product  # shape (dim, dim)

    # 4. Concatenate expected single gate
    t = dim / norm / 3
    log_diag = jnp.log1p(
        jnp.sum(
            prob[:, None]
            * jnp.expm1(
                1j
                * (random_signs * vector)[:, None]
                * inner_prod_signs
                * t
                / unit_num_samples
            ),
            axis=0,
        )
    )
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)  # shape (dim,)

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

    sin = (diag - jnp.conj(diag)) / (2j)  # shape (dim,)
    cos = (diag + jnp.conj(diag)) / 2  # shape (dim,)

    block_encoding = jnp.stack([sin, cos, cos, -sin], axis=0).reshape(
        2, 2, dim
    )  # shape (2, 2, dim)

    # 6. Apply QSVT to approximate arcsin(x)
    block_encoding = qsvt.apply_qsvt_diag(
        block_encoding, num_ancilla=1, angle_set=angle_set
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
    state = state[:orig_dim]  # truncate back to the original dimension

    return state, unit_num_samples * (angle_set.shape[0] - 1)


def q_oracle_sketch_boolean(truth_table, unit_num_samples):
    """
    Construct the quantum oracle sketch of a boolean function.

    Use 0 ancilla qubit.

    Args:
        truth_table: array of shape (dim,), the truth table of the boolean function
        unit_num_samples: int, number of samples to use in each sketch

    Returns:
        (diagonal, num_samples): diagonal of the phase oracle sketch as an array of shape (dim,)
            and the number of samples used
    """
    dim = truth_table.shape[0]

    prob = jnp.ones_like(truth_table, dtype=real_dtype) / dim
    t = jnp.pi * dim

    # expected single gate
    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples * truth_table))
    # concatenate all gates
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)

    return diag, unit_num_samples


def q_oracle_sketch_matrix_element(matrix, unit_num_samples):
    """
    Construct a Hermitian block encoding of
    the sparse element oracle |i>|j> -> A_{ij} |i>|j> from matrix data samples.

    Use 1 ancilla qubit.

    Assume that the matrix elements are in [-1,1].

    Args:
        matrix: array of shape (num_rows, num_cols), the input matrix
        unit_num_samples: int, number of samples to use in each sketch

    Returns:
        (diag, num_samples): the diagonal of the relevant block of the block encoding
            as an array of shape (num_rows * num_cols,) and the number of samples used
    """

    dims = matrix.shape
    nnz = jnp.count_nonzero(matrix)

    t = nnz

    # 1. construct the unitary U: |i>|j> -> exp( i * B_{ij} ) |i>|j>
    # where B_{ij} = arcsin( A_{ij} )

    # prob is uniform over all non-zero elements
    prob = jnp.zeros_like(matrix, dtype=real_dtype)
    prob = prob.at[matrix != 0].set(1.0 / nnz)

    # expected single gate
    log_diag = jnp.log1p(
        prob * jnp.expm1(1j * t / unit_num_samples * jnp.arcsin(matrix))
    )  # shape (num_rows, num_cols)
    # concatenate all gates
    log_diag = unit_num_samples * log_diag  # shape (num_rows, num_cols)
    diag = jnp.exp(log_diag)  # shape (num_rows, num_cols)

    diag = diag.reshape(dims[0] * dims[1])

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

    sin = (diag - diag.conj()) / (2j)

    # return only the diagonal of the relevant block
    return sin, unit_num_samples


def q_oracle_sketch_matrix_row_index(matrix, unit_num_samples):
    """
    Construct a block encoding of
    the sparse row index oracle |i>|k>|0> -> |i>|k>|j(i, k)>,
    where j(i, k) is the column index of the k-th non-zero element in row i,
    when the data are random rows of the matrix.

    Use 2 ancilla qubits.

    The index register is padded to the next power of 2 internally to support binary representation.

    Args:
        matrix: array of shape (num_rows, num_cols), the input matrix with row sparsity 'sparsity'
        unit_num_samples: int, number of samples to use in each sketch

    Returns:
        (oracle, num_samples): the oracle as an array of shape (num_rows, sparsity, num_cols),
            where sparsity is the row sparsity of the matrix; and the number of samples used
    """

    dims = matrix.shape
    sparsity = int(jnp.max(jnp.count_nonzero(matrix, axis=1)))

    bitlength_col = int(jnp.ceil(jnp.log2(dims[1])))

    # 0. Calculate the binary representation of j(i, k)_m
    # non-zero column indices for each row
    nz_col_indices = jnp.argsort(matrix != 0, axis=1, descending=True)[
        :, :sparsity
    ]  # shape (dims[0], sparsity)

    bit_positions = jnp.arange(bitlength_col - 1, -1, -1)
    col_bits = (
        nz_col_indices[..., None] >> bit_positions
    ) & 1  # shape (dims[0], sparsity, bitlength_col)

    # 1. Construct the controlled phase oracle
    # |i>|k>|1>_m -> (-1)^{j(i, k)_m} |i>|k>|1>_m
    # for each bit m of the column index j(i, k)
    # this is done simultaneously for all m with the same samples
    t = jnp.pi * dims[0]
    prob = jnp.ones((dims[0],), dtype=real_dtype) / (dims[0])

    # expected single gate
    log_diag = jnp.log1p(
        prob[:, None, None] * jnp.expm1(1j * t / unit_num_samples * col_bits)
    )  # shape (dims[0], sparsity, bitlength_col)
    # concatenate all gates
    log_diag = unit_num_samples * log_diag  # shape (dims[0], sparsity, bitlength_col)
    diag = jnp.exp(log_diag)  # shape (dims[0], sparsity, bitlength_col)

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

    return state, unit_num_samples


def q_oracle_sketch_matrix_index(
    matrix, unit_num_samples, axis, degree=101, scale=0.9999
):
    """
    Construct a block encoding of
    the sparse row or column index oracle,
    without erasing the rank register |k>.

    For example, the row index oracle is |i>|k>|0> -> |i>|k>|j(i, k)>,
    where j(i, k) is the column index of the k-th non-zero element in row i.

    Similarly, the column index oracle is |j>|k>|0> -> |j>|k>|i(j, k)>,
    where i(j, k) is the row index of the k-th non-zero element in column j.

    Use 2 ancilla qubits.

    The index register is padded to the next power of 2 internally to support binary search.

    Args:
        matrix: array of shape (num_rows, num_cols), the input matrix with row or column sparsity 'sparsity'
        unit_num_samples: int, number of samples to use in each sketch
        axis: int, 0 for row index oracle, 1 for column index oracle
        degree: odd int, degree of the polynomial approximation for sign function in QSVT, default 51
        scale: float in (0, 1), target magnitude of the sign function, default 0.99

    Returns:
        (oracle, num_samples): the oracle as an array of shape (num_rows, sparsity, num_cols)
            if axis == 0 or (num_cols, sparsity, num_rows) if axis == 1; and the number of samples used
    """

    # regard axis as 'rows', 1 - axis as 'columns'
    num_rows = matrix.shape[axis]
    bitlength_col = int(jnp.ceil(jnp.log2(matrix.shape[1 - axis])))
    orig_num_cols = matrix.shape[1 - axis]
    num_cols = 2**bitlength_col  # pad index register to power of 2

    sparsity = int(jnp.max(jnp.count_nonzero(matrix, axis=1 - axis)))
    nnz = jnp.count_nonzero(matrix)

    k_indices = jnp.arange(sparsity, dtype=int_dtype) + 1  # shape (sparsity,)
    t = jnp.pi * nnz / (2 * sparsity + 1)
    k_phase_scale = jnp.pi / (2 * sparsity + 1)

    # 1. Construct cumulative counter unitary
    # |i>|k>|l> -> exp( i * theta(i,k,l) ) |i>|k>|l>
    # theta(i, k, l) = pi / (2s+1) * (|{j: A_{ij} \neq 0, j < l}| - k + 1/2)
    # using random gates of the form
    # |i>|k>|l> -> exp( i * pi * nnz / (2s+1) / num_samples * \sum_t 1[i_t=i, j_t<l] ) |i>|k>|l>
    # note that the -k + 1/2 part can be done separately
    # and the remaining terms does not depend on k

    # prob is uniform over all non-zero elements
    prob = jnp.zeros_like(matrix if axis == 0 else matrix.T, dtype=real_dtype)
    prob = prob.at[matrix != 0 if axis == 0 else matrix.T != 0].set(1.0 / nnz)
    # pad the columns to power of 2
    prob = jnp.pad(
        prob,
        ((0, 0), (0, num_cols - orig_num_cols)),
        mode="constant",
        constant_values=0,
    )  # shape (num_rows, num_cols)
    # probability of row = i, column < l
    prob = jnp.cumsum(prob, axis=1) - prob  # shape (num_rows, num_cols)

    # expected single gate
    log_diag = jnp.log1p(
        prob * jnp.expm1(1j * t / unit_num_samples)
    )  # shape (num_rows, num_cols)
    # concatenate all gates
    log_diag = unit_num_samples * log_diag  # shape (num_rows, num_cols)

    # expand to include sparsity index k
    log_diag = jnp.repeat(
        log_diag[:, None, :], sparsity, axis=1
    )  # shape (num_rows, sparsity, num_cols)

    # - k + 1/2
    # Note that k starts from 1 to sparsity
    log_diag = (
        log_diag - 1j * (k_indices[None, :, None] - 0.5) * k_phase_scale
    )  # shape (num_rows, sparsity, num_cols)

    diag = jnp.exp(log_diag).reshape(
        num_rows * sparsity * num_cols
    )  # shape (num_rows * sparsity * num_cols,)

    # 2. LCU to convert phase to sin
    sin = (diag - jnp.conj(diag)) / (2j)
    cos = (diag + jnp.conj(diag)) / 2
    block_encoding = jnp.stack(
        [jnp.stack([sin, cos], axis=0), jnp.stack([cos, -sin], axis=0)],
        axis=0,
    )  # shape (2, 2, num_rows * sparsity * num_cols)

    # 3. Use QSVT to apply the sign function to get the phase oracle
    # |i>|k>|l> -> (-1)^{ 1[C(i, l) < k] } |i>|k>|l>

    # QSVT setup
    threshold = jnp.pi / (4 * sparsity + 2) * 0.8
    print("QSVT sign function threshold:", threshold)
    angle_set, scale = qsvt.get_qsvt_angles_sign(
        degree=degree, threshold=threshold, rescale=scale
    )
    angle_set = angle_set.astype(real_dtype)

    block_encoding = qsvt.apply_qsvt_diag(
        block_encoding, num_ancilla=1, angle_set=angle_set
    )  # shape (2, 2, num_rows * sparsity * num_cols)

    block_encoding = jnp.real(block_encoding[0, 0])
    block_encoding = block_encoding.reshape(
        num_rows, sparsity, num_cols
    )  # shape (num_rows, sparsity, num_cols)
    # now we have obtained the phase oracle
    # |i>|k>|l> -> (-1)^{ 1[C(i, l) < k] } |i>|k>|l>

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
    # initial state on |l>|o> = |0^n>|0> for binary search
    state_lo = jnp.zeros((num_cols, 2), dtype=real_dtype)
    state_lo = state_lo.at[0, 0].set(1.0)  # shape (num_cols, 2)
    state_lo = state_lo[None, None, :, :]  # shape (1, 1, num_cols, 2)
    state_lo = jnp.tile(
        state_lo, (num_rows, sparsity, 1, 1)
    )  # shape (num_rows, sparsity, num_cols, 2)

    for bit in range(bitlength_col - 1, -1, -1):  # MSB-first (bit is LSB index)
        # apply SWAP_{l_t, o} X_{l_t} O X_{l_t}

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
    state_lo = state_lo[
        :, :, : matrix.shape[1 - axis], 0
    ]  # shape (dims[axis], sparsity, dims[1 - axis])

    return state_lo, unit_num_samples * (angle_set.shape[0] - 1) * bitlength_col


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

    state, num_samples = q_state_sketch_flat(x, num_samples)

    # test norm
    print(f"State norm squared: 1 + {jnp.linalg.norm(state) ** 2 - 1:.3e}")

    # test reconstruction
    recon_x = state
    error = jnp.linalg.norm(recon_x - x / jnp.sqrt(N))
    print(f"State reconstruction error in Euclidean distance: {error:.3e}")
    print(f"Total number of samples used: {num_samples:.3e}")
    assert jnp.allclose(error, 0, atol=1e-2)


def _test_q_state_sketch(key):
    print("-" * 10)
    print("Testing quantum state sketching for general vectors...")
    N = 1000
    num_samples = int(1e4)

    print(f"Testing general vector with dimension N = {N}")

    key, subkey = random.split(key)
    v = random.normal(subkey, (N,), dtype=real_dtype)
    v = v / jnp.linalg.norm(v)

    state, num_samples = q_state_sketch(v, key, num_samples, degree=20)

    prob = jnp.linalg.norm(state) ** 2
    print(f"Success probability: {prob:.3f}")

    error = jnp.linalg.norm(v - state / jnp.linalg.norm(state))
    print(f"State reconstruction error in l2 norm: {error:.3e}")
    print(f"Total number of samples used: {num_samples:.3e}")
    assert jnp.isclose(error, 0, atol=1e-1)

    # amplitude amplification to boost success probability
    print("Testing amplitude amplification to boost success probability...")

    degree = 51
    target_norm = 0.98

    # apply amplitude amplification
    state_aa = primitives.amplitude_amplification(
        state, degree=degree, target_norm=target_norm
    )
    num_samples = num_samples * (degree - 1)

    prob_aa = jnp.linalg.norm(state_aa) ** 2
    print(
        f"Post-amplification success probability: {prob_aa:.3f}, was {prob:.3f} before"
    )
    error_aa = jnp.linalg.norm(v - state_aa / jnp.linalg.norm(state_aa))
    print(
        f"Post-amplification state reconstruction error in l2 norm: {error_aa:.3e}, was {error:.3e} before"
    )
    assert jnp.isclose(error_aa, 0, atol=1e-1)

    print(f"Total number of samples used: {num_samples:.3e}")


def _test_q_oracle_sketch_boolean(key):
    print("-" * 10)
    print("Testing quantum oracle sketching for Boolean functions...")
    # random boolean function
    N = 1000
    num_samples = int(1e7)

    print(f"Testing boolean function with dimension N = {N:.2e}")

    f = random.randint(key, (N,), minval=0, maxval=2, dtype=int_dtype)

    diag, num_samples = q_oracle_sketch_boolean(f, num_samples)

    # test reconstruction
    target_diag = jnp.exp(1j * jnp.pi * f)
    error = jnp.max(jnp.abs(diag - target_diag))
    print(f"Boolean function phase oracle error in spectral norm: {error:.3e}")
    assert jnp.allclose(error, 0, atol=1e-1)
    print(f"Total number of samples used: {num_samples:.3e}")


def _test_q_oracle_sketch_matrix_element(key):
    print("-" * 10)
    print("Testing quantum oracle sketching for matrix element oracle...")
    # random matrix
    N1 = 1000
    N2 = 10000
    num_samples = int(1e6)
    nnz = N1 * 3
    A = utils.random_sparse_matrix(key, (N1, N2), nnz)

    print(f"Testing matrix with dimension N1 = {N1:.2e}, N2 = {N2:.2e}")
    print(
        f"Note that the oracle has dimension (N1*N2) x (N1*N2) = {(N1 * N2):.2e} x {(N1 * N2):.2e}"
    )

    start_time = time.time()

    oracle_diag, num_samples = q_oracle_sketch_matrix_element(A, num_samples)

    end_time = time.time()

    print(f"Oracle construction time: {end_time - start_time:.3e} seconds")

    target_diag = A.reshape(N1 * N2)
    error = jnp.max(jnp.abs(oracle_diag - target_diag))
    print(f"Matrix reconstruction error in spectral norm: {error:.3e}")
    assert jnp.isclose(error, 0, atol=1e-1)
    print(f"Total number of samples used: {num_samples:.3e}")


def _test_q_oracle_sketch_matrix_row_index(key):
    print("-" * 10)
    print("Testing quantum oracle sketching for matrix row index oracle...")
    # random sparse matrix
    dim1 = 100
    dim2 = 1000
    nnz = dim2 * 3
    num_samples = int(1e7)

    print(f"Testing sparse matrix with dimension {dim1} x {dim2}, nnz = {nnz}")

    A = utils.random_sparse_matrix(key, (dim1, dim2), nnz)
    row_counts = jnp.sum(A != 0, axis=1)
    row_sparsity = int(jnp.max(row_counts))

    print(f"Matrix row sparsity: {row_sparsity}")
    print("Matrix row sparsity distribution:", jnp.bincount(row_counts))

    start_time = time.time()
    index_oracle, num_samples = q_oracle_sketch_matrix_row_index(A, num_samples)
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
    print("-" * 10)
    print("Testing quantum oracle sketching for matrix index oracle...")
    # random sparse matrix
    dim1 = 100
    dim2 = 10
    nnz = dim1 * 3
    num_samples = int(1e7)

    print(f"Testing sparse matrix with dimension {dim1} x {dim2}, nnz = {nnz}")

    A = utils.random_sparse_matrix(key, (dim1, dim2), nnz)
    row_counts = jnp.sum(A != 0, axis=1)
    row_sparsity = int(jnp.max(row_counts))

    print(f"Matrix row sparsity: {row_sparsity}")
    print("Matrix row sparsity distribution:", jnp.bincount(row_counts))

    start_time = time.time()
    index_oracle, num_samples = q_oracle_sketch_matrix_index(
        A, num_samples, axis=0, degree=101, scale=0.9999
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

    norm = jnp.linalg.norm(index_oracle, axis=-1, keepdims=True)
    index_oracle_normalized = index_oracle / norm

    succ_prob = jnp.min(norm[valid] ** 2)
    print(f"Minimum success probability of block encoding: {succ_prob:.3e}")

    pred_value = jnp.take_along_axis(index_oracle_normalized, pred[..., None], axis=-1)[
        ..., 0
    ]
    error = jnp.max(jnp.where(valid, jnp.abs(1.0 - pred_value), 0.0))
    print(f"Matrix index oracle reconstruction error after postselection: {error:.3e}")
    assert jnp.isclose(error, 0, atol=1e-1)
    print(f"Number of samples used: {num_samples:.3e}")


def _test_matrix_block_encoding(key):
    # random sparse matrix
    dim1 = 100
    dim2 = 1000
    nnz = dim2 * 3
    unit_num_samples = int(1e7)

    print(f"Testing sparse matrix with dimension {dim1} x {dim2}, nnz = {nnz}")

    A = utils.random_sparse_matrix(key, (dim1, dim2), nnz)

    row_sparsity = int(jnp.max(jnp.count_nonzero(A, axis=1)))
    col_sparsity = int(jnp.max(jnp.count_nonzero(A, axis=0)))
    print(f"Matrix row sparsity: {row_sparsity}, column sparsity: {col_sparsity}")

    # construct element oracle
    element_oracle, num_samples_element = q_oracle_sketch_matrix_element(
        A, unit_num_samples
    )
    print(f"Number of samples for element oracle: {num_samples_element:.3e}")

    # construct row index oracle
    row_index_oracle, num_samples_row = q_oracle_sketch_matrix_row_index(
        A, unit_num_samples
    )
    print(f"Number of samples for row index oracle: {num_samples_row:.3e}")

    # construct column index oracle
    col_index_oracle, num_samples_col = q_oracle_sketch_matrix_index(
        A, unit_num_samples, axis=1, degree=151, scale=0.9999
    )
    print(f"Number of samples for column index oracle: {num_samples_col:.3e}")

    print(row_index_oracle.shape, col_index_oracle.shape, element_oracle.shape)

    # construct block encoding
    block_encoding = utils.block_encoding_from_sparse_oracles(
        row_index_oracle, col_index_oracle, element_oracle
    )
    # normalized_block_encoding = block_encoding * jnp.sqrt(row_sparsity * col_sparsity)
    normalized_block_encoding = block_encoding / jnp.linalg.norm(block_encoding, ord=2)

    error_fro = jnp.linalg.norm(normalized_block_encoding - A) / jnp.linalg.norm(A)
    error_spec = jnp.linalg.norm(
        normalized_block_encoding - A, ord=2
    ) / jnp.linalg.norm(A, ord=2)

    print(f"Row sparsity: {row_sparsity}, Col sparsity: {col_sparsity}")
    print(
        f"Frobenius norm of A: {jnp.linalg.norm(A):.3e}, Spectral norm of A: {jnp.linalg.norm(A, ord=2):.3e}"
    )

    print(
        f"Matrix block encoding reconstruction error: relative Frobenius norm error = {error_fro:.3e}, relative spectral norm error = {error_spec:.3e}"
    )
    assert jnp.isclose(error_spec, 0, atol=1e-1)
    print(
        f"Number of samples used: {num_samples_element + num_samples_row + num_samples_col:.3e}"
    )


if __name__ == "__main__":
    key = random.PRNGKey(0)

    _test_q_state_sketch_flat(key)

    _test_q_state_sketch(key)

    _test_q_oracle_sketch_boolean(key)

    _test_q_oracle_sketch_matrix_element(key)

    _test_q_oracle_sketch_matrix_row_index(key)

    _test_q_oracle_sketch_matrix_index(key)

    _test_matrix_block_encoding(key)

    print("-" * 10)
    print("All tests passed.")

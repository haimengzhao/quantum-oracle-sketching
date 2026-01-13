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


def q_state_sketch_flat(vector, num_samples):
    """
    Construct the quantum state sketch of a flat vector.

    Use 1 ancilla qubit.

    Assume that the vector is flat: all components are plus or minus one.

    Args:
        vector: array of shape (dim,), the input vector
        num_samples: int, number of samples to use in the sketch

    Returns:
        (state, num_samples): quantum state sketch as an array of shape (dim,) and the number of samples used
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

    return state, num_samples


def q_state_sketch(vector, key, unit_num_samples, degree=4):
    """
    Construct the quantum state sketch of a general vector.

    Use 2 ancilla qubit.
    One for the first LCU and QSVT, one for the second LCU to extract real part.

    Args:
        vector: array of shape (dim,), the input vector
        key: jax.random.PRNGKey, random key for generating random signs
        unit_num_samples: int, number of samples to use in each sketch
        degree: even int, degree of the polynomial approximation for arcsin(x),
            default 4, data size should be a multiple of degree
    Returns:
        (state, num_samples): quantum state sketch as an array of shape (dim,) and the number of samples used

    """

    dim = vector.shape[0]
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

    return state, unit_num_samples * (angle_set.shape[0] - 1)


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
    N = 1024
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


if __name__ == "__main__":
    key = random.PRNGKey(0)

    _test_q_state_sketch_flat(key)

    _test_q_state_sketch(key)

    print("-" * 10)
    print("All tests passed.")

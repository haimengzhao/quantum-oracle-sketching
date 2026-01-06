import jax
import jax.numpy as jnp
from jax import random


def unnormalized_hadamard_transform(n):
    H = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    H_n = H
    for _ in range(n - 1):
        H_n = jnp.kron(H_n, H)
    return H_n


def generate_random_unitary(key, dim):
    """Generate a random unitary matrix of size n x n."""
    A = random.normal(key, (dim, dim)) + 1j * random.normal(key, (dim, dim))
    Q, R = jnp.linalg.qr(A)
    D = jnp.diag(jnp.diag(R) / jnp.abs(jnp.diag(R)))
    return jnp.dot(Q, D)


def get_block_encoded(U, num_ancilla=1):
    """Get the block-encoded matrix of a unitary U."""
    dim = U.shape[0]
    assert dim % (2**num_ancilla) == 0, (
        "Unitary size must be divisible by 2^num_ancilla."
    )
    matrix = U[: dim // (2**num_ancilla), : dim // (2**num_ancilla)]
    return matrix


def generate_random_hermitian(key, dim):
    """Generate a random Hermitian matrix of size n x n with norm bounded by one."""
    A = random.normal(key, (dim, dim)) + 1j * random.normal(key, (dim, dim))
    A = (A + A.conj().T) / 2
    # Normalize the matrix to have norm bounded by 1
    norm = jnp.linalg.norm(A, ord=2)
    A = A / norm / 5
    return A


def halmos_dilation(A):
    """Construct the Halmos dilation of a Hermitian matrix A."""
    dim = A.shape[0]
    # matrix square root
    sqrt_A = jax.scipy.linalg.sqrtm(jnp.eye(dim) - A @ A)
    U = jnp.block([[A, sqrt_A], [sqrt_A, -A]])
    return U


def random_halsmos_dilation(key, dim):
    """Generate a random Halmos dilation of size 2N x 2N."""
    A = generate_random_hermitian(key, dim)
    U = halmos_dilation(A)
    # scramble the other blocks
    key, key1, key2 = random.split(key, 3)
    U1 = generate_random_unitary(key1, dim)
    U2 = generate_random_unitary(key2, dim)
    U = (
        jnp.block([[jnp.eye(dim), jnp.zeros((dim, dim))], [jnp.zeros((dim, dim)), U2]])
        @ U
        @ jnp.block(
            [[jnp.eye(dim), jnp.zeros((dim, dim))], [jnp.zeros((dim, dim)), U1]]
        )
    )
    return U


def hermitian_block_encoding(U):
    """
    Get the Hermitian unitary block encoding from any unitary block encoding U.
    Appendix C of https://arxiv.org/pdf/2002.11649

    We are safe to do this even with quantum oracle sketching,
    since we can implement the c^0U and c^1U† unitaries simultaneously
    using the same samples.
    This observation also applies to LCU constructions.
    For example, in taking the real part of a block encoding,
    or in constructing sin and cos block encodings from phase oracles.
    """
    hadamard = jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2)
    zero_to_one = hadamard @ jnp.array([[0, 0], [1, 0]]) @ hadamard
    one_to_zero = hadamard @ jnp.array([[0, 1], [0, 0]]) @ hadamard
    V = jnp.kron(zero_to_one, U) + jnp.kron(one_to_zero, U.conj().T)
    return V


"""
Tests
"""


if __name__ == "__main__":
    dim = 4
    key = random.PRNGKey(0)
    U = random_halsmos_dilation(key, dim)
    A = get_block_encoded(U)

    # test unitary
    print(
        jnp.isclose(jnp.linalg.norm(U @ U.conj().T - jnp.eye(2 * dim)), 0, atol=1e-2)
    )  # Should be close to 0

    # test Hermitian
    print(jnp.isclose(jnp.linalg.norm(A - A.conj().T), 0))  # Should be close to 0

    # test norm bounded by one
    print(jnp.isclose(jnp.linalg.norm(A, ord=2), 1))  # Should be close to 1 or less

    # dilation twice and than take block-encoding should give back the original matrix
    A2 = get_block_encoded(halmos_dilation(halmos_dilation(A)), num_ancilla=2)
    print(jnp.isclose(jnp.linalg.norm(A - A2), 0))  # Should be close to 0

    # test hermitian block encoding
    V = hermitian_block_encoding(U)
    # unitary
    print(
        jnp.isclose(jnp.linalg.norm(V @ V.conj().T - jnp.eye(4 * dim)), 0, atol=1e-2)
    )  # Should be close to 0
    # hermitian
    print(jnp.isclose(jnp.linalg.norm(V - V.conj().T), 0))  # Should be close to 0
    # block encoding
    A3 = get_block_encoded(V, num_ancilla=2)
    print(jnp.isclose(jnp.linalg.norm(A - A3), 0, atol=1e-5))  # Should be close to 0

import jax
import jax.numpy as jnp
import pyqsp
from jax import random
from pyqsp import angle_sequence

import utils


def get_qsvt_angles(
    func,
    degree,
    rescale,
):
    """
    Get QSVT angles for a given target function.

    Args:
        func: target function to approximate
        degree: degree of the polynomial approximation
        rescale: scaling factor to ensure the function is bounded within [-1, 1]

    Returns:
        angles: array of QSVT angles
    """
    poly = pyqsp.poly.PolyTaylorSeries().taylor_series(
        func=func,
        degree=degree,
        max_scale=rescale,
        chebyshev_basis=True,
        cheb_samples=2 * degree,
    )

    (phi_set, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
        poly, method="sym_qsp", chebyshev_basis=True
    )

    # Eq. (15) in https://arxiv.org/pdf/2002.11649
    phi_to_angle = (
        jnp.array([1 / 4] + [1 / 2] * (phi_set.shape[0] - 2) + [1 / 4]) * jnp.pi
    )
    angle_set = phi_set + phi_to_angle

    return angle_set


def qsp_op(angle, dim, num_ancilla=2):
    """QSP operator for a given rotation angle. dim is the dimension of the block encoded matrix."""
    mask = jnp.array([1] + [-1] * (2**num_ancilla - 1))
    angle = jnp.ones(2**num_ancilla) * angle
    angle = angle * mask
    op = jnp.diag(jnp.exp(1j * angle))
    return jnp.kron(op, jnp.eye(dim))


def apply_qsvt(U, num_ancilla, angle_set):
    """
    Apply QSVT to a Hermitian unitary U that block encodes some matrix using the given angles.

    Args:
        U: a Hermitian unitary
        num_ancilla: number of ancilla qubits in U
        angle_set: array of QSVT angles

    Returns:
        V: unitary of the QSVT circuit
    """

    # dimension of the block encoded matrix
    dim = U.shape[-1] // (2**num_ancilla)

    # test Hermitian
    assert jnp.allclose(U, U.conj().T), "U is not Hermitian"

    circ = jnp.exp(1j * (-jnp.pi / 2) * (angle_set.shape[0])) * qsp_op(
        angle_set[0], dim, num_ancilla=num_ancilla
    )

    for angle in angle_set[1:]:
        circ = circ @ V @ qsp_op(angle, dim, num_ancilla=num_ancilla)

    return circ


# tests
if __name__ == "__main__":

    def func(x):
        return 2 * x**2 - 1

    dim = 100
    polydeg = 12
    rescale = 0.5

    def scaled_func(x):
        return rescale * func(x)

    print("Testing QSVT...")

    angle_set = get_qsvt_angles(
        func=scaled_func,
        degree=polydeg,
        rescale=rescale,
    )

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    U = utils.random_halsmos_dilation(subkey, dim)
    V = utils.hermitian_block_encoding(U)
    A = utils.get_block_encoded(V, num_ancilla=2)

    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)

    eigvals_target = jnp.sort(scaled_func(eigvals))

    print("QSVT approximation error:", jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)))

    assert jnp.allclose(eigvals_qsvt, eigvals_target, atol=1e-1)

    print("All tests passed.")

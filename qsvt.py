from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pyqsp
from jax import random
from pyqsp import angle_sequence

import utils

jax.config.update("jax_enable_x64", True)


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
        angle_set: array of QSVT angles
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

    assert isinstance(phi_set, np.ndarray), "Failed to compute QSVT angles."

    # Eq. (15) in https://arxiv.org/pdf/2002.11649
    phi_to_angle = (
        jnp.array([1 / 4] + [1 / 2] * (phi_set.shape[0] - 2) + [1 / 4]) * jnp.pi
    )
    angle_set = phi_set + phi_to_angle

    return angle_set


def get_qsvt_angles_inverse(
    kappa,
    epsilon=0.1,
):
    """
    Get QSVT angles for approximating the inverse function.

    Args:
        kappa: condition number
        epsilon: approximation error

    Returns:
        angle_set: array of QSVT angles
        scale: scaling factor
    """
    poly = pyqsp.poly.PolyOneOverX()
    pcoefs, scale = poly.generate(
        kappa=kappa,
        epsilon=epsilon,
        return_coef=True,
        ensure_bounded=True,
        return_scale=True,
        chebyshev_basis=True,
    )

    (phi_set, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
        pcoefs, method="sym_qsp", chebyshev_basis=True
    )

    assert isinstance(phi_set, np.ndarray), "Failed to compute QSVT angles."

    # Eq. (15) in https://arxiv.org/pdf/2002.11649
    phi_to_angle = (
        jnp.array([1 / 4] + [1 / 2] * (phi_set.shape[0] - 2) + [1 / 4]) * jnp.pi
    )
    angle_set = phi_set + phi_to_angle

    return angle_set, scale


@partial(jax.jit, static_argnums=(1,))
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

    # diagonal phase signs of the QSP operator, angle not multiplied yet
    mask = jnp.concatenate([jnp.array([1.0]), -jnp.ones((2**num_ancilla) - 1)])
    qsp_op_phase_pattern = jnp.repeat(mask, dim)

    circ = jnp.exp(1j * (-jnp.pi / 2) * (angle_set.shape[0])) * jnp.diag(
        jnp.exp(1j * angle_set[0] * qsp_op_phase_pattern)
    )

    # apply QSP rotations and unitaries
    # Eq. (12) in https://arxiv.org/pdf/2002.11649
    # using diagonal element-wise multiplication for efficiency
    for angle in angle_set[1:]:
        circ = circ @ U * jnp.exp(1j * angle * qsp_op_phase_pattern)[None, :]

    return circ


@partial(jax.jit, static_argnums=(1,))
def apply_qsvt_imperfect(U_sequence, num_ancilla, angle_set):
    """
    Apply QSVT to a sequence of imperfect implementation of a Hermitian unitary U that block encodes some matrix using the given angles.

    Args:
        U_sequence: a sequence of imperfect implementations of a Hermitian unitary, shape (num_gates, dim * 2^num_ancilla, dim * 2^num_ancilla)
        num_ancilla: number of ancilla qubits in U
        angle_set: array of QSVT angles, shape (num_gates+1,)

    Returns:
        V: unitary of the QSVT circuit
    """

    # dimension of the block encoded matrix
    dim = U_sequence.shape[-1] // (2**num_ancilla)

    assert U_sequence.shape[0] == angle_set.shape[0] - 1, (
        "Number of imperfect gates must match number of QSVT angles minus one."
    )

    # diagonal phase signs of the QSP operator, angle not multiplied yet
    mask = jnp.concatenate([jnp.array([1.0]), -jnp.ones((2**num_ancilla) - 1)])
    qsp_op_phase_pattern = jnp.repeat(mask, dim)

    circ = jnp.exp(1j * (-jnp.pi / 2) * (angle_set.shape[0])) * jnp.diag(
        jnp.exp(1j * angle_set[0] * qsp_op_phase_pattern)
    )

    # apply QSP rotations and unitaries
    # Eq. (12) in https://arxiv.org/pdf/2002.11649
    # using diagonal element-wise multiplication for efficiency
    for angle, U in zip(angle_set[1:], U_sequence):
        circ = circ @ U * jnp.exp(1j * angle * qsp_op_phase_pattern)[None, :]

    return circ


def _test_qsvt(func):
    dim = 100
    polydeg = 12
    rescale = 0.5

    def scaled_func(x):
        return rescale * func(x)

    angle_set = get_qsvt_angles(
        func=scaled_func,
        degree=polydeg,
        rescale=rescale,
    )

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    U = utils.random_halsmos_dilation(subkey, dim).astype(jnp.complex128)
    V = utils.hermitian_block_encoding(U)
    A = utils.get_block_encoded(V, num_ancilla=2)

    start_time = time.time()
    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    end_time = time.time()
    print(f"Time for QSVT application: {end_time - start_time} seconds")

    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)

    eigvals_target = jnp.sort(scaled_func(eigvals))

    print("QSVT approximation error:", jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)))

    assert jnp.allclose(eigvals_qsvt, eigvals_target, atol=1e-4)


def _test_qsvt_imperfect(func, noise_level=0.001):
    print(f"Testing imperfect QSVT with noise level: {noise_level}")

    dim = 100
    polydeg = 6
    rescale = 0.5

    def scaled_func(x):
        return rescale * func(x)

    angle_set = get_qsvt_angles(
        func=scaled_func,
        degree=polydeg,
        rescale=rescale,
    )

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    U = utils.random_halsmos_dilation(subkey, dim).astype(jnp.complex128)
    A = utils.get_block_encoded(U, num_ancilla=1)

    # generate imperfect implementations of U with noisy phases
    num_gates = angle_set.shape[0] - 1
    V_sequence = []
    for i in range(num_gates):
        key, subkey = random.split(key)
        noise = random.normal(subkey, (U.shape[0],)) * noise_level
        noisy_phase = jnp.exp(1j * noise)
        U_noisy = U * noisy_phase[None, :]
        V_sequence.append(utils.hermitian_block_encoding(U_noisy))

    V_sequence = jnp.stack(V_sequence, axis=0)

    start_time = time.time()
    V_qsvt = apply_qsvt_imperfect(V_sequence, num_ancilla=2, angle_set=angle_set)
    end_time = time.time()
    print(f"Time for imperfect QSVT application: {end_time - start_time} seconds")

    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)

    eigvals_target = jnp.sort(scaled_func(eigvals))

    print(
        "Imperfect QSVT approximation error:",
        jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)),
    )

    print(
        "Imperfect QSVT approximation relative error:",
        jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)) / noise_level,
    )

    assert (
        jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)) / noise_level < 1.0
        or noise_level < 1e-5
    )


def _test_inverse():
    dim = 5
    kappa = 5
    epsilon = 0.1

    print("Testing QSVT with 1/x function...")
    print(f"Condition number: {kappa}, Approximation error: {epsilon}")

    angle_set, scale = get_qsvt_angles_inverse(
        kappa=kappa,
        epsilon=epsilon,
    )

    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    U = utils.random_halsmos_dilation(subkey, dim).astype(jnp.complex128)
    V = utils.hermitian_block_encoding(U)
    A = utils.get_block_encoded(V, num_ancilla=2)

    start_time = time.time()
    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    end_time = time.time()
    print(f"Time for inverse QSVT application: {end_time - start_time} seconds")

    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)

    eigvals_target = jnp.sort(scale / eigvals)

    mask = jnp.abs(eigvals_target) < 1.0
    eigvals_qsvt = eigvals_qsvt * mask
    eigvals_target = eigvals_target * mask

    print(
        "Inverse QSVT approximation error:",
        jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)),
    )

    assert jnp.allclose(eigvals_qsvt, eigvals_target, atol=1e-2)


if __name__ == "__main__":

    def func_even(x):
        return 2 * x**2 - 1

    def func_even1(x):
        return x**2

    def func_odd(x):
        return 4 * x**3 - 3 * x

    # time profiling
    import time

    print("Testing QSVT...")

    print("---" * 10)
    print("Testing even function...")
    _test_qsvt(func_even)

    print("---" * 10)
    print("Testing imperfect QSVT...")
    _test_qsvt_imperfect(func_even)

    print("---" * 10)
    print("Testing another even function...")
    _test_qsvt(func_even1)

    print("---" * 10)
    print("Testing imperfect QSVT...")
    _test_qsvt_imperfect(func_even1)

    print("---" * 10)
    print("Testing odd function...")
    _test_qsvt(func_odd)

    print("---" * 10)
    print("Testing imperfect QSVT...")
    _test_qsvt_imperfect(func_odd)

    print("---" * 10)
    print("Testing imperfect QSVT with no noise...")
    _test_qsvt_imperfect(func_odd, noise_level=0.0)

    print("---" * 10)
    print("Testing imperfect QSVT with higher noise...")
    _test_qsvt_imperfect(func_odd, noise_level=0.1)

    print("---" * 10)
    print("Testing 1/x function...")
    _test_inverse()

    print("---" * 10)

    print("All tests passed.")

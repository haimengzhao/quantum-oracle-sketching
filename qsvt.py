from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pyqsp
import scipy
from jax import random
from pyqsp import angle_sequence
from pyqsp.poly import PolyGenerator

import utils

jax.config.update("jax_enable_x64", True)


class PolyTaylorSeries(PolyGenerator):
    """
    Base class for PolySign and PolyThreshold,
    modified from pyqsp.poly.PolyTaylorSeries to include cheb_domain parameter,
    which specifies the domain over which the Chebyshev fit is to be performed.
    """

    def taylor_series(
        self,
        func,
        degree,
        ensure_bounded=True,
        return_scale=False,
        npts=100,
        max_scale=0.9,
        chebyshev_basis=False,
        cheb_samples=20,
        cheb_domain=(-1, 1),  # ADDED
    ):
        """
        If chebyshev_basis is True:
            Return numpy Chebyshev approximation for func, using numpy methods for Chebyshev approximation of specified degree.
            We also evaluate the mean absolute difference on equispaced points over the interval [-1,1].

        If chebyshev_basis is False:
            Return numpy Polynomial approximation for func, constructed using
            taylor series, of specified degree.
            We also evaluate the mean absolute difference on equispaced points over the interval [-1,1].
        """

        # Note: PolyTaylorSeries now no longer generates approximating Taylor polynomials, but only Chebyshev interpolations as contained in the assured branch indicated below. This exhibits better stability and convergence.

        cheb_samples = (
            2 * degree
        )  # Set to prevent aliasing; note that all methods calling TaylorSeries implicitly have their cheb_samples specifications overruled here.
        # Generate x and y values for fit according to func; note use of chebyshev nodes of the first kind.
        samples = np.polynomial.chebyshev.chebpts1(cheb_samples)
        scale = 1.0  # Binding variable.

        vals = np.array(list(map(func, samples)))

        # ADDED
        mask = (samples >= cheb_domain[0]) & (samples <= cheb_domain[1])

        # Generate cheb fit for function.
        # ORIGINAL: cheb_coefs = np.polynomial.chebyshev.chebfit(samples, vals, degree)
        cheb_coefs = np.polynomial.chebyshev.chebfit(samples, vals, degree, w=mask)

        # Generate chebyshev polynomial object from coefs.
        cheb_poly = np.polynomial.chebyshev.Chebyshev(cheb_coefs)

        # Determine maximum over interval and rescale.
        if ensure_bounded:
            # Minimize polynomial and negative of polynomial to find overall bound on absolute value.
            res_1 = scipy.optimize.minimize(
                -1 * cheb_poly, (0.1,), bounds=[cheb_domain]
            )
            res_2 = scipy.optimize.minimize(cheb_poly, (0.1,), bounds=[cheb_domain])
            pmax_1 = res_1.x[0]
            pmax_2 = res_2.x[0]

            # Compute the smaller of the two rescaling values.
            arg_array = np.array([pmax_1, pmax_2])
            max_index = np.argmax([abs(cheb_poly(pmax_1)), abs(cheb_poly(pmax_2))])
            scale = 1.0 / np.max([abs(cheb_poly(pmax_1)), abs(cheb_poly(pmax_2))])

            # Compute overal rescaling factor and apply to poly approx.
            scale = scale * max_scale
            print(
                f"[PolyTaylorSeries] (Cheb) max {scale} is at {arg_array[max_index]}: normalizing"
            )
            cheb_poly = scale * cheb_poly

        # Determine average error on interval and print.
        adat = np.linspace(cheb_domain[0], cheb_domain[1], npts)
        pdat = cheb_poly(adat)
        edat = scale * func(adat)  # Compare to rescaled function.
        avg_err = abs(edat - pdat).mean()
        print(
            f"[PolyTaylorSeries] (Cheb) average error = {avg_err} in the domain [{cheb_domain[0]}, {cheb_domain[1]}] using degree {degree}"
        )

        if ensure_bounded and return_scale:
            return cheb_poly, scale
        else:
            return cheb_poly


def get_qsvt_angles(func, degree, rescale, cheb_domain=(-1, 1)):
    """
    Get QSVT angles for a given target function.

    Args:
        func: target function to approximate
        degree: degree of the polynomial approximation
        rescale: scaling factor to ensure the function is bounded within [-1, 1]
        cheb_domain: domain over which the Chebyshev fit is to be performed, default is (-1, 1)
    Returns:
        angle_set: array of QSVT angles
    """
    poly = PolyTaylorSeries().taylor_series(
        func=func,
        degree=degree,
        max_scale=rescale,
        ensure_bounded=False,
        chebyshev_basis=True,
        cheb_samples=100 * degree,
        cheb_domain=cheb_domain,
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


def get_qsvt_angles_inverse(kappa, epsilon=0.1):
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


def get_qsvt_angles_sign(degree, threshold=0.1, rescale=0.9):
    """
    Get QSVT angles for approximating the sign function.

    Args:
        degree: degree of the polynomial approximation
        threshold: equals one when |x| >= threshold
        rescale: scaling factor to ensure the function is bounded within [-1, 1]
    Returns:
        angle_set: array of QSVT angles
    """

    poly = pyqsp.poly.PolySign()
    pcoefs, scale = poly.generate(
        degree=degree,
        delta=np.ceil(2.0 / threshold),
        ensure_bounded=True,
        return_scale=True,
        chebyshev_basis=True,
        max_scale=rescale,
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

    # apply QSP rotations and unitaries
    # Eq. (12) in https://arxiv.org/pdf/2002.11649
    # using diagonal element-wise multiplication for efficiency
    # we add an additional -i phase to turn imaginary parts into real parts
    # so that the polynomial is implemented as the real part of the block encoding

    circ = jnp.exp(1j * (-jnp.pi / 2) * (angle_set.shape[0])) * jnp.diag(
        jnp.exp(1j * angle_set[0] * qsp_op_phase_pattern)
    )

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


"""
Tests
"""


def _test_qsvt(func):
    dim = 100
    polydeg = 10
    rescale = 0.9

    def scaled_func(x):
        return rescale * func(x)

    angle_set = get_qsvt_angles(
        func=func,
        degree=polydeg,
        rescale=rescale,
    )

    print("QSVT angle set length:", angle_set.shape[0])

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    U = utils.random_halsmos_dilation(subkey, dim).astype(jnp.complex128)
    V = utils.hermitian_block_encoding(U)
    A = utils.get_block_encoded(V, num_ancilla=2)

    start_time = time.time()
    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    end_time = time.time()
    print(f"Time for QSVT application: {end_time - start_time:.3e} seconds")

    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)

    eigvals_target = jnp.sort(scaled_func(eigvals))

    print(
        f"QSVT approximation error: {jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)):.3e}"
    )

    assert jnp.allclose(eigvals_qsvt, eigvals_target, atol=1e-4)


def _test_qsvt_imperfect(func, noise_level=0.001):
    print(f"Testing imperfect QSVT with noise level: {noise_level:.3e}")

    dim = 100
    polydeg = 10
    rescale = 0.5

    def scaled_func(x):
        return rescale * func(x)

    angle_set = get_qsvt_angles(
        func=func,
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
    print(f"Time for imperfect QSVT application: {end_time - start_time:.3e} seconds")

    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)

    eigvals_target = jnp.sort(scaled_func(eigvals))

    print(
        f"Imperfect QSVT approximation error: {jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)):.3e}",
    )

    print(
        f"Imperfect QSVT approximation relative error: {jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)) / noise_level:.3e}",
    )

    assert (
        jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)) / noise_level < 1.0
        or noise_level < 1e-5
    )


def _test_inverse():
    dim = 100
    kappa = 5
    epsilon = 0.01

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
    print(f"Time for inverse QSVT application: {end_time - start_time:.3e} seconds")

    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)

    eigvals_target = jnp.sort(scale / eigvals)

    # remove outliers that are smaller than 1/kappa
    num_outliers = jnp.sum(jnp.abs(eigvals) < 1.0 / kappa)
    error = jnp.abs(eigvals_qsvt - eigvals_target)
    error = jnp.sort(error)[:-num_outliers]  # remove outliers

    print(
        f"Inverse QSVT approximation error: {jnp.max(error):.3e}",
    )

    assert jnp.allclose(error, 0, atol=10 * epsilon)


def _test_sign():
    dim = 100
    threshold = 0.1
    degree = 51
    scale = 0.9

    print("Testing QSVT with sign function...")
    print(f"Threshold: {threshold}, Polynomial degree: {degree}")

    angle_set, _ = get_qsvt_angles_sign(
        threshold=threshold,
        degree=degree,
        rescale=scale,
    )

    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    U = utils.random_halsmos_dilation(subkey, dim).astype(jnp.complex128)
    V = utils.hermitian_block_encoding(U)
    A = utils.get_block_encoded(V, num_ancilla=2)

    start_time = time.time()
    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    end_time = time.time()
    print(f"Time for sign QSVT application: {end_time - start_time:.3e} seconds")

    A_qsvt = utils.get_block_encoded(V_qsvt, num_ancilla=2)

    # test QSVT approximation
    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)
    eigvals_target = jnp.sort(scale * jnp.sign(eigvals))

    # remove outliers that are smaller than threshold
    num_outliers = jnp.sum(jnp.abs(eigvals) < 1.5 * threshold)
    error = jnp.abs(eigvals_qsvt - eigvals_target)
    error = jnp.sort(error)[:-num_outliers]  # remove outliers

    print(
        f"Sign QSVT approximation error: {jnp.max(error):.3e}",
    )

    assert jnp.allclose(error, 0, atol=1e-2)


if __name__ == "__main__":

    def func_even(x):
        return 2 * x**2 - 1

    def func_even1(x):
        return x**2

    def func_odd(x):
        return np.arcsin(np.sin(1.0) * x)

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
    print("Testing sign function...")
    _test_sign()

    print("---" * 10)
    print("All tests passed.")

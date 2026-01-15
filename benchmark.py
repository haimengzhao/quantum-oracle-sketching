from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from matplotlib.lines import Line2D
from tqdm import tqdm

import qos
import qsvt
import utils

# Publication-style plotting defaults (serif + larger sizes).
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (3.5, 2.5),
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

# arcsin QSVT for general state sketching
arcsin_degree = 20
arcsin_angle_set = qsvt.get_qsvt_angles(
    func=lambda x: jnp.arcsin(x) / jnp.arcsin(1),
    degree=arcsin_degree,
    rescale=1.0,
    cheb_domain=(-jnp.sin(1), jnp.sin(1)),
    ensure_bounded=False,
    parity=1,
)
target_norm = 1 / (jnp.arcsin(1) * 5)

# vectorized versions of qos functions
q_state_sketch_vectorized = jax.vmap(
    partial(qos.q_state_sketch, angle_set=arcsin_angle_set, degree=arcsin_degree),
    in_axes=(0, 0, None),
    out_axes=(0, 0),
)

q_state_sketch_flat_vectorized = jax.vmap(
    qos.q_state_sketch_flat, in_axes=(0, None), out_axes=(0, 0)
)

q_oracle_sketch_boolean_vectorized = jax.vmap(
    qos.q_oracle_sketch_boolean, in_axes=(0, None), out_axes=(0, 0)
)

q_oracle_sketch_matrix_element_vectorized = jax.vmap(
    qos.q_oracle_sketch_matrix_element, in_axes=(0, None), out_axes=(0, 0)
)


def benchmark_random_vector(key, dim, unit_num_samples, repetition=10):
    key, subkey = random.split(key)
    vector = utils.random_unit_vector(subkey, dim, batch_size=repetition)

    keys = random.split(key, repetition)
    state_sketch, num_samples = q_state_sketch_vectorized(
        vector, keys, unit_num_samples
    )
    norm = jnp.linalg.norm(state_sketch, axis=-1)

    error = jnp.linalg.norm(state_sketch - vector * target_norm, axis=-1)
    error_mean = jnp.mean(error)
    error_std = jnp.std(error) / jnp.sqrt(repetition)

    norm_error = jnp.abs(norm - target_norm)
    norm_error_mean = jnp.mean(norm_error)
    norm_error_std = jnp.std(norm_error) / jnp.sqrt(repetition)

    num_samples = jnp.mean(num_samples)

    return {
        "error_mean": error_mean,
        "error_std": error_std,
        "norm_error_mean": norm_error_mean,
        "norm_error_std": norm_error_std,
        "num_samples": num_samples,
    }


def benchmark_random_flat_vector(key, dim, unit_num_samples, repetition=10):
    key, subkey = random.split(key)
    vector = utils.random_flat_vector(subkey, dim, batch_size=repetition)

    state_sketch, num_samples = q_state_sketch_flat_vectorized(vector, unit_num_samples)

    error = jnp.linalg.norm(
        state_sketch - vector / jnp.linalg.norm(vector, axis=-1, keepdims=True), axis=-1
    )
    error_mean = jnp.mean(error)
    error_std = jnp.std(error) / jnp.sqrt(repetition)
    num_samples = jnp.mean(num_samples)

    return {
        "error_mean": error_mean,
        "error_std": error_std,
        "num_samples": num_samples,
    }


def benchmark_random_boolean_function(key, dim, unit_num_samples, repetition=10):
    key, subkey = random.split(key)
    truth_table = random.randint(
        key, (repetition, dim), minval=0, maxval=2, dtype=utils.int_dtype
    )

    oracle_sketch, num_samples = q_oracle_sketch_boolean_vectorized(
        truth_table, unit_num_samples
    )

    target_diag = jnp.exp(1j * jnp.pi * truth_table)
    error = jnp.max(jnp.abs(oracle_sketch - target_diag), axis=-1)
    error_mean = jnp.mean(error)
    error_std = jnp.std(error) / jnp.sqrt(repetition)
    num_samples = jnp.mean(num_samples)

    return {
        "error_mean": error_mean,
        "error_std": error_std,
        "num_samples": num_samples,
    }


def benchmark_random_sparse_matrix_element(
    key, dim, nnz, unit_num_samples, repetition=10
):
    key, subkey = random.split(key)
    sparse_matrix = utils.random_sparse_matrix_constant_magnitude(
        subkey, (dim, dim), nnz=nnz, magnitude=1, batch_size=repetition
    )

    oracle_sketch, num_samples = q_oracle_sketch_matrix_element_vectorized(
        sparse_matrix, unit_num_samples
    )

    target_oracle = sparse_matrix.reshape(repetition, -1)

    # operator norm error of diagonal matrices
    error = jnp.max(jnp.abs(oracle_sketch - target_oracle), axis=-1)
    error_mean = jnp.mean(error)
    error_std = jnp.std(error) / jnp.sqrt(repetition)
    num_samples = jnp.mean(num_samples)

    return {
        "error_mean": error_mean,
        "error_std": error_std,
        "num_samples": num_samples,
    }


def benchmark(
    key,
    benchmark_function,
    dim_list,
    unit_num_samples_list,
    repetition,
    matrix_dim=None,
    verbose=False,
):
    results = {}
    for dim in tqdm(dim_list, desc="Dimensions", position=0):
        results[int(dim)] = {}
        for unit_num_samples in tqdm(
            unit_num_samples_list, desc="Num Samples", position=1, leave=False
        ):
            key, subkey = random.split(key)
            if matrix_dim is not None:
                # when benchmarking sparse matrix
                # dim here refers to nnz
                res = benchmark_function(
                    subkey, matrix_dim, dim, unit_num_samples, repetition
                )
            else:
                res = benchmark_function(subkey, dim, unit_num_samples, repetition)
            results[int(dim)][int(unit_num_samples)] = res

            if verbose:
                tqdm.write(", ".join([f"{f}: {res[f]:3e}" for f in res.keys()]))
    return results


def plot_benchmark_results(
    results, title, dim_list=None, matrix=False, fit=None, save_path=None, show=False
):
    plt.figure(figsize=(4, 3))

    color_counter = 0
    for dim, dim_results in results.items():
        if dim_list is not None and dim not in dim_list:
            continue

        num_samples_list = []
        error_mean_list = []
        error_std_list = []
        for unit_num_samples, res in dim_results.items():
            num_samples_list.append(res["num_samples"])
            error_mean_list.append(res["error_mean"])
            error_std_list.append(res["error_std"])
        plt.errorbar(
            num_samples_list,
            error_mean_list,
            yerr=error_std_list,
            fmt="o",
            label=rf"$N = {dim}$" if not matrix else f"$N_{{nnz}} = {dim}$",
            color=rf"C{color_counter}",
            capsize=5,
        )
        if fit is not None:
            # Plot fitted curve for this dimension
            num_samples_fit = jnp.array(num_samples_list)
            error_fit = (
                fit["C"] * (dim ** fit["alpha"]) / (num_samples_fit ** fit["beta"])
            )
            plt.plot(
                num_samples_fit,
                error_fit,
                "-",
                color=f"C{color_counter}",
            )
            color_counter += 1

    fit_handles = None
    if fit is not None:
        if matrix:
            label_str = rf"Fit: $M = {fit['C']:.1f} N_{{nnz}}^{{{fit['alpha']:.2f}}}/\epsilon^{{{fit['beta']:.2f}}}$"
        else:
            label_str = rf"Fit: $M = {fit['C']:.1f} N^{{{fit['alpha']:.2f}}}/\epsilon^{{{fit['beta']:.2f}}}$"
        rmse_str = rf"RMS rel. err.: ${fit['rmse'] * 100:.1f}\%$"
        fit_handles = [
            Line2D([], [], color="grey", linestyle="-", label=label_str),
            Line2D([], [], color="none", label=rmse_str),
        ]

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Number of samples $M$")
    plt.ylabel(r"Error $\epsilon$")
    plt.title(title)
    # 1. Turn on Major Grid (Darker)
    plt.grid(True, which="major", linestyle="-", linewidth=0.8, color="gray", alpha=0.4)
    # 2. Turn on Minor Grid (Lighter & Subtle)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
    data_legend = plt.legend(loc="upper right")
    if fit_handles is not None:
        plt.gca().add_artist(data_legend)
        plt.legend(handles=fit_handles, loc="lower left", handlelength=2)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def fit_sample_complexity(results):
    """
    Fits the sample complexity ansatz: num_samples = C * dim^alpha / error^beta
    using linear least squares in log-space.
    """
    num_samples = []
    dims = []
    errors = []
    for dim, dim_results in results.items():
        for unit_num_samples, res in dim_results.items():
            num_samples.append(res["num_samples"])
            dims.append(dim)
            errors.append(res["error_mean"])
    # 1. Log-transform the data
    ln_sample = jnp.log(jnp.array(num_samples))
    ln_dim = jnp.log(jnp.array(dims))
    ln_error = jnp.log(jnp.array(errors))

    # 2. Construct Design Matrix A for equation: ln(num_samples) = ln(C) + alpha*ln(dim) - beta*ln(error)
    # Form: [1, ln(dim), ln(error)]
    ones = jnp.ones_like(ln_sample)
    A = jnp.stack([ones, ln_dim, ln_error], axis=1)

    # 3. Solve A * x = ln_sample via SVD-based pseudo-inverse
    # x = [ln(C), alpha, -beta]
    coeffs, residuals, rank, s = jnp.linalg.lstsq(A, ln_sample)

    ln_C, alpha, neg_beta = coeffs

    cov_matrix = jnp.linalg.pinv(A.T @ A) * residuals[0] / (A.shape[0] - A.shape[1])
    std = jnp.sqrt(jnp.diag(cov_matrix))
    ln_C_std, alpha_std, neg_beta_std = std

    # 4. Extract physical parameters
    fit = {
        "C": jnp.exp(ln_C),
        "alpha": alpha,
        "beta": -neg_beta,  # Note the sign flip because term is -beta*ln(error)
        "rmse": jnp.sqrt(residuals / (A.shape[0] - A.shape[1]))[0],
        "C_std": jnp.exp(ln_C) * ln_C_std,
        "alpha_std": alpha_std,
        "beta_std": neg_beta_std,
    }

    print("Fitted sample complexity parameters:")
    print(
        f"C: {fit['C']:.3e} ± {fit['C_std']:.3e}, alpha: {fit['alpha']:.3f} ± {fit['alpha_std']:.3e}, beta: {fit['beta']:.3f} ± {fit['beta_std']:.3e}, rmse: {fit['rmse']:.3e}"
    )

    return fit


if __name__ == "__main__":
    key = random.PRNGKey(42)

    # 1. benchmark flat vector quantum state sketch
    print("-" * 40)
    print("Benchmarking quantum state sketch for flat vectors...")
    dim_list = 100 * jnp.arange(1, 11)
    unit_num_samples_list = (10 ** jnp.linspace(5, 8, num=10)).astype(int)
    repetition = 100
    print("Dimensions:", dim_list)
    print("Number of samples:", unit_num_samples_list)
    print("Repetitions:", repetition)

    results = benchmark(
        key, benchmark_random_flat_vector, dim_list, unit_num_samples_list, repetition
    )
    fit = fit_sample_complexity(results)

    plot_benchmark_results(
        results,
        "Quantum State Sketching: Flat Vector",
        dim_list=[100, 200, 500, 1000],
        fit=fit,
        save_path="benchmark_flat_vector.pdf",
    )

    # 2. benchmark general vector quantum state sketch
    print("-" * 40)
    print("Benchmarking quantum state sketch for general vectors...")
    # general quantum state sketch works internally by padding dimension to next power of 2
    # so to fit the sample complexity correctly we use dimensions that are powers of 2
    dim_list = 100 * jnp.arange(1, 11)
    unit_num_samples_list = (10 ** jnp.linspace(5, 8, num=10)).astype(int)
    repetition = 10
    print("Dimensions:", dim_list)
    print("Number of samples:", unit_num_samples_list)
    print("Repetitions:", repetition)

    results = benchmark(
        key, benchmark_random_vector, dim_list, unit_num_samples_list, repetition
    )
    fit = fit_sample_complexity(results)

    plot_benchmark_results(
        results,
        "Quantum State Sketching: General Vector",
        dim_list=[100, 200, 500, 1000],
        fit=fit,
        save_path="benchmark_general_vector.pdf",
    )

    # 3. benchmark boolean function oracle sketch
    print("-" * 40)
    print("Benchmarking oracle sketch for boolean functions...")
    dim_list = 100 * jnp.arange(1, 11)
    unit_num_samples_list = (10 ** jnp.linspace(5, 8, num=10)).astype(int)
    repetition = 100
    print("Dimensions:", dim_list)
    print("Number of samples:", unit_num_samples_list)
    print("Repetitions:", repetition)

    results = benchmark(
        key,
        benchmark_random_boolean_function,
        dim_list,
        unit_num_samples_list,
        repetition,
    )
    fit = fit_sample_complexity(results)

    plot_benchmark_results(
        results,
        "Quantum Oracle Sketching: Boolean Functions",
        dim_list=[100, 200, 500, 1000],
        fit=fit,
        save_path="benchmark_boolean_function.pdf",
    )

    # 4. benchmark sparse matrix element oracle sketch
    print("-" * 40)
    print("Benchmarking oracle sketch for sparse matrix element oracle...")
    dim = 100
    nnz_list = jnp.array([200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000])
    unit_num_samples_list = (10 ** jnp.linspace(5, 8, num=10)).astype(int)
    repetition = 200
    print(f"Dimensions (square matrix): {dim} x {dim}, NNZ: ", nnz_list)
    print("Number of samples:", unit_num_samples_list)
    print("Repetitions:", repetition)
    results = benchmark(
        key,
        benchmark_random_sparse_matrix_element,
        nnz_list,
        unit_num_samples_list,
        repetition,
        matrix_dim=dim,
    )
    fit = fit_sample_complexity(results)

    plot_benchmark_results(
        results,
        "Quantum Oracle Sketching: Sparse Matrix Element",
        dim_list=[200, 500, 1000, 2000],
        fit=fit,
        save_path="benchmark_matrix_element.pdf",
        matrix=True,
    )

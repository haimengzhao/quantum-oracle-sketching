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

arcsin_degree = 20
arcsin_angle_set = qsvt.get_qsvt_angles(
    func=lambda x: jnp.arcsin(x) / jnp.arcsin(1),
    degree=arcsin_degree,
    rescale=1.0,
    cheb_domain=(-jnp.sin(1), jnp.sin(1)),
    ensure_bounded=False,
    parity=1,
)

q_state_sketch_vectorized = jax.vmap(
    partial(qos.q_state_sketch, angle_set=arcsin_angle_set, degree=arcsin_degree),
    in_axes=(0, 0, None),
    out_axes=(0, 0),
)

q_state_sketch_flat_vectorized = jax.vmap(
    qos.q_state_sketch_flat, in_axes=(0, None), out_axes=(0, 0)
)


def benchmark_random_vector(key, dim, unit_num_samples, repetition=10):
    key, subkey = random.split(key)
    vector = utils.random_unit_vector(subkey, dim, batch_size=repetition)

    keys = random.split(key, repetition)
    state_sketch, num_samples = q_state_sketch_vectorized(
        vector, keys, unit_num_samples
    )
    norm = jnp.linalg.norm(state_sketch, axis=-1)
    target_norm = 1 / (jnp.arcsin(1) * 5)

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


def benchmark(key, benchmark_function, dim_list, unit_num_samples_list, repetition):
    results = {}
    for dim in tqdm(dim_list, desc="Dimensions", position=0):
        results[int(dim)] = {}
        for unit_num_samples in tqdm(
            unit_num_samples_list, desc="Num Samples", position=1, leave=False
        ):
            key, subkey = random.split(key)
            res = benchmark_function(subkey, dim, unit_num_samples, repetition)
            results[int(dim)][int(unit_num_samples)] = res
            tqdm.write(", ".join([f"{f}: {res[f]:3e}" for f in res.keys()]))
    return results


def plot_benchmark_results(
    results, title, dim_list=None, fit=None, save_path=None, show=False
):
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()

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
            fmt="o-",
            label=f"$N = {dim}$",
            color=f"C{color_counter}",
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
                "--",
                color=f"C{color_counter}",
            )
            color_counter += 1

    fit_handles = None
    if fit is not None:
        label_str = rf"Fit: $M = {fit['C']:.1f} N^{{{fit['alpha']:.2f}}}/\epsilon^{{{fit['beta']:.2f}}}$"
        rmse_str = rf"RMS rel. err.: ${fit['rmse'] * 100:.1f}\%$"
        fit_handles = [
            Line2D([], [], color="grey", linestyle="--", label=label_str),
            Line2D([], [], color="none", label=rmse_str),
        ]

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Number of samples $M$")
    plt.ylabel(r"Error $\epsilon$")
    plt.title(title)
    plt.grid(True)
    data_legend = ax.legend(loc="upper right")
    if fit_handles is not None:
        ax.add_artist(data_legend)
        ax.legend(handles=fit_handles, loc="lower left", handlelength=2)
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
    repetition = 100

    dim_list = 100 * jnp.arange(1, 11)
    unit_num_samples_list = (10 ** jnp.linspace(5, 8, num=10)).astype(int)

    results = benchmark(
        key, benchmark_random_flat_vector, dim_list, unit_num_samples_list, repetition
    )
    fit = fit_sample_complexity(results)

    plot_benchmark_results(
        results,
        "Quantum State Sketching: Flat Vector",
        dim_list=[100, 200, 500, 1000],
        fit=fit,
        save_path="flat_vector_benchmark.pdf",
    )

    # general quantum state sketch works internally by padding dimension to next power of 2
    # so to fit the sample complexity correctly we use dimensions that are powers of 2
    dim_list = jnp.array([64, 128, 256, 512, 1024, 2048])
    unit_num_samples_list = (10 ** jnp.linspace(5, 8, num=10)).astype(int)
    repetition = 10

    results = benchmark(
        key, benchmark_random_vector, dim_list, unit_num_samples_list, repetition
    )
    fit = fit_sample_complexity(results)

    plot_benchmark_results(
        results,
        "Quantum State Sketching: General Vector",
        dim_list=[128, 256, 512, 1024],
        fit=fit,
        save_path="general_vector_benchmark.pdf",
    )

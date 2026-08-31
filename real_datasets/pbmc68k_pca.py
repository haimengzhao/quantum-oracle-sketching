"""PBMC68k sweep: machine size vs PCA variance recovery.

Top singular direction of the normalized UMI matrix, computed after
rare-gene truncation (min_samples) or an oblivious feature sketch, lifted
back to the full gene space and scored by relative explained variance; see
sweep_utils for the shared pipeline.
"""

import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pbmc68k_utils
import sketch_utils
import sweep_utils
from scipy.sparse.linalg import svds
from tqdm import tqdm

np.random.seed(42)
sweep_utils.apply_plot_style()

min_samples_list = pbmc68k_utils.get_min_samples_sweep()
sketch_n_features = [
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    12288,
    16384,
    20000,
    24576,
    29491,
]
n_sketch_seeds = 5
rare_extra_markers = [-5, -10, -15, -20]


def load_data():
    tqdm.write("Loading PBMC68k dataset (all classes)...")
    X_full, _, label_names = pbmc68k_utils.load_pbmc68k_data(
        min_samples=1, normalize=True, binary=False
    )
    tqdm.write(f"Dataset shape: {X_full.shape}, Classes: {label_names}")
    return X_full


def get_pca_results_full():
    X_full = load_data()

    tqdm.write("Computing Top Singular Vector (Ground Truth)...")
    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]
    var_max = np.linalg.norm(X_full @ v_full) ** 2
    D_full = X_full.shape[1]

    results = {
        "min_samples": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovery": [],
    }

    tqdm.write("Sweeping min_samples for PBMC68k...")

    for min_samp in tqdm(min_samples_list, desc="min_samples Sweep"):
        X_trunc, gene_indices = pbmc68k_utils.filter_genes_by_frequency(
            X_full, min_samp
        )

        if X_trunc.shape[1] == 0:
            continue

        num_samples, feature_dim = X_trunc.shape
        sparsity = sketch_utils.max_sparsity(X_trunc)

        _, _, vt_trunc = svds(X_trunc.asfptype(), k=1)
        v_trunc = vt_trunc[0]

        v_lifted = np.zeros(D_full)
        v_lifted[gene_indices] = v_trunc
        v_lifted = v_lifted / np.linalg.norm(v_lifted)

        var_captured = np.linalg.norm(X_full @ v_lifted) ** 2

        results["min_samples"].append(min_samp)
        results["space_streaming"].append(feature_dim)
        results["space_sparse"].append(sketch_utils.matrix_nnz(X_trunc))
        results["space_quantum"].append(
            sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "pca")
        )
        results["variance_recovery"].append(var_captured / var_max)

    return results


def get_pca_results_sketch(sketch_seeds, mode):
    X_full = load_data()

    tqdm.write("Computing Top Singular Vector (Ground Truth)...")
    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]
    var_max = np.linalg.norm(X_full @ v_full) ** 2

    full_dim = X_full.shape[1]
    n_features_list = [k for k in sketch_n_features if k < full_dim] + [full_dim]

    results = {
        "n_features": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovery_by_seed": [],
    }

    tqdm.write(f"Sweeping {mode} sketch dimension for PBMC68k...")

    for n_features in tqdm(n_features_list, desc=f"{mode} Sweep"):
        seed_space_streaming = []
        seed_space_sparse = []
        seed_space_quantum = []
        seed_recoveries = []

        for seed in sketch_seeds:
            X_sketch, sketch_info = sketch_utils.sketch_features(
                X_full, n_features, mode, seed=seed
            )

            num_samples, feature_dim = X_sketch.shape
            sparsity = sketch_utils.max_sparsity(X_sketch)

            seed_space_streaming.append(feature_dim)
            seed_space_sparse.append(sketch_utils.matrix_nnz(X_sketch))
            seed_space_quantum.append(
                sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "pca")
            )

            if sketch_info is None:
                seed_recoveries.append(1.0)
            else:
                _, _, vt_sketch = svds(sketch_utils.svds_input(X_sketch), k=1)
                v_sketch = vt_sketch[0]
                v_lifted = sketch_utils.lift_sketched_vector(
                    v_sketch, sketch_info, mode
                )
                v_lifted = v_lifted / sketch_utils.lifted_vector_norm(v_lifted)
                var_captured = np.linalg.norm(X_full @ v_lifted) ** 2
                seed_recoveries.append(var_captured / var_max)

        results["n_features"].append(n_features)
        results["space_streaming"].append(seed_space_streaming)
        results["space_sparse"].append(seed_space_sparse)
        results["space_quantum"].append(seed_space_quantum)
        results["variance_recovery_by_seed"].append(seed_recoveries)

    return results


def run_analysis(load_file=None, mode=None, n_sketch_seeds=n_sketch_seeds):
    sketch_mode = sweep_utils.is_sketch_mode(mode)

    if load_file is None:
        if sketch_mode:
            sketch = sketch_utils.get_sketch(mode)
            print(f"Running PCA Analysis on {sketch.name}-sketched PBMC68k Dataset...")
            seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
            print(f"Averaging over random sketch seeds: {seeds}")
            results = get_pca_results_sketch(seeds, mode)
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="variance",
                dataset_name=f"PBMC68k ({sketch.name})",
                output_json=f"pbmc68k_{mode}_size_vs_variance.json",
                sketch_mode=True,
                param_name="n_features",
                raw_key="raw_data_by_n_features",
                mode=mode,
                seeds=seeds,
            )
        else:
            print("Running PCA Analysis on PBMC68k Dataset (Binary Classification)...")
            results = get_pca_results_full()
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="variance",
                dataset_name="PBMC68k (Binary)",
                output_json="pbmc68k_size_vs_variance.json",
                sketch_mode=False,
                param_name="min_samples",
                raw_key="raw_data_by_min_samples",
            )
    else:
        print(f"Loading analysis from {load_file}...")

    with open(load_file, "r") as f:
        data = json.load(f)
    sketch_utils.validate_truncation_data(data, mode, load_file)
    stats = sweep_utils.load_sweep_stats(
        data, "variance", rare_raw_key="raw_data_by_min_samples"
    )
    plot_results(stats, mode, sketch_mode)


def plot_results(stats, mode, sketch_mode):
    plt.figure(figsize=sweep_utils.FIGSIZE)
    ax = plt.gca()
    for k in sweep_utils.METHOD_KEYS:
        xm, xs, ym = sweep_utils.sort_by_space(
            stats[k]["metric_mean"], stats[k]["metric_sem"], stats[k]["space_mean"]
        )
        sweep_utils.plot_curve(
            ax,
            k,
            xm,
            xs,
            ym,
            label=(sketch_utils.streaming_label(mode) if k == "streaming" else None),
            show_all_markers=sketch_mode,
            extra_marker_indices=rare_extra_markers,
            fill_when_zero_err=False,
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if sketch_mode:
        streaming_label_x, streaming_label_y, streaming_label_ha = (
            (0.9, 0.45, "right") if mode == "jl" else (0.9, 0.62, "right")
        )
        plt.text(
            0.2,
            0.9,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        plt.text(
            streaming_label_x,
            streaming_label_y,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
            ha=streaming_label_ha,
        )
        plt.text(
            0.15,
            0.04,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
    else:
        plt.text(
            1,
            1e6,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            0.996,
            1.2e4,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            1,
            1.7e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    plt.yscale("log")
    plt.ylim(1e1, 1e7)
    plt.xlabel("Relative explained variance")

    if sketch_mode:
        plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        plt.xlim(-0.1, 1.1)
    else:
        plt.xticks(
            [0.92, 0.94, 0.96, 0.98, 1],
            ["92%", "94%", "96%", "98%", "100%"],
        )
        plt.xlim(0.915, 1.005)

    plt.tick_params(direction="in", which="both", top=False, right=True)
    ax.set_ylabel("Machine size")
    ax.tick_params(axis="y")
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Dimension reduction")
    plt.tight_layout()
    if sketch_mode:
        output_pdf = f"pbmc68k_{mode}_size_vs_variance.pdf"
    else:
        output_pdf = "pbmc68k_size_vs_variance.pdf"
    plt.savefig(output_pdf)
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    sweep_utils.sweep_main("PBMC68k Machine Size vs Variance Analysis", run_analysis)

"""Dorothea sweep: machine size vs LS-SVM classification accuracy.

Ridge classification (5-fold CV) on the binary drug-discovery features,
sweeping rare-feature truncation (min_df) or an oblivious feature sketch;
see sweep_utils for the shared pipeline.
"""

import json

import dorothea_utils
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import sketch_utils
import sweep_utils
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)
sweep_utils.apply_plot_style()

min_dfs = [
    1,
    2,
    5,
    12,
    19,
    26,
    34,
    43,
    55,
    69,
    82,
    90,
    97,
    114,
    128,
    143,
    164,
    197,
    226,
    240,
    308,
]
sketch_n_features = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
n_sketch_seeds = 5


def make_classifier():
    return RidgeClassifier(
        random_state=42, alpha=200, solver="auto", class_weight="balanced"
    )


def get_svm_results_full():
    print("Loading Dorothea data...")
    X_full, y_full = dorothea_utils.load_dorothea_data(valid=True)
    tqdm.write(f"Dataset shape: {X_full.shape}")

    # Pre-compute document frequencies for the min_df filter.
    print("Computing document frequencies...")
    X_bin = X_full.copy()
    X_bin.data[:] = 1
    doc_freqs = np.array(X_bin.sum(axis=0)).flatten()
    del X_bin

    results = {
        "min_dfs": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracy_scores": [],
    }

    tqdm.write("Sweeping min_df for Dorothea (SVM)...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep"):
        keep_indices = np.where(doc_freqs >= mdf)[0]

        if len(keep_indices) == 0:
            print(f"Skipping min_df={mdf} (0 features kept)")
            continue

        X_trunc = X_full[:, keep_indices]  # type: ignore

        num_samples, feature_dim = X_trunc.shape
        sparsity = sketch_utils.max_sparsity(X_trunc)

        scores = cross_val_score(make_classifier(), X_trunc, y_full, cv=5)

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(feature_dim)
        results["space_sparse"].append(sketch_utils.matrix_nnz(X_trunc))
        results["space_quantum"].append(
            sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "svm")
        )
        results["accuracy_scores"].append([float(s) for s in scores])

    return results


def get_svm_results_sketch(sketch_seeds, mode):
    print("Loading Dorothea data...")
    X_full, y_full = dorothea_utils.load_dorothea_data(valid=True)
    tqdm.write(f"Dataset shape: {X_full.shape}")

    full_dim = X_full.shape[1]
    n_features_list = [k for k in sketch_n_features if k < full_dim] + [full_dim]

    results = {
        "n_features": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracy_scores": [],
    }

    tqdm.write(f"Sweeping {mode} sketch dimension for Dorothea (SVM)...")

    for n_features in tqdm(n_features_list, desc=f"{mode} Sweep"):
        seed_space_streaming = []
        seed_space_sparse = []
        seed_space_quantum = []
        seed_accuracy_scores = []

        for seed in sketch_seeds:
            X_sketch, _ = sketch_utils.sketch_features(
                X_full, n_features, mode, seed=seed
            )

            num_samples, feature_dim = X_sketch.shape
            sparsity = sketch_utils.max_sparsity(X_sketch)

            seed_space_streaming.append(feature_dim)
            seed_space_sparse.append(sketch_utils.matrix_nnz(X_sketch))
            seed_space_quantum.append(
                sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "svm")
            )

            scores = cross_val_score(make_classifier(), X_sketch, y_full, cv=5)
            seed_accuracy_scores.append([float(s) for s in scores])

        results["n_features"].append(n_features)
        results["space_streaming"].append(seed_space_streaming)
        results["space_sparse"].append(seed_space_sparse)
        results["space_quantum"].append(seed_space_quantum)
        results["accuracy_scores"].append(seed_accuracy_scores)

    return results


def run_analysis(load_file=None, mode=None, n_sketch_seeds=n_sketch_seeds):
    sketch_mode = sweep_utils.is_sketch_mode(mode)

    if load_file is None:
        if sketch_mode:
            sketch = sketch_utils.get_sketch(mode)
            print(f"Running SVM Analysis on {sketch.name}-sketched Dorothea Dataset...")
            seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
            print(f"Averaging over random sketch seeds: {seeds}")
            results = get_svm_results_sketch(seeds, mode)
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="accuracy",
                dataset_name=f"Dorothea ({sketch.name})",
                output_json=f"dorothea_{mode}_size_vs_accuracy.json",
                sketch_mode=True,
                param_name="n_features",
                raw_key="raw_data_by_n_features",
                mode=mode,
                seeds=seeds,
            )
        else:
            print("Running SVM Analysis on Dorothea Dataset...")
            results = get_svm_results_full()
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="accuracy",
                dataset_name="Dorothea",
                output_json="dorothea_size_vs_accuracy.json",
                sketch_mode=False,
                param_name="min_dfs",
                raw_key="raw_data_by_min_df",
                rare_scores_key="accuracy_scores",
            )
    else:
        print(f"Loading analysis from {load_file}...")

    with open(load_file, "r") as f:
        data = json.load(f)
    sketch_utils.validate_truncation_data(data, mode, load_file)
    stats = sweep_utils.load_sweep_stats(data, "accuracy")
    plot_results(stats, mode, sketch_mode)


def plot_results(stats, mode, sketch_mode):
    plt.figure(figsize=sweep_utils.FIGSIZE)
    ax = plt.gca()
    for k in sweep_utils.METHOD_KEYS:
        xm, xs, ym = sweep_utils.sort_by_space(
            stats[k]["metric_mean"], stats[k]["metric_sem"], stats[k]["space_mean"]
        )
        ind = xm >= 0.6
        sweep_utils.plot_curve(
            ax,
            k,
            xm[ind],
            xs[ind],
            ym[ind],
            label=(sketch_utils.streaming_label(mode) if k == "streaming" else None),
            show_all_markers=True,
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if sketch_mode:
        streaming_label_x, streaming_label_y, streaming_label_ha = (
            (0.6, 0.35, "right") if mode == "jl" else (0.85, 0.5, "right")
        )
        plt.text(
            0.05,
            0.92,
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
            0.035,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
    else:
        plt.text(
            0.9,
            6e5,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            0.9,
            4e3,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            0.95,
            1.4e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    plt.yscale("log")
    plt.xlabel("Accuracy")
    plt.ylabel("Machine size")
    plt.ylim(1e1, 2e6)
    if sketch_mode:
        plt.xlim(0.79, 0.95)
        plt.xticks([0.80, 0.84, 0.88, 0.92], ["80%", "84%", "88%", "92%"])
    else:
        plt.xlim(0.58, 0.97)
        plt.xticks([0.60, 0.70, 0.80, 0.90], ["60%", "70%", "80%", "90%"])
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification")
    plt.tight_layout()
    if sketch_mode:
        output_pdf = f"dorothea_{mode}_size_vs_accuracy.pdf"
    else:
        output_pdf = "dorothea_size_vs_accuracy.pdf"
    plt.savefig(output_pdf)
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    sweep_utils.sweep_main("Dorothea Machine Size vs Accuracy Analysis", run_analysis)

"""IMDB sweep: machine size vs LS-SVM classification accuracy.

Ridge classification (5-fold CV) on TF-IDF features, sweeping rare-feature
truncation (min_df) or an oblivious feature sketch; see sweep_utils for the
shared pipeline.
"""

import json

import imdb_utils
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import sketch_utils
import sweep_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)
sweep_utils.apply_plot_style()

min_dfs = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    14,
    16,
    19,
    21,
    24,
    28,
    32,
    36,
    42,
    48,
    55,
    62,
    71,
    81,
    93,
    106,
    122,
    139,
    159,
    181,
    207,
    236,
    270,
    308,
    352,
    402,
    459,
    524,
    599,
    684,
    781,
    891,
    1018,
    1162,
    1327,
    1515,
    1730,
    1976,
    2256,
    2576,
    2941,
    3358,
    3835,
    4379,
    5000,
]
sketch_n_features = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
n_sketch_seeds = 5
rare_extra_markers = [-1, -3, -9, -13, -18, -21, -28]


def make_classifier():
    return RidgeClassifier(random_state=42, alpha=10, solver="auto")


def get_ridge_results_full():
    X_all_raw, y_all = imdb_utils.load_imdb_data()

    results = {
        "min_dfs": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracy_scores": [],
    }

    tqdm.write("Sweeping min_df for Full IMDB...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep"):
        vectorizer = TfidfVectorizer(
            min_df=mdf, stop_words="english", dtype=np.float32
        )
        X_all = vectorizer.fit_transform(X_all_raw)
        X_all.eliminate_zeros()

        num_samples, feature_dim = X_all.shape
        sparsity = sketch_utils.max_sparsity(X_all)

        scores = cross_val_score(make_classifier(), X_all, y_all, cv=5)

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(feature_dim)
        results["space_sparse"].append(sketch_utils.matrix_nnz(X_all))
        results["space_quantum"].append(
            sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "svm")
        )
        results["accuracy_scores"].append([float(s) for s in scores])

    return results


def get_ridge_results_sketch(sketch_seeds, mode):
    X_all_raw, y_all = imdb_utils.load_imdb_data()

    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", dtype=np.float32)
    X_full = vectorizer.fit_transform(X_all_raw)
    X_full.eliminate_zeros()

    full_dim = X_full.shape[1]
    n_features_list = [k for k in sketch_n_features if k < full_dim] + [full_dim]

    results = {
        "n_features": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracy_scores": [],
    }

    tqdm.write(f"Sweeping {mode} sketch dimension for Full IMDB...")

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

            scores = cross_val_score(make_classifier(), X_sketch, y_all, cv=5)
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
            print(f"Running Ridge Analysis on {sketch.name}-sketched IMDB Dataset...")
            seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
            print(f"Averaging over random sketch seeds: {seeds}")
            results = get_ridge_results_sketch(seeds, mode)
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="accuracy",
                dataset_name=f"IMDB Full ({sketch.name})",
                output_json=f"imdb_{mode}_size_vs_accuracy.json",
                sketch_mode=True,
                param_name="n_features",
                raw_key="raw_data_by_n_features",
                mode=mode,
                seeds=seeds,
            )
        else:
            print("Running Ridge Analysis on Full IMDB Dataset...")
            results = get_ridge_results_full()
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="accuracy",
                dataset_name="IMDB Full",
                output_json="imdb_size_vs_accuracy.json",
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
        sweep_utils.plot_curve(
            ax,
            k,
            xm,
            xs,
            ym,
            label=(sketch_utils.streaming_label(mode) if k == "streaming" else None),
            show_all_markers=sketch_mode,
            extra_marker_indices=rare_extra_markers,
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if sketch_mode:
        streaming_label_x, streaming_label_y, streaming_label_ha = (
            (0.7, 0.5, "right") if mode == "jl" else (0.9, 0.7, "right")
        )
        plt.text(
            0.2,
            0.85,
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
        plt.xticks([0.60, 0.70, 0.80, 0.90], ["60%", "70%", "80%", "90%"])
        plt.xlim(0.57, 0.92)
    else:
        plt.text(
            0.70,
            4e6,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
        )
        plt.text(
            0.88,
            9e4,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            0.90,
            1.9e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.xticks(
            [0.70, 0.75, 0.80, 0.85, 0.90],
            ["70%", "75%", "80%", "85%", "90%"],
        )
        plt.xlim(0.69, 0.91)

    plt.yscale("log")
    plt.xlabel("Accuracy")
    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.ylabel("Machine size")
    plt.ylim(1e1, 1e7)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification")
    plt.tight_layout()
    if sketch_mode:
        output_pdf = f"imdb_{mode}_size_vs_accuracy.pdf"
    else:
        output_pdf = "imdb_size_vs_accuracy.pdf"
    plt.savefig(output_pdf)
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    sweep_utils.sweep_main("IMDB Machine Size vs Accuracy Analysis", run_analysis)

"""PBMC68k sweep: machine size vs LS-SVM classification accuracy.

Ridge classification (5-fold CV) averaged over random cell-type pairs,
sweeping rare-gene truncation (min_samples) or an oblivious feature sketch;
see sweep_utils for the shared pipeline.
"""

import json
from itertools import combinations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pbmc68k_utils
import sketch_utils
import sweep_utils
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)
sweep_utils.apply_plot_style()

N_PAIRS = 100  # Number of random cell-type pairs to average over

min_samples_list = pbmc68k_utils.get_min_samples_sweep()
sketch_n_features = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
n_sketch_seeds = 5


def make_classifier():
    return RidgeClassifier(
        random_state=42, alpha=200.0, solver="auto", class_weight="balanced"
    )


def get_random_pairs(n_classes, n_pairs, rng):
    """Generate n_pairs random pairs from n_classes."""
    all_pairs = list(combinations(range(n_classes), 2))
    indices = rng.choice(len(all_pairs), size=n_pairs, replace=True)
    return [all_pairs[i] for i in indices]


def load_data_and_pairs():
    tqdm.write("Loading PBMC68k dataset (all classes)...")
    X_full, y_full, label_names = pbmc68k_utils.load_pbmc68k_data(
        min_samples=1, normalize=True, binary=False
    )
    tqdm.write(f"Dataset shape: {X_full.shape}, Classes: {label_names}")

    rng = np.random.default_rng(42)
    pairs = get_random_pairs(len(label_names), N_PAIRS, rng)
    return X_full, y_full, label_names, pairs


def evaluate_pair(X, y_full, c1, c2):
    """Machine sizes and CV scores for one binary cell-type pair."""
    mask = (y_full == c1) | (y_full == c2)
    X_pair = X[mask]
    y_pair = (y_full[mask] == c2).astype(int)

    num_samples, feature_dim = X_pair.shape
    sparsity = sketch_utils.max_sparsity(X_pair)

    spaces = (
        feature_dim,
        sketch_utils.matrix_nnz(X_pair),
        sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "svm"),
    )
    scores = cross_val_score(make_classifier(), X_pair, y_pair, cv=5)
    return spaces, [float(s) for s in scores]


def get_ridge_results_full():
    X_full, y_full, label_names, pairs = load_data_and_pairs()
    tqdm.write(f"Using {len(pairs)} random class pairs for binary classification")
    for i, (c1, c2) in enumerate(pairs):
        tqdm.write(f"  Pair {i + 1}: {label_names[c1]} vs {label_names[c2]}")

    results = {
        "min_samples": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracy_scores": [],
        "pairs": [(label_names[c1], label_names[c2]) for c1, c2 in pairs],
    }

    tqdm.write("Sweeping min_samples for PBMC68k (averaged over random pairs)...")

    for min_samp in tqdm(min_samples_list, desc="min_samples Sweep"):
        X_filtered, _ = pbmc68k_utils.filter_genes_by_frequency(X_full, min_samp)

        if X_filtered.shape[1] == 0:
            continue

        pair_spaces = []
        pair_accuracy_scores = []
        for c1, c2 in pairs:
            spaces, scores = evaluate_pair(X_filtered, y_full, c1, c2)
            pair_spaces.append(spaces)
            pair_accuracy_scores.append(scores)

        space_streaming, space_sparse, space_quantum = zip(*pair_spaces)

        results["min_samples"].append(min_samp)
        results["space_streaming"].append(np.mean(space_streaming))
        results["space_sparse"].append(np.mean(space_sparse))
        results["space_quantum"].append(np.mean(space_quantum))
        results["accuracy_scores"].append(pair_accuracy_scores)

    return results


def get_ridge_results_sketch(sketch_seeds, mode):
    X_full, y_full, label_names, pairs = load_data_and_pairs()

    full_dim = X_full.shape[1]
    n_features_list = [k for k in sketch_n_features if k < full_dim] + [full_dim]

    results = {
        "n_features": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracy_scores": [],
        "pairs": [(label_names[c1], label_names[c2]) for c1, c2 in pairs],
    }

    tqdm.write(f"Sweeping {mode} sketch dimension for PBMC68k...")

    for n_features in tqdm(n_features_list, desc=f"{mode} Sweep"):
        seed_pair_space_streaming = []
        seed_pair_space_sparse = []
        seed_pair_space_quantum = []
        seed_pair_accuracy_scores = []

        for seed in sketch_seeds:
            X_sketch_full, _ = sketch_utils.sketch_features(
                X_full, n_features, mode, seed=seed
            )

            pair_spaces = []
            pair_accuracy_scores = []
            for c1, c2 in pairs:
                spaces, scores = evaluate_pair(X_sketch_full, y_full, c1, c2)
                pair_spaces.append(spaces)
                pair_accuracy_scores.append(scores)

            space_streaming, space_sparse, space_quantum = zip(*pair_spaces)
            seed_pair_space_streaming.append(list(space_streaming))
            seed_pair_space_sparse.append(list(space_sparse))
            seed_pair_space_quantum.append(list(space_quantum))
            seed_pair_accuracy_scores.append(pair_accuracy_scores)

        results["n_features"].append(n_features)
        results["space_streaming"].append(seed_pair_space_streaming)
        results["space_sparse"].append(seed_pair_space_sparse)
        results["space_quantum"].append(seed_pair_space_quantum)
        results["accuracy_scores"].append(seed_pair_accuracy_scores)

    return results


def run_analysis(load_file=None, mode=None, n_sketch_seeds=n_sketch_seeds):
    sketch_mode = sweep_utils.is_sketch_mode(mode)

    if load_file is None:
        if sketch_mode:
            sketch = sketch_utils.get_sketch(mode)
            print(f"Running Ridge Analysis on {sketch.name}-sketched PBMC68k Dataset...")
            seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
            print(f"Averaging over random sketch seeds: {seeds}")
            results = get_ridge_results_sketch(seeds, mode)
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="accuracy",
                dataset_name=f"PBMC68k ({sketch.name}, averaged over random pairs)",
                output_json=f"pbmc68k_{mode}_size_vs_accuracy.json",
                sketch_mode=True,
                param_name="n_features",
                raw_key="raw_data_by_n_features",
                mode=mode,
                seeds=seeds,
                by_suffix="by_seed_pair",
                extra_top={"pairs": results["pairs"], "n_pairs": len(results["pairs"])},
            )
        else:
            print("Running Ridge Analysis on PBMC68k Dataset (Binary Classification)...")
            results = get_ridge_results_full()
            load_file = sweep_utils.save_sweep_json(
                results,
                metric="accuracy",
                dataset_name="PBMC68k (Binary, averaged over random pairs)",
                output_json="pbmc68k_size_vs_accuracy.json",
                sketch_mode=False,
                param_name="min_samples",
                raw_key="raw_data_by_min_samples",
                rare_scores_key="accuracy_scores_by_pair",
                extra_top={"pairs": results["pairs"], "n_pairs": len(results["pairs"])},
            )
    else:
        print(f"Loading analysis from {load_file}...")

    with open(load_file, "r") as f:
        data = json.load(f)
    sketch_utils.validate_truncation_data(data, mode, load_file)
    stats = sweep_utils.load_sweep_stats(
        data, "accuracy", rare_raw_key="raw_data_by_min_samples"
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
            show_all_markers=True,
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if sketch_mode:
        streaming_label_x, streaming_label_y, streaming_label_ha = (
            (0.7, 0.45, "right") if mode == "jl" else (0.9, 0.62, "right")
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
            0.81,
            2e6,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
        )
        plt.text(
            0.888,
            1.2e4,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            0.90,
            1.7e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    plt.yscale("log")
    plt.xlabel("Accuracy")

    plt.xticks(
        [0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
        ["80%", "82%", "84%", "86%", "88%", "90%"],
    )
    if mode == "bucket":
        plt.xlim(0.795, 0.91)
    else:
        plt.xlim(0.795, 0.915)

    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.ylabel("Machine size")
    plt.ylim(1e1, 1e7)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification")
    plt.tight_layout()
    if sketch_mode:
        output_pdf = f"pbmc68k_{mode}_size_vs_accuracy.pdf"
    else:
        output_pdf = "pbmc68k_size_vs_accuracy.pdf"
    plt.savefig(output_pdf)
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    sweep_utils.sweep_main("PBMC68k Machine Size vs Accuracy Analysis", run_analysis)

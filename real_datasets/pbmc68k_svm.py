import argparse
import json
from itertools import combinations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pbmc68k_utils
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)
N_PAIRS = 100  # Number of random pairs to average over

# Same Plotting Style
plt.rcParams.update(
    {
        "font.family": "sans",
        "font.serif": ["Google Sans"],
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

# min_samples sweep - similar to min_df
min_samples_list = pbmc68k_utils.get_min_samples_sweep()
num_markers = 40

colors = {
    "quantum": "#CD591A",
    "streaming": "#2657AF",
    "sparse": "#606060",
}
labels = {
    "streaming": "Classical streaming",
    "sparse": "Classical sparse / QRAM",
    "quantum": "Quantum oracle sketching",
}
markers = {"streaming": "P", "sparse": "X", "quantum": "D"}
figsize = (3.5, 3.5)
markersize = {"streaming": 50, "sparse": 50, "quantum": 30}
linewidth_marker = {"streaming": 0, "sparse": 0, "quantum": 0}


def get_random_pairs(n_classes, n_pairs, rng):
    """Generate n_pairs random pairs from n_classes."""
    all_pairs = list(combinations(range(n_classes), 2))
    indices = rng.choice(len(all_pairs), size=n_pairs, replace=True)
    return [all_pairs[i] for i in indices]


def get_ridge_results_full():
    # 1. Load Full Data (normalized, all genes, all classes)
    tqdm.write("Loading PBMC68k dataset (all classes)...")
    X_full, y_full, label_names = pbmc68k_utils.load_pbmc68k_data(
        min_samples=1, normalize=True, binary=False
    )
    tqdm.write(f"Dataset shape: {X_full.shape}, Classes: {label_names}")

    # 2. Generate random pairs of cell types
    rng = np.random.default_rng(42)
    n_classes = len(label_names)
    pairs = get_random_pairs(n_classes, N_PAIRS, rng)
    tqdm.write(f"Using {len(pairs)} random class pairs for binary classification")
    for i, (c1, c2) in enumerate(pairs):
        tqdm.write(f"  Pair {i + 1}: {label_names[c1]} vs {label_names[c2]}")

    # 3. Sweep min_samples
    results = {
        "min_samples": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracies_mean": [],
        "accuracies_std": [],
        "pairs": [(label_names[c1], label_names[c2]) for c1, c2 in pairs],
    }

    tqdm.write("Sweeping min_samples for PBMC68k (averaged over random pairs)...")

    for min_samp in tqdm(min_samples_list, desc="min_samples Sweep"):
        # Filter genes by min_samples on full data
        X_filtered, gene_indices = pbmc68k_utils.filter_genes_by_frequency(
            X_full, min_samp
        )

        # Skip if no genes remain
        if X_filtered.shape[1] == 0:
            continue

        # Collect results across all pairs
        pair_accuracies = []
        pair_space_streaming = []
        pair_space_sparse = []
        pair_space_quantum = []

        for c1, c2 in pairs:
            # Create binary subset for this pair
            mask = (y_full == c1) | (y_full == c2)
            X_pair = X_filtered[mask]
            y_pair = (y_full[mask] == c2).astype(
                int
            )  # Binary labels: 0 for c1, 1 for c2

            shape = X_pair.get_shape()
            feature_dim = shape[1]
            num_samples = shape[0]

            # Sparsity calculation
            row_sparsity = int(X_pair.getnnz(axis=1).max())
            col_sparsity = int(X_pair.getnnz(axis=0).max())
            sparsity = max(row_sparsity, col_sparsity)

            # --- Space Calculations ---
            space_stream = feature_dim
            space_sparse = X_pair.getnnz()
            space_quantum = (
                2 * np.ceil(np.log2(num_samples + 2 * feature_dim))
                + np.ceil(np.log2(sparsity + 1))
                + 4
            )

            # --- Ridge Training & Eval (CV) ---
            clf = RidgeClassifier(
                random_state=42, alpha=200.0, solver="auto", class_weight="balanced"
            )
            # 5-Fold Cross Validation
            scores = cross_val_score(clf, X_pair, y_pair, cv=5)

            pair_accuracies.append(scores.mean())
            pair_space_streaming.append(space_stream)
            pair_space_sparse.append(space_sparse)
            pair_space_quantum.append(space_quantum)

        # Average across pairs
        acc_mean = np.mean(pair_accuracies)
        # Standard error across pairs (variation between pairs)
        acc_sem = np.std(pair_accuracies) / np.sqrt(len(pair_accuracies))

        # Average space metrics (they vary slightly due to different sample sizes per pair)
        space_streaming_mean = np.mean(pair_space_streaming)
        space_sparse_mean = np.mean(pair_space_sparse)
        space_quantum_mean = np.mean(pair_space_quantum)

        results["min_samples"].append(min_samp)
        results["space_streaming"].append(space_streaming_mean)
        results["space_sparse"].append(space_sparse_mean)
        results["space_quantum"].append(space_quantum_mean)
        results["accuracies_mean"].append(acc_mean)
        results["accuracies_std"].append(acc_sem)

    return results


def plot_parametric_hybrid(
    x_mean, x_std, y_mean, color, marker, label, linewidth, marker_size
):
    y_vals = np.array(y_mean)
    x_vals = np.array(x_mean)
    x_errs = np.array(x_std)

    plt.fill_betweenx(
        y_vals,
        x_vals - x_errs,
        x_vals + x_errs,
        color=color,
        alpha=0.2,
        edgecolor="none",
    )

    # Line
    plt.plot(x_vals, y_vals, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    # # Markers
    # x_min, x_max = np.min(x_vals), np.max(x_vals)
    # target_x = np.linspace(x_min, x_max, num=num_markers)
    # marker_indices = []
    # for tx in target_x:
    #     idx = (np.abs(x_vals - tx)).argmin()
    #     if idx not in marker_indices:
    #         marker_indices.append(idx)

    plt.scatter(
        x_vals,
        y_vals,
        marker=marker,
        color=color,
        label=label,
        alpha=0.9,
        s=marker_size,
        linewidth=linewidth,
    )


def get_sorted_arrays(x_mean, x_std, y_mean):
    data = sorted(zip(x_mean, x_std, y_mean), key=lambda x: x[2])
    return (
        np.array([d[0] for d in data]),
        np.array([d[1] for d in data]),
        np.array([d[2] for d in data]),
    )


def run_analysis(load_file=None):
    keys = ["streaming", "sparse", "quantum"]

    if load_file is not None:
        print(f"Loading analysis from {load_file}...")
        with open(load_file, "r") as f:
            data = json.load(f)
        raw_data = data["raw_data_by_min_samples"]
        min_samps = sorted([int(k) for k in raw_data.keys()])

        final_stats = {
            k: {"mean_space": [], "mean_acc": [], "sem_acc": []} for k in keys
        }
        for ms in min_samps:
            for k in keys:
                final_stats[k]["mean_space"].append(raw_data[str(ms)][k]["space"])
                final_stats[k]["mean_acc"].append(raw_data[str(ms)][k]["accuracy_mean"])
                final_stats[k]["sem_acc"].append(raw_data[str(ms)][k]["accuracy_sem"])

    else:
        print("Running Ridge Analysis on PBMC68k Dataset (Binary Classification)...")
        results = get_ridge_results_full()

        final_stats = {
            k: {"mean_space": [], "mean_acc": [], "sem_acc": []} for k in keys
        }

        data_to_save = {
            "dataset": "PBMC68k (Binary, averaged over random pairs)",
            "pairs": results.get("pairs", []),
            "n_pairs": len(results.get("pairs", [])),
            "raw_data_by_min_samples": {},
        }

        for i, ms in enumerate(results["min_samples"]):
            ms_str = str(ms)
            data_to_save["raw_data_by_min_samples"][ms_str] = {}
            for k in keys:
                space = results[f"space_{k}"][i]
                acc_mean = results["accuracies_mean"][i]
                acc_sem = results["accuracies_std"][i]

                final_stats[k]["mean_space"].append(space)
                final_stats[k]["mean_acc"].append(acc_mean)
                final_stats[k]["sem_acc"].append(acc_sem)

                data_to_save["raw_data_by_min_samples"][ms_str][k] = {
                    "space": space,
                    "accuracy_mean": acc_mean,
                    "accuracy_sem": acc_sem,
                }

        with open("pbmc68k_size_vs_accuracy.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to pbmc68k_size_vs_accuracy.json")

    # Plot
    plt.figure(figsize=figsize)
    for k in keys:
        xm, xs, ym = get_sorted_arrays(
            final_stats[k]["mean_acc"],
            final_stats[k]["sem_acc"],
            final_stats[k]["mean_space"],
        )
        plot_parametric_hybrid(
            xm,
            xs,
            ym,
            colors[k],
            markers[k],
            labels[k],
            linewidth_marker[k],
            markersize[k],
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]

    plt.text(
        0.8,
        1.8e6,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.888,
        1.2e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        0.90,
        1.5e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
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
    plt.xlim(0.795, 0.915)

    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.ylabel("Machine size")
    plt.ylim(1e1, 1e7)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification")
    plt.tight_layout()
    plt.savefig("pbmc68k_size_vs_accuracy.pdf")
    print("Saved pbmc68k_size_vs_accuracy.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PBMC68k Machine Size vs Accuracy Analysis"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    args = parser.parse_args()

    run_analysis(load_file=args.load)

import argparse
import json

import dorothea_utils
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)

# Plotting Style
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

# Min DF Sweep Range (Same as PCA)
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


def get_svm_results_full():
    # 1. Load Data
    print("Loading Dorothea data...")
    X_full, y_full = dorothea_utils.load_dorothea_data(valid=True)
    tqdm.write(f"Dataset shape: {X_full.shape}")
    row_sparsity = int(X_full.getnnz(axis=1).max())
    col_sparsity = int(X_full.getnnz(axis=0).max())
    sparsity = max(row_sparsity, col_sparsity)
    tqdm.write(
        f"Max row sparsity: {row_sparsity}, Max col sparsity: {col_sparsity}, Overall sparsity: {sparsity}"
    )

    # Pre-compute document frequencies
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
        "accuracy_mean": [],
        "accuracy_std": [],
    }

    tqdm.write("Sweeping min_df for Dorothea (SVM)...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep"):
        keep_mask = doc_freqs >= mdf
        keep_indices = np.where(keep_mask)[0]

        if len(keep_indices) == 0:
            print(f"Skipping min_df={mdf} (0 features kept)")
            continue

        X_trunc = X_full[:, keep_indices]  # type: ignore

        shape = X_trunc.shape
        feature_dim = shape[1]
        num_samples = shape[0]

        # Space metrics
        row_sparsity = int(X_trunc.getnnz(axis=1).max()) if shape[0] > 0 else 0
        col_sparsity = int(X_trunc.getnnz(axis=0).max()) if shape[1] > 0 else 0
        sparsity = max(row_sparsity, col_sparsity)

        space_stream = feature_dim
        space_sparse = X_trunc.getnnz()
        space_quantum = (
            2 * np.ceil(np.log2(num_samples + feature_dim + 1))
            + np.ceil(np.log2(sparsity + 1))
            + 4
        )

        clf = RidgeClassifier(
            random_state=42, alpha=200, solver="auto", class_weight="balanced"
        )

        # 5-Fold Cross Validation
        scores = cross_val_score(clf, X_trunc, y_full, cv=5, n_jobs=-1)

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(space_stream)
        results["space_sparse"].append(space_sparse)
        results["space_quantum"].append(space_quantum)
        results["accuracy_mean"].append(scores.mean())
        results["accuracy_std"].append(scores.std() / np.sqrt(len(scores)))

    return results


def plot_parametric_hybrid(
    x_mean, x_std, y_mean, color, marker, label, linewidth, marker_size
):
    # 1. Horizontal Tube (Accuracy SEM/STD)
    y_vals = np.array(y_mean)
    x_vals = np.array(x_mean)
    x_errs = np.array(x_std) if x_std is not None else np.zeros_like(x_vals)

    plt.fill_betweenx(
        y_vals,
        x_vals - x_errs,
        x_vals + x_errs,
        color=color,
        alpha=0.2,
        edgecolor="none",
    )

    # 2. Line
    plt.plot(x_vals, y_vals, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

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

    if load_file:
        print(f"Loading analysis from {load_file}...")
        with open(load_file, "r") as f:
            data_to_save = json.load(f)
        raw_data = data_to_save["raw_data_by_min_df"]
        mdfs = sorted([int(k) for k in raw_data.keys()])

        final_stats = {
            k: {"mean_space": [], "mean_acc": [], "std_acc": []} for k in keys
        }
        for m in mdfs:
            entry = raw_data[str(m)]
            for k in keys:
                final_stats[k]["mean_space"].append(entry[k]["space"])
                final_stats[k]["mean_acc"].append(entry[k]["accuracy"])
                final_stats[k]["std_acc"].append(entry[k]["accuracy_std"])

    else:
        print("Running SVM Analysis on Dorothea Dataset...")
        results = get_svm_results_full()

        data_to_save = {"dataset": "Dorothea", "raw_data_by_min_df": {}}

        final_stats = {
            k: {"mean_space": [], "mean_acc": [], "std_acc": []} for k in keys
        }

        for i, mdf in enumerate(results["min_dfs"]):
            mdf_str = str(mdf)
            data_to_save["raw_data_by_min_df"][mdf_str] = {}

            for k in keys:
                space = results[f"space_{k}"][i]
                acc = results["accuracy_mean"][i]
                std = results["accuracy_std"][i]

                final_stats[k]["mean_space"].append(space)
                final_stats[k]["mean_acc"].append(acc)
                final_stats[k]["std_acc"].append(std)

                data_to_save["raw_data_by_min_df"][mdf_str][k] = {
                    "space": space,
                    "accuracy": acc,
                    "accuracy_std": std,
                }

        with open("dorothea_size_vs_accuracy.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to dorothea_size_vs_accuracy.json")

    # Plot
    plt.figure(figsize=figsize)
    for k in keys:
        xm, xs, ym = get_sorted_arrays(
            final_stats[k]["mean_acc"],
            final_stats[k]["std_acc"],
            final_stats[k]["mean_space"],
        )
        ind = xm >= 0.6
        plot_parametric_hybrid(
            xm[ind],
            xs[ind],
            ym[ind],
            colors[k],
            markers[k],
            labels[k],
            linewidth_marker[k],
            markersize[k],
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    plt.text(
        0.9,
        6e5,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        0.9,
        4e3,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        0.95,
        1.4e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.xlabel("Accuracy")
    plt.ylabel("Machine size")
    plt.ylim(1e1, 2e6)
    plt.xlim(0.58, 0.97)
    plt.xticks([0.60, 0.70, 0.80, 0.90], ["60%", "70%", "80%", "90%"])
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Constraint vs Accuracy (Dorothea)")
    plt.tight_layout()
    plt.savefig("dorothea_size_vs_accuracy.pdf")
    print("Saved dorothea_size_vs_accuracy.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dorothea Machine Size vs Accuracy Analysis"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    args = parser.parse_args()

    run_analysis(load_file=args.load)

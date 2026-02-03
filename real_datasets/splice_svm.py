import argparse
import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import splice_utils
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)

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

min_samples_list = splice_utils.get_min_samples_sweep()
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


def get_ridge_results_full():
    tqdm.write("Loading Splice Junction dataset (binary: EI vs IE)...")
    X_full, y, label_names = splice_utils.load_splice_data(binary=True, min_samples=1)
    tqdm.write(f"Dataset shape: {X_full.shape}, Classes: {label_names}")
    row_sparsity = int(X_full.getnnz(axis=1).max())
    col_sparsity = int(X_full.getnnz(axis=0).max())
    sparsity = max(row_sparsity, col_sparsity)
    tqdm.write(
        f"Max row sparsity: {row_sparsity}, Max col sparsity: {col_sparsity}, Overall sparsity: {sparsity}"
    )

    results = {
        "min_samples": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracies_mean": [],
        "accuracies_std": [],
    }

    tqdm.write("Sweeping min_samples for Splice Junction...")

    for min_samp in tqdm(min_samples_list, desc="min_samples Sweep"):
        X, _ = splice_utils.filter_features_by_frequency(X_full, min_samp)

        if X.shape[1] == 0:
            continue

        shape = X.get_shape()
        feature_dim = shape[1]
        num_samples = shape[0]

        row_sparsity = int(X.getnnz(axis=1).max())
        col_sparsity = int(X.getnnz(axis=0).max())
        sparsity = max(row_sparsity, col_sparsity)

        space_stream = feature_dim
        space_sparse = X.getnnz()

        space_quantum = (
            2 * np.ceil(np.log2(num_samples + 2 * feature_dim))
            + np.ceil(np.log2(sparsity + 1))
            + 4
        )

        clf = RidgeClassifier(
            random_state=42, alpha=1, solver="auto", class_weight="balanced"
        )
        scores = cross_val_score(clf, X, y, cv=5)

        acc_mean = scores.mean()
        acc_sem = scores.std() / np.sqrt(len(scores))

        results["min_samples"].append(min_samp)
        results["space_streaming"].append(space_stream)
        results["space_sparse"].append(space_sparse)
        results["space_quantum"].append(space_quantum)
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
        print("Running Ridge Analysis on Splice Junction Dataset (Binary)...")
        results = get_ridge_results_full()

        final_stats = {
            k: {"mean_space": [], "mean_acc": [], "sem_acc": []} for k in keys
        }

        data_to_save = {
            "dataset": "Splice Junction (Binary: EI vs IE)",
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

        with open("splice_size_vs_accuracy.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to splice_size_vs_accuracy.json")

    # Plot
    plt.figure(figsize=figsize)
    for k in keys:
        xm, xs, ym = get_sorted_arrays(
            final_stats[k]["mean_acc"],
            final_stats[k]["sem_acc"],
            final_stats[k]["mean_space"],
        )
        ind = xm > 0.775
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

    sparse_spaces = final_stats["sparse"]["mean_space"]
    streaming_spaces = final_stats["streaming"]["mean_space"]
    quantum_spaces = final_stats["quantum"]["mean_space"]
    acc_vals = final_stats["sparse"]["mean_acc"]

    plt.text(
        0.78,
        8e4,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.845,
        1e3,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        0.86,
        1.5e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.xlabel("Accuracy")

    plt.xticks([0.78, 0.8, 0.82, 0.84, 0.86], ["78%", "80%", "82%", "84%", "86%"])
    plt.xlim(0.77, 0.875)

    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.ylabel("Machine size")
    plt.ylim(1e1, 2e5)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification (Splice)")
    plt.tight_layout()
    plt.savefig("splice_size_vs_accuracy.pdf")
    print("Saved splice_size_vs_accuracy.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splice Junction Machine Size vs Accuracy Analysis"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    args = parser.parse_args()

    run_analysis(load_file=args.load)

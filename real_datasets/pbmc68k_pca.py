import argparse
import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pbmc68k_utils
from scipy.sparse.linalg import svds
from tqdm import tqdm

np.random.seed(42)

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


def get_pca_results_full():
    # 1. Load Full Data (binary classification)
    tqdm.write("Loading PBMC68k dataset (binary classification)...")
    X_full, y, label_names = pbmc68k_utils.load_pbmc68k_data(
        min_samples=1, normalize=True, binary=False
    )
    tqdm.write(f"Dataset shape: {X_full.shape}, Classes: {label_names}")

    # Compute "Ground Truth" (Full Dimension)
    tqdm.write("Computing Top Singular Vector (Ground Truth)...")
    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]

    # Max Norm Squared (energy of the top component)
    var_max = np.linalg.norm(X_full @ v_full) ** 2
    D_full = X_full.shape[1]

    # Storage
    results = {
        "min_samples": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovery": [],
    }

    tqdm.write("Sweeping min_samples for PBMC68k...")

    for min_samp in tqdm(min_samples_list, desc="min_samples Sweep"):
        # Filter genes by min_samples
        X_trunc, gene_indices = pbmc68k_utils.filter_genes_by_frequency(
            X_full, min_samp
        )

        # Skip if no genes remain
        if X_trunc.shape[1] == 0:
            continue

        shape = X_trunc.get_shape()
        feature_dim = shape[1]
        num_samples = shape[0]

        # Sparsity calculation
        row_sparsity = int(X_trunc.getnnz(axis=1).max())
        col_sparsity = int(X_trunc.getnnz(axis=0).max())
        sparsity = max(row_sparsity, col_sparsity)

        # --- Space Calculations ---
        space_stream = feature_dim
        space_sparse = X_trunc.getnnz()

        space_quantum = (
            2 * np.ceil(np.log2(num_samples + feature_dim))
            + np.ceil(np.log2(sparsity))
            + 4
        )

        # --- SVD Calculation (Sparse) ---
        _, _, vt_trunc = svds(X_trunc.asfptype(), k=1)
        v_trunc = vt_trunc[0]

        # --- Variance Recovery Calculation ---
        # Lift to Full Space
        v_lifted = np.zeros(D_full)
        v_lifted[gene_indices] = v_trunc

        # Normalize lifted vector
        norm = np.linalg.norm(v_lifted)
        v_lifted = v_lifted / norm

        # Variance (Energy) Captured
        var_captured = np.linalg.norm(X_full @ v_lifted) ** 2
        recovery = var_captured / var_max

        results["min_samples"].append(min_samp)
        results["space_streaming"].append(space_stream)
        results["space_sparse"].append(space_sparse)
        results["space_quantum"].append(space_quantum)
        results["variance_recovery"].append(recovery)

    return results


def plot_parametric_hybrid(
    x_mean, y_mean, color, marker, label, linewidth, marker_size
):
    # Line
    plt.plot(x_mean, y_mean, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    # Markers
    x_min, x_max = np.min(x_mean), np.max(x_mean)
    target_x = np.linspace(x_min, x_max, num=num_markers)
    marker_indices = []
    for tx in target_x:
        idx = (np.abs(x_mean - tx)).argmin()
        if idx not in marker_indices:
            marker_indices.append(idx)
    marker_indices += [-5, -10, -15, -20]

    plt.scatter(
        np.array(x_mean)[marker_indices],
        np.array(y_mean)[marker_indices],
        marker=marker,
        color=color,
        label=label,
        alpha=0.9,
        s=marker_size,
        linewidth=linewidth,
    )


def get_sorted_arrays(x_mean, y_mean):
    data = sorted(zip(x_mean, y_mean), key=lambda x: x[1])
    return (
        np.array([d[0] for d in data]),
        np.array([d[1] for d in data]),
    )


def run_analysis(load_file=None):
    keys = ["streaming", "sparse", "quantum"]

    if load_file is not None:
        print(f"Loading analysis from {load_file}...")
        with open(load_file, "r") as f:
            data = json.load(f)
        raw_data = data["raw_data_by_min_samples"]
        min_samps = sorted([int(k) for k in raw_data.keys()])

        final_stats = {k: {"mean_space": [], "mean_var": []} for k in keys}

        for ms in min_samps:
            for k in keys:
                final_stats[k]["mean_space"].append(raw_data[str(ms)][k]["space"])
                final_stats[k]["mean_var"].append(
                    raw_data[str(ms)][k]["variance_recovery"]
                )

    else:
        print("Running PCA Analysis on PBMC68k Dataset (Binary Classification)...")
        results = get_pca_results_full()

        final_stats = {k: {"mean_space": [], "mean_var": []} for k in keys}

        data_to_save = {"dataset": "PBMC68k (Binary)", "raw_data_by_min_samples": {}}

        for i, ms in enumerate(results["min_samples"]):
            ms_str = str(ms)
            data_to_save["raw_data_by_min_samples"][ms_str] = {}

            for k in keys:
                space = results[f"space_{k}"][i]
                rec = results["variance_recovery"][i]

                final_stats[k]["mean_space"].append(space)
                final_stats[k]["mean_var"].append(rec)

                data_to_save["raw_data_by_min_samples"][ms_str][k] = {
                    "space": space,
                    "variance_recovery": rec,
                }

        with open("pbmc68k_size_vs_variance.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to pbmc68k_size_vs_variance.json")

    # Plot
    plt.figure(figsize=figsize)
    for k in keys:
        xm, ym = get_sorted_arrays(
            final_stats[k]["mean_var"],
            final_stats[k]["mean_space"],
        )
        plot_parametric_hybrid(
            xm,
            ym,
            colors[k],
            markers[k],
            labels[k],
            linewidth_marker[k],
            markersize[k],
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]

    plt.text(
        1,
        1e6,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        0.996,
        1.5e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        1,
        2e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.ylim(1e1, 1e7)
    plt.xlabel("Relative explained variance")

    plt.xticks(
        [0.92, 0.94, 0.96, 0.98, 1],
        ["92%", "94%", "96%", "98%", "100%"],
    )
    plt.xlim(0.915, 1.005)

    plt.tick_params(direction="in", which="both", top=False, right=True)
    ax = plt.gca()
    ax.set_ylabel("Machine size")
    ax.tick_params(axis="y")
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Dimension reduction")
    plt.tight_layout()
    plt.savefig("pbmc68k_size_vs_variance.pdf")
    print("Saved pbmc68k_size_vs_variance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PBMC68k Machine Size vs Variance Analysis"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    args = parser.parse_args()

    run_analysis(load_file=args.load)

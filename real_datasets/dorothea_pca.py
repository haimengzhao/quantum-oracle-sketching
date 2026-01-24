import argparse
import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from tqdm import tqdm

import dorothea_utils

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

# Min DF Sweep Range (Dorothea has 800 samples total, so min_df should be small)
# For Dorothea: Train (800) + Valid (350).
# So max DF is 1150.
# Let's sweep logarithmic from 1 to ~1000
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
    "quantum": "#E69F00",
    "streaming": "#005AB5",
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
    # 1. Load Data
    print("Loading Dorothea data...")
    try:
        X_full, _ = dorothea_utils.load_dorothea_data(valid=True)
    except FileNotFoundError:
        print("Valid set issues, loading only train")
        X_full, _ = dorothea_utils.load_dorothea_data(valid=False)

    X_full = X_full.astype(np.float64)

    # 2. Compute "Ground Truth" (Full Dimension)
    print("Computing Top Singular Vector (Ground Truth)...")
    # Dorothea is 1150 x 100000. SVD is feasible.
    try:
        _, _, vt_full = svds(X_full, k=1)
        v_full = vt_full[0]  # type: ignore # Shape (D,)
    except Exception as e:
        print(f"SVD failed: {e}")
        raise e

    # Max Norm Squared
    var_max = np.linalg.norm(X_full @ v_full) ** 2

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
        "variance_recovery": [],
    }

    tqdm.write("Sweeping min_df for Dorothea...")

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

        if min(shape) <= 1:
            if shape[1] == 1:
                # 1 feature: direction is [1.0]
                v_trunc = np.array([1.0])
            else:
                print(f"Skipping min_df={mdf} (Dimensions too small: {shape})")
                continue
        else:
            try:
                _, _, vt_trunc = svds(X_trunc, k=1)
                v_trunc = vt_trunc[0]  # type: ignore
            except Exception as e:
                print(f"SVD failed for shape {shape}: {e}")
                continue

        v_lifted = np.zeros(X_full.shape[1])  # type: ignore
        v_lifted[keep_indices] = v_trunc
        norm = np.linalg.norm(v_lifted)
        v_lifted = v_lifted / norm

        var_captured = np.linalg.norm(X_full @ v_lifted) ** 2
        recovery = var_captured / var_max

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(space_stream)
        results["space_sparse"].append(space_sparse)
        results["space_quantum"].append(space_quantum)
        results["variance_recovery"].append(recovery)

    return results


def plot_parametric_hybrid(
    x_mean, y_mean, color, marker, label, linewidth, marker_size
):
    # No Standard Deviation bands for single dataset run

    # 1. Line
    plt.plot(x_mean, y_mean, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    plt.scatter(
        np.array(x_mean),
        np.array(y_mean),
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

    if load_file:
        print(f"Loading analysis from {load_file}...")
        with open(load_file, "r") as f:
            data_to_save = json.load(f)
        raw_data = data_to_save["raw_data_by_min_df"]
        mdfs = sorted([int(k) for k in raw_data.keys()])

        final_stats = {k: {"mean_space": [], "mean_var": []} for k in keys}
        for m in mdfs:
            entry = raw_data[str(m)]
            for k in keys:
                final_stats[k]["mean_space"].append(entry[k]["space"])
                final_stats[k]["mean_var"].append(entry[k]["variance_recovery"])

    else:
        print("Running Analysis on Dorothea Dataset...")
        results = get_pca_results_full()

        data_to_save = {"dataset": "Dorothea", "raw_data_by_min_df": {}}

        final_stats = {k: {"mean_space": [], "mean_var": []} for k in keys}

        for i, mdf in enumerate(results["min_dfs"]):
            mdf_str = str(mdf)
            data_to_save["raw_data_by_min_df"][mdf_str] = {}

            for k in keys:
                space = results[f"space_{k}"][i]
                rec = results["variance_recovery"][i]

                final_stats[k]["mean_space"].append(space)
                final_stats[k]["mean_var"].append(rec)

                data_to_save["raw_data_by_min_df"][mdf_str][k] = {
                    "space": space,
                    "variance_recovery": rec,
                }

        with open("dorothea_size_vs_variance.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to dorothea_size_vs_variance.json")

    # Plot
    plt.figure(figsize=figsize)
    for k in keys:
        xm, ym = get_sorted_arrays(
            final_stats[k]["mean_var"],
            final_stats[k]["mean_space"],
        )
        ind = (xm >= 0.1) * (xm <= 1)
        xm = xm[ind]
        ym = ym[ind]
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
        0.15,
        8e5,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.9,
        4e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        1,
        1.4e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.ylim(1e1, 2e6)
    plt.xlabel("Relative explained variance")
    plt.xticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        ["20%", "40%", "60%", "80%", "100%"],
    )
    plt.xlim(0.08, 1.05)
    plt.tick_params(direction="in", which="both", top=False, right=True)
    ax = plt.gca()
    ax.set_ylabel("Machine size")
    ax.tick_params(axis="y")

    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Dimension reduction (Dorothea)")
    plt.tight_layout()
    plt.savefig("dorothea_size_vs_variance.pdf")
    print("Saved dorothea_size_vs_variance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dorothea Machine Size vs Variance Analysis"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    args = parser.parse_args()

    run_analysis(load_file=args.load)

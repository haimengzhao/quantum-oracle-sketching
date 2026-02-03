import argparse
import json

import imdb_utils
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

np.random.seed(42)

# Same Plotting Style as 20NG
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

# Min DF Sweep Range - Optimized for even spacing
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
    # 1. Load Data (Full IMDB)
    raw_documents, _ = imdb_utils.load_imdb_data()

    # 2. Compute "Ground Truth" (Full Dimension)
    print("Vectorizing Full Dimension Reference (min_df=1)...")
    full_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    X_full = full_vectorizer.fit_transform(raw_documents)
    # X_full.eliminate_zeros() # Often slower for very large matrices if not needed

    print("Computing Top Singular Vector (Ground Truth)...")
    # Note: We use X_full.asfptype() to ensure float compatibility for svds
    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]  # type: ignore # Shape (D,)

    # Max Norm Squared (energy of the top component)
    var_max = np.linalg.norm(X_full @ v_full) ** 2

    # Vocabulary map
    vocab_full = full_vectorizer.vocabulary_
    D_full = X_full.shape[1]

    # Storage
    results = {
        "min_dfs": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovery": [],
    }

    tqdm.write("Sweeping min_df for Full IMDB...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep"):
        vectorizer = TfidfVectorizer(min_df=mdf, stop_words="english")
        X_trunc = vectorizer.fit_transform(raw_documents)
        # X_trunc.eliminate_zeros()

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

        # Quantum Space Calculation
        space_quantum = (
            2 * np.ceil(np.log2(num_samples + feature_dim))
            + np.ceil(np.log2(sparsity))
            + 4
        )

        # --- SVD Calculation (Sparse) ---
        # Compute Top Singular Vector in Truncated Space
        _, _, vt_trunc = svds(X_trunc.asfptype(), k=1)
        v_trunc = vt_trunc[0]  # type: ignore # Shape (D,)

        # --- Variance Recovery Calculation ---
        # Lift to Full Space
        v_lifted = np.zeros(D_full)  # type: ignore

        # Fast Lifting
        # 1. Get feature names of truncated
        trunc_feature_names = vectorizer.get_feature_names_out()
        # 2. Get corresponding indices in full vocab
        #    Many words will be present since min_df > 1 implies present in min_df=1
        full_indices = []
        trunc_indices = []

        for i, word in enumerate(trunc_feature_names):
            if word in vocab_full:
                full_indices.append(vocab_full[word])
                trunc_indices.append(i)

        v_lifted[full_indices] = v_trunc[trunc_indices]

        # Normalize lifted vector (unit direction)
        norm = np.linalg.norm(v_lifted)
        v_lifted = v_lifted / norm

        # Variance (Energy) Captured
        var_captured = np.linalg.norm(X_full @ v_lifted) ** 2  # type: ignore
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

    # 2. Markers
    x_min, x_max = np.min(x_mean), np.max(x_mean)
    target_x = np.linspace(x_min, x_max, num=num_markers)
    marker_indices = []
    for tx in target_x:
        idx = (np.abs(x_mean - tx)).argmin()
        if idx not in marker_indices:
            marker_indices.append(idx)
    marker_indices += [-2, -5, -10, -15, -19, -24]

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
    # Sort by Y-axis metric (Space)
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
        # Extract data directly
        raw_data = data["raw_data_by_min_df"]
        # Convert keys to int and sort
        mdfs = sorted([int(k) for k in raw_data.keys()])

        final_stats = {k: {"mean_space": [], "mean_var": []} for k in keys}

        for mdf in mdfs:
            for k in keys:
                final_stats[k]["mean_space"].append(raw_data[str(mdf)][k]["space"])
                final_stats[k]["mean_var"].append(
                    raw_data[str(mdf)][k]["variance_recovery"]
                )

    else:
        print("Running Analysis on Full IMDB Dataset...")
        results = get_pca_results_full()

        # Reshape into final_stats format for plotting and saving
        final_stats = {k: {"mean_space": [], "mean_var": []} for k in keys}

        data_to_save = {"dataset": "IMDB Full", "raw_data_by_min_df": {}}

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

        with open("imdb_size_vs_variance.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to imdb_size_vs_variance.json")

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
        0.75,
        4e6,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.99,
        9e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        1,
        1.9e1,
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
        [0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        ["75%", "80%", "85%", "90%", "95%", "100%"],
    )
    plt.xlim(0.73, 1.03)
    plt.tick_params(direction="in", which="both", top=False, right=True)
    ax = plt.gca()
    ax.set_ylabel("Machine size")
    ax.tick_params(axis="y")
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Dimension reduction")
    plt.tight_layout()
    plt.savefig("imdb_size_vs_variance.pdf")
    print("Saved imdb_size_vs_variance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IMDB Machine Size vs Variance Analysis"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    args = parser.parse_args()

    run_analysis(load_file=args.load)

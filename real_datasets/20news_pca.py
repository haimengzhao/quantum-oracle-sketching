import argparse
import json
import random
from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
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

min_dfs = list(range(2, 21)) + list(range(25, 105, 5))
num_markers = 20

# --- Plotting ---
colors = {
    "quantum": "#E69F00",  # Rich, readable Gold (Okabe-Ito palette)
    "streaming": "#005AB5",  # Strong Blue
    "sparse": "#606060",  # Dark Grey (De-emphasized)
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


def load_data(categories=None):
    tqdm.write(f"Loading 20Newsgroups {categories}...")
    data_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        return_X_y=True,
    )
    data_test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        return_X_y=True,
    )
    # For PCA, we can use both train and test data to get a better estimate of the covariance
    return list(data_train[0]) + list(data_test[0])


def get_pca_results(categories):
    # 1. Load Data
    raw_documents = load_data(categories)

    # 2. Compute "Ground Truth" (Full Dimension)
    # min_df=1 for full dimension reference
    full_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    X_full = full_vectorizer.fit_transform(raw_documents)
    X_full.eliminate_zeros()  # type: ignore

    # Compute Top Singular Vector (Ground Truth) on Sparse Data (No Centering)
    # Note: We use X_full.asfptype() to ensure float compatibility for svds
    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]  # type: ignore # Shape (D,)

    # Max Norm Squared (energy of the top component)
    # X_full @ v_full is (N_samples,), norm squared gives sum of squared projections
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

    tqdm.write(f"Sweeping min_df for categories {categories}...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep", leave=False):
        vectorizer = TfidfVectorizer(min_df=mdf, stop_words="english")
        X_trunc = vectorizer.fit_transform(raw_documents)
        X_trunc.eliminate_zeros()  # type: ignore

        shape = X_trunc.get_shape()
        feature_dim = shape[1]
        num_samples = shape[0]

        # Sparsity calculation
        row_sparsity = int(X_trunc.getnnz(axis=1).max())
        col_sparsity = int(X_trunc.getnnz(axis=0).max())
        sparsity = max(row_sparsity, col_sparsity)

        # --- Space Calculations ---
        # --- Space Calculations ---
        # Any classical streaming algorithm must at least store the principal component,
        #    which is of size d. So classical streaming space is d floats.
        # Any classical standard algorithm that stores the whole sparse matrix
        #    must at least store all non-zero entries, so classical sparse space is nnz.
        space_stream = feature_dim  # Classical Streaming Space: d
        space_sparse = X_trunc.getnnz()  # Classical Sparse Space: nnz

        # We use quantum oracle sketching to build:
        # 1. The block encoding of the data matrix X in R^{n x d},
        #    Its Hermitian dilation is in R^{(n+d) x (n+d)}. This requires building the
        #    sparse index/element oracle for the dilated matrix, which has sparsity = sparsity.
        #    Hence, building index oracle requires 2log2(n+d) + log2(sparsity) + 2 (QSVT & binary search output) qubits.
        #    Building element oracle requires can reuse the same qubits, so no extra qubits needed.
        # 2. The state preparation unitary block encoding of the guiding vector g in R^d, which
        #    requires log2(d) + 2 (first LCU+QSVT and second LCU) qubits. These qubits are contained
        #    in the previous count and they can be reused.
        # Then we perform quantum ground state preparation algorithm using QSVT,
        #    which requires 1 ancilla qubit for the QSVT, contained in the previous count because we can reuse
        #    the ancilla qubit from quantum oracle sketching.
        # Finally, we need to perform interferometric measurement to calculate the signed overlap with test state,
        #    which requires 1 extra ancilla qubit.
        # The final estimate of the label is stored classically on a running average, so only 1 extra float is needed.
        # Therefore, total quantum space is:
        #   (2log2(n + d) + log2(sparsity) + 3) qubits + 1 float
        space_quantum = (
            2 * np.ceil(np.log2(num_samples + feature_dim))
            + np.ceil(np.log2(sparsity))
            + 4  # + 3 qubits + 1 float
        )

        # --- SVD Calculation (Sparse) ---
        # Compute Top Singular Vector in Truncated Space
        _, _, vt_trunc = svds(X_trunc.asfptype(), k=1)
        v_trunc = vt_trunc[0]  # type: ignore

        # --- Variance Recovery Calculation ---
        # Lift to Full Space
        trunc_vocab = vectorizer.vocabulary_
        v_lifted = np.zeros(D_full)

        # Lifting loop
        for word, idx_trunc in trunc_vocab.items():
            if word in vocab_full:
                idx_full = vocab_full[word]
                v_lifted[idx_full] = v_trunc[idx_trunc]

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
    x_mean, x_std, y_mean, y_std, color, marker, label, linewidth, marker_size, ax=None
):
    # 1. Horizontal Tube (Metric Variance)
    plt.fill_betweenx(
        y_mean, x_mean - x_std, x_mean + x_std, color=color, alpha=0.2, edgecolor="none"
    )

    # 2. Line (Full data)
    plt.plot(x_mean, y_mean, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    # 3. Markers
    # only display markers for evenly spaced points over accuracy to avoid clutter
    # calculate marker indices based on x_mean, note that x_mean is not necessarily evenly spaced
    x_min, x_max = np.min(x_mean), np.max(x_mean)
    target_x = np.linspace(x_min, x_max, num=num_markers)
    marker_indices = []
    for tx in target_x:
        idx = (np.abs(x_mean - tx)).argmin()
        if idx not in marker_indices:
            marker_indices.append(idx)

    plt.scatter(
        x_mean[marker_indices],
        y_mean[marker_indices],
        marker=marker,
        color=color,
        label=label,
        alpha=0.9,
        s=marker_size,
        linewidth=linewidth,
    )


def get_sorted_arrays(x_mean, x_std, y_mean, y_std):
    # Sort by Y-axis metric (Space) to ensure clean trajectories
    data = sorted(zip(x_mean, x_std, y_mean, y_std), key=lambda x: x[2])
    return (
        np.array([d[0] for d in data]),
        np.array([d[1] for d in data]),
        np.array([d[2] for d in data]),
        np.array([d[3] for d in data]),
    )


def run_analysis(n_pairs=10, from_json_data=None):
    keys = ["streaming", "sparse", "quantum"]
    final_stats = {
        k: {
            "mean_space": [],
            "std_space": [],
            "mean_var": [],
            "std_var": [],
        }
        for k in keys
    }

    if from_json_data is not None:
        print("Restoring analysis from JSON data...")
        n_pairs = from_json_data["n_pairs"]
        raw_data = from_json_data["raw_data_by_min_df"]
        by_min_df = {int(k): v for k, v in raw_data.items()}

        # Target Range for Subsampling (matches fresh run range)
        target_min_dfs = min_dfs

        # Re-compute stats
        for mdf in sorted(list(set(target_min_dfs))):
            if mdf not in by_min_df:
                continue

            for k in keys:
                spaces = np.array(by_min_df[mdf][k]["space"])

                # Handle variance/recovery key
                if "variance_recovery" in by_min_df[mdf][k]:
                    vars_ = np.array(by_min_df[mdf][k]["variance_recovery"])
                elif "variance" in by_min_df[mdf][k]:
                    vars_ = np.array(by_min_df[mdf][k]["variance"])
                else:
                    vars_ = np.zeros_like(spaces)

                if len(spaces) > 0:
                    sqrt_n = np.sqrt(len(spaces))
                    final_stats[k]["mean_space"].append(np.mean(spaces))
                    final_stats[k]["std_space"].append(np.std(spaces) / sqrt_n)
                    final_stats[k]["mean_var"].append(np.mean(vars_))
                    final_stats[k]["std_var"].append(np.std(vars_) / sqrt_n)

    else:
        print(f"Running Analysis over {n_pairs} random sets of 1v1 categories...")
        all_cats = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes")
        ).target_names  # type: ignore

        by_min_df = defaultdict(
            lambda: {k: {"space": [], "variance_recovery": []} for k in keys}
        )

        for i in tqdm(range(n_pairs), desc="Category Pairs", leave=True):
            # Randomly select 2 categories
            try:
                cats = random.sample(all_cats, 2)
            except ValueError:
                continue

            tqdm.write(f"[{i + 1}/{n_pairs}] Group: {cats}")

            # Calculate PCA variance and space
            res = get_pca_results(cats)

            for j, mdf in enumerate(res["min_dfs"]):
                for k in keys:
                    by_min_df[mdf][k]["space"].append(res[f"space_{k}"][j])
                    by_min_df[mdf][k]["variance_recovery"].append(
                        res["variance_recovery"][j]
                    )

        # Compute Stats
        for mdf in sorted(by_min_df.keys()):
            for k in keys:
                spaces = np.array(by_min_df[mdf][k]["space"])
                vars_ = np.array(by_min_df[mdf][k]["variance_recovery"])

                if len(spaces) > 0:
                    sqrt_n = np.sqrt(len(spaces))
                    final_stats[k]["mean_space"].append(np.mean(spaces))
                    final_stats[k]["std_space"].append(np.std(spaces) / sqrt_n)
                    final_stats[k]["mean_var"].append(np.mean(vars_))
                    final_stats[k]["std_var"].append(np.std(vars_) / sqrt_n)

        # Save Data
        data_to_save = {
            "n_pairs": n_pairs,
            "cats_per_class": 1,
            "raw_data_by_min_df": {
                mdf: {
                    metric: {
                        field: list(np.array(vals).astype(float))
                        for field, vals in sub_dict.items()
                    }
                    for metric, sub_dict in mdf_dict.items()
                }
                for mdf, mdf_dict in by_min_df.items()
            },
        }
        with open("20newsgroups_size_vs_variance.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to 20newsgroups_size_vs_variance.json")

    # Plot: Size vs Variance Recovery
    plt.figure(figsize=figsize)
    for k in keys:
        xm, xs, ym, ys = get_sorted_arrays(
            final_stats[k]["mean_var"],
            final_stats[k]["std_var"],
            final_stats[k]["mean_space"],
            final_stats[k]["std_space"],
        )
        plot_parametric_hybrid(
            xm,
            xs,
            ym,
            ys,
            colors[k],
            markers[k],
            labels[k],
            linewidth_marker[k],
            markersize[k],
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    plt.text(
        0.56,
        8e4,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.98,
        1e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        1,
        7e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.ylim(1e1, 2e5)
    plt.xlabel("Relative explained variance")
    plt.xticks(
        [0.6, 0.7, 0.8, 0.9, 1.0],
        ["60%", "70%", "80%", "90%", "100%"],
    )
    plt.xlim(0.52, 1.03)
    plt.tick_params(direction="in", which="both", top=False, right=True)
    ax = plt.gca()
    ax.set_ylabel("Machine size")
    ax.tick_params(axis="y")
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Dimension reduction")
    plt.tight_layout()
    plt.savefig("20newsgroups_size_vs_variance.pdf")
    print("Saved 20newsgroups_size_vs_variance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="20newsgroups Machine Size vs Variance Analysis"
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=100,
        help="Number of random pairs to average (for new run)",
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    args = parser.parse_args()

    if args.load is not None:
        with open(args.load, "r") as f:
            data = json.load(f)
        run_analysis(from_json_data=data)
    elif args.n_pairs is not None:
        run_analysis(n_pairs=args.n_pairs)
    else:
        print("Please specify --n_pairs <N> (to run) or --load <file.json> (to plot).")

import argparse
import json
import random
from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
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
    return data_train[0], data_train[1], data_test[0], data_test[1]


def get_ridge_results(categories):
    # 1. Load Data
    X_train_raw, y_train, X_test_raw, y_test = load_data(categories)

    # Combine for Cross Validation
    X_all_raw = list(X_train_raw) + list(X_test_raw)
    y_all = np.concatenate([y_train, y_test])

    # 2. Sweep min_df (Sparse Range for Smooth Plots)

    # Storage
    results = {
        "min_dfs": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracies": [],
        "error_rates": [],
    }

    tqdm.write(f"Sweeping min_df for categories {categories}...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep", leave=False):
        vectorizer = TfidfVectorizer(min_df=mdf, stop_words="english")
        X_all = vectorizer.fit_transform(X_all_raw)
        X_all.eliminate_zeros()  # type: ignore

        shape = X_all.get_shape()
        feature_dim = shape[1]
        num_samples = shape[0]

        # Sparsity calculation
        row_sparsity = int(X_all.getnnz(axis=1).max())
        col_sparsity = int(X_all.getnnz(axis=0).max())
        sparsity = max(row_sparsity, col_sparsity)

        # --- Space Calculations ---
        # Any classical streaming algorithm must at least store the weight vector,
        #    which is of size d. So classical streaming space is d floats.
        # Any classical standard algorithm that stores the whole sparse matrix
        #    must at least store all non-zero entries, so classical sparse space is nnz.
        space_stream = feature_dim  # Classical Streaming Space: d
        space_sparse = X_all.getnnz()  # Classical Sparse Space: nnz

        # We use quantum oracle sketching to build:
        # 1. The block encoding of the augmented data matrix [X; \lambda I] in R^{(n+d) x d},
        #    Its Hermitian dilation is in R^{(n+2d) x (n+2d)}. This requires building the
        #    sparse index/element oracle for the augmented matrix, which has sparsity = sparsity + 1.
        #    Hence, building index oracle requires 2log2(n+2d) + log2(sparsity + 1) + 2 (QSVT & binary search output) qubits.
        #    Building element oracle requires can reuse the same qubits, so no extra qubits needed.
        # 2. The state preparation unitary block encoding of the label vector y in R^n, which
        #    requires log2(n) + 2 (first LCU+QSVT and second LCU) qubits. These qubits are contained
        #    in the previous count and they can be reused.
        # Then we perform quantum ridge regression with amplitude amplification using QSVT-based quantum linear system solver,
        #    which requires 1 ancilla qubit for the QSVT, contained in the previous count because we can reuse
        #    the ancilla qubit from quantum oracle sketching.
        # Finally, we need to perform interferometric measurement to calculate the signed overlap with test state,
        #    which requires 1 extra ancilla qubit.
        # The final estimate of the label is stored classically on a running average, so only 1 extra float is needed.
        # Therefore, total quantum space is:
        #   (2log2(n + 2d) + log2(sparsity + 1) + 3) qubits + 1 float
        space_quantum = (
            2 * np.ceil(np.log2(num_samples + 2 * feature_dim))
            + np.ceil(np.log2(sparsity + 1))
            + 4  # + 3 qubits + 1 float
        )

        # --- Ridge Training & Eval (CV) ---
        clf = RidgeClassifier(random_state=42, alpha=1.0, solver="auto")
        # 5-Fold Cross Validation
        # That means in each fold, we train on 80% data and test on 20% data
        scores = cross_val_score(clf, X_all, y_all, cv=5)
        acc = scores.mean()

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(space_stream)
        results["space_sparse"].append(space_sparse)
        results["space_quantum"].append(space_quantum)
        results["accuracies"].append(acc)
        results["error_rates"].append(1.0 - acc)

    return results


def plot_parametric_hybrid(
    x_mean, x_std, y_mean, y_std, color, marker, label, linewidth, marker_size, ax=None
):
    # 1. Horizontal Tube (Accuracy/Error Variance)
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
    marker_indices.append(-3)
    # marker_indices = np.arange(len(x_mean))

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
            "mean_acc": [],
            "std_acc": [],
            "mean_err": [],
            "std_err": [],
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
                accs = np.array(by_min_df[mdf][k]["accuracy"])
                errs = np.array(by_min_df[mdf][k]["error"])

                if len(spaces) > 0:
                    # calculate mean and std error of the mean
                    sqrt_n = np.sqrt(len(spaces))
                    final_stats[k]["mean_space"].append(np.mean(spaces))
                    final_stats[k]["std_space"].append(np.std(spaces) / sqrt_n)
                    final_stats[k]["mean_acc"].append(np.mean(accs))
                    final_stats[k]["std_acc"].append(np.std(accs) / sqrt_n)
                    final_stats[k]["mean_err"].append(np.mean(errs))
                    final_stats[k]["std_err"].append(np.std(errs) / sqrt_n)

    else:
        print(f"Running Analysis over {n_pairs} random sets of 1v1 categories...")
        all_cats = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes")
        ).target_names  # type: ignore

        by_min_df = defaultdict(
            lambda: {k: {"space": [], "error": [], "accuracy": []} for k in keys}
        )

        for i in tqdm(range(n_pairs), desc="Category Pairs", leave=True):
            # Randomly select 2 categories
            cats = random.sample(all_cats, 2)

            tqdm.write(f"[{i + 1}/{n_pairs}] Group: {cats}")

            # Calculate Ridge accuracy and space
            res = get_ridge_results(cats)

            for j, mdf in enumerate(res["min_dfs"]):
                for k in keys:
                    by_min_df[mdf][k]["space"].append(res[f"space_{k}"][j])
                    by_min_df[mdf][k]["error"].append(res["error_rates"][j])
                    by_min_df[mdf][k]["accuracy"].append(res["accuracies"][j])

        # Compute Stats
        for mdf in sorted(by_min_df.keys()):
            for k in keys:
                spaces = np.array(by_min_df[mdf][k]["space"])
                accs = np.array(by_min_df[mdf][k]["accuracy"])
                errs = np.array(by_min_df[mdf][k]["error"])

                if len(spaces) > 0:
                    # calculate mean and std error of the mean
                    sqrt_n = np.sqrt(len(spaces))
                    final_stats[k]["mean_space"].append(np.mean(spaces))
                    final_stats[k]["std_space"].append(np.std(spaces) / sqrt_n)
                    final_stats[k]["mean_acc"].append(np.mean(accs))
                    final_stats[k]["std_acc"].append(np.std(accs) / sqrt_n)
                    final_stats[k]["mean_err"].append(np.mean(errs))
                    final_stats[k]["std_err"].append(np.std(errs) / sqrt_n)

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
        with open("20newsgroups_size_vs_accuracy.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to 20newsgroups_size_vs_accuracy.json")

    # Plot: Size vs Accuracy
    plt.figure(figsize=figsize)
    for k in keys:
        xm, xs, ym, ys = get_sorted_arrays(
            final_stats[k]["mean_acc"],
            final_stats[k]["std_acc"],
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
        0.835,
        8e4,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.873,
        9e3,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.94,
        7e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.xlabel("Accuracy")
    plt.xticks(
        [0.84, 0.86, 0.88, 0.90, 0.92, 0.94],
        ["84%", "86%", "88%", "90%", "92%", "94%"],
    )
    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.ylabel("Model size")
    plt.ylim(1e1, 2e5)
    # plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification")
    plt.tight_layout()
    plt.savefig("20newsgroups_size_vs_accuracy.pdf")
    print("Saved 20newsgroups_size_vs_accuracy.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="20newsgroups Model Size vs Accuracy Analysis"
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

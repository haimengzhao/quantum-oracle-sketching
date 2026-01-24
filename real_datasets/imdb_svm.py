
import argparse
import json
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import imdb_utils

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

min_dfs = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 19, 21, 24, 28, 32, 36, 42, 48, 55, 62, 71, 81, 93, 106, 122, 139, 159, 181, 207, 236, 270, 308, 352, 402, 459, 524, 599, 684, 781, 891, 1018, 1162, 1327, 1515, 1730, 1976, 2256, 2576, 2941, 3358, 3835, 4379, 5000]
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

def get_ridge_results_full():
    # 1. Load Data
    X_all_raw, y_all = imdb_utils.load_imdb_data()

    # 2. Sweep min_df
    results = {
        "min_dfs": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracies_mean": [],
        "accuracies_std": [], # For error bars (SEM)
    }

    tqdm.write("Sweeping min_df for Full IMDB...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep"):
        vectorizer = TfidfVectorizer(min_df=mdf, stop_words="english")
        X_all = vectorizer.fit_transform(X_all_raw)
        # X_all.eliminate_zeros()

        shape = X_all.get_shape()
        feature_dim = shape[1]
        num_samples = shape[0]

        # Sparsity calculation
        row_sparsity = int(X_all.getnnz(axis=1).max())
        col_sparsity = int(X_all.getnnz(axis=0).max())
        sparsity = max(row_sparsity, col_sparsity)

        # --- Space Calculations ---
        space_stream = feature_dim
        space_sparse = X_all.getnnz()

        space_quantum = (
            2 * np.ceil(np.log2(num_samples + 2 * feature_dim))
            + np.ceil(np.log2(sparsity + 1))
            + 4 
        )

        # --- Ridge Training & Eval (CV) ---
        clf = RidgeClassifier(random_state=42, alpha=1.0, solver="auto")
        # 5-Fold Cross Validation
        scores = cross_val_score(clf, X_all, y_all, cv=5)
        
        acc_mean = scores.mean()
        # Calculate Standard Error of the Mean (SEM) = std / sqrt(n)
        acc_sem = scores.std() / np.sqrt(len(scores))

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(space_stream)
        results["space_sparse"].append(space_sparse)
        results["space_quantum"].append(space_quantum)
        results["accuracies_mean"].append(acc_mean)
        results["accuracies_std"].append(acc_sem)

    return results

def plot_parametric_hybrid(x_mean, x_std, y_mean, color, marker, label, linewidth, marker_size):
    # 1. Horizontal Tube (Accuracy SEM)
    # y_mean is Space (Log scale), which doesn't have variance here (single run)
    # x_mean is Accuracy, which has SEM
    
    # We want shade around x_mean defined by x_std
    y_vals = np.array(y_mean)
    x_vals = np.array(x_mean)
    x_errs = np.array(x_std)
    
    plt.fill_betweenx(
        y_vals, x_vals - x_errs, x_vals + x_errs, color=color, alpha=0.2, edgecolor="none"
    )

    # 2. Line
    plt.plot(x_vals, y_vals, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    # 3. Markers
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    target_x = np.linspace(x_min, x_max, num=num_markers)
    marker_indices = []
    for tx in target_x:
        idx = (np.abs(x_vals - tx)).argmin()
        if idx not in marker_indices:
            marker_indices.append(idx)
    marker_indices.append(-3) # Ensure high accuracy points are shown

    plt.scatter(
        x_vals[marker_indices],
        y_vals[marker_indices],
        marker=marker,
        color=color,
        label=label,
        alpha=0.9,
        s=marker_size,
        linewidth=linewidth,
    )

def get_sorted_arrays(x_mean, x_std, y_mean):
    # Sort by Y-axis metric (Space)
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
        raw_data = data["raw_data_by_min_df"]
        mdfs = sorted([int(k) for k in raw_data.keys()])
        
        final_stats = {k: {"mean_space": [], "mean_acc": [], "sem_acc": []} for k in keys}
        for mdf in mdfs:
             for k in keys:
                 final_stats[k]["mean_space"].append(raw_data[str(mdf)][k]["space"])
                 final_stats[k]["mean_acc"].append(raw_data[str(mdf)][k]["accuracy_mean"])
                 final_stats[k]["sem_acc"].append(raw_data[str(mdf)][k]["accuracy_sem"])

    else:
        print("Running Ridge Analysis on Full IMDB Dataset...")
        results = get_ridge_results_full()
        
        final_stats = {k: {"mean_space": [], "mean_acc": [], "sem_acc": []} for k in keys}
        
        data_to_save = {
            "dataset": "IMDB Full",
            "raw_data_by_min_df": {}
        }
        
        for i, mdf in enumerate(results["min_dfs"]):
            mdf_str = str(mdf)
            data_to_save["raw_data_by_min_df"][mdf_str] = {}
            for k in keys:
                space = results[f"space_{k}"][i]
                acc_mean = results["accuracies_mean"][i]
                acc_sem = results["accuracies_std"][i]
                
                final_stats[k]["mean_space"].append(space)
                final_stats[k]["mean_acc"].append(acc_mean)
                final_stats[k]["sem_acc"].append(acc_sem)
                
                data_to_save["raw_data_by_min_df"][mdf_str][k] = {
                    "space": space,
                    "accuracy_mean": acc_mean,
                    "accuracy_sem": acc_sem
                }

        with open("imdb_size_vs_accuracy.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to imdb_size_vs_accuracy.json")

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
        0.69,
        4e6,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.88,
        9e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        0.90,
        1.9e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.xlabel("Accuracy")
    plt.xticks(
        [0.70, 0.75, 0.80, 0.85, 0.90],
        ["70%", "75%", "80%", "85%", "90%"]
    )
    plt.xlim(0.68, 0.92)
    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.ylabel("Machine size")
    plt.ylim(1e1, 1e7)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification (IMDB)")
    plt.tight_layout()
    plt.savefig("imdb_size_vs_accuracy.pdf")
    print("Saved imdb_size_vs_accuracy.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB Machine Size vs Accuracy Analysis")
    parser.add_argument("--load", type=str, default=None, help="Load analysis data from JSON file")
    args = parser.parse_args()

    run_analysis(load_file=args.load)

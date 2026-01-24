
import argparse
import json
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from tqdm import tqdm
import splice_utils

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
    tqdm.write("Loading Splice Junction dataset (binary: EI vs IE)...")
    X_full, y, label_names = splice_utils.load_splice_data(binary=True, min_samples=1)
    tqdm.write(f"Dataset shape: {X_full.shape}, Classes: {label_names}")
    
    tqdm.write("Computing Top Singular Vector (Ground Truth)...")
    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]

    var_max = np.linalg.norm(X_full @ v_full) ** 2
    D_full = X_full.shape[1]

    results = {
        "min_samples": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovery": [],
    }

    tqdm.write("Sweeping min_samples for Splice Junction...")

    for min_samp in tqdm(min_samples_list, desc="min_samples Sweep"):
        X_trunc, feature_indices = splice_utils.filter_features_by_frequency(X_full, min_samp)
        
        # Skip if matrix too small for SVD
        if X_trunc.shape[1] < 2 or X_trunc.shape[0] < 2:
            continue

        shape = X_trunc.get_shape()
        feature_dim = shape[1]
        num_samples = shape[0]

        row_sparsity = int(X_trunc.getnnz(axis=1).max())
        col_sparsity = int(X_trunc.getnnz(axis=0).max())
        sparsity = max(row_sparsity, col_sparsity)

        space_stream = feature_dim 
        space_sparse = X_trunc.getnnz() 

        space_quantum = (
            2 * np.ceil(np.log2(num_samples + feature_dim))
            + np.ceil(np.log2(sparsity))
            + 4 
        )

        _, _, vt_trunc = svds(X_trunc.asfptype(), k=1)
        v_trunc = vt_trunc[0]

        v_lifted = np.zeros(D_full)
        v_lifted[feature_indices] = v_trunc

        norm = np.linalg.norm(v_lifted)
        v_lifted = v_lifted / norm

        var_captured = np.linalg.norm(X_full @ v_lifted) ** 2 
        recovery = var_captured / var_max

        results["min_samples"].append(min_samp)
        results["space_streaming"].append(space_stream)
        results["space_sparse"].append(space_sparse)
        results["space_quantum"].append(space_quantum)
        results["variance_recovery"].append(recovery)

    return results

def plot_parametric_hybrid(x_mean, y_mean, color, marker, label, linewidth, marker_size):
    plt.plot(x_mean, y_mean, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    x_min, x_max = np.min(x_mean), np.max(x_mean)
    target_x = np.linspace(x_min, x_max, num=num_markers)
    marker_indices = []
    for tx in target_x:
        idx = (np.abs(x_mean - tx)).argmin()
        if idx not in marker_indices:
            marker_indices.append(idx)
    # Add final points, but cap to array size
    for offset in [-5, -10, -15]:
        if abs(offset) < len(x_mean) and offset not in marker_indices:
            marker_indices.append(offset)

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
                 final_stats[k]["mean_var"].append(raw_data[str(ms)][k]["variance_recovery"])

    else:
        print("Running PCA Analysis on Splice Junction Dataset (Binary)...")
        results = get_pca_results_full()
        
        final_stats = {k: {"mean_space": [], "mean_var": []} for k in keys}
        
        data_to_save = {
            "dataset": "Splice Junction (Binary: EI vs IE)",
            "raw_data_by_min_samples": {}
        }
        
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
                    "variance_recovery": rec
                }
                
        with open("splice_size_vs_variance.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
        print("Saved raw data to splice_size_vs_variance.json")

    # Plot
    plt.figure(figsize=figsize)
    for k in keys:
        xm, ym = get_sorted_arrays(
            final_stats[k]["mean_var"],
            final_stats[k]["mean_space"],
        )
        ind = xm >=0.3
        plot_parametric_hybrid(
            xm[ind],
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
    var_vals = final_stats["sparse"]["mean_var"]
    
    plt.text(
        0.4,
        8e4,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    plt.text(
        0.98,
        4e3,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    plt.text(
        0.98,
        1.5e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    plt.yscale("log")
    plt.ylim(1e1, 2e5)
    plt.xlabel("Relative explained variance")
    
    plt.xticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"])
    plt.xlim(0.3, 1.05)
    
    plt.tick_params(direction="in", which="both", top=False, right=True)
    ax = plt.gca()
    ax.set_ylabel("Machine size")
    ax.tick_params(axis="y")
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Dimension reduction (Splice)")
    plt.tight_layout()
    plt.savefig("splice_size_vs_variance.pdf")
    print("Saved splice_size_vs_variance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splice Junction Machine Size vs Variance Analysis")
    parser.add_argument("--load", type=str, default=None, help="Load analysis data from JSON file")
    args = parser.parse_args()

    run_analysis(load_file=args.load)

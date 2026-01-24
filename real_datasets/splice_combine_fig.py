
import argparse
import json
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

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
markersize = {"streaming": 50, "sparse": 50, "quantum": 30}
linewidth_marker = {"streaming": 0, "sparse": 0, "quantum": 0}


def plot_parametric_hybrid(ax, x_mean, x_std, y_mean, color, marker, label, linewidth, marker_size):
    x_vals = np.array(x_mean)
    y_vals = np.array(y_mean)
    x_errs = np.array(x_std) if x_std is not None else None
    
    if x_errs is not None and np.any(x_errs > 0):
        ax.fill_betweenx(
            y_vals, x_vals - x_errs, x_vals + x_errs, color=color, alpha=0.2, edgecolor="none"
        )
    
    ax.plot(x_vals, y_vals, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    ax.scatter(
        x_vals,
        y_vals,
        marker=marker,
        color=color,
        label=label,
        alpha=0.9,
        s=marker_size,
        linewidth=linewidth,
    )


def get_sorted_arrays(x_mean, x_std, y_mean, y_std):
    data = sorted(zip(x_mean, x_std, y_mean, y_std), key=lambda x: x[2])
    return (
        np.array([d[0] for d in data]),
        np.array([d[1] for d in data]),
        np.array([d[2] for d in data]),
        np.array([d[3] for d in data]),
    )


def compute_stats_from_json(data, metric_type):
    keys = ["streaming", "sparse", "quantum"]
    final_stats = {
        k: {
            "mean_space": [],
            "std_space": [],
            "mean_x": [],
            "std_x": [],
        }
        for k in keys
    }

    raw_data = data["raw_data_by_min_samples"]
    min_samps = sorted([int(k) for k in raw_data.keys()])

    for ms in min_samps:
        for k in keys:
            entry = raw_data[str(ms)][k]
            
            space = float(entry["space"])
            final_stats[k]["mean_space"].append(space)
            final_stats[k]["std_space"].append(0.0)
            
            if metric_type == "accuracy":
                acc_mean = float(entry["accuracy_mean"])
                acc_sem = float(entry["accuracy_sem"])
                final_stats[k]["mean_x"].append(acc_mean)
                final_stats[k]["std_x"].append(acc_sem)
            
            elif metric_type == "variance":
                var_rec = float(entry["variance_recovery"])
                final_stats[k]["mean_x"].append(var_rec)
                final_stats[k]["std_x"].append(0.0)

    for k in keys:
        for field in final_stats[k]:
            final_stats[k][field] = np.array(final_stats[k][field])

    return final_stats


def plot_accuracy_panel(ax, stats):
    keys = ["streaming", "sparse", "quantum"]
    for k in keys:
        xm, xs, ym, ys = get_sorted_arrays(
            stats[k]["mean_x"],
            stats[k]["std_x"],
            stats[k]["mean_space"],
            stats[k]["std_space"],
        )
        # Filter to xm > 0.775
        ind = xm > 0.775
        plot_parametric_hybrid(
            ax,
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
    
    ax.text(
        0.775,
        8e4,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    ax.text(
        0.845,
        1e3,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        0.86,
        1.5e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e5)
    ax.set_xlabel("Accuracy")
    
    ax.set_xticks([0.78, 0.8, 0.82, 0.84, 0.86])
    ax.set_xticklabels(["78%", "80%", "82%", "84%", "86%"])
    ax.set_xlim(0.77, 0.875)
    
    ax.set_ylabel("Machine size")
    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Binary classification")


def plot_variance_panel(ax, stats):
    keys = ["streaming", "sparse", "quantum"]
    for k in keys:
        xm, xs, ym, ys = get_sorted_arrays(
            stats[k]["mean_x"],
            stats[k]["std_x"],
            stats[k]["mean_space"],
            stats[k]["std_space"],
        )
        # Filter to xm >= 0.3
        ind = xm >= 0.3
        plot_parametric_hybrid(
            ax,
            xm[ind],
            None,
            ym[ind],
            colors[k],
            markers[k],
            labels[k],
            linewidth_marker[k],
            markersize[k],
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    
    ax.text(
        0.4,
        8e4,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    ax.text(
        0.98,
        4e3,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        0.98,
        1.5e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e5)
    ax.set_xlabel("Relative explained variance")
    
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["40%", "60%", "80%", "100%"])
    ax.set_xlim(0.3, 1.05)
    
    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Dimension reduction")


def main():
    parser = argparse.ArgumentParser(
        description="Combine Splice size-vs-accuracy and size-vs-variance plots."
    )
    parser.add_argument(
        "--accuracy-json",
        type=str,
        default="splice_size_vs_accuracy.json",
        help="Path to accuracy JSON file.",
    )
    parser.add_argument(
        "--variance-json",
        type=str,
        default="splice_size_vs_variance.json",
        help="Path to variance JSON file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="splice_combine.pdf",
        help="Output figure path.",
    )
    args = parser.parse_args()

    with open(args.accuracy_json, "r") as f:
        accuracy_data = json.load(f)
    with open(args.variance_json, "r") as f:
        variance_data = json.load(f)

    accuracy_stats = compute_stats_from_json(accuracy_data, "accuracy")
    variance_stats = compute_stats_from_json(variance_data, "variance")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True)
    plot_accuracy_panel(ax_left, accuracy_stats)
    plot_variance_panel(ax_right, variance_stats)

    ax_right.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

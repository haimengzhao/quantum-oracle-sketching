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
markersize = {"streaming": 50, "sparse": 50, "quantum": 30}
linewidth_marker = {"streaming": 0, "sparse": 0, "quantum": 0}


def plot_parametric_hybrid(
    ax,
    x_mean,
    x_std,
    y_mean,
    y_std,
    color,
    marker,
    label,
    linewidth,
    marker_size,
    accuracy_panel=True,
):
    if x_std is not None and np.any(x_std > 0):
        ax.fill_betweenx(
            y_mean,
            x_mean - x_std,
            x_mean + x_std,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )

    ax.plot(x_mean, y_mean, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    if accuracy_panel:
        marker_indices = np.arange(len(x_mean))
    else:
        x_min, x_max = np.min(x_mean), np.max(x_mean)
        target_x = np.linspace(x_min, x_max, num=num_markers)
        marker_indices = []
        for tx in target_x:
            idx = (np.abs(x_mean - tx)).argmin()
            if idx not in marker_indices:
                marker_indices.append(idx)
        marker_indices += [-5, -10, -15, -20]

    ax.scatter(
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
        plot_parametric_hybrid(
            ax,
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

    ax.text(
        0.81,
        2e6,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    ax.text(
        0.888,
        1.2e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        0.90,
        1.7e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 1e7)
    ax.set_xlabel("Accuracy")

    ax.set_xticks([0.80, 0.82, 0.84, 0.86, 0.88, 0.90])
    ax.set_xticklabels(["80%", "82%", "84%", "86%", "88%", "90%"])
    ax.set_xlim(0.795, 0.915)

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
        plot_parametric_hybrid(
            ax,
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

    ax.text(
        1,
        1e6,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        0.996,
        1.2e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        1,
        1.7e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 1e7)
    ax.set_xlabel("Relative explained variance")

    ax.set_xticks([0.92, 0.94, 0.96, 0.98, 1.0])
    ax.set_xticklabels(["92%", "94%", "96%", "98%", "100%"])
    ax.set_xlim(0.915, 1.005)

    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Dimension reduction")


def main():
    parser = argparse.ArgumentParser(
        description="Combine PBMC68k size-vs-accuracy and size-vs-variance plots."
    )
    parser.add_argument(
        "--accuracy-json",
        type=str,
        default="pbmc68k_size_vs_accuracy.json",
        help="Path to accuracy JSON file.",
    )
    parser.add_argument(
        "--variance-json",
        type=str,
        default="pbmc68k_size_vs_variance.json",
        help="Path to variance JSON file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pbmc68k_combine.pdf",
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

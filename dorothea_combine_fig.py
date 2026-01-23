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
    color,
    marker,
    label,
    linewidth,
    marker_size,
    accuracy_panel=True,
):
    # 1. Horizontal Tube (Accuracy SEM/STD)
    if x_std is not None and np.any(x_std > 0):
        ax.fill_betweenx(
            y_mean,
            x_mean - x_std,
            x_mean + x_std,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )

    # 2. Line
    ax.plot(x_mean, y_mean, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    ax.scatter(
        x_mean,
        y_mean,
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


def compute_stats_from_json(data, metric_type):
    keys = ["streaming", "sparse", "quantum"]
    final_stats = {
        k: {
            "mean_space": [],
            "mean_x": [],
            "std_x": [],
        }
        for k in keys
    }

    raw_data = data["raw_data_by_min_df"]
    mdfs = sorted([int(k) for k in raw_data.keys()])

    for mdf in mdfs:
        for k in keys:
            if k not in raw_data[str(mdf)]:
                continue
            entry = raw_data[str(mdf)][k]

            space = float(entry["space"])
            final_stats[k]["mean_space"].append(space)

            if metric_type == "accuracy":
                acc_mean = float(entry["accuracy"])
                acc_std = float(entry.get("accuracy_std", 0.0))
                final_stats[k]["mean_x"].append(acc_mean)
                final_stats[k]["std_x"].append(acc_std)

            elif metric_type == "variance":
                var_rec = float(entry["variance_recovery"])
                final_stats[k]["mean_x"].append(var_rec)
                final_stats[k]["std_x"].append(0.0)

    for k in keys:
        for field in final_stats[k]:
            final_stats[k][field] = np.array(final_stats[k][field])  # type: ignore

    return final_stats


def plot_accuracy_panel(ax, stats):
    keys = ["streaming", "sparse", "quantum"]
    for k in keys:
        xm, xs, ym = get_sorted_arrays(
            stats[k]["mean_x"],
            stats[k]["std_x"],
            stats[k]["mean_space"],
        )
        ind = xm >= 0.6
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
            accuracy_panel=True,
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]

    ax.text(
        0.9,
        6e5,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        0.9,
        4e3,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        0.95,
        1.4e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e6)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0.58, 0.97)
    ax.set_xticks([0.60, 0.70, 0.80, 0.90])
    ax.set_xticklabels(["60%", "70%", "80%", "90%"])

    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Binary classification")


def plot_variance_panel(ax, stats):
    keys = ["streaming", "sparse", "quantum"]
    for k in keys:
        xm, xs, ym = get_sorted_arrays(
            stats[k]["mean_x"],
            stats[k]["std_x"],
            stats[k]["mean_space"],
        )
        ind = (xm >= 0.04) * (xm <= 1)
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
            accuracy_panel=False,
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]

    ax.text(
        0.15,
        8e5,
        "Classical sparse / QRAM",
        color=colors["sparse"],
        fontsize=10,
        path_effects=halo,
    )
    ax.text(
        0.9,
        4e4,
        "Classical streaming",
        color=colors["streaming"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )
    ax.text(
        1,
        1.4e1,
        "Quantum oracle sketching",
        color=colors["quantum"],
        fontsize=10,
        path_effects=halo,
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e6)
    ax.set_xlabel("Relative explained variance")
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Dimension reduction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-json", default="dorothea_size_vs_accuracy.json")
    parser.add_argument("--variance-json", default="dorothea_size_vs_variance.json")
    parser.add_argument("--out", default="dorothea_combine.pdf")
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

    ax_left.set_ylabel("Machine size")
    ax_right.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

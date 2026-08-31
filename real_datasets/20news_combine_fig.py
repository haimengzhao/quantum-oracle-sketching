"""Combined two-panel 20 Newsgroups figure: classification and dimension reduction.

Loads the accuracy and variance sweep JSONs written by 20news_svm.py and
20news_pca.py and draws both panels side by side; mode bucket_jl
additionally overlays the sparse-JL streaming curve on the bucket panels.
"""

import argparse
import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import sketch_utils
import sweep_utils

np.random.seed(42)
sweep_utils.apply_plot_style()

num_markers = 20
rare_extra_markers = [-3]


def plot_accuracy_panel(ax, stats, mode="rare", jl_stats=None):
    for k in sweep_utils.METHOD_KEYS:
        xm, xs, ym = sweep_utils.sort_by_space(
            stats[k]["metric_mean"], stats[k]["metric_sem"], stats[k]["space_mean"]
        )
        sweep_utils.plot_curve(
            ax,
            k,
            xm,
            xs,
            ym,
            label=(sketch_utils.streaming_label(mode) if k == "streaming" else None),
            show_all_markers=(mode != "rare"),
            num_markers=num_markers,
            extra_marker_indices=rare_extra_markers,
        )
    if jl_stats is not None:
        sweep_utils.plot_jl_overlay(ax, jl_stats)

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if mode in ("bucket", "jl"):
        ax.text(
            0.2,
            0.82,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        ax.text(
            0.78,
            0.58,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
            ha="right",
        )
        ax.text(
            0.15,
            0.06,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        if jl_stats is not None:
            ax.text(
                0.58,
                0.46,
                "Classical sparse JL",
                color=sweep_utils.COLORS["jl_streaming"],
                fontsize=10,
                path_effects=halo,
                transform=ax.transAxes,
                ha="right",
            )
    else:
        ax.text(
            0.83,
            9e4,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
        )
        ax.text(
            0.934,
            9e3,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        ax.text(
            0.94,
            7e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e5)
    ax.set_xlabel("Accuracy")
    if mode in ("bucket", "jl"):
        ax.set_xticks([0.65, 0.75, 0.85, 0.95])
        ax.set_xticklabels(["65%", "75%", "85%", "95%"])
        ax.set_xlim(0.62, 0.97)
    else:
        ax.set_xticks([0.84, 0.86, 0.88, 0.90, 0.92, 0.94])
        ax.set_xticklabels(["84%", "86%", "88%", "90%", "92%", "94%"])
    ax.set_ylabel("Machine size")
    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Binary classification")


def plot_variance_panel(ax, stats, mode="rare", jl_stats=None):
    for k in sweep_utils.METHOD_KEYS:
        xm, xs, ym = sweep_utils.sort_by_space(
            stats[k]["metric_mean"], stats[k]["metric_sem"], stats[k]["space_mean"]
        )
        sweep_utils.plot_curve(
            ax,
            k,
            xm,
            xs,
            ym,
            label=(sketch_utils.streaming_label(mode) if k == "streaming" else None),
            show_all_markers=(mode != "rare"),
            num_markers=num_markers,
            extra_marker_indices=rare_extra_markers,
        )
    if jl_stats is not None:
        sweep_utils.plot_jl_overlay(ax, jl_stats)

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if mode in ("bucket", "jl"):
        ax.text(
            0.2,
            0.85,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        ax.text(
            0.98,
            0.58,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
            ha="right",
        )
        ax.text(
            0.15,
            0.06,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        if jl_stats is not None:
            ax.text(
                0.58,
                0.75,
                "Classical sparse JL",
                color=sweep_utils.COLORS["jl_streaming"],
                fontsize=10,
                path_effects=halo,
                transform=ax.transAxes,
                ha="right",
            )
    else:
        ax.text(
            0.535,
            9e4,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
        )
        ax.text(
            0.98,
            9e3,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        ax.text(
            1,
            7e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e5)
    ax.set_xlabel("Relative explained variance")
    if mode in ("bucket", "jl"):
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["25%", "50%", "75%", "100%"])
        ax.set_xlim(0.08, 1.03)
    else:
        ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticklabels(["60%", "70%", "80%", "90%", "100%"])
        ax.set_xlim(0.52, 1.03)
    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Dimension reduction")


def main():
    parser = argparse.ArgumentParser(
        description="Combine 20newsgroups size-vs-accuracy and size-vs-variance plots."
    )
    parser.add_argument(
        "--accuracy-json",
        type=str,
        default=None,
        help="Path to accuracy JSON file.",
    )
    parser.add_argument(
        "--variance-json",
        type=str,
        default=None,
        help="Path to variance JSON file.",
    )
    parser.add_argument(
        "--jl-accuracy-json",
        type=str,
        default=None,
        help="Path to JL accuracy JSON for bucket_jl mode.",
    )
    parser.add_argument(
        "--jl-variance-json",
        type=str,
        default=None,
        help="Path to JL variance JSON for bucket_jl mode.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output figure path.",
    )
    parser.add_argument(
        "--mode",
        choices=["rare", "bucket", "jl", "bucket_jl"],
        required=True,
    )
    args = parser.parse_args()

    if args.mode in ("bucket", "bucket_jl"):
        if args.accuracy_json is None:
            args.accuracy_json = "20newsgroups_bucket_size_vs_accuracy.json"
        if args.variance_json is None:
            args.variance_json = "20newsgroups_bucket_size_vs_variance.json"
        if args.mode == "bucket_jl":
            if args.jl_accuracy_json is None:
                args.jl_accuracy_json = "20newsgroups_jl_size_vs_accuracy.json"
            if args.jl_variance_json is None:
                args.jl_variance_json = "20newsgroups_jl_size_vs_variance.json"
    elif args.mode == "jl":
        if args.accuracy_json is None:
            args.accuracy_json = "20newsgroups_jl_size_vs_accuracy.json"
        if args.variance_json is None:
            args.variance_json = "20newsgroups_jl_size_vs_variance.json"
    else:
        if args.accuracy_json is None:
            args.accuracy_json = "20newsgroups_size_vs_accuracy.json"
        if args.variance_json is None:
            args.variance_json = "20newsgroups_size_vs_variance.json"

    with open(args.accuracy_json, "r") as f:
        accuracy_data = json.load(f)
    with open(args.variance_json, "r") as f:
        variance_data = json.load(f)
    if args.mode != "bucket_jl":
        sketch_utils.validate_truncation_data(
            accuracy_data, args.mode, args.accuracy_json
        )
        sketch_utils.validate_truncation_data(
            variance_data, args.mode, args.variance_json
        )
    jl_accuracy_data = None
    jl_variance_data = None
    if args.mode == "bucket_jl":
        with open(args.jl_accuracy_json, "r") as f:
            jl_accuracy_data = json.load(f)
        with open(args.jl_variance_json, "r") as f:
            jl_variance_data = json.load(f)

    for path, data in [
        (args.accuracy_json, accuracy_data),
        (args.variance_json, variance_data),
    ]:
        is_feature_mode = "raw_data_by_n_features" in data
        if is_feature_mode != (args.mode in ("bucket", "jl", "bucket_jl")):
            raise ValueError(f"{path} does not match --mode {args.mode}")
        if args.mode == "bucket_jl" and data.get("truncation_mode") != "bucket":
            raise ValueError(f"{path} is not bucket data")
    if args.mode == "bucket_jl":
        for path, data in [
            (args.jl_accuracy_json, jl_accuracy_data),
            (args.jl_variance_json, jl_variance_data),
        ]:
            if "raw_data_by_n_features" not in data:
                raise ValueError(f"{path} does not contain JL feature-sweep data")
            if data.get("truncation_mode") != "jl":
                raise ValueError(f"{path} is not JL data")
            if data.get("jl_transform") != sketch_utils.SPARSE_JL_TRANSFORM:
                raise ValueError(
                    f"{path} was not generated with balanced signed sparse JL"
                )

    if args.out is None:
        if args.mode == "bucket":
            args.out = "20newsgroups_bucket_combine.pdf"
        elif args.mode == "bucket_jl":
            args.out = "20newsgroups_bucket_jl_combine.pdf"
        elif args.mode == "jl":
            args.out = "20newsgroups_jl_combine.pdf"
        else:
            args.out = "20newsgroups_combine.pdf"

    plot_mode = "bucket" if args.mode == "bucket_jl" else args.mode
    accuracy_stats = sweep_utils.load_sweep_stats(accuracy_data, "accuracy")
    variance_stats = sweep_utils.load_sweep_stats(variance_data, "variance")
    jl_accuracy_stats = None
    jl_variance_stats = None
    if args.mode == "bucket_jl":
        jl_accuracy_stats = sweep_utils.load_sweep_stats(
            jl_accuracy_data, "accuracy", keys=["streaming"]
        )
        jl_variance_stats = sweep_utils.load_sweep_stats(
            jl_variance_data, "variance", keys=["streaming"]
        )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3.5), sharey=True)
    plot_accuracy_panel(
        ax_left, accuracy_stats, mode=plot_mode, jl_stats=jl_accuracy_stats
    )
    plot_variance_panel(
        ax_right, variance_stats, mode=plot_mode, jl_stats=jl_variance_stats
    )

    ax_right.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

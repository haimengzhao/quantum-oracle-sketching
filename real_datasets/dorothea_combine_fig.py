"""Combined two-panel Dorothea figure: classification and dimension reduction.

Loads the accuracy and variance sweep JSONs written by dorothea_svm.py and
dorothea_pca.py and draws both panels side by side; mode bucket_jl
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


def plot_accuracy_panel(
    ax, stats, mode="rare", jl_stats=None, labels="text", classical_alpha=1.0,
    adaptive=None,
):
    # Floating in-panel labels only in text mode; an external legend otherwise.
    text = ax.text if labels == "text" else (lambda *args, **kwargs: None)
    for k in sweep_utils.METHOD_KEYS:
        xm, xs, ym = sweep_utils.sort_by_space(
            stats[k]["metric_mean"], stats[k]["metric_sem"], stats[k]["space_mean"]
        )
        ind = xm >= 0.6
        sweep_utils.plot_curve(
            ax,
            k,
            xm[ind],
            xs[ind],
            ym[ind],
            alpha=(1.0 if k == "quantum" else classical_alpha),
            emphasize=(k == "quantum" and classical_alpha < 1.0),
            label=(sketch_utils.streaming_label(mode) if k == "streaming" else None),
            show_all_markers=True,
            fill_when_zero_err=False,
        )
    if jl_stats is not None:
        sweep_utils.plot_jl_overlay(ax, jl_stats, alpha=classical_alpha)

    if adaptive is not None:
        sweep_utils.plot_adaptive_overlay(ax, adaptive, alpha=classical_alpha)
    halo = [pe.withStroke(linewidth=3, foreground="white")]

    if mode in ("bucket", "jl"):
        text(
            0.05,
            0.92,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        text(
            0.85,
            0.5,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
            ha="right",
        )
        text(
            0.15,
            0.035,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        if jl_stats is not None:
            text(
                0.6,
                0.35,
                "Classical sparse JL",
                color=sweep_utils.COLORS["jl_streaming"],
                fontsize=10,
                path_effects=halo,
                transform=ax.transAxes,
                ha="right",
            )
    else:
        text(
            0.9,
            6e5,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        text(
            0.9,
            4e3,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        text(
            0.95,
            1.4e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e6)
    ax.set_xlabel("Accuracy")
    if mode in ("bucket", "jl"):
        ax.set_xlim(0.79, 0.95)
        ax.set_xticks([0.80, 0.84, 0.88, 0.92])
        ax.set_xticklabels(["80%", "84%", "88%", "92%"])
    else:
        ax.set_xlim(0.58, 0.97)
        ax.set_xticks([0.60, 0.70, 0.80, 0.90])
        ax.set_xticklabels(["60%", "70%", "80%", "90%"])

    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Binary classification")


def plot_variance_panel(
    ax, stats, mode="rare", jl_stats=None, labels="text", classical_alpha=1.0
):
    # Floating in-panel labels only in text mode; an external legend otherwise.
    text = ax.text if labels == "text" else (lambda *args, **kwargs: None)
    for k in sweep_utils.METHOD_KEYS:
        xm, xs, ym = sweep_utils.sort_by_space(
            stats[k]["metric_mean"], stats[k]["metric_sem"], stats[k]["space_mean"]
        )
        if mode in ("bucket", "jl"):
            ind = (xm >= 0) * (xm <= 1)
        else:
            ind = (xm >= 0.04) * (xm <= 1)
        sweep_utils.plot_curve(
            ax,
            k,
            xm[ind],
            xs[ind],
            ym[ind],
            alpha=(1.0 if k == "quantum" else classical_alpha),
            emphasize=(k == "quantum" and classical_alpha < 1.0),
            label=(sketch_utils.streaming_label(mode) if k == "streaming" else None),
            show_all_markers=True,
            fill_when_zero_err=False,
        )
    if jl_stats is not None:
        sweep_utils.plot_jl_overlay(ax, jl_stats, alpha=classical_alpha)

    halo = [pe.withStroke(linewidth=3, foreground="white")]

    if mode in ("bucket", "jl"):
        text(
            0.2,
            0.85,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        text(
            0.22,
            0.5,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
            ha="left",
        )
        text(
            0.15,
            0.035,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        if jl_stats is not None:
            text(
                0.7,
                0.76,
                "Classical sparse JL",
                color=sweep_utils.COLORS["jl_streaming"],
                fontsize=10,
                path_effects=halo,
                transform=ax.transAxes,
                ha="right",
            )
    else:
        text(
            0.15,
            8e5,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
        )
        text(
            0.9,
            4e4,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        text(
            1,
            1.4e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    ax.set_yscale("log")
    ax.set_ylim(1e1, 2e6)
    ax.set_xlabel("Relative explained variance")
    if mode in ("bucket", "jl"):
        ax.set_xlim(0.35, 1.05)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["40%", "60%", "80%", "100%"])
    else:
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    ax.set_title("Dimension reduction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-json", default=None)
    parser.add_argument("--variance-json", default=None)
    parser.add_argument("--jl-accuracy-json", default=None)
    parser.add_argument("--jl-variance-json", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--mode",
        choices=["rare", "bucket", "jl", "bucket_jl"],
        required=True,
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="overlay the adaptive sketching curves (AWM-Sketch, MISSION) from "
        "the adaptive survey JSON on the classification panel",
    )
    parser.add_argument(
        "--adaptive-json",
        type=str,
        default="survey_adaptive_dorothea.json",
        help="adaptive survey JSON (with --adaptive)",
    )
    parser.add_argument(
        "--labels",
        choices=["text", "none"],
        default="text",
        help="in-panel method labels (text) or none (for an external legend)",
    )
    parser.add_argument(
        "--classical-alpha",
        type=float,
        default=1.0,
        help="opacity of the classical curves; <1 also draws the quantum curve "
        "heavier and on top",
    )
    args = parser.parse_args()

    if args.mode in ("bucket", "bucket_jl"):
        if args.accuracy_json is None:
            args.accuracy_json = "dorothea_bucket_size_vs_accuracy.json"
        if args.variance_json is None:
            args.variance_json = "dorothea_bucket_size_vs_variance.json"
        if args.mode == "bucket_jl":
            if args.jl_accuracy_json is None:
                args.jl_accuracy_json = "dorothea_jl_size_vs_accuracy.json"
            if args.jl_variance_json is None:
                args.jl_variance_json = "dorothea_jl_size_vs_variance.json"
    elif args.mode == "jl":
        if args.accuracy_json is None:
            args.accuracy_json = "dorothea_jl_size_vs_accuracy.json"
        if args.variance_json is None:
            args.variance_json = "dorothea_jl_size_vs_variance.json"
    else:
        if args.accuracy_json is None:
            args.accuracy_json = "dorothea_size_vs_accuracy.json"
        if args.variance_json is None:
            args.variance_json = "dorothea_size_vs_variance.json"

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

    out_given = args.out is not None
    if args.out is None:
        if args.mode == "bucket":
            args.out = "dorothea_bucket_combine.pdf"
        elif args.mode == "bucket_jl":
            args.out = "dorothea_bucket_jl_combine.pdf"
        elif args.mode == "jl":
            args.out = "dorothea_jl_combine.pdf"
        else:
            args.out = "dorothea_combine.pdf"

    if args.adaptive and not out_given:
        args.out = args.out.replace("_combine.pdf", "_adaptive_combine.pdf")
    adaptive_curves = (
        sweep_utils.load_adaptive_curves(args.adaptive_json) if args.adaptive else None
    )

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
        ax_left,
        accuracy_stats,
        mode=plot_mode,
        jl_stats=jl_accuracy_stats,
        labels=args.labels,
        classical_alpha=args.classical_alpha,
        adaptive=adaptive_curves,
    )
    plot_variance_panel(
        ax_right,
        variance_stats,
        mode=plot_mode,
        jl_stats=jl_variance_stats,
        labels=args.labels,
        classical_alpha=args.classical_alpha,
    )

    ax_left.set_ylabel("Machine size")
    ax_right.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

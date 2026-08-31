"""Shared machinery for the machine-size vs performance sweeps.

Every dataset script pairs a compute kernel (load the dataset, evaluate the
task on truncated or sketched features) with the common pipeline implemented
here: machine-size accounting, JSON persistence, statistics, plotting
primitives, and the command line. Both the fresh-run path and --load plot
through the same loader, load_sweep_stats, so a figure is always a pure
function of its JSON.

The canonical JSON layout written by save_sweep_json is

    {"dataset": ..., <extra top-level fields>, <raw key>: {param: {method:
        {"space", <metric mean>, <metric sem>[, raw score arrays]}}},
        <sketch provenance from sketch_utils.sketch_metadata>}

where method is one of METHOD_KEYS and the raw arrays are stored per seed,
per pair, or per seed and pair, matching the sweep's structure.
load_sweep_stats reads exactly this layout.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import sketch_utils


METHOD_KEYS = ("streaming", "sparse", "quantum")
FIGSIZE = (3.5, 3.5)

RC_PARAMS = {
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

COLORS = {
    "quantum": "#CD591A",
    "streaming": "#2657AF",
    "sparse": "#606060",
    "jl_streaming": "#2A8C55",
}
LABELS = {
    "streaming": "Classical streaming",
    "sparse": "Classical sparse / QRAM",
    "quantum": "Quantum oracle sketching",
    "jl_streaming": "Classical sparse JL",
}
MARKERS = {"streaming": "P", "sparse": "X", "quantum": "D", "jl_streaming": "o"}
MARKER_SIZES = {"streaming": 50, "sparse": 50, "quantum": 30, "jl_streaming": 42}
MARKER_LINEWIDTHS = {"streaming": 0, "sparse": 0, "quantum": 0, "jl_streaming": 0}


def apply_plot_style():
    """Apply the shared publication plot style."""
    plt.rcParams.update(RC_PARAMS)


def qos_machine_size(n_samples, n_features, sparsity, task):
    """Machine size of quantum oracle sketching, in fundamental memory units.

    Following the accounting in the Supplementary Information, the total is
    the logical qubits maintained throughout the algorithm plus one classical
    scalar register for the running average of the prediction:

        LS-SVM: 2 ceil(log2(N + 2D)) + ceil(log2(s + 1)) + 3 qubits + 1 float
        PCA:    2 ceil(log2(N + D))  + ceil(log2(s))     + 3 qubits + 1 float

    For LS-SVM, the block-encoded augmented matrix [X; lambda I] has a
    Hermitian embedding of dimension (N + 2D) x (N + 2D) and sparsity s + 1;
    for PCA, X itself is block-encoded with no regularization augmentation,
    giving dimension (N + D) x (N + D) and sparsity s. In both cases the
    sparse index oracle needs 2 ceil(log2(dimension)) + ceil(log2(sparsity))
    + 2 qubits (the 2 for QSVT); the element oracle, the state preparation of
    the label/guiding vector, and the linear-system or ground-state QSVT
    ancilla all reuse those qubits, and the interferometric overlap
    measurement adds 1 extra ancilla qubit.
    """
    if task == "svm":
        return (
            2 * np.ceil(np.log2(n_samples + 2 * n_features))
            + np.ceil(np.log2(sparsity + 1))
            + 4
        )
    if task == "pca":
        return (
            2 * np.ceil(np.log2(n_samples + n_features))
            + np.ceil(np.log2(sparsity))
            + 4
        )
    raise ValueError("task must be 'svm' or 'pca'")


def is_sketch_mode(mode):
    """True for a registered sketch mode, False for 'rare'; raise otherwise."""
    if mode == "rare":
        return False
    if mode in sketch_utils.OBLIVIOUS_SKETCHES:
        return True
    raise ValueError(
        f"mode must be 'rare' or one of {list(sketch_utils.OBLIVIOUS_SKETCHES)}"
    )


def build_arg_parser(description):
    """Argument parser shared by the sweep scripts (--load, --mode, --n-seeds)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--load", type=str, default=None, help="Load analysis data from JSON file"
    )
    parser.add_argument(
        "--mode",
        choices=["rare", *sketch_utils.OBLIVIOUS_SKETCHES],
        required=True,
        help="rare: original rare-feature truncation; "
        + "; ".join(
            f"{key}: {sketch.description}"
            for key, sketch in sketch_utils.OBLIVIOUS_SKETCHES.items()
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="number of random sketch seeds to average over",
    )
    return parser


def sweep_main(description, run_analysis):
    """Command-line entry point shared by the single-run sweep scripts."""
    args = build_arg_parser(description).parse_args()
    run_analysis(load_file=args.load, mode=args.mode, n_sketch_seeds=args.n_seeds)


# --- statistics -------------------------------------------------------------

_ACCURACY_SCORE_KEYS = (
    "accuracy_scores_by_seed_pair",
    "accuracy_scores_by_pair",
    "accuracy_scores_by_seed",
    "accuracy_scores",
)
_VARIANCE_SCORE_KEYS = (
    "variance_recovery_by_seed_pair",
    "variance_recovery_by_pair",
    "variance_recovery_by_seed",
)
_SPACE_ARRAY_KEYS = ("space_by_seed_pair", "space_by_pair", "space_by_seed")


def metric_stats(entry, metric):
    """(mean, SEM) of one method record, recomputed from the raw scores."""
    if metric == "accuracy":
        for key in _ACCURACY_SCORE_KEYS:
            if key in entry:
                return sketch_utils.mean_and_sem(entry[key])
        raise ValueError(f"record has no accuracy scores: {sorted(entry)}")
    if metric == "variance":
        for key in _VARIANCE_SCORE_KEYS:
            if key in entry:
                return sketch_utils.mean_and_sem(entry[key])
        return (
            float(entry["variance_recovery"]),
            float(entry["variance_recovery_sem"]),
        )
    raise ValueError("metric must be 'accuracy' or 'variance'")


def space_stats(entry):
    """(mean, SEM) of one method record's machine size."""
    for key in _SPACE_ARRAY_KEYS:
        if key in entry:
            _, sem = sketch_utils.mean_and_sem(entry[key])
            return float(entry["space"]), sem
    return float(entry["space"]), 0.0


def load_sweep_stats(data, metric, rare_raw_key="raw_data_by_min_df", keys=METHOD_KEYS):
    """Normalize a sweep JSON into per-method plotting statistics.

    Returns {method: {"space_mean", "space_sem", "metric_mean", "metric_sem"}}
    with values as numpy arrays ordered by increasing sweep parameter.
    """
    if "raw_data_by_n_features" in data:
        raw_data = data["raw_data_by_n_features"]
    else:
        raw_data = data[rare_raw_key]
    by_param = {int(p): v for p, v in raw_data.items()}

    stats = {
        k: {"space_mean": [], "space_sem": [], "metric_mean": [], "metric_sem": []}
        for k in keys
    }
    for param in sorted(by_param):
        for k in keys:
            entry = by_param[param][k]
            space_mean, space_sem = space_stats(entry)
            metric_mean, metric_sem = metric_stats(entry, metric)
            stats[k]["space_mean"].append(space_mean)
            stats[k]["space_sem"].append(space_sem)
            stats[k]["metric_mean"].append(metric_mean)
            stats[k]["metric_sem"].append(metric_sem)

    for k in keys:
        for field in stats[k]:
            stats[k][field] = np.array(stats[k][field])
    return stats


# --- persistence ------------------------------------------------------------

def save_sweep_json(
    results,
    *,
    metric,
    dataset_name,
    output_json,
    sketch_mode,
    param_name,
    raw_key,
    mode=None,
    seeds=None,
    by_suffix="by_seed",
    rare_scores_key=None,
    extra_top=None,
):
    """Assemble the canonical sweep JSON from kernel results and write it.

    Kernel results hold, per sweep parameter, the machine sizes (per seed for
    sketch sweeps, scalar for rare sweeps) and the raw scores; the recorded
    means and SEMs are recomputed here from those raw values so the file is
    self-consistent. Sketch sweeps store the raw arrays under "*_{by_suffix}"
    together with the provenance from sketch_utils.sketch_metadata; rare
    accuracy sweeps store the raw scores under rare_scores_key.
    """
    data_to_save = {"dataset": dataset_name}
    if extra_top:
        data_to_save.update(extra_top)
    data_to_save[raw_key] = {}
    if sketch_mode:
        data_to_save.update(
            sketch_utils.sketch_metadata(mode, seeds, results[param_name])
        )

    for i, param in enumerate(results[param_name]):
        record = {}
        for k in METHOD_KEYS:
            spaces = results[f"space_{k}"][i]
            if metric == "accuracy":
                scores = results["accuracy_scores"][i]
                mean, sem = sketch_utils.mean_and_sem(scores)
                if sketch_mode:
                    entry = {
                        "space": float(np.mean(spaces)),
                        "accuracy_mean": mean,
                        "accuracy_sem": sem,
                        f"space_{by_suffix}": spaces,
                        f"accuracy_scores_{by_suffix}": scores,
                    }
                else:
                    entry = {
                        "space": spaces,
                        "accuracy_mean": mean,
                        "accuracy_sem": sem,
                        rare_scores_key: scores,
                    }
            else:
                if sketch_mode:
                    recoveries = results["variance_recovery_by_seed"][i]
                    mean, sem = sketch_utils.mean_and_sem(recoveries)
                    entry = {
                        "space": float(np.mean(spaces)),
                        "variance_recovery": mean,
                        "variance_recovery_sem": sem,
                        f"space_{by_suffix}": spaces,
                        f"variance_recovery_{by_suffix}": recoveries,
                    }
                else:
                    entry = {
                        "space": spaces,
                        "variance_recovery": results["variance_recovery"][i],
                        "variance_recovery_sem": 0.0,
                    }
            record[k] = entry
        data_to_save[raw_key][str(param)] = record

    with open(output_json, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved raw data to {output_json}")
    return output_json


# --- plotting ---------------------------------------------------------------

def sort_by_space(metric_mean, metric_sem, space_mean):
    """Sort one curve by machine size; returns (metric, sem, space) arrays."""
    data = sorted(zip(metric_mean, metric_sem, space_mean), key=lambda x: x[2])
    return (
        np.array([d[0] for d in data]),
        np.array([d[1] for d in data]),
        np.array([d[2] for d in data]),
    )


def marker_indices(x_vals, num_markers, extra=()):
    """Thin markers to the curve points nearest num_markers even x targets.

    Deduplicates in target order and appends the fixed extra indices, exactly
    reproducing the marker placement of the published figures.
    """
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    target_x = np.linspace(x_min, x_max, num=num_markers)
    indices = []
    for tx in target_x:
        idx = (np.abs(x_vals - tx)).argmin()
        if idx not in indices:
            indices.append(idx)
    indices += list(extra)
    return indices


def plot_jl_overlay(ax, stats):
    """Overlay the sparse-JL streaming curve on a combined-figure panel."""
    xm, xs, ym = sort_by_space(
        stats["streaming"]["metric_mean"],
        stats["streaming"]["metric_sem"],
        stats["streaming"]["space_mean"],
    )
    if np.any(xs > 0):
        ax.fill_betweenx(
            ym,
            xm - xs,
            xm + xs,
            color=COLORS["jl_streaming"],
            alpha=0.10,
            edgecolor="none",
            zorder=1,
        )
    ax.plot(
        xm,
        ym,
        linestyle=(0, (1, 1.15)),
        color=COLORS["jl_streaming"],
        linewidth=1.8,
        alpha=0.95,
        dash_capstyle="round",
        zorder=4,
    )
    ax.scatter(
        xm,
        ym,
        marker=MARKERS["jl_streaming"],
        facecolors="none",
        edgecolors=COLORS["jl_streaming"],
        label=LABELS["jl_streaming"],
        alpha=0.98,
        s=MARKER_SIZES["jl_streaming"],
        linewidth=1.2,
        zorder=5,
    )


def plot_curve(
    ax,
    key,
    x_mean,
    x_err,
    y_mean,
    *,
    label=None,
    show_all_markers=False,
    num_markers=40,
    extra_marker_indices=(),
    fill_when_zero_err=True,
):
    """Draw one method curve: SEM tube, line, and markers.

    key selects color/marker/label from the shared style; label overrides the
    default. When show_all_markers is False, markers are thinned via
    marker_indices. fill_when_zero_err=False suppresses the SEM tube when all
    errors vanish.
    """
    color = COLORS[key]
    if fill_when_zero_err or np.any(x_err > 0):
        ax.fill_betweenx(
            y_mean,
            x_mean - x_err,
            x_mean + x_err,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )

    ax.plot(x_mean, y_mean, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    if show_all_markers:
        indices = np.arange(len(x_mean))
    else:
        indices = marker_indices(x_mean, num_markers, extra_marker_indices)

    ax.scatter(
        x_mean[indices],
        y_mean[indices],
        marker=MARKERS[key],
        color=color,
        label=label if label is not None else LABELS[key],
        alpha=0.9,
        s=MARKER_SIZES[key],
        linewidth=MARKER_LINEWIDTHS[key],
    )

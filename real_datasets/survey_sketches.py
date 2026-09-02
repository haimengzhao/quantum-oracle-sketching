"""Survey of oblivious sketching methods on IMDb and PBMC68k.

Compares the terminal task performance of data-independent sketches at equal
streaming machine size (the sketch dimension): uniform feature subsampling,
feature hashing, and the balanced signed sparse JL family at sparsity 1, 2,
4, and 8. Every method reuses the preprocessing, models, grids, and seeds of
the corresponding main-figure sweep, so the feature hashing and sparsity-1 JL
curves reproduce the published ones.

Compute one panel's data with

    python survey_sketches.py --dataset imdb --task svm

(writes survey_imdb_svm.json), and assemble the 2x4 figure from the four
JSONs with

    python survey_sketches.py --plot

The top row shows machine size vs terminal performance on the main-figure
axes, together with the full-matrix quantum oracle sketching point (the
quantum endpoint of the corresponding main-figure sweep, read from its
JSON); the bottom row shows the per-seed paired performance difference to
the feature hashing baseline, which is therefore identically zero.

Compute jobs checkpoint after every sketch dimension into
survey_{dataset}_{task}.partial.json (removed once the final JSON is
written), so an interrupted run resumes where it left off. The PBMC68k
classification job parallelizes its independent per-pair cross-validations
over --n-jobs joblib workers; results are independent of the worker count.
"""

import argparse
import json
import os
from itertools import combinations

import imdb_utils
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sketch_utils
import sweep_utils
from joblib import Parallel, delayed
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)
sweep_utils.apply_plot_style()

SURVEY_METHODS = ("subsamp", "bucket", "jl", "jl2", "jl4", "jl8")
BASELINE_METHOD = "bucket"
n_sketch_seeds = 5

# Sketch-dimension grids: identical to the corresponding main-figure sweeps.
SKETCH_N_FEATURES = {
    ("imdb", "svm"): [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    ("imdb", "pca"): [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        49152,
        65536,
        81000,
    ],
    ("pbmc68k", "svm"): [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    ("pbmc68k", "pca"): [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        12288,
        16384,
        20000,
        24576,
        29491,
    ],
}

N_PAIRS = 100  # PBMC68k classification: random cell-type pairs, as in pbmc68k_svm.py


# --- compute ----------------------------------------------------------------

def make_imdb_classifier():
    # Matches imdb_svm.py.
    return RidgeClassifier(random_state=42, alpha=10, solver="auto")


def make_pbmc68k_classifier():
    # Matches pbmc68k_svm.py.
    return RidgeClassifier(
        random_state=42, alpha=200.0, solver="auto", class_weight="balanced"
    )


def get_random_pairs(n_classes, n_pairs, rng):
    # Matches pbmc68k_svm.py, so the survey averages over the same pairs.
    all_pairs = list(combinations(range(n_classes), 2))
    indices = rng.choice(len(all_pairs), size=n_pairs, replace=True)
    return [all_pairs[i] for i in indices]


def load_imdb():
    X_all_raw, y_all = imdb_utils.load_imdb_data()
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", dtype=np.float32)
    X_full = vectorizer.fit_transform(X_all_raw)
    X_full.eliminate_zeros()
    return X_full, np.asarray(y_all)


def load_pbmc68k():
    import pbmc68k_utils  # imports scvelo; keep IMDb-only runs light

    return pbmc68k_utils.load_pbmc68k_data(
        min_samples=1, normalize=True, binary=False
    )


def grid_with_endpoint(grid, full_dim):
    return [k for k in grid if k < full_dim] + [full_dim]


def _pair_cv_scores(X_pair, y_pair):
    """5-fold CV scores of one PBMC68k cell-type pair (joblib worker)."""
    scores = cross_val_score(make_pbmc68k_classifier(), X_pair, y_pair, cv=5)
    return [float(s) for s in scores]


def svm_record(X_full, seeds, k, evaluate, scores_key):
    """One LS-SVM grid point: evaluate every method at every sketch seed."""
    record = {"space": int(k)}
    endpoint_scores = None
    for method in SURVEY_METHODS:
        scores_by_seed = []
        for seed in seeds:
            X_sketch, info = sketch_utils.sketch_features(
                X_full, k, method, seed=seed
            )
            if info is None:
                # Full-dimension endpoint: every method returns the exact
                # data, so the (deterministic) scores are computed once.
                if endpoint_scores is None:
                    endpoint_scores = evaluate(X_sketch)
                scores_by_seed.append(endpoint_scores)
            else:
                scores_by_seed.append(evaluate(X_sketch))
        record[method] = {scores_key: scores_by_seed}
    return record


def pca_record(X_full, seeds, k, var_max):
    """One PCA grid point: variance recovery of the lifted top direction."""
    record = {"space": int(k)}
    for method in SURVEY_METHODS:
        recoveries = []
        for seed in seeds:
            X_sketch, info = sketch_utils.sketch_features(
                X_full, k, method, seed=seed
            )
            if info is None:
                recoveries.append(1.0)
                continue
            _, _, vt_sketch = svds(sketch_utils.svds_input(X_sketch), k=1)
            v_lifted = sketch_utils.lift_sketched_vector(vt_sketch[0], info, method)
            v_lifted = v_lifted / sketch_utils.lifted_vector_norm(v_lifted)
            var_captured = np.linalg.norm(X_full @ v_lifted) ** 2
            recoveries.append(float(var_captured / var_max))
        record[method] = {"variance_recovery_by_seed": recoveries}
    return record


def _top_component_energy(X_full):
    print("Computing Top Singular Vector (Ground Truth)...")
    _, _, vt_full = svds(sketch_utils.svds_input(X_full), k=1)
    return np.linalg.norm(X_full @ vt_full[0]) ** 2


def survey_imdb_svm(seeds, n_jobs):
    X_full, y_all = load_imdb()

    def evaluate(X_sketch):
        scores = cross_val_score(make_imdb_classifier(), X_sketch, y_all, cv=5)
        return [float(s) for s in scores]

    def compute(k):
        return svm_record(X_full, seeds, k, evaluate, "accuracy_scores_by_seed")

    grid = grid_with_endpoint(SKETCH_N_FEATURES[("imdb", "svm")], X_full.shape[1])
    return grid, compute, {}


def survey_pbmc68k_svm(seeds, n_jobs):
    X_full, y_full, label_names = load_pbmc68k()
    rng = np.random.default_rng(42)
    pairs = get_random_pairs(len(label_names), N_PAIRS, rng)

    # Row indices and binary labels per pair do not depend on the sketch.
    pair_rows = [
        np.flatnonzero((y_full == c1) | (y_full == c2)) for c1, c2 in pairs
    ]
    pair_labels = [
        (y_full[rows] == c2).astype(int)
        for rows, (_, c2) in zip(pair_rows, pairs)
    ]
    parallel = Parallel(n_jobs=n_jobs)

    def evaluate(X_sketch):
        # The per-pair cross-validations are independent; joblib preserves
        # input order, so results match a serial loop exactly.
        return parallel(
            delayed(_pair_cv_scores)(X_sketch[rows], y_pair)
            for rows, y_pair in zip(pair_rows, pair_labels)
        )

    def compute(k):
        return svm_record(
            X_full, seeds, k, evaluate, "accuracy_scores_by_seed_pair"
        )

    grid = grid_with_endpoint(SKETCH_N_FEATURES[("pbmc68k", "svm")], X_full.shape[1])
    extra = {
        "pairs": [(label_names[c1], label_names[c2]) for c1, c2 in pairs],
        "n_pairs": len(pairs),
    }
    return grid, compute, extra


def survey_imdb_pca(seeds, n_jobs):
    X_full, _ = load_imdb()
    var_max = _top_component_energy(X_full)

    def compute(k):
        return pca_record(X_full, seeds, k, var_max)

    grid = grid_with_endpoint(SKETCH_N_FEATURES[("imdb", "pca")], X_full.shape[1])
    return grid, compute, {}


def survey_pbmc68k_pca(seeds, n_jobs):
    X_full, _, _ = load_pbmc68k()
    var_max = _top_component_energy(X_full)

    def compute(k):
        return pca_record(X_full, seeds, k, var_max)

    grid = grid_with_endpoint(SKETCH_N_FEATURES[("pbmc68k", "pca")], X_full.shape[1])
    return grid, compute, {}


KERNELS = {
    ("imdb", "svm"): survey_imdb_svm,
    ("imdb", "pca"): survey_imdb_pca,
    ("pbmc68k", "svm"): survey_pbmc68k_svm,
    ("pbmc68k", "pca"): survey_pbmc68k_pca,
}


def _write_json_atomic(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def run_survey(dataset, task, n_seeds, n_jobs=-1):
    seeds = sketch_utils.sample_sketch_seeds(n_seeds)
    print(f"Running sketching survey on {dataset} ({task})...")
    print(f"Methods: {', '.join(SURVEY_METHODS)}")
    print(f"Averaging over random sketch seeds: {seeds}")
    grid, compute, extra = KERNELS[(dataset, task)](seeds, n_jobs)

    # A checkpoint is valid only for the exact same survey configuration.
    fingerprint = {
        "dataset": dataset,
        "task": task,
        "methods": list(SURVEY_METHODS),
        "sketch_seeds": seeds,
        "n_features_grid": [int(k) for k in grid],
    }
    if "n_pairs" in extra:
        fingerprint["n_pairs"] = extra["n_pairs"]

    checkpoint_path = f"survey_{dataset}_{task}.partial.json"
    raw = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        if {key: checkpoint.get(key) for key in fingerprint} != fingerprint:
            raise ValueError(
                f"{checkpoint_path} was written by an incompatible survey "
                "configuration; delete it to start from scratch"
            )
        raw = checkpoint["raw_data_by_n_features"]
        print(
            f"Resuming from {checkpoint_path}: "
            f"{len(raw)}/{len(grid)} sketch dimensions already done"
        )

    for k in tqdm(grid, desc="sketch dimension"):
        if str(k) in raw:
            continue
        raw[str(k)] = compute(k)
        _write_json_atomic(
            checkpoint_path, {**fingerprint, "raw_data_by_n_features": raw}
        )

    data = {
        "dataset": dataset,
        "task": task,
        "metric": "accuracy" if task == "svm" else "variance",
        "methods": list(SURVEY_METHODS),
        "baseline": BASELINE_METHOD,
        "method_transforms": {
            method: sketch_utils.OBLIVIOUS_SKETCHES[method].transform
            for method in SURVEY_METHODS
        },
        "sketch_seeds": seeds,
        "sketch_seed_sample_seed": sketch_utils.DEFAULT_SKETCH_SEED_SAMPLE_SEED,
        **extra,
        "raw_data_by_n_features": raw,
    }
    output_json = f"survey_{dataset}_{task}.json"
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"Saved raw data to {output_json}")
    return output_json


# --- plotting ---------------------------------------------------------------

STYLES = {
    "subsamp": dict(
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.5,
        line_alpha=0.9,
        marker="s",
        marker_size=36,
        filled=False,
        label="Subsampling",
    ),
    "bucket": dict(
        color=sweep_utils.COLORS["streaming"],
        linestyle="-",
        linewidth=1.5,
        line_alpha=0.9,
        marker="P",
        marker_size=50,
        filled=True,
        label="Feature hashing",
    ),
    "jl": dict(
        color="#2A8C55",
        linestyle=(0, (1, 1.15)),
        linewidth=1.8,
        line_alpha=0.95,
        marker="o",
        marker_size=42,
        filled=False,
        label="Sparse JL (s=1)",
    ),
    "jl2": dict(
        color="#227044",
        linestyle=(0, (1, 1.15)),
        linewidth=1.8,
        line_alpha=0.95,
        marker="o",
        marker_size=42,
        filled=False,
        label="Sparse JL (s=2)",
    ),
    "jl4": dict(
        color="#195433",
        linestyle=(0, (1, 1.15)),
        linewidth=1.8,
        line_alpha=0.95,
        marker="o",
        marker_size=42,
        filled=False,
        label="Sparse JL (s=4)",
    ),
    "jl8": dict(
        color="#113822",
        linestyle=(0, (1, 1.15)),
        linewidth=1.8,
        line_alpha=0.95,
        marker="o",
        marker_size=42,
        filled=False,
        label="Sparse JL (s=8)",
    ),
}

PANELS = [("imdb", "svm"), ("imdb", "pca"), ("pbmc68k", "svm"), ("pbmc68k", "pca")]
PANEL_TITLES = {
    ("imdb", "svm"): "IMDb classification",
    ("imdb", "pca"): "IMDb dimension reduction",
    ("pbmc68k", "svm"): "PBMC68k classification",
    ("pbmc68k", "pca"): "PBMC68k dimension reduction",
}
# Top-row axes: identical to the sketch-mode main figures.
TOP_AXES = {
    ("imdb", "svm"): dict(
        xlim=(0.57, 0.92), xticks=[0.60, 0.70, 0.80, 0.90], xlabel="Accuracy"
    ),
    ("imdb", "pca"): dict(
        xlim=(-0.1, 1.1),
        xticks=[0, 0.25, 0.5, 0.75, 1.0],
        xlabel="Relative explained variance",
    ),
    ("pbmc68k", "svm"): dict(
        xlim=(0.795, 0.91),
        xticks=[0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
        xlabel="Accuracy",
    ),
    ("pbmc68k", "pca"): dict(
        xlim=(-0.1, 1.1),
        xticks=[0, 0.25, 0.5, 0.75, 1.0],
        xlabel="Relative explained variance",
    ),
}
BOTTOM_XLABEL = "Diff. from feature hashing"
YLIM = (1e1, 2e5)

QOS_STYLE = dict(
    color=sweep_utils.COLORS["quantum"],
    marker="D",
    marker_size=45,
    label="Quantum oracle sketching",
)
# The full-matrix QOS reference is the quantum curve's full-dimension
# endpoint of the corresponding main-figure sweep, read verbatim from its
# JSON: (performance mean, performance sem, machine size) at exact recovery.
QOS_MAIN_JSON = {
    ("imdb", "svm"): (
        "imdb_bucket_size_vs_accuracy.json", "accuracy_mean", "accuracy_sem"
    ),
    ("imdb", "pca"): (
        "imdb_bucket_size_vs_variance.json",
        "variance_recovery",
        "variance_recovery_sem",
    ),
    ("pbmc68k", "svm"): (
        "pbmc68k_bucket_size_vs_accuracy.json", "accuracy_mean", "accuracy_sem"
    ),
    ("pbmc68k", "pca"): (
        "pbmc68k_bucket_size_vs_variance.json",
        "variance_recovery",
        "variance_recovery_sem",
    ),
}


def qos_endpoint(json_dir, dataset, task):
    """(mean, sem, machine size) of the published full-matrix QOS point."""
    fname, mean_key, sem_key = QOS_MAIN_JSON[(dataset, task)]
    path = os.path.join(json_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is required for the full-matrix QOS reference point; "
            "run the corresponding main-figure sweep first"
        )
    with open(path, "r") as f:
        main = json.load(f)
    raw = main["raw_data_by_n_features"]
    endpoint = raw[max(raw, key=int)]["quantum"]
    return endpoint[mean_key], endpoint[sem_key], endpoint["space"]


def method_scores(record, method):
    """The sole raw score array stored for a method at one sketch dimension."""
    return np.array(next(iter(record[method].values())), dtype=float)


def panel_stats(data):
    """Per-method performance and paired baseline-difference statistics."""
    raw = data["raw_data_by_n_features"]
    ks = sorted(int(k) for k in raw)
    stats = {
        m: {"space": [], "mean": [], "sem": [], "dmean": [], "dsem": []}
        for m in data["methods"]
    }
    for k in ks:
        record = raw[str(k)]
        base = method_scores(record, data["baseline"])
        for m in data["methods"]:
            scores = method_scores(record, m)
            mean, sem = sketch_utils.mean_and_sem(scores)
            dmean, dsem = sketch_utils.mean_and_sem(scores - base)
            stats[m]["space"].append(record["space"])
            stats[m]["mean"].append(mean)
            stats[m]["sem"].append(sem)
            stats[m]["dmean"].append(dmean)
            stats[m]["dsem"].append(dsem)
    for m in stats:
        for field in stats[m]:
            stats[m][field] = np.array(stats[m][field])
    return stats


def plot_survey_curve(ax, method, x_mean, x_sem, y):
    style = STYLES[method]
    if np.any(x_sem > 0):
        ax.fill_betweenx(
            y,
            x_mean - x_sem,
            x_mean + x_sem,
            color=style["color"],
            alpha=0.2,
            edgecolor="none",
        )
    ax.plot(
        x_mean,
        y,
        linestyle=style["linestyle"],
        color=style["color"],
        linewidth=style["linewidth"],
        alpha=style["line_alpha"],
        dash_capstyle="round",
    )
    if style["filled"]:
        ax.scatter(
            x_mean,
            y,
            marker=style["marker"],
            color=style["color"],
            alpha=0.9,
            s=style["marker_size"],
            linewidth=0,
        )
    else:
        ax.scatter(
            x_mean,
            y,
            marker=style["marker"],
            facecolors="none",
            edgecolors=style["color"],
            alpha=0.98,
            s=style["marker_size"],
            linewidth=1.2,
        )


def _legend_handle(method):
    style = STYLES[method]
    return mlines.Line2D(
        [],
        [],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        marker=style["marker"],
        markersize=7,
        markerfacecolor=style["color"] if style["filled"] else "none",
        markeredgecolor=style["color"],
        markeredgewidth=0 if style["filled"] else 1.2,
        label=style["label"],
    )


def _percent(value, _):
    return f"{100 * value:g}%"


def _signed_percent(value, _):
    return "0%" if value == 0 else f"{100 * value:+g}%"


def plot_survey(json_dir, output_pdf):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))

    for col, (dataset, task) in enumerate(PANELS):
        path = os.path.join(json_dir, f"survey_{dataset}_{task}.json")
        with open(path, "r") as f:
            data = json.load(f)
        if (data["dataset"], data["task"]) != (dataset, task):
            raise ValueError(f"{path} does not hold the {dataset}/{task} survey")
        stats = panel_stats(data)
        ax_top, ax_bottom = axes[0, col], axes[1, col]

        for m in data["methods"]:
            plot_survey_curve(
                ax_top, m, stats[m]["mean"], stats[m]["sem"], stats[m]["space"]
            )
            plot_survey_curve(
                ax_bottom, m, stats[m]["dmean"], stats[m]["dsem"], stats[m]["space"]
            )

        qos_mean, qos_sem, qos_space = qos_endpoint(json_dir, dataset, task)
        if qos_sem > 0:
            ax_top.plot(
                [qos_mean - qos_sem, qos_mean + qos_sem],
                [qos_space, qos_space],
                color=QOS_STYLE["color"],
                linewidth=1.2,
                alpha=0.5,
            )
        ax_top.scatter(
            [qos_mean],
            [qos_space],
            marker=QOS_STYLE["marker"],
            color=QOS_STYLE["color"],
            alpha=0.9,
            s=QOS_STYLE["marker_size"],
            linewidth=0,
            zorder=5,
        )

        top_cfg = TOP_AXES[(dataset, task)]
        ax_top.set_xlim(*top_cfg["xlim"])
        ax_top.set_xticks(top_cfg["xticks"])
        ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(_percent))
        ax_top.set_xlabel(top_cfg["xlabel"])
        ax_top.set_title(PANEL_TITLES[(dataset, task)])

        # Signed differences span decades (JL family within ~1%, subsampling
        # 10-30%), so use a symmetric log axis. The linear window covers
        # +-1%: seed-noise-level differences stay unmagnified around the zero
        # baseline while the large deviations are log-compressed.
        dmax = max(
            np.max(np.abs(stats[m]["dmean"]) + stats[m]["dsem"])
            for m in data["methods"]
        )
        dmax = 1.6 * max(dmax, 1e-2)
        ax_bottom.set_xscale("symlog", linthresh=1e-2, linscale=0.75)
        ax_bottom.set_xlim(-dmax, dmax)
        decades = [10.0**e for e in range(-2, 1) if 10.0**e <= dmax]
        ax_bottom.set_xticks([-t for t in reversed(decades)] + [0] + decades)
        ax_bottom.xaxis.set_major_formatter(mticker.FuncFormatter(_signed_percent))
        ax_bottom.set_xlabel(BOTTOM_XLABEL)

        for ax in (ax_top, ax_bottom):
            ax.set_yscale("log")
            ax.set_ylim(*YLIM)
            ax.tick_params(direction="in", which="both", top=False, right=True)
            ax.grid(True, which="major", ls="-", alpha=0.1)
            if col == 0:
                ax.set_ylabel("Machine size")
            else:
                ax.tick_params(axis="y", labelleft=False)

    handles = [_legend_handle(m) for m in SURVEY_METHODS]
    handles.append(
        mlines.Line2D(
            [],
            [],
            color=QOS_STYLE["color"],
            linestyle="none",
            marker=QOS_STYLE["marker"],
            markersize=7,
            markerfacecolor=QOS_STYLE["color"],
            markeredgewidth=0,
            label=QOS_STYLE["label"],
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_pdf)
    print(f"Saved {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Survey of oblivious sketching methods (IMDb and PBMC68k)."
    )
    parser.add_argument("--dataset", choices=["imdb", "pbmc68k"])
    parser.add_argument("--task", choices=["svm", "pca"])
    parser.add_argument(
        "--plot",
        action="store_true",
        help="assemble the 2x4 figure from the four survey JSONs",
    )
    parser.add_argument(
        "--json-dir", type=str, default=".", help="directory holding survey JSONs"
    )
    parser.add_argument("--out", type=str, default="survey_sketches.pdf")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=n_sketch_seeds,
        help="number of random sketch seeds to average over",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="joblib workers for the pair-level cross-validations "
        "(PBMC68k classification); results do not depend on this",
    )
    args = parser.parse_args()

    ran = False
    if args.dataset or args.task:
        if not (args.dataset and args.task):
            parser.error("--dataset and --task must be given together")
        run_survey(args.dataset, args.task, args.n_seeds, n_jobs=args.n_jobs)
        ran = True
    if args.plot:
        plot_survey(args.json_dir, args.out)
        ran = True
    if not ran:
        parser.error("nothing to do: pass --dataset/--task to compute, or --plot")


if __name__ == "__main__":
    main()

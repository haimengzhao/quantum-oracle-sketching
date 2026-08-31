"""20 Newsgroups sweep: machine size vs LS-SVM classification accuracy.

Ridge classification (5-fold CV) on TF-IDF features, averaged over random
1v1 category pairs (each pair is vectorized separately), sweeping
rare-feature truncation (min_df) or an oblivious feature sketch; see
sweep_utils for the shared pipeline.
"""

import json
import random
from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import sketch_utils
import sweep_utils
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

np.random.seed(42)
random.seed(42)
sweep_utils.apply_plot_style()

min_dfs = list(range(2, 21)) + list(range(25, 105, 5))
sketch_n_features = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
n_sketch_seeds = 5
num_markers = 20
rare_extra_markers = [-3]


def make_classifier():
    return RidgeClassifier(random_state=42, alpha=1.0, solver="auto")


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
    X_train_raw, y_train, X_test_raw, y_test = load_data(categories)

    X_all_raw = list(X_train_raw) + list(X_test_raw)
    y_all = np.concatenate([y_train, y_test])

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

        num_samples, feature_dim = X_all.shape
        sparsity = sketch_utils.max_sparsity(X_all)

        scores = cross_val_score(make_classifier(), X_all, y_all, cv=5)

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(feature_dim)
        results["space_sparse"].append(sketch_utils.matrix_nnz(X_all))
        results["space_quantum"].append(
            sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "svm")
        )
        results["accuracies"].append([float(s) for s in scores])
        results["error_rates"].append([float(1.0 - s) for s in scores])

    return results


def get_ridge_results_sketch(categories, sketch_seeds, mode):
    X_train_raw, y_train, X_test_raw, y_test = load_data(categories)

    X_all_raw = list(X_train_raw) + list(X_test_raw)
    y_all = np.concatenate([y_train, y_test])

    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    X_full = vectorizer.fit_transform(X_all_raw)
    X_full.eliminate_zeros()  # type: ignore

    results = {
        "n_features": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracies": [],
        "error_rates": [],
    }

    tqdm.write(f"Sweeping {mode} sketch dimension for categories {categories}...")

    for n_features in tqdm(sketch_n_features, desc=f"{mode} Sweep", leave=False):
        seed_space_streaming = []
        seed_space_sparse = []
        seed_space_quantum = []
        seed_accuracy_scores = []
        seed_error_scores = []

        for seed in sketch_seeds:
            X_sketch, _ = sketch_utils.sketch_features(
                X_full, n_features, mode, seed=seed
            )

            num_samples, feature_dim = X_sketch.shape
            sparsity = sketch_utils.max_sparsity(X_sketch)

            seed_space_streaming.append(feature_dim)
            seed_space_sparse.append(sketch_utils.matrix_nnz(X_sketch))
            seed_space_quantum.append(
                sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "svm")
            )

            scores = cross_val_score(make_classifier(), X_sketch, y_all, cv=5)
            seed_accuracy_scores.append([float(s) for s in scores])
            seed_error_scores.append([float(1.0 - s) for s in scores])

        results["n_features"].append(n_features)
        results["space_streaming"].append(seed_space_streaming)
        results["space_sparse"].append(seed_space_sparse)
        results["space_quantum"].append(seed_space_quantum)
        results["accuracies"].append(seed_accuracy_scores)
        results["error_rates"].append(seed_error_scores)

    return results


def run_and_save(n_pairs, mode, n_sketch_seeds, sketch_mode):
    print(f"Running Analysis over {n_pairs} random sets of 1v1 categories...")
    if sketch_mode:
        feature_seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
        print(f"Averaging over random sketch seeds: {feature_seeds}")
    all_cats = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    ).target_names  # type: ignore

    keys = sweep_utils.METHOD_KEYS
    by_param = defaultdict(
        lambda: {k: {"space": [], "error": [], "accuracy": []} for k in keys}
    )

    for i in tqdm(range(n_pairs), desc="Category Pairs", leave=True):
        cats = random.sample(all_cats, 2)
        tqdm.write(f"[{i + 1}/{n_pairs}] Group: {cats}")

        if sketch_mode:
            res = get_ridge_results_sketch(cats, feature_seeds, mode)
            params = res["n_features"]
        else:
            res = get_ridge_results(cats)
            params = res["min_dfs"]

        for j, param in enumerate(params):
            for k in keys:
                by_param[param][k]["space"].append(res[f"space_{k}"][j])
                by_param[param][k]["error"].append(res["error_rates"][j])
                by_param[param][k]["accuracy"].append(res["accuracies"][j])

    raw_key = "raw_data_by_n_features" if sketch_mode else "raw_data_by_min_df"
    if sketch_mode:
        output_json = f"20newsgroups_{mode}_size_vs_accuracy.json"
    else:
        output_json = "20newsgroups_size_vs_accuracy.json"
    data_to_save = {"n_pairs": n_pairs, "cats_per_class": 1, raw_key: {}}
    for param, param_dict in by_param.items():
        data_to_save[raw_key][param] = {}
        for k, sub_dict in param_dict.items():
            spaces = np.array(sub_dict["space"], dtype=float)
            accuracies = np.array(sub_dict["accuracy"], dtype=float)
            errors = np.array(sub_dict["error"], dtype=float)

            if sketch_mode:
                # Stored arrays are transposed from (pair, seed, ...) to
                # (seed, pair, ...) so seeds index the outer axis.
                data_to_save[raw_key][param][k] = {
                    "space": float(np.mean(spaces)),
                    "space_by_seed_pair": np.moveaxis(spaces, 0, 1).tolist(),
                    "accuracy_mean": float(np.mean(accuracies)),
                    "accuracy_sem": float(
                        np.std(accuracies.reshape(-1)) / np.sqrt(accuracies.size)
                    ),
                    "accuracy_scores_by_seed_pair": np.moveaxis(
                        accuracies, 0, 1
                    ).tolist(),
                    "error_mean": float(np.mean(errors)),
                    "error_sem": float(
                        np.std(errors.reshape(-1)) / np.sqrt(errors.size)
                    ),
                    "error_scores_by_seed_pair": np.moveaxis(errors, 0, 1).tolist(),
                }
            else:
                data_to_save[raw_key][param][k] = {
                    "space": float(np.mean(spaces)),
                    "space_by_pair": spaces.tolist(),
                    "accuracy_mean": float(np.mean(accuracies)),
                    "accuracy_sem": float(
                        np.std(accuracies.reshape(-1)) / np.sqrt(accuracies.size)
                    ),
                    "accuracy_scores_by_pair": accuracies.tolist(),
                    "error_mean": float(np.mean(errors)),
                    "error_sem": float(
                        np.std(errors.reshape(-1)) / np.sqrt(errors.size)
                    ),
                    "error_scores_by_pair": errors.tolist(),
                }
    if sketch_mode:
        data_to_save.update(
            sketch_utils.sketch_metadata(mode, feature_seeds, sketch_n_features)
        )
        data_to_save[f"{mode}_key_note"] = (
            "raw_data_by_n_features keys are requested sketch dimensions; "
            "a pair with fewer original features uses its full matrix."
        )
    with open(output_json, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved raw data to {output_json}")
    return output_json


def run_analysis(n_pairs=10, load_file=None, mode=None, n_sketch_seeds=n_sketch_seeds):
    sketch_mode = sweep_utils.is_sketch_mode(mode)

    if load_file is None:
        load_file = run_and_save(n_pairs, mode, n_sketch_seeds, sketch_mode)
    else:
        print(f"Loading analysis from {load_file}...")

    with open(load_file, "r") as f:
        data = json.load(f)
    sketch_utils.validate_truncation_data(data, mode, load_file)
    stats = sweep_utils.load_sweep_stats(data, "accuracy")
    plot_results(stats, mode, sketch_mode)


def plot_results(stats, mode, sketch_mode):
    plt.figure(figsize=sweep_utils.FIGSIZE)
    ax = plt.gca()
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
            show_all_markers=sketch_mode,
            num_markers=num_markers,
            extra_marker_indices=rare_extra_markers,
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if sketch_mode:
        streaming_label_x, streaming_label_y, streaming_label_ha = (
            (0.58, 0.46, "right") if mode == "jl" else (0.78, 0.58, "right")
        )
        plt.text(
            0.2,
            0.82,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
        plt.text(
            streaming_label_x,
            streaming_label_y,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
            ha=streaming_label_ha,
        )
        plt.text(
            0.15,
            0.06,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            transform=ax.transAxes,
        )
    else:
        plt.text(
            0.83,
            9e4,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
        )
        plt.text(
            0.934,
            9e3,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            0.94,
            7e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    plt.yscale("log")
    plt.xlabel("Accuracy")
    if sketch_mode:
        plt.xticks([0.65, 0.75, 0.85, 0.95], ["65%", "75%", "85%", "95%"])
        plt.xlim(0.62, 0.97)
    else:
        plt.xticks(
            [0.84, 0.86, 0.88, 0.90, 0.92, 0.94],
            ["84%", "86%", "88%", "90%", "92%", "94%"],
        )
    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.ylabel("Machine size")
    plt.ylim(1e1, 2e5)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Binary classification")
    plt.tight_layout()
    if sketch_mode:
        output_pdf = f"20newsgroups_{mode}_size_vs_accuracy.pdf"
    else:
        output_pdf = "20newsgroups_size_vs_accuracy.pdf"
    plt.savefig(output_pdf)
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    parser = sweep_utils.build_arg_parser(
        "20newsgroups Machine Size vs Accuracy Analysis"
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=100,
        help="Number of random pairs to average (for new run)",
    )
    args = parser.parse_args()
    run_analysis(
        n_pairs=args.n_pairs,
        load_file=args.load,
        mode=args.mode,
        n_sketch_seeds=args.n_seeds,
    )

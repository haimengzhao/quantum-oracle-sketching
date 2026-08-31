"""20 Newsgroups sweep: machine size vs PCA variance recovery.

Top singular direction of the per-pair TF-IDF matrix, averaged over random
1v1 category pairs, computed after rare-feature truncation (min_df) or an
oblivious feature sketch, lifted back to the full feature space and scored
by relative explained variance; see sweep_utils for the shared pipeline.
"""

import json
import random
from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import sketch_utils
import sweep_utils
from scipy.sparse.linalg import svds
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

np.random.seed(42)
random.seed(42)
sweep_utils.apply_plot_style()

min_dfs = list(range(2, 21)) + list(range(25, 105, 5))
sketch_n_features = [
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
    32768,
]
n_sketch_seeds = 5
num_markers = 20


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
    # For PCA, both train and test data give a better covariance estimate.
    return list(data_train[0]) + list(data_test[0])


def get_pca_results(categories):
    raw_documents = load_data(categories)

    full_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    X_full = full_vectorizer.fit_transform(raw_documents)
    X_full.eliminate_zeros()  # type: ignore

    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]  # type: ignore
    var_max = np.linalg.norm(X_full @ v_full) ** 2

    vocab_full = full_vectorizer.vocabulary_
    D_full = X_full.shape[1]

    results = {
        "min_dfs": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovery": [],
    }

    tqdm.write(f"Sweeping min_df for categories {categories}...")

    for mdf in tqdm(min_dfs, desc="min_df Sweep", leave=False):
        vectorizer = TfidfVectorizer(min_df=mdf, stop_words="english")
        X_trunc = vectorizer.fit_transform(raw_documents)
        X_trunc.eliminate_zeros()  # type: ignore

        num_samples, feature_dim = X_trunc.shape
        sparsity = sketch_utils.max_sparsity(X_trunc)

        # Top singular direction in the truncated space.
        _, _, vt_trunc = svds(X_trunc.asfptype(), k=1)
        v_trunc = vt_trunc[0]  # type: ignore

        # Lift to the full vocabulary via the word identities, then score.
        trunc_vocab = vectorizer.vocabulary_
        v_lifted = np.zeros(D_full)
        for word, idx_trunc in trunc_vocab.items():
            if word in vocab_full:
                v_lifted[vocab_full[word]] = v_trunc[idx_trunc]
        v_lifted = v_lifted / np.linalg.norm(v_lifted)

        var_captured = np.linalg.norm(X_full @ v_lifted) ** 2  # type: ignore

        results["min_dfs"].append(mdf)
        results["space_streaming"].append(feature_dim)
        results["space_sparse"].append(sketch_utils.matrix_nnz(X_trunc))
        results["space_quantum"].append(
            sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "pca")
        )
        results["variance_recovery"].append(var_captured / var_max)

    return results


def get_pca_results_sketch(categories, sketch_seeds, mode):
    raw_documents = load_data(categories)

    full_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    X_full = full_vectorizer.fit_transform(raw_documents)
    X_full.eliminate_zeros()  # type: ignore

    _, _, vt_full = svds(X_full.asfptype(), k=1)
    v_full = vt_full[0]  # type: ignore
    var_max = np.linalg.norm(X_full @ v_full) ** 2

    results = {
        "n_features": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovery": [],
    }

    tqdm.write(f"Sweeping {mode} sketch dimension for categories {categories}...")

    for n_features in tqdm(sketch_n_features, desc=f"{mode} Sweep", leave=False):
        seed_space_streaming = []
        seed_space_sparse = []
        seed_space_quantum = []
        seed_recoveries = []

        for seed in sketch_seeds:
            X_sketch, sketch_info = sketch_utils.sketch_features(
                X_full, n_features, mode, seed=seed
            )

            num_samples, feature_dim = X_sketch.shape
            sparsity = sketch_utils.max_sparsity(X_sketch)

            seed_space_streaming.append(feature_dim)
            seed_space_sparse.append(sketch_utils.matrix_nnz(X_sketch))
            seed_space_quantum.append(
                sweep_utils.qos_machine_size(num_samples, feature_dim, sparsity, "pca")
            )

            if sketch_info is None:
                seed_recoveries.append(1.0)
            else:
                _, _, vt_sketch = svds(sketch_utils.svds_input(X_sketch), k=1)
                v_sketch = vt_sketch[0]  # type: ignore
                v_lifted = sketch_utils.lift_sketched_vector(
                    v_sketch, sketch_info, mode
                )
                v_lifted = v_lifted / sketch_utils.lifted_vector_norm(v_lifted)
                var_captured = np.linalg.norm(X_full @ v_lifted) ** 2
                seed_recoveries.append(var_captured / var_max)

        results["n_features"].append(n_features)
        results["space_streaming"].append(seed_space_streaming)
        results["space_sparse"].append(seed_space_sparse)
        results["space_quantum"].append(seed_space_quantum)
        results["variance_recovery"].append(seed_recoveries)

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
        lambda: {k: {"space": [], "variance_recovery": []} for k in keys}
    )

    for i in tqdm(range(n_pairs), desc="Category Pairs", leave=True):
        try:
            cats = random.sample(all_cats, 2)
        except ValueError:
            continue

        tqdm.write(f"[{i + 1}/{n_pairs}] Group: {cats}")

        if sketch_mode:
            res = get_pca_results_sketch(cats, feature_seeds, mode)
            params = res["n_features"]
        else:
            res = get_pca_results(cats)
            params = res["min_dfs"]

        for j, param in enumerate(params):
            for k in keys:
                by_param[param][k]["space"].append(res[f"space_{k}"][j])
                by_param[param][k]["variance_recovery"].append(
                    res["variance_recovery"][j]
                )

    raw_key = "raw_data_by_n_features" if sketch_mode else "raw_data_by_min_df"
    if sketch_mode:
        output_json = f"20newsgroups_{mode}_size_vs_variance.json"
    else:
        output_json = "20newsgroups_size_vs_variance.json"
    data_to_save = {"n_pairs": n_pairs, "cats_per_class": 1, raw_key: {}}
    for param, param_dict in by_param.items():
        data_to_save[raw_key][param] = {}
        for k, sub_dict in param_dict.items():
            spaces = np.array(sub_dict["space"], dtype=float)
            recoveries = np.array(sub_dict["variance_recovery"], dtype=float)

            if sketch_mode:
                # Stored arrays are transposed from (pair, seed) to
                # (seed, pair) so seeds index the outer axis.
                data_to_save[raw_key][param][k] = {
                    "space": float(np.mean(spaces)),
                    "space_by_seed_pair": np.moveaxis(spaces, 0, 1).tolist(),
                    "variance_recovery": float(np.mean(recoveries)),
                    "variance_recovery_sem": float(
                        np.std(recoveries.reshape(-1)) / np.sqrt(recoveries.size)
                    ),
                    "variance_recovery_by_seed_pair": np.moveaxis(
                        recoveries, 0, 1
                    ).tolist(),
                }
            else:
                data_to_save[raw_key][param][k] = {
                    "space": float(np.mean(spaces)),
                    "space_by_pair": spaces.tolist(),
                    "variance_recovery": float(np.mean(recoveries)),
                    "variance_recovery_sem": float(
                        np.std(recoveries.reshape(-1)) / np.sqrt(recoveries.size)
                    ),
                    "variance_recovery_by_pair": recoveries.tolist(),
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
    stats = sweep_utils.load_sweep_stats(data, "variance")
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
        )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    if sketch_mode:
        streaming_label_x, streaming_label_y, streaming_label_ha = (
            (0.58, 0.75, "right") if mode == "jl" else (0.98, 0.58, "right")
        )
        plt.text(
            0.2,
            0.85,
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
            0.535,
            9e4,
            "Classical sparse / QRAM",
            color=sweep_utils.COLORS["sparse"],
            fontsize=10,
            path_effects=halo,
        )
        plt.text(
            0.98,
            9e3,
            sketch_utils.streaming_label(mode),
            color=sweep_utils.COLORS["streaming"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )
        plt.text(
            1,
            7e1,
            "Quantum oracle sketching",
            color=sweep_utils.COLORS["quantum"],
            fontsize=10,
            path_effects=halo,
            ha="right",
        )

    plt.yscale("log")
    plt.ylim(1e1, 2e5)
    plt.xlabel("Relative explained variance")
    if sketch_mode:
        plt.xticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"])
        plt.xlim(0.08, 1.03)
    else:
        plt.xticks(
            [0.6, 0.7, 0.8, 0.9, 1.0],
            ["60%", "70%", "80%", "90%", "100%"],
        )
        plt.xlim(0.52, 1.03)
    plt.tick_params(direction="in", which="both", top=False, right=True)
    ax.set_ylabel("Machine size")
    ax.tick_params(axis="y")
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title("Dimension reduction")
    plt.tight_layout()
    if sketch_mode:
        output_pdf = f"20newsgroups_{mode}_size_vs_variance.pdf"
    else:
        output_pdf = "20newsgroups_size_vs_variance.pdf"
    plt.savefig(output_pdf)
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    parser = sweep_utils.build_arg_parser(
        "20newsgroups Machine Size vs Variance Analysis"
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

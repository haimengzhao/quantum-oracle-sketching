"""Adaptive sketching survey on IMDb and PBMC68k classification.

Compares feature hashing — both the exact ridge terminal accuracy (as in
the main figures; dashed) and the same hashed model trained by the shared
streaming SGD protocol (solid) — against two adaptive sketching methods
trained by streaming ridge SGD under a total scalar-register budget:

- AWM-Sketch: Tai, Sharan, Bailis, Valiant, "Sketching Linear Classifiers
  over Data Streams", SIGMOD 2018 (arXiv:1711.02305), Algorithms 1-2.
- MISSION: Aghazadeh, Spring, LeJeune, Dasarathy, Shrivastava, Baraniuk,
  "MISSION: Ultra Large-Scale Feature Selection using Count-Sketches",
  ICML 2018 (arXiv:1806.04310), Algorithm 1.

Protocol (fixed by design review):
- Per seed, an 80/20 stratified split; the stream draws single rows
  uniformly WITH replacement from the train split; accuracy is measured on
  the held-out 20%.
- Ridge loss on +-1 labels: l = 0.5*(w.x + b - y)^2 + (lambda/2)|w|^2 with
  lambda = alpha/N_train and the main figures' alpha (IMDb 10, PBMC68k 200);
  PBMC68k uses balanced class weights, matching class_weight="balanced".
  The bias b is unregularized and excluded from every budget uniformly.
- One fixed learning-rate schedule for all methods, eta_t = ETA0/sqrt(t),
  with ETA0 selected PER DATASET by --tune-eta: full-dimensional SGD ridge
  on the untruncated data (IMDb: the single task; PBMC68k: the first
  cell-type pair), first sampled seed, candidates ETA0_CANDIDATES, chosen
  by minimal training loss.
- Stopping and selection: every N/5 samples (N = train rows) the TRAINING
  ridge objective of the decoded model is evaluated; the stream stops once
  the lowest training loss so far has not decreased by a relative 1e-4
  over a window of N samples (hard cap 300*N). The reported accuracy is
  the held-out accuracy at MINIMAL training loss, so neither stopping nor
  model selection ever touches the test split.
- Register accounting counts ALL method state: sketch counters and heap /
  active-set ids and values (one register each). Hash functions are O(1)
  seeds (the arrays materialized here are a simulation convenience, exactly
  as for the oblivious sketches); the current sample and per-step workspace
  are not counted (SI convention).

Compute one dataset with `python survey_adaptive.py --dataset imdb`
(checkpoints per budget, resumes automatically), and assemble the 1x4
figure with `python survey_adaptive.py --plot`. A second 1x4 figure,
`--plot-convergence`, shows the feature hashing SGD training dynamics
(training ridge loss relative to the exact same-budget solution, and test
accuracy, vs samples / training set size; one semi-transparent curve per
task x seed run) at a middle and a large budget per dataset.

The figure's difference panels pair every run against the feature hashing
baseline at equal MEASURED machine size (interpolating the baseline's
accuracy-size curve; relevant for MISSION, whose measured register count is
slightly below its nominal budget). The accuracy panels also show the full-matrix quantum
oracle sketching reference: the exact full-dimensional ridge accuracy under
the same 80/20 protocol at the SI machine size of QOS on the training
stream (`--add-qos-ref` inserts it into existing survey JSONs).
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
sweep_utils.apply_plot_style()

METHODS = ("hashing", "hashing_sgd", "awm", "mission")
BASELINE_METHOD = "hashing"
n_sketch_seeds = 5

# Selected per dataset by --tune-eta on 2026-08-31: full-dimensional SGD
# ridge on the untruncated data (IMDb: the single task; PBMC68k: the first
# cell-type pair), first sampled seed, chosen by minimal training loss.
# IMDb min losses 0.4994 / 0.4708 / 0.2960 / 0.2488 and PBMC68k 0.0853 /
# 0.0763 / 0.0739 / diverged for candidates 0.001 / 0.01 / 0.1 / 1.0; the
# record is kept in survey_adaptive_eta0.json. One value per dataset,
# shared by all methods.
ETA0 = {"imdb": 1.0, "pbmc68k": 0.1}
ETA0_CANDIDATES = [0.001, 0.01, 0.1, 1.0]

# Marks the stopping/selection convention in every JSON fingerprint, so
# results recorded under a different convention (e.g. best-so-far or
# terminal test accuracy) are never reused.
REPORTED_VALUE = "accuracy_at_min_train_loss"

ALPHA = {"imdb": 10.0, "pbmc68k": 200.0}  # ridge alpha, as in the main sweeps
BALANCED = {"imdb": False, "pbmc68k": True}

# Register budgets: the main-figure sketch grids WITHOUT the full-dimension
# endpoint (at full dimension the comparison would measure optimizer and
# architecture residuals rather than the budget effect).
BUDGETS = {
    "imdb": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "pbmc68k": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
}

N_PAIRS = 20  # first 20 of the main figures' 100 random PBMC68k class pairs

EVAL_DIVISOR = 5  # evaluate every N/EVAL_DIVISOR samples
IMPROVEMENT_TOL = 1e-4
SAMPLE_CAP_FACTOR = 300  # hard cap: SAMPLE_CAP_FACTOR * N samples

RNG_BLOCK = 1 << 14  # stream indices drawn in blocks (bit-exact vs singles)


# --- data -------------------------------------------------------------------

def load_imdb():
    X_all_raw, y_all = imdb_utils.load_imdb_data()
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", dtype=np.float32)
    X_full = vectorizer.fit_transform(X_all_raw)
    X_full.eliminate_zeros()
    y = np.where(np.asarray(y_all) > 0, 1.0, -1.0)
    return X_full.tocsr(), y


def get_random_pairs(n_classes, n_pairs, rng):
    # Matches pbmc68k_svm.py, so the pairs subset the published protocol.
    all_pairs = list(combinations(range(n_classes), 2))
    indices = rng.choice(len(all_pairs), size=n_pairs, replace=True)
    return [all_pairs[i] for i in indices]


def load_pbmc68k_pairs():
    import pbmc68k_utils  # imports scvelo; keep IMDb-only runs light

    X_full, y_full, label_names = pbmc68k_utils.load_pbmc68k_data(
        min_samples=1, normalize=True, binary=False
    )
    rng = np.random.default_rng(42)
    pairs = get_random_pairs(len(label_names), 100, rng)[:N_PAIRS]
    tasks = []
    for c1, c2 in pairs:
        rows = np.flatnonzero((y_full == c1) | (y_full == c2))
        y_pair = np.where(y_full[rows] == c2, 1.0, -1.0)
        tasks.append((X_full[rows].tocsr(), y_pair, (label_names[c1], label_names[c2])))
    return tasks


# --- streaming ridge SGD models ----------------------------------------------
#
# Shared conventions: labels are +-1 and the margin loss is
# l = 0.5*(tau - y)^2 so the gradient factor is g = c*(tau - y), where c is
# the (balanced) class weight; this is y*dl(y*tau) in the papers' notation
# and the standard SGD formula for ridge regression. The L2 decay
# (1 - lambda*eta) is applied through one global scale factor (WM-Sketch
# paper Sec. 5.1 "Efficient Regularization"; bit-equivalent to shrinking
# every stored weight each step). Hash/sign functions are materialized as
# arrays purely as a simulation device; the algorithm requires only O(1)
# seeds, the same convention used for the oblivious sketches.


class DenseRidgeSGD:
    """Dense SGD ridge: selects ETA0 (--tune-eta) and, on hashed features,
    implements the streaming feature hashing method ("hashing_sgd")."""

    def __init__(self, n_features, seed):
        self.w = np.zeros(n_features)
        self.bias = 0.0
        self.scale = 1.0

    def update(self, fi, fv, y, c, eta, lam):
        tau = self.scale * float(self.w[fi] @ fv) + self.bias
        g = c * (tau - y)
        self.scale *= 1.0 - lam * eta
        self.w[fi] -= eta * g * fv / self.scale
        self.bias -= eta * g

    def decode(self, fi):
        return self.scale * self.w[fi]


class AWMSketch:
    """Active-Set Weight-Median Sketch (arXiv:1711.02305, Algorithm 2).

    Configuration follows the paper's best configurations (Table 2): depth 1
    and sketch width = 2 * active-set capacity, i.e. capacity = B//4 and
    width = B - 2*capacity, so ids + values + counters total exactly B
    registers. With depth 1 the count-sketch query is the single signed
    counter. Promotion/eviction follows Algorithm 2 literally: a feature is
    promoted when its updated estimate exceeds the smallest active weight in
    magnitude; the evicted weight is written back into the sketch as the
    difference S[i_min] - Query(i_min). Features within one sample are
    handled against a single min snapshot, with promotion candidates then
    processed sequentially (a fixed feature ordering of the per-sample loop).
    """

    def __init__(self, budget, n_features, seed):
        self.capacity = budget // 4
        self.width = budget - 2 * self.capacity
        rng = np.random.default_rng(seed)
        self.h = rng.integers(0, self.width, size=n_features)
        self.sgn = rng.choice(np.array([-1.0, 1.0]), size=n_features)
        self.z = np.zeros(self.width)
        # Active set: fixed-capacity parallel arrays + id -> slot map.
        self.ids = np.full(self.capacity, -1, dtype=np.int64)
        self.vals = np.zeros(self.capacity)
        self.slot = {}
        self.fill = 0
        self.bias = 0.0
        self.scale = 1.0
        self.space = 2 * self.capacity + self.width

    def _query_raw(self, fi):
        return self.sgn[fi] * self.z[self.h[fi]]

    def update(self, fi, fv, y, c, eta, lam):
        in_active = np.fromiter(
            (f in self.slot for f in fi), dtype=bool, count=len(fi)
        )
        tau = self.bias
        if in_active.any():
            slots = np.fromiter(
                (self.slot[f] for f in fi[in_active]),
                dtype=np.int64,
                count=int(in_active.sum()),
            )
            tau += self.scale * float(self.vals[slots] @ fv[in_active])
        sk = ~in_active
        if sk.any():
            tau += self.scale * float(self._query_raw(fi[sk]) @ fv[sk])
        g = c * (tau - y)

        # Uniform L2 decay of active set and sketch via the global scale.
        self.scale *= 1.0 - lam * eta
        step = eta * g / self.scale
        if in_active.any():
            self.vals[slots] -= step * fv[in_active]

        if sk.any():
            fs, vs = fi[sk], fv[sk]
            wtilde = self._query_raw(fs) - step * vs
            if self.fill < self.capacity:
                # Fill spare capacity first (no eviction needed); once full,
                # the remaining features get the plain sketch update.
                for f, w, v in zip(fs.tolist(), wtilde.tolist(), vs.tolist()):
                    if self.fill < self.capacity:
                        self._insert(f, w)
                    else:
                        self.z[self.h[f]] -= self.sgn[f] * step * v
            else:
                cur_min = np.abs(self.vals).min()
                cand = np.abs(wtilde) > cur_min
                plain = ~cand
                if plain.any():
                    np.add.at(
                        self.z,
                        self.h[fs[plain]],
                        -self.sgn[fs[plain]] * step * vs[plain],
                    )
                for f, w, v in zip(
                    fs[cand].tolist(), wtilde[cand].tolist(), vs[cand].tolist()
                ):
                    imin = int(np.argmin(np.abs(self.vals)))
                    if abs(w) > abs(self.vals[imin]):
                        evicted = int(self.ids[imin])
                        # Write the evicted weight back into the sketch.
                        delta = self.vals[imin] - self._query_raw(
                            np.array([evicted])
                        )[0]
                        self.z[self.h[evicted]] += self.sgn[evicted] * delta
                        del self.slot[evicted]
                        self.ids[imin] = f
                        self.vals[imin] = w
                        self.slot[f] = imin
                    else:
                        self.z[self.h[f]] -= self.sgn[f] * step * v
        self.bias -= eta * g

    def _insert(self, f, w):
        i = self.fill
        self.ids[i] = f
        self.vals[i] = w
        self.slot[f] = i
        self.fill += 1

    def decode(self, fi):
        w = self._query_raw(fi).copy()
        for pos, f in enumerate(fi.tolist()):
            s = self.slot.get(f)
            if s is not None:
                w[pos] = self.vals[s]
        return self.scale * w


class TopKSketchModel:
    """Machinery for MISSION: a depth-d Count-Sketch of the
    weights plus a top-k set of feature ids whose queried values form the
    (k-sparse) model used for every prediction (arXiv:1806.04310, Alg. 1:
    "Get the top-k heavy-hitters from the sketch"). Neither paper prescribes
    a budget split between sketch and heap is prescribed by the paper; we
    mirror the even
    exact-vs-sketch split found optimal for the AWM-Sketch (arXiv:1711.02305
    Table 2): k = B//4 (2k registers for ids+values), remainder to the
    sketch rows.
    """

    depth = None  # set by subclass

    def __init__(self, budget, n_features, seed):
        self.k = budget // 4
        self.width = (budget - 2 * self.k) // self.depth
        rng = np.random.default_rng(seed)
        self.h = rng.integers(0, self.width, size=(self.depth, n_features))
        self.sgn = rng.choice(np.array([-1.0, 1.0]), size=(self.depth, n_features))
        self.z = np.zeros((self.depth, self.width))
        self.ids = np.full(self.k, -1, dtype=np.int64)
        self.vals = np.zeros(self.k)
        self.slot = {}
        self.fill = 0
        self.bias = 0.0
        self.scale = 1.0
        self.space = 2 * self.k + self.depth * self.width

    def _query_raw(self, fi):
        # Median over the depth of signed counters (Count-Sketch query).
        est = self.sgn[:, fi] * self.z[np.arange(self.depth)[:, None], self.h[:, fi]]
        return np.median(est, axis=0)

    def _sketch_add(self, fi, delta):
        for j in range(self.depth):
            np.add.at(self.z[j], self.h[j, fi], self.sgn[j, fi] * delta)

    def _topk_margin(self, fi, fv):
        tau = self.bias
        hit = [
            (pos, self.slot[f]) for pos, f in enumerate(fi.tolist()) if f in self.slot
        ]
        if hit:
            pos, slots = zip(*hit)
            tau += self.scale * float(
                self.vals[np.array(slots)] @ fv[np.array(pos)]
            )
        return tau, hit

    def _refresh_topk(self, fi):
        est = self._query_raw(fi)
        for f, e in zip(fi.tolist(), est.tolist()):
            s = self.slot.get(f)
            if s is not None:
                self.vals[s] = e
            elif self.fill < self.k:
                self.ids[self.fill] = f
                self.vals[self.fill] = e
                self.slot[f] = self.fill
                self.fill += 1
            else:
                imin = int(np.argmin(np.abs(self.vals)))
                if abs(e) > abs(self.vals[imin]):
                    del self.slot[int(self.ids[imin])]
                    self.ids[imin] = f
                    self.vals[imin] = e
                    self.slot[f] = imin

    def decode(self, fi):
        w = np.zeros(len(fi))
        for pos, f in enumerate(fi.tolist()):
            s = self.slot.get(f)
            if s is not None:
                w[pos] = self.vals[s]
        return self.scale * w


class Mission(TopKSketchModel):
    """MISSION (arXiv:1806.04310, Algorithm 1) with depth 3 ("3 in our
    Count-Sketch implementation", Sec. 4). Per step: predict with the
    k-sparse top-k model, add the (ridge) gradient's nonzero entries to the
    sketch, re-query the touched features, refresh top-k membership. The L2
    decay via the global scale is our ridge-protocol adaptation (MISSION's
    own Algorithm 1 optimizes the unregularized quadratic loss).
    """

    depth = 3

    def update(self, fi, fv, y, c, eta, lam):
        tau, _ = self._topk_margin(fi, fv)
        g = c * (tau - y)
        self.scale *= 1.0 - lam * eta
        self._sketch_add(fi, -eta * g * fv / self.scale)
        self._refresh_topk(fi)
        self.bias -= eta * g


ADAPTIVE_MODELS = {"awm": AWMSketch, "mission": Mission}


# --- streaming protocol -------------------------------------------------------

def class_weights(y_train, balanced):
    if not balanced:
        return {1.0: 1.0, -1.0: 1.0}
    n = len(y_train)
    n_pos = int((y_train > 0).sum())
    return {1.0: n / (2.0 * n_pos), -1.0: n / (2.0 * (n - n_pos))}


def test_scores(model, X_test, n_features):
    """Margin scores on the test split via the model's decoded weights."""
    return (
        X_test @ decoded_weights(model, np.unique(X_test.indices), n_features)
        + model.bias
    )


def decoded_weights(model, feats, n_features):
    w_hat = np.zeros(n_features)
    if len(feats):
        w_hat[feats] = model.decode(feats)
    return w_hat


def ridge_objective(scores, y_train, sample_weights, lam, w_sq):
    """The streamed training objective: mean weighted margin loss + penalty."""
    return float(
        np.mean(sample_weights * 0.5 * (scores - y_train) ** 2) + 0.5 * lam * w_sq
    )


def stream_until_terminal(model, X_train, y_train, X_test, y_test, seed, lam, balanced, eta0):
    """Run the shared streaming protocol.

    Every N/EVAL_DIVISOR samples the TRAINING ridge objective of the decoded
    model is evaluated (alongside the held-out accuracy, recorded for
    reporting only). The stream stops once the lowest training loss so far
    has not decreased by a relative IMPROVEMENT_TOL over a window of N
    samples (hard cap SAMPLE_CAP_FACTOR*N). The reported accuracy is the one
    observed at MINIMAL training loss, so neither stopping nor model
    selection ever touches the test split.

    Returns (acc_at_min_loss, min_loss, samples, trajectory); trajectory
    entries are [samples, train_loss, test_accuracy].
    """
    # One up-front float64 copy instead of a per-sample astype in the loop.
    if X_train.dtype != np.float64:
        X_train = X_train.astype(np.float64)
    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    weights = class_weights(y_train, balanced)
    sample_weights = np.where(y_train > 0, weights[1.0], weights[-1.0])
    train_feats = np.unique(X_train.indices)
    rng = np.random.default_rng(seed)
    eval_every = max(1, n_train // EVAL_DIVISOR)
    patience = n_train
    cap = SAMPLE_CAP_FACTOR * n_train

    indptr, indices, data = X_train.indptr, X_train.indices, X_train.data
    best_loss = np.inf
    best_acc = float("nan")
    last_improved = 0
    trajectory = []
    t = 0
    block, bpos = None, 0
    while t < cap:
        if block is None or bpos == len(block):
            # Blocked draws reproduce the single-draw sequence bit-exactly.
            block, bpos = rng.integers(n_train, size=RNG_BLOCK), 0
        idx = int(block[bpos])
        bpos += 1
        lo, hi = indptr[idx], indptr[idx + 1]
        t += 1
        model.update(
            indices[lo:hi],
            data[lo:hi],
            y_train[idx],
            weights[y_train[idx]],
            eta0 / np.sqrt(t),
            lam,
        )
        if t % eval_every == 0:
            # decoded_weights fills a float64 vector, so the loss matvec
            # and reductions run in float64.
            w_hat = decoded_weights(model, train_feats, n_features)
            loss = ridge_objective(
                X_train @ w_hat + model.bias,
                y_train,
                sample_weights,
                lam,
                float(w_hat @ w_hat),
            )
            acc = float(np.mean((test_scores(model, X_test, n_features) >= 0) == (y_test > 0)))
            trajectory.append([t, round(loss, 8), round(acc, 6)])
            if loss < best_loss * (1.0 - IMPROVEMENT_TOL):
                last_improved = t
            if loss < best_loss:
                best_loss = loss
                best_acc = acc
            if t - last_improved >= patience:
                break
    return float(best_acc), float(best_loss), t, trajectory


def run_one(method, budget, X, y, seed, alpha, balanced, eta0):
    """One (method, budget, seed) run on one task; returns a record dict."""
    idx_train, idx_test = train_test_split(
        np.arange(X.shape[0]),
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    X_train, X_test = X[idx_train].tocsr(), X[idx_test].tocsr()
    y_train, y_test = y[idx_train], y[idx_test]

    if method == "hashing":
        # Exact ridge terminal accuracy on the hashed features (Q1): the
        # bucket map is applied to the whole task so train/test share it.
        X_hashed, _ = sketch_utils.sketch_features(X, budget, "bucket", seed=seed)
        X_hashed = X_hashed.tocsr()
        clf = RidgeClassifier(
            random_state=42,
            alpha=alpha,
            solver="auto",
            class_weight="balanced" if balanced else None,
        )
        clf.fit(X_hashed[idx_train], y_train)
        acc = float(clf.score(X_hashed[idx_test], y_test))
        # The exact minimizer's value of the streamed objective (lambda =
        # alpha/N correspondence), directly comparable to the SGD min loss.
        weights = class_weights(y_train, balanced)
        w_exact = clf.coef_.ravel()
        lam = alpha / len(idx_train)
        train_loss = ridge_objective(
            X_hashed[idx_train] @ w_exact + float(clf.intercept_[0]),
            y_train,
            np.where(y_train > 0, weights[1.0], weights[-1.0]),
            lam,
            float(w_exact @ w_exact),
        )
        return {
            "accuracy": acc,
            "train_loss": train_loss,
            "space": int(min(budget, X.shape[1])),
            "samples": 0,
        }

    if method == "hashing_sgd":
        # Feature hashing trained by the shared streaming protocol: the same
        # bucket map as the exact branch (same seed), with the hashed bucket
        # weights learned by the same SGD schedule and stopping rule as the
        # adaptive methods.
        X_hashed, _ = sketch_utils.sketch_features(X, budget, "bucket", seed=seed)
        X_hashed = X_hashed.tocsr()
        model = DenseRidgeSGD(X_hashed.shape[1], seed)
        lam = alpha / len(idx_train)
        acc, train_loss, samples, trajectory = stream_until_terminal(
            model,
            X_hashed[idx_train].tocsr(),
            y_train,
            X_hashed[idx_test].tocsr(),
            y_test,
            seed,
            lam,
            balanced,
            eta0,
        )
        return {
            "accuracy": acc,
            "train_loss": train_loss,
            "space": int(min(budget, X.shape[1])),
            "samples": int(samples),
            "trajectory": trajectory,
        }

    model = ADAPTIVE_MODELS[method](budget, X.shape[1], seed)
    lam = alpha / len(idx_train)
    acc, train_loss, samples, trajectory = stream_until_terminal(
        model, X_train, y_train, X_test, y_test, seed, lam, balanced, eta0
    )
    space = model.space
    return {
        "accuracy": acc,
        "train_loss": train_loss,
        "space": int(space),
        "samples": int(samples),
        "trajectory": trajectory,
    }


def qos_reference_runs(dataset, tasks):
    """Full-matrix QOS reference point for one dataset.

    Quantum oracle sketching reaches the exact full-dimensional solution, so
    its terminal accuracy is the exact ridge accuracy on the untruncated data
    under the survey's own 80/20 protocol, and its machine size is the SI
    formula evaluated on the training stream the algorithm would consume
    (train rows, full feature dimension, train max row/column sparsity).
    """
    seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
    alpha, balanced = ALPHA[dataset], BALANCED[dataset]
    runs = []
    for name, X, y in tasks:
        for seed in seeds:
            idx_train, idx_test = train_test_split(
                np.arange(X.shape[0]),
                test_size=0.2,
                stratify=y,
                random_state=seed,
            )
            clf = RidgeClassifier(
                random_state=42,
                alpha=alpha,
                solver="auto",
                class_weight="balanced" if balanced else None,
            )
            clf.fit(X[idx_train], y[idx_train])
            space = sweep_utils.qos_machine_size(
                len(idx_train),
                X.shape[1],
                sketch_utils.max_sparsity(X[idx_train]),
                "svm",
            )
            runs.append(
                {
                    "task": name,
                    "seed": seed,
                    "accuracy": float(clf.score(X[idx_test], y[idx_test])),
                    "space": float(space),
                }
            )
    return runs


# --- drivers ------------------------------------------------------------------

def _write_json_atomic(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def tune_eta(datasets):
    """Select ETA0 per dataset by a full-dimensional SGD ridge run.

    One run per candidate on the untruncated data (IMDb: the single task;
    PBMC68k: the first cell-type pair), first sampled seed, the dataset's
    own alpha and class weighting; chosen by MINIMAL TRAINING LOSS (ties
    break toward the smaller eta0), so the selection is test-blind like the
    runs themselves. Records survey_adaptive_eta0.json.
    """
    seed = sketch_utils.sample_sketch_seeds(1)[0]
    record = {}
    if os.path.exists("survey_adaptive_eta0.json"):
        with open("survey_adaptive_eta0.json", "r") as f:
            record = json.load(f)
    for dataset in datasets:
        name, X, y = dataset_tasks(dataset)[0]
        alpha, balanced = ALPHA[dataset], BALANCED[dataset]
        idx_train, idx_test = train_test_split(
            np.arange(X.shape[0]), test_size=0.2, stratify=y, random_state=seed
        )
        lam = alpha / len(idx_train)
        entry = {"task": name}
        for eta0 in ETA0_CANDIDATES:
            model = DenseRidgeSGD(X.shape[1], seed)
            acc, train_loss, samples, _ = stream_until_terminal(
                model,
                X[idx_train].tocsr(),
                y[idx_train],
                X[idx_test].tocsr(),
                y[idx_test],
                seed,
                lam,
                balanced,
                eta0,
            )
            entry[str(eta0)] = {
                "train_loss": train_loss,
                "accuracy": acc,
                "samples": samples,
            }
            print(
                f"{dataset} eta0={eta0}: min train loss {train_loss:.6f} "
                f"(accuracy there {acc:.4f}) after {samples} samples"
            )
        chosen = min(
            ETA0_CANDIDATES,
            key=lambda e: (entry[str(e)]["train_loss"], e),
        )
        entry["chosen"] = chosen
        record[dataset] = entry
        _write_json_atomic("survey_adaptive_eta0.json", record)
        print(f"{dataset}: selected ETA0 = {chosen}")
    return record


def dataset_tasks(dataset):
    """List of (task_name, X, y) classification tasks for one dataset."""
    if dataset == "imdb":
        X, y = load_imdb()
        return [("imdb", X, y)]
    tasks = load_pbmc68k_pairs()
    return [(f"{c1}|{c2}", X, y) for X, y, (c1, c2) in tasks]


def run_dataset(dataset, n_jobs):
    eta0 = ETA0[dataset]
    if eta0 is None:
        raise RuntimeError(
            f"ETA0[{dataset!r}] is unset; run --tune-eta and record the value"
        )
    seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
    alpha, balanced = ALPHA[dataset], BALANCED[dataset]
    budgets = BUDGETS[dataset]
    print(f"Adaptive sketching survey on {dataset}: methods {METHODS}")
    print(f"Budgets: {budgets}; seeds: {seeds}; eta0={eta0}")
    tasks = dataset_tasks(dataset)
    print(f"{len(tasks)} classification task(s)")

    fingerprint = {
        "dataset": dataset,
        "methods": list(METHODS),
        "sketch_seeds": seeds,
        "budgets": budgets,
        "n_tasks": len(tasks),
        "eta0": eta0,
        "reported": REPORTED_VALUE,
        "arithmetic": "float64",
    }

    def compatible(checkpoint):
        # A checkpoint (or an earlier final JSON) is reusable if it matches
        # the configuration exactly, except that it may hold a SUBSET of the
        # current methods: only the missing methods are then computed.
        others = {k: v for k, v in fingerprint.items() if k != "methods"}
        return {k: checkpoint.get(k) for k in others} == others and set(
            checkpoint.get("methods", [])
        ) <= set(METHODS)

    checkpoint_path = f"survey_adaptive_{dataset}.partial.json"
    output_json = f"survey_adaptive_{dataset}.json"
    raw = {}
    qos_reference = None
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        if not compatible(checkpoint):
            raise ValueError(
                f"{checkpoint_path} was written by an incompatible "
                "configuration; delete it to start from scratch"
            )
        raw = checkpoint["raw_data_by_budget"]
        print(f"Resuming: {len(raw)}/{len(budgets)} budgets already present")
    elif os.path.exists(output_json):
        with open(output_json, "r") as f:
            previous = json.load(f)
        if compatible(previous):
            raw = previous["raw_data_by_budget"]
            qos_reference = previous.get("qos_reference")
            print(f"Extending {output_json} with any missing methods")

    from tqdm import tqdm

    # One flat job pool across ALL budgets: within a budget the slowest
    # method's runs are stragglers, so a per-budget barrier would
    # serialize them; a flat pool packs them across every core. Jobs are ordered budget-major and
    # joblib's ordered generator yields them in that order, so each budget
    # is checkpointed exactly when its last job arrives, while the workers
    # run ahead into later budgets. Results are scheduling-independent.
    missing = {
        budget: [m for m in METHODS if m not in raw.get(str(budget), {})]
        for budget in budgets
    }
    jobs = [
        (budget, method, ti, seed)
        for budget in budgets
        for method in missing[budget]
        for ti in range(len(tasks))
        for seed in seeds
    ]
    if jobs:
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(run_one)(
                method, budget, tasks[ti][1], tasks[ti][2], seed, alpha, balanced, eta0
            )
            for budget, method, ti, seed in jobs
        )
        pending = {
            budget: {m: {"runs": []} for m in missing[budget]}
            for budget in budgets
            if missing[budget]
        }
        remaining = {
            budget: len(missing[budget]) * len(tasks) * len(seeds)
            for budget in pending
        }
        for (budget, method, ti, seed), res in zip(
            jobs, tqdm(results, total=len(jobs), desc="runs")
        ):
            pending[budget][method]["runs"].append(
                {"task": tasks[ti][0], "seed": seed, **res}
            )
            remaining[budget] -= 1
            if remaining[budget] == 0:
                record = dict(raw.get(str(budget), {}))
                record.update(pending.pop(budget))
                raw[str(budget)] = {m: record[m] for m in METHODS}
                _write_json_atomic(
                    checkpoint_path, {**fingerprint, "raw_data_by_budget": raw}
                )

    if qos_reference is None:
        print("Computing full-matrix QOS reference (exact full-dimensional ridge)...")
        qos_reference = {"runs": qos_reference_runs(dataset, tasks)}
    data = {
        **fingerprint,
        "alpha": alpha,
        "balanced": balanced,
        "baseline": BASELINE_METHOD,
        "tasks": [name for name, _, _ in tasks],
        "qos_reference": qos_reference,
        "raw_data_by_budget": raw,
    }
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"Saved raw data to {output_json}")
    return output_json


# --- plotting -----------------------------------------------------------------

# Solid lines are streaming-SGD results under the shared protocol; the
# exact-solver feature hashing reference is dashed.
STYLES = {
    "hashing": dict(
        color=sweep_utils.COLORS["streaming"],
        linestyle="--",
        marker="P",
        marker_size=50,
        filled=False,
        label="Feature hashing (exact)",
    ),
    "hashing_sgd": dict(
        color=sweep_utils.COLORS["streaming"],
        linestyle="-",
        marker="P",
        marker_size=50,
        filled=True,
        label="Feature hashing (SGD)",
    ),
    "awm": dict(
        color="#2A8C55", linestyle="-", marker="o", marker_size=42, filled=False,
        label="AWM-Sketch",
    ),
    "mission": dict(
        color="#7B3294", linestyle="-", marker="^", marker_size=42, filled=False,
        label="MISSION",
    ),
}
QOS_STYLE = dict(
    color=sweep_utils.COLORS["quantum"],
    marker="D",
    marker_size=45,
    label="Quantum oracle sketching",
)
# Slightly wider than the main-figure windows: the streaming-SGD feature
# hashing curve reaches chance level on IMDb at small budgets and spans
# 0.71-0.92 on PBMC68k.
ACC_AXES = {
    "imdb": dict(xlim=(0.48, 0.92), xticks=[0.50, 0.60, 0.70, 0.80, 0.90]),
    "pbmc68k": dict(
        xlim=(0.69, 0.93), xticks=[0.70, 0.75, 0.80, 0.85, 0.90]
    ),
}
PANEL_TITLES = {"imdb": "IMDb classification", "pbmc68k": "PBMC68k classification"}
YLIM = (1e1, 1e7)


def _interp_with_end_slopes(x, xp, fp):
    """Piecewise-linear interpolation with end-slope extrapolation."""
    if x <= xp[0]:
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return float(fp[0] + slope * (x - xp[0]))
    if x >= xp[-1]:
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return float(fp[-1] + slope * (x - xp[-1]))
    return float(np.interp(x, xp, fp))


def _baseline_curves(data):
    """Per (task, seed): the baseline's accuracy vs log10(machine size)."""
    raw = data["raw_data_by_budget"]
    points = {}
    for b in raw:
        for r in raw[b][data["baseline"]]["runs"]:
            points.setdefault((r["task"], r["seed"]), []).append(
                (np.log10(r["space"]), r["accuracy"])
            )
    curves = {}
    for key, pts in points.items():
        pts.sort()
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        # PBMC68k pairs are drawn with replacement, so a repeated pair yields
        # identical duplicate points on its curve; collapse equal sizes.
        ux, inverse = np.unique(xs, return_inverse=True)
        uy = np.bincount(inverse, weights=ys) / np.bincount(inverse)
        curves[key] = (ux, uy)
    return curves


def panel_stats(data):
    """Per-method curve statistics.

    The difference panels pair each run against the feature hashing baseline
    AT EQUAL MEASURED MACHINE SIZE: for a run of measured size S (which can
    differ from the nominal budget), the paired reference is the same
    task/seed baseline curve linearly interpolated at log10(S), with
    end-slope extrapolation beyond the swept range. For the baseline itself
    the interpolant passes through its own grid points, so its difference is
    identically zero.
    """
    raw = data["raw_data_by_budget"]
    budgets = sorted(raw, key=int)
    base_curves = _baseline_curves(data)
    stats = {
        m: {"space": [], "mean": [], "sem": [], "dmean": [], "dsem": []}
        for m in data["methods"]
    }
    for b in budgets:
        for m in data["methods"]:
            runs = raw[b][m]["runs"]
            acc = np.array([r["accuracy"] for r in runs])
            space = float(np.mean([r["space"] for r in runs]))
            diffs = np.array(
                [
                    r["accuracy"]
                    - _interp_with_end_slopes(
                        np.log10(r["space"]), *base_curves[(r["task"], r["seed"])]
                    )
                    for r in runs
                ]
            )
            mean, sem = sketch_utils.mean_and_sem(acc)
            dmean, dsem = sketch_utils.mean_and_sem(diffs)
            for field, value in (
                ("space", space),
                ("mean", mean),
                ("sem", sem),
                ("dmean", dmean),
                ("dsem", dsem),
            ):
                stats[m][field].append(value)
    for m in stats:
        for field in stats[m]:
            stats[m][field] = np.array(stats[m][field])
    return stats


def plot_curve(ax, method, x, x_err, y):
    style = STYLES[method]
    if np.any(x_err > 0):
        ax.fill_betweenx(
            y, x - x_err, x + x_err, color=style["color"], alpha=0.2, edgecolor="none"
        )
    ax.plot(
        x, y, linestyle=style["linestyle"], color=style["color"],
        linewidth=1.5, alpha=0.9,
    )
    if style["filled"]:
        ax.scatter(
            x, y, marker=style["marker"], color=style["color"], alpha=0.9,
            s=style["marker_size"], linewidth=0,
        )
    else:
        ax.scatter(
            x, y, marker=style["marker"], facecolors="none",
            edgecolors=style["color"], alpha=0.98, s=style["marker_size"],
            linewidth=1.2,
        )


def _percent(value, _):
    return f"{100 * value:g}%"


def _signed_percent(value, _):
    return "0%" if value == 0 else f"{100 * value:+g}%"


def plot_qos_point(ax, qos_reference):
    """The full-matrix QOS reference as a single point (mean +- SEM)."""
    acc = np.array([r["accuracy"] for r in qos_reference["runs"]])
    mean, sem = sketch_utils.mean_and_sem(acc)
    space = float(np.mean([r["space"] for r in qos_reference["runs"]]))
    if sem > 0:
        ax.plot(
            [mean - sem, mean + sem],
            [space, space],
            color=QOS_STYLE["color"],
            linewidth=1.2,
            alpha=0.5,
        )
    ax.scatter(
        [mean],
        [space],
        marker=QOS_STYLE["marker"],
        color=QOS_STYLE["color"],
        alpha=0.9,
        s=QOS_STYLE["marker_size"],
        linewidth=0,
        zorder=5,
    )


def qos_legend_handle():
    return mlines.Line2D(
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


def plot_survey(json_dir, output_pdf):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    datasets = ["imdb", "pbmc68k"]
    stats_by_dataset = {}
    for dataset in datasets:
        with open(os.path.join(json_dir, f"survey_adaptive_{dataset}.json")) as f:
            data = json.load(f)
        stats_by_dataset[dataset] = (data, panel_stats(data))

    # Panels grouped per dataset: accuracy then difference, IMDb first.
    for col, dataset in enumerate(datasets):
        data, stats = stats_by_dataset[dataset]
        ax = axes[2 * col]
        for m in data["methods"]:
            order = np.argsort(stats[m]["space"])
            plot_curve(
                ax,
                m,
                stats[m]["mean"][order],
                stats[m]["sem"][order],
                stats[m]["space"][order],
            )
        if "qos_reference" in data:
            plot_qos_point(ax, data["qos_reference"])
        cfg = ACC_AXES[dataset]
        ax.set_xlim(*cfg["xlim"])
        ax.set_xticks(cfg["xticks"])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_percent))
        ax.set_xlabel("Accuracy")
        ax.set_title(PANEL_TITLES[dataset])

    for col, dataset in enumerate(datasets):
        data, stats = stats_by_dataset[dataset]
        ax = axes[2 * col + 1]
        for m in data["methods"]:
            order = np.argsort(stats[m]["space"])
            plot_curve(
                ax,
                m,
                stats[m]["dmean"][order],
                stats[m]["dsem"][order],
                stats[m]["space"][order],
            )
        dmax = max(
            np.max(np.abs(stats[m]["dmean"]) + stats[m]["dsem"])
            for m in data["methods"]
        )
        # Linear within +-5%, log-compressed beyond. Ticks step by factors
        # of 5 (5%, 25%, ...): decade steps from 5% collide at panel width.
        dmax = 1.6 * max(dmax, 5e-2)
        ax.set_xscale("symlog", linthresh=5e-2, linscale=0.5)
        ax.set_xlim(-dmax, dmax)
        ticks = [t for t in (5e-2, 25e-2, 125e-2) if t <= dmax]
        ax.set_xticks([-t for t in reversed(ticks)] + [0] + ticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_signed_percent))
        ax.set_xlabel("Diff. from feature hashing (exact)")
        ax.set_title(PANEL_TITLES[dataset])

    for col, ax in enumerate(axes):
        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.tick_params(direction="in", which="both", top=False, right=True)
        ax.grid(True, which="major", ls="-", alpha=0.1)
        if col == 0:
            ax.set_ylabel("Machine size")
        else:
            ax.tick_params(axis="y", labelleft=False)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=STYLES[m]["color"],
            linestyle=STYLES[m]["linestyle"],
            linewidth=1.5,
            marker=STYLES[m]["marker"],
            markersize=7,
            markerfacecolor=STYLES[m]["color"] if STYLES[m]["filled"] else "none",
            markeredgecolor=STYLES[m]["color"],
            markeredgewidth=0 if STYLES[m]["filled"] else 1.2,
            label=STYLES[m]["label"],
        )
        for m in METHODS
    ]
    handles.append(qos_legend_handle())
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output_pdf)
    print(f"Saved {output_pdf}")


# Convergence figure: feature hashing SGD training dynamics at one middle and
# one large budget per dataset (the largest swept budget and, one panel to its
# left, the budget a factor 8 below it; at the sweeps' small budgets the
# near-chance accuracy trajectories are pure noise).
CONV_PANELS = [("imdb", 8192), ("imdb", 65536), ("pbmc68k", 4096), ("pbmc68k", 32768)]
CONV_LOSS_COLOR = sweep_utils.COLORS["sparse"]
CONV_ACC_COLOR = sweep_utils.COLORS["streaming"]
# Some PBMC68k pairs at large D' transiently explode by orders of magnitude
# before the 1/sqrt(t) schedule recovers them (converging runs stay within a
# factor of a few of the exact loss, so 10 separates the two cleanly). Axis
# limits follow the never-exploding runs; exploding curves exit the frame and
# re-enter where they recover.
CONV_RATIO_SANE_MAX = 10.0

# The PBMC68k panels display the single pair of the two LARGEST cell-type
# classes, which the survey's 20 random pairs happen not to include;
# --pbmc-top-pair computes it (same protocol and seeds) into TOP_PAIR_JSON.
TOP_PAIR_JSON = "survey_adaptive_pbmc68k_top2.json"
CONV_SOURCES = {"imdb": "survey_adaptive_imdb.json", "pbmc68k": TOP_PAIR_JSON}


def _convergence_ready(data, budget):
    """True if a survey JSON carries what the convergence figure needs at
    this budget: hashing_sgd trajectories plus the matched exact-solution
    ridge losses ('train_loss' on the hashing records; absent from JSONs
    written before the current convention)."""
    raw = data.get("raw_data_by_budget", {}).get(str(budget))
    if raw is None or "hashing" not in raw or "hashing_sgd" not in raw:
        return False
    return all("train_loss" in r for r in raw["hashing"]["runs"]) and all(
        "trajectory" in r for r in raw["hashing_sgd"]["runs"]
    )


def run_pbmc_top_pair(n_jobs):
    """Compute the convergence figure's PBMC68k task.

    Feature hashing (exact and streaming SGD) on the pair of the two largest
    cell-type classes at the two PBMC68k convergence budgets, under exactly
    the survey's protocol, seeds, and record shape; written to TOP_PAIR_JSON
    for --plot-convergence.
    """
    import pbmc68k_utils  # imports scvelo; keep IMDb-only runs light

    X_full, y_full, label_names = pbmc68k_utils.load_pbmc68k_data(
        min_samples=1, normalize=True, binary=False
    )
    counts = np.bincount(y_full)
    c1, c2 = sorted(np.argsort(counts)[-2:])
    rows = np.flatnonzero((y_full == c1) | (y_full == c2))
    X = X_full[rows].tocsr()
    y = np.where(y_full[rows] == c2, 1.0, -1.0)
    name = f"{label_names[c1]}|{label_names[c2]}"
    print(f"PBMC68k top-2-classes pair: {name} ({counts[c1]} + {counts[c2]} cells)")

    seeds = sketch_utils.sample_sketch_seeds(n_sketch_seeds)
    alpha, balanced, eta0 = ALPHA["pbmc68k"], BALANCED["pbmc68k"], ETA0["pbmc68k"]
    budgets = [b for ds, b in CONV_PANELS if ds == "pbmc68k"]
    methods = ("hashing", "hashing_sgd")
    jobs = [(b, m, s) for b in budgets for m in methods for s in seeds]

    from tqdm import tqdm

    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(run_one)(m, b, X, y, s, alpha, balanced, eta0) for b, m, s in jobs
    )
    raw = {}
    for (b, m, s), res in zip(jobs, tqdm(results, total=len(jobs), desc="runs")):
        raw.setdefault(str(b), {}).setdefault(m, {"runs": []})["runs"].append(
            {"task": name, "seed": s, **res}
        )
    data = {
        "dataset": "pbmc68k",
        "methods": list(methods),
        "sketch_seeds": seeds,
        "budgets": budgets,
        "eta0": eta0,
        "alpha": alpha,
        "balanced": balanced,
        "reported": REPORTED_VALUE,
        "arithmetic": "float64",
        "tasks": [name],
        "raw_data_by_budget": raw,
    }
    _write_json_atomic(TOP_PAIR_JSON, data)
    print(f"Saved {TOP_PAIR_JSON}")


def convergence_stats(data, budget):
    """Per-run hashing_sgd trajectories at one budget, on the pass grid.

    Every run evaluates at multiples of eval_every = n_train // EVAL_DIVISOR,
    so the k-th trajectory entry sits at k/EVAL_DIVISOR passes for every task
    and seed; each run extends to its own patience point. The loss is the
    training ridge objective as a RATIO to the exact solution of the SAME
    hashed problem (matched task/seed 'hashing' record): the exact reference
    is the constant 1, and the stopping rule's relative-improvement tolerance
    acts on this scale (a log excess-loss axis would magnify the tail into
    apparent non-convergence). The accuracy is likewise the ratio of the test
    accuracy to the matched exact solution's, so every run is normalized by
    its own seed's reference rather than a cross-seed mean.
    """
    raw = data["raw_data_by_budget"][str(budget)]
    exact = {(r["task"], r["seed"]): r for r in raw["hashing"]["runs"]}
    runs = []
    for r in raw["hashing_sgd"]["runs"]:
        ex = exact[(r["task"], r["seed"])]
        traj = np.asarray(r["trajectory"], dtype=float)
        runs.append(
            {
                "passes": np.arange(1, len(traj) + 1) / EVAL_DIVISOR,
                "loss_ratio": traj[:, 1] / ex["train_loss"],
                "acc_ratio": traj[:, 2] / ex["accuracy"],
            }
        )
    return {"runs": runs}


def plot_convergence(json_dir, output_pdf):
    """1x4 feature hashing SGD convergence figure (CONV_PANELS).

    Per panel: left axis the training ridge loss relative to the exact
    same-budget solution (log scale, dashed grey reference at 1), right axis
    the test accuracy relative to the same matched exact solution (dashed
    blue reference at 1). Every run (task x seed) is drawn as its own
    semi-transparent curve; the axes are scaled per panel.
    """
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7))
    data_cache = {}
    panel_stats = []
    for dataset, budget in CONV_PANELS:
        if dataset not in data_cache:
            path = os.path.join(json_dir, CONV_SOURCES[dataset])
            data_cache[dataset] = None
            if os.path.exists(path):
                with open(path) as f:
                    data_cache[dataset] = json.load(f)
            else:
                print(f"{path} not found; leaving the {dataset} panel(s) empty")
        data = data_cache[dataset]
        if data is not None and not _convergence_ready(data, budget):
            # e.g. a stale JSON from an earlier convention (no exact-solution
            # train loss recorded) while the rerun is still in progress.
            print(
                f"{CONV_SOURCES[dataset]} lacks the current fields at "
                f"budget {budget}; leaving that panel empty"
            )
            data = None
        panel_stats.append(convergence_stats(data, budget) if data else None)

    # Axis limits follow the never-exploding runs (CONV_RATIO_SANE_MAX), so
    # a transient blow-up cannot stretch a panel until every converged
    # trajectory flattens; axhline is excluded from autoscaling, so the
    # exact references (1 on both axes) enter the limits explicitly.
    def sane_runs(stats):
        runs = [
            r for r in stats["runs"]
            if r["loss_ratio"].max() <= CONV_RATIO_SANE_MAX
        ]
        return runs or stats["runs"]

    # The PBMC68k panels show the single top-2-classes pair; tag the second
    # title line (the first would collide with the neighboring panels).
    subtitles = {"imdb": "", "pbmc68k": "top-2, "}
    right_axes = []
    for ax, (dataset, budget), stats in zip(axes, CONV_PANELS, panel_stats):
        ax.set_title(f"{PANEL_TITLES[dataset]}\n{subtitles[dataset]}$D' = {budget}$")
        ax_r = ax.twinx()
        right_axes.append(ax_r)
        if stats is None:
            ax.text(
                0.5, 0.5, "pending", transform=ax.transAxes,
                ha="center", va="center", color="0.6",
            )
            continue
        runs = stats["runs"]
        # Fewer runs (IMDb: 5 seeds) can carry more opacity than many
        # (PBMC68k: 20 pairs x 5 seeds) before the overlay saturates.
        alpha = max(0.12, min(0.45, 4.0 / len(runs)))
        for run in runs:
            ax.plot(
                run["passes"], run["loss_ratio"],
                color=CONV_LOSS_COLOR, linewidth=0.9, alpha=alpha,
            )
            ax_r.plot(
                run["passes"], run["acc_ratio"],
                color=CONV_ACC_COLOR, linewidth=0.9, alpha=alpha,
            )
        ax.axhline(
            1.0, color=CONV_LOSS_COLOR, linestyle="--", linewidth=1.2, alpha=0.9
        )
        ax_r.axhline(
            1.0, color=CONV_ACC_COLOR, linestyle="--", linewidth=1.2, alpha=0.9
        )
        # Per-panel limits from the never-exploding runs; axhline is excluded
        # from autoscaling, so the exact references enter them explicitly.
        sane = sane_runs(stats)
        hi = max(r["loss_ratio"].max() for r in sane)
        lo = min(1.0, min(r["loss_ratio"].min() for r in sane))
        pad = (hi / lo) ** 0.05
        ax.set_ylim(lo / pad, hi * pad)
        hi = max(max(r["acc_ratio"].max() for r in runs), 1.0)
        lo = min(min(r["acc_ratio"].min() for r in runs), 1.0)
        pad = 0.05 * (hi - lo)
        ax_r.set_ylim(lo - pad, hi + pad)
        # The pass axis is also per panel: one exploded run streaming to the
        # sample cap must not squeeze the converged majority.
        x_hi = max(r["passes"][-1] for r in sane)
        ax.set_xlim(-0.02 * x_hi, 1.02 * x_hi)

    plain = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    for col, (ax, ax_r) in enumerate(zip(axes, right_axes)):
        ax.set_yscale("log")
        # Label the log ticks as plain numbers instead of 1.1x10^0 style. Up
        # to about a half-decade span the minor ticks are labeled too; wider
        # panels get the sparse 1-2-5 sequence (all nine minors would crowd).
        lo, hi = ax.get_ylim()
        ax.yaxis.set_major_formatter(plain)
        if hi / lo < 6:
            ax.yaxis.set_minor_formatter(plain)
        else:
            ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("samples / training set size")
        ax.tick_params(direction="in", which="both", top=False)
        ax_r.tick_params(direction="in", which="both")
        ax.grid(True, which="major", ls="-", alpha=0.1)
        if col == 0:
            ax.set_ylabel("Loss / exact loss")
        if col == len(axes) - 1:
            ax_r.set_ylabel("Test accuracy / exact")

    handles = [
        mlines.Line2D(
            [], [], color=CONV_LOSS_COLOR, linewidth=1.5,
            label="Feature hashing (SGD): loss",
        ),
        mlines.Line2D(
            [], [], color=CONV_LOSS_COLOR, linewidth=1.2, linestyle="--",
            label="Feature hashing (exact): loss",
        ),
        mlines.Line2D(
            [], [], color=CONV_ACC_COLOR, linewidth=1.5,
            label="Feature hashing (SGD): accuracy",
        ),
        mlines.Line2D(
            [], [], color=CONV_ACC_COLOR, linewidth=1.2, linestyle="--",
            label="Feature hashing (exact): accuracy",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output_pdf)
    print(f"Saved {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive sketching survey (IMDb and PBMC68k classification)."
    )
    parser.add_argument("--dataset", choices=["imdb", "pbmc68k"])
    parser.add_argument(
        "--tune-eta",
        action="store_true",
        help="select ETA0 per dataset on the untruncated data and record the "
        "result (both datasets, or only --dataset when given)",
    )
    parser.add_argument(
        "--add-qos-ref",
        action="store_true",
        help="compute the full-matrix QOS reference and insert it into the "
        "existing survey JSONs without redoing the budget sweeps",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-convergence",
        action="store_true",
        help="assemble the 1x4 feature hashing SGD convergence figure "
        "(loss and accuracy vs samples / training set size)",
    )
    parser.add_argument(
        "--pbmc-top-pair",
        action="store_true",
        help="compute the convergence figure's PBMC68k task (feature hashing "
        "on the pair of the two largest classes) into "
        "survey_adaptive_pbmc68k_top2.json",
    )
    parser.add_argument("--json-dir", type=str, default=".")
    parser.add_argument("--out", type=str, default="survey_adaptive.pdf")
    parser.add_argument(
        "--out-convergence", type=str, default="survey_adaptive_convergence.pdf"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="joblib workers over independent runs; results do not depend on this",
    )
    args = parser.parse_args()

    ran = False
    if args.tune_eta:
        tune_eta([args.dataset] if args.dataset else ["imdb", "pbmc68k"])
        ran = True
    if args.dataset and not args.tune_eta:
        run_dataset(args.dataset, args.n_jobs)
        ran = True
    if args.add_qos_ref:
        for dataset in ("imdb", "pbmc68k"):
            path = os.path.join(args.json_dir, f"survey_adaptive_{dataset}.json")
            if not os.path.exists(path):
                print(f"{path} not found; skipping")
                continue
            with open(path, "r") as f:
                data = json.load(f)
            data["qos_reference"] = {
                "runs": qos_reference_runs(dataset, dataset_tasks(dataset))
            }
            _write_json_atomic(path, data)
            print(f"Inserted the full-matrix QOS reference into {path}")
        ran = True
    if args.plot:
        plot_survey(args.json_dir, args.out)
        ran = True
    if args.pbmc_top_pair:
        run_pbmc_top_pair(args.n_jobs)
        ran = True
    if args.plot_convergence:
        plot_convergence(args.json_dir, args.out_convergence)
        ran = True
    if not ran:
        parser.error(
            "nothing to do: pass --tune-eta, --dataset, --plot, or "
            "--plot-convergence"
        )


if __name__ == "__main__":
    main()

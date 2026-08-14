import os
import sys
import time
import warnings

import dill as pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import effective_n_jobs, Parallel
from jupyterthemes import jtplot
from scipy.special import erf
from scipy.stats import hmean, skew, skewnorm

from sklearn.base import BaseEstimator
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    DistanceMetric,
    pairwise_distances,
    pairwise_distances_chunked,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import BallTree, KDTree, KNeighborsClassifier
from sklearn.neighbors._base import KNeighborsMixin, NeighborsBase
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils import gen_even_slices
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from knn_utils import default, timer

pd.options.display.max_columns = 999
warnings.simplefilter(action="ignore", category=DataConversionWarning)

jtplot.style(theme="grade3", context="notebook", ticks=True, grid=False)


current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
DATA_PATH = default(os.environ.get("DATA_PATH"), os.path.join(SRC_ROOT, "data"))


def default(value, default_val):
    return default_val if value is None else value


def plot_accuracy_vs_trust_score(accuracy, trust_score):
    """
    plot the histogram of the trust score vs correctness
    accuracy: boolean array of the prediction correctness
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_ylabel(r"# samples", fontsize=14)
    ax.set_xlabel("Trust score", fontsize=14)
    ax.set_axisbelow(True)
    ax.grid()
    _min = min(trust_score)
    _max = max(trust_score)
    n_bins = 15
    width = (_max - _min) / n_bins

    hist, bin_edges = np.histogram(trust_score, bins=n_bins, range=(_min, _max))

    wrong = trust_score[~accuracy]
    hist_wrong, _ = np.histogram(wrong, bins=n_bins, range=(_min, _max))
    ax.bar(
        np.linspace(_min, _max, len(hist)),
        hist_wrong,
        width=width,
        alpha=0.95,
        color="#d65757",
        capsize=4,
        label=f'{"Wrong":-<10} {sum(hist_wrong)} samples avg score: {wrong.mean():.2f}',
    )

    correct = trust_score[accuracy]
    hist_correct, _ = np.histogram(correct, bins=n_bins, range=(_min, _max))
    ax.bar(
        np.linspace(_min, _max, len(hist)),
        hist_correct,
        bottom=hist_wrong,
        width=width,
        alpha=0.95,
        color="#088cad",
        capsize=4,
        label=f'{"Correct":-<10} {sum(hist_correct)} samples avg score: {correct.mean():.2f}',
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=2, fontsize=14)
    plt.show()


def get_knn_probability(
    X_train,
    y_train,
    X_valid,
    k=9,
    tol=1,
    num_classes=4,
    weights="distance",
    class_weights=None,
):
    """
    a small compact function that get the
    distance weighted voting using k nearest neighbors
    using class weights and inverse distance weights
    the full sklearn API is below

    Ref:
    Tan, S. (2005). Neighbor-weighted k-nearest neighbor for unbalanced text corpus. Expert Systems with Applications, 28(4), 667-671.
    """

    def Hamming_distance(A, B):
        # MOD(hamming_cgmlst): EnteroBase-like cgMLST distance (normalized by valid loci per pair).
        # Only positive integers are valid alleles; others are treated as 0 (missing).
        A = np.asarray(A)
        B = np.asarray(B)

        A = np.where(A > 0, A, 0)
        B = np.where(B > 0, B, 0)

        valid = (A[:, None, :] > 0) & (B[None, :, :] > 0)
        valid_n = valid.sum(axis=2)
        diff = ((A[:, None, :] != B[None, :, :]) & valid).sum(axis=2)

        dist = np.where(valid_n > 0, diff / valid_n, 1.0)
        return dist

    dist = Hamming_distance(X_valid.values, X_train.values)

    # if distance is too small, rescale it
    eps = tol if tol < 1 else 1e-6
    dist[dist < eps] = eps

    # sort by columns for each row, then return the first k columns' indices
    index_neighbors = np.argsort(dist, axis=1)
    sorted_dist = np.take_along_axis(dist, index_neighbors, axis=1)

    index_knn = index_neighbors[:, :k]

    # the above are the indices, this is computing the inverse distances
    if weights == "distance":
        distance_weights = 2 / sorted_dist[:, :k]
    elif weights == "uniform":
        distance_weights = 1

    # retrieving the labels of these k neighbors
    label_knn = y_train[index_knn]

    # computing the vote
    vote = np.zeros((X_valid.shape[0], num_classes))
    class_weights = np.ones(num_classes) if class_weights is None else class_weights
    for j in range(num_classes):
        vote[:, j] = class_weights[j] * np.sum(
            distance_weights * (label_knn == j), axis=1
        )

    # normalize the vote to become a probability
    # proba = vote/np.sum(vote,axis=1)[:, None]
    proba = np.exp(vote) / np.exp(vote).sum(axis=1)[:, None]

    return proba


class WeightKNeighborsClassifier(NeighborsBase, KNeighborsMixin):
    def __init__(
        self,
        *,
        n_neighbors=9,
        algorithm="ball_tree",
        leaf_size=18,
        p=2,
        metric="hamming_cgmlst",  # default to penalty distance
        kcand=100,  # candidate pool size for re-ranking
        
        metric_params=None,
        n_jobs=None,
        weights="distance",
        tree_type="ball_tree",  # or "kd_tree"
        gamma=2.5,  # default exp for class weights
        delta=1,  # default knn distance exp if not softmax
        proba_method="softmax",
        train_labels=None,
        add_class_weights=True,
        sampling=False,  # whether using sampling to compute pairwise distance
        sampling_ratio=0.2,
        n_samples_fit=None,
        tree_path=None,
        tree_by_cls_path=None,
        sanity_verbose=False,  # verbose per-assert logging
        sanity_summary=True,  # print compact summaries at key steps
        alpha=0.0,  # alpha is parameter for the filtering
        filtering_method="none",  # "none", "density", "uncertainty"
        pre_filtering=False,  # whether to pre-filter the training data
        score_max_query_neighbors=20,  # defeault maximum number of neighbors to compute the trust score when query a single sample
        score_mean_train=None,  # mean of the distance ratio for each class in train data
        score_std_train=None,  # std of the distance ratio for each class in train data
        score_skew_train=None,  # skew of the distance ratio for each class in train data
        query_agg_func=np.nanmean,  # np.mean, np.median, np.nanmean, np.nanmedian
        score_normalization="sigmoid",  # normal or sigmoid,
        n_filtered_samples=None,  # for trust score
        min_dist=1e-5,
        eps=1e-7,
        beta=1,  # score exponent
        random_state=1127825,
        verbose=1,
        **kwargs,
    ):
        self.metric_mode = metric
        self.kcand = kcand
        self.sanity_verbose = bool(sanity_verbose)
        self.sanity_summary = bool(sanity_summary)
        tree_metric = "hamming" if metric in ("hamming", "hamming_cgmlst") else metric

        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=tree_metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

        self.weights = weights
        self.train_labels = train_labels
        self.add_class_weights = add_class_weights
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self._fit_method = self.algorithm
        self.proba_method = proba_method
        self.score_max_query_neighbors = score_max_query_neighbors
        self.filtering_method = filtering_method
        self.pre_filtering = pre_filtering
        self.is_filtered = False
        self.query_agg_func = query_agg_func
        self.score_normalization = score_normalization
        self.score_mean_train = score_mean_train
        self.score_std_train = score_std_train
        self.score_skew_train = score_skew_train
        self.sampling = sampling
        self.sampling_ratio = sampling_ratio
        self.n_samples_fit = n_samples_fit
        self.random_state = random_state
        self.min_dist = min_dist
        self.eps = eps
        self.tree_type = tree_type
        self.tree_path = tree_path
        self.tree_by_cls_path = tree_by_cls_path
        self.n_filtered_samples = n_filtered_samples
        self.verbose = verbose

    @staticmethod
    def hamming_cgmlst_batch(X, Y, return_counts=False):
        """
        Rules:
        - Only positive integers are treated as valid allele calls.
        - All other values are treated as 0 (missing).
        - A locus contributes to distance only if both isolates have valid (positive) allele calls.
        - For each pair, distance is normalized by the number of valid loci for that pair:
              dist = diff_count / valid_loci_count
          where diff_count counts loci with valid calls in both isolates and different alleles.

        Parameters:
        X : array-like, shape (n_query, p) or (p,)
        Y : array-like, shape (n_query, k, p) or (k, p)
        return_counts : bool
            If True, also return (diff_count, valid_loci_count).

        Returns:
        dist : ndarray, shape (n_query, k)
        diff_count : ndarray, shape (n_query, k), optional
        valid_loci_count : ndarray, shape (n_query, k), optional
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        Y = np.asarray(Y)
        if Y.ndim == 2:
            Y = Y[None, :, :]

        # Keep only positive integers as valid alleles; everything else -> 0
        Xv = np.where(X > 0, X, 0)
        Yv = np.where(Y > 0, Y, 0)

        valid = (Xv[:, None, :] > 0) & (Yv > 0)  # (nq, k, p)
        valid_n = valid.sum(axis=2)              # (nq, k)
        diff = ((Yv != Xv[:, None, :]) & valid).sum(axis=2)  # (nq, k)

        # Avoid divide-by-zero; if no valid loci, set dist=1.0 (maximal distance)
        dist = np.where(valid_n > 0, diff / valid_n, 1.0)

        if return_counts:
            return dist, diff, valid_n
        return dist

    @staticmethod
    def load_trees(path):
        if path is not None:
            with open(path, "rb") as f:
                trees = pickle.load(f)
            return trees
        else:
            return None

    @staticmethod
    def save_trees(trees, path):
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(trees, f)

    def build_tree(self, X, path=None):
        with timer(f"Building tree"):
            """
            to check the leaf and tree shape
            tr = self._tree.get_arrays()
            for leaf in tr:
                print(leaf.shape)
            """
            if self._fit_method == "ball_tree":
                self._tree = BallTree(
                    X,
                    self.leaf_size,
                    metric=self.metric,
                    **self.effective_metric_params_,
                )
            elif self._fit_method == "kd_tree":
                if (
                    self.metric == "minkowski"
                    and self.effective_metric_params_.get("w") is not None
                ):
                    raise ValueError(
                        "algorithm='kd_tree' is not valid for "
                        "metric='minkowski' with a weight parameter 'w': "
                        "try algorithm='ball_tree' "
                    )
                self._tree = KDTree(
                    X,
                    self.leaf_size,
                    metric=self.metric,
                    **self.effective_metric_params_,
                )
        if self.verbose > 0:
            print(f"Tree size: {self._tree.__sizeof__()}")
            print(f"Saving tree to {path}")
        self.save_trees(self._tree, path)

    def _assert_(self, cond, msg):
        if getattr(self, 'sanity_verbose', False):
            print(f"[SANITY] {msg} -> {'OK' if cond else 'FAIL'}")
        if not cond:
            raise ValueError(f"[SANITY] {msg}")

    def _sanity_print(self, msg):
        if getattr(self, 'sanity_summary', False):
            print(f"[SANITY] {msg}")


    def _sanity_check_inputs(self, X, y, additional_labels=None):
        self._sanity_print('Input checks: lengths/shapes')
        self._assert_(X is not None, "X is None")
        self._assert_(y is not None, "y is None")
        self._assert_(len(X) == len(y), f"len(X)={len(X)} != len(y)={len(y)}")
        if additional_labels is not None:
            self._assert_(len(additional_labels) == len(X),
                          f"len(additional_labels)={len(additional_labels)} != len(X)={len(X)}")
        self._assert_(hasattr(X, "shape") and len(X.shape) == 2, f"X must be 2D, got shape={getattr(X,'shape',None)}")

    def _sanity_check_state_after_fit(self):
        fold = getattr(self, 'fold_id', None)
        n_fit = len(self._fit_X) if hasattr(self, '_fit_X') and self._fit_X is not None else None
        k = int(getattr(self, 'n_neighbors', -1))
        kcand = int(getattr(self, 'kcand', -1))
        mode = getattr(self, 'metric_mode', 'hamming')
        self._sanity_print(f'fold={fold} n_fit={n_fit} k={k} kcand={kcand} metric={mode}')
        if hasattr(self, '_fit_y') and self._fit_y is not None:
            counts = {int(lab): int((self._fit_y==lab).sum()) for lab in getattr(self, 'train_labels', [])}
            self._sanity_print(f'per-class train counts={counts}')
        if hasattr(self, 'tree_path') and hasattr(self, 'tree_by_cls_path'):
            self._sanity_print(f'tree_path={getattr(self, "tree_path", None)} tree_by_cls_path={getattr(self, "tree_by_cls_path", None)}')
        self._sanity_print('Post-fit checks: trees/caches consistency')
        self._assert_(hasattr(self, "_fit_X") and self._fit_X is not None, "_fit_X missing after fit")
        self._assert_(hasattr(self, "_fit_y") and self._fit_y is not None, "_fit_y missing after fit")
        self._assert_(len(self._fit_X) == len(self._fit_y),
                      f"_fit_X/_fit_y length mismatch: {len(self._fit_X)} vs {len(self._fit_y)}")
        self._assert_(hasattr(self, "_tree") and self._tree is not None, "_tree missing after filter/build")

        self._assert_(hasattr(self, "X_by_cls_used") and self.X_by_cls_used is not None, "X_by_cls_used missing")
        self._assert_(hasattr(self, "trees") and self.trees is not None, "per-class trees missing")
        self._assert_(len(self.trees) == len(self.train_labels), "per-class trees length mismatch train_labels")

        total = 0
        for i, lab in enumerate(self.train_labels):
            Xi = self.X_by_cls_used[i]
            self._assert_(Xi is not None, f"X_by_cls_used[{i}] is None")
            total += len(Xi)
        self._assert_(total == len(self._fit_X),
                      f"sum(len(X_by_cls_used))={total} != len(_fit_X)={len(self._fit_X)}")

        if hasattr(self, "_fit_additional_labels") and self._fit_additional_labels is not None:
            if hasattr(self, "additional_labels_cls") and self.additional_labels_cls is not None:
                total2 = sum(len(a) for a in self.additional_labels_cls)
                self._assert_(total2 == len(self._fit_additional_labels),
                              f"sum(len(additional_labels_cls))={total2} != len(_fit_additional_labels)={len(self._fit_additional_labels)}")

        kcand = int(getattr(self, "kcand", 100))
        self._assert_(kcand >= 1, f"kcand must be >=1, got {kcand}")
        self._assert_(kcand >= int(self.n_neighbors), f"kcand={kcand} must be >= n_neighbors={self.n_neighbors}")

    def _sanity_check_query(self, ind, dist, expected_k):
        self._assert_(ind.shape == dist.shape, f"neighbors index shape {ind.shape} != dist shape {dist.shape}")
        self._assert_(ind.shape[1] == int(expected_k), f"returned k={ind.shape[1]} != expected_k={expected_k}")
        self._assert_(ind.min() >= 0, "neighbor index < 0")
        self._assert_(ind.max() < len(self._fit_X), "neighbor index out of range")
        self._assert_(np.all(dist >= 0.0), "distance has negative values")
        self._assert_(np.all(dist <= 1.0 + 1e-9), "distance > 1.0 (normalized)")


    def fit(self, X, y, additional_labels=None, fold_id=None):
        self._sanity_check_inputs(X, y, additional_labels)
        """
        X: feature vector
        y: label vector
        additional_labels: potential identifier of the sample, e.g., barcodes
        """
        self.fold_id = fold_id

        if isinstance(X, pd.DataFrame):
            X = X.values
        X, y = check_X_y(X, y)
        self.n_samples, self.n_features = (
            X.shape
        )
        self.labels = np.unique(y)  # y is the training data labels
        self.train_labels = default(self.train_labels, self.labels)
        if additional_labels is None:
            additional_labels = np.empty_like(y, dtype=str)
        self.n_labels = len(self.train_labels)

        if set(self.labels) != set(self.train_labels):
            if self.verbose > 0:
                print(
                    f"Training labels: {self.train_labels}, all labels: {self.labels}"
                )
            tr_idx = np.isin(y, self.train_labels)
            X = X[tr_idx]
            y = y[tr_idx]
        else:
            tr_idx = np.ones_like(y).astype(bool)

        self.n_samples_class = np.asarray([np.sum(y == l) for l in self.train_labels])
        if self.add_class_weights:
            self.class_weights = (
                self.n_samples_class / (min(self.n_samples_class) + 1)
            ) ** (-1 / self.gamma)
        else:
            self.class_weights = np.ones(self.n_labels)

        sampling_idx = np.arange(self.n_samples)[tr_idx]
        if self.sampling and self.sampling_ratio is not None:
            if self.verbose > 0:
                print(f"Sampling {self.sampling_ratio*100}% of data")
            np.random.seed(seed=self.random_state)
            assert len(X) == len(y)
            idx = np.random.randint(0, len(X), size=int(self.sampling_ratio * len(X)))
            X, y = X[idx], y[idx]
            assert len(additional_labels) == self.n_samples
            additional_labels = additional_labels[idx]
            sampling_idx = sampling_idx[idx]
        self.sampling_idx = sampling_idx
        self.n_samples = len(X)

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        tree_name = f"./cgMLST_tree_Nsample_{len(X)}_Nfeat_{X.shape[1]}_Nclass_{len(self.train_labels)}" + (f"_fold{fold_id}" if fold_id is not None else "") + ".pkl"  # MOD(CV-fold)
        self.tree_name = tree_name
        if self.tree_path is not None:
            _tree_path = os.path.join(self.tree_path, tree_name)
            try:
                with timer(f"Loading tree from {_tree_path}"):
                    self._tree = self.load_trees(_tree_path)
            except:
                self.build_tree(X, path=_tree_path)
        else:
            self.build_tree(X, path=tree_name)
            

        self.n_samples_fit = default(self.n_samples_fit, X.shape[0])
        self._fit_X = X
        self._fit_y = y
        self._fit_additional_labels = additional_labels
        self._query_dist = (None, None)
        if self.pre_filtering:
            self._filter()
            # validate fitted state/tree/cache consistency
            self._sanity_check_state_after_fit()
            # ensure fold-specific caching to prevent leakage
            fold = getattr(self, 'fold_id', None)
            if fold is not None:
                for attr in ['_tree_path', '_tree_path_by_cls']:
                    p = getattr(self, attr, None)
                    if isinstance(p, str):
                        self._assert_(f'_fold{fold}' in p, f'{attr} missing fold suffix: {p}')
            if self.score_normalization == "normal":
                if self.score_mean_train is None or self.score_std_train is None:
                    self._get_filter_stats(
                        max_query_neighbors=self.score_max_query_neighbors
                    )
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            assert isinstance(n_neighbors, int)

        if X is None:  # train data
            X = self._fit_X

        n_samples_fit = self.n_samples_fit
        if n_neighbors > n_samples_fit:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                f" but n_samples = {n_samples_fit}, n_neighbors = {n_neighbors}"
            )

        if self._fit_method == "brute":
            dist = pairwise_distances(
                X,
                self._fit_X,
                metric=self.metric,
                **self.effective_metric_params_,
            )
            index_neighbors = np.argsort(dist, axis=1)
            index_neighbors = index_neighbors[:, : self.n_neighbors]
            dist = np.take_along_axis(dist, index_neighbors, axis=1)

        elif self._fit_method in ["ball_tree", "kd_tree"]:
            # retrieve a larger candidate pool from the fast tree metric, then
            # recompute distances using hamming_cgmlst and re-rank to get final top-k.
            kcand = int(getattr(self, "kcand", 100))
            kcand = max(kcand, int(n_neighbors))
            _, index_neighbors = self._tree.query(X, kcand, return_distance=True)

            if getattr(self, "metric_mode", "hamming") == "hamming_cgmlst":
                Y_nb = self._fit_X[index_neighbors]
                # compute EnteroBase-like distance on candidates
                dist, diff_cnt, valid_cnt = self.hamming_cgmlst_batch(
                    X,
                    Y_nb,
                    return_counts=True,
                )
                # re-rank within candidate pool using cgMLST distance
                order = np.argsort(dist, axis=1)[:, :n_neighbors]
                dist = np.take_along_axis(dist, order, axis=1)
                index_neighbors = np.take_along_axis(index_neighbors, order, axis=1)
                diff_cnt = np.take_along_axis(diff_cnt, order, axis=1)
                valid_cnt = np.take_along_axis(valid_cnt, order, axis=1)

                # store counts for reporting
                self._last_diff_count = diff_cnt
                self._last_valid_loci = valid_cnt
            else:
                # fallback: standard tree distance
                dist = np.zeros((X.shape[0], n_neighbors), dtype=float)
                # recompute exact distances for returned neighbors under tree metric
                # (kept for compatibility)
                _d, _idx = self._tree.query(X, n_neighbors, return_distance=True)
                dist, index_neighbors = _d, _idx
        # dist *= self.n_features
        dist[dist < self.min_dist] = self.min_dist

        if return_distance:
            # validate neighbor indices/distances
            self._sanity_check_query(index_neighbors, dist, expected_k=n_neighbors)
            return dist, index_neighbors
        else:
            return index_neighbors

    @staticmethod
    def softmax(Z, weight=1):
        """Z: (*, n_classes)"""
        Z = np.exp(Z * weight)
        return Z / Z.sum(axis=1)[:, None]

    @staticmethod
    def softmin(Z, weight=1):
        Z = np.exp(-Z * weight)
        return Z / Z.sum(axis=1)[:, None]

    @staticmethod
    def polynomialmax(Z, q=2):
        Z = Z**q
        return Z / Z.sum(axis=1)[:, None]

    def compute_proba(self, Z: np.ndarray):
        if self.proba_method == "softmax":
            return self.softmax(Z, self.delta)
        elif self.proba_method == "softmin":
            return self.softmin(Z, self.delta)
        elif self.proba_method == "polynomial":
            return self.polynomialmax(Z, self.delta)
        else:
            raise NotImplementedError

    def predict_proba(self, X, n_neighbors=None):
        """
        the probability computation can use multiple way to compute:

        1. a 'soft' knn: exponential weighting.
          Refs:
            J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov. (2005) Neighbourhood Components Analysis. Advances in Neural Information Processing Systems. 17, 513-520, 2005.
            S. A. Dudani. (1976) The distance-weighted k-nearest neighbor rule IEEE trans. on system, man, and cybernetics. 6, 325-327
            Bicego, M., & Loog, M. (2016, December). Weighted K-nearest neighbor revisited. In 2016 23rd International Conference on Pattern Recognition (ICPR) (pp. 1642-1647). IEEE.
            Fukunaga & Hostetler (1975) Fukunaga K, Hostetler L. K-nearest-neighbor Bayes-risk estimation. IEEE Transactions on Information Theory. 1975;21(3):285-293.
            Mensink, T., Verbeek, J., Perronnin, F., & Csurka, G. (2013). Distance-based image classification: Generalizing to new classes at near-zero cost. IEEE transactions on pattern analysis and machine intelligence, 35(11), 2624-2637.

        2. hard version: L^q weighting.
          Refs:
            (q=1) Dudani, S. A. (1976). The distance-weighted k-nearest-neighbor rule. IEEE Transactions on Systems, Man, and Cybernetics, (4), 325-327.
        """
        check_is_fitted(self)
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors

        if self.weights == "uniform":
            index_neighbors = self.kneighbors(X, n_neighbors, return_distance=False)
            dist = None
            distance_weights = 1
        elif self.weights == "distance":
            dist, index_neighbors = self.kneighbors(X, n_neighbors)
            distance_weights = np.ones_like(dist)
            distance_diff = dist[:, -1] - dist[:, 0] + self.min_dist
            distance_weights[:, 1:] = (dist[:, -1, None] - dist[:, 1:]) / distance_diff[
                :, None
            ]

        label_knn = self._fit_y[index_neighbors]

        vote = np.zeros((X.shape[0], self.n_labels))
        for j in range(self.n_labels):
            if self.proba_method == "softmax":
                vote[:, j] = self.class_weights[j] * np.sum(
                    distance_weights * (label_knn == j), axis=1
                )
            elif self.proba_method == "softmin":
                vote[:, j] = (
                    1
                    / self.class_weights[j]
                    * np.mean(1 / distance_weights * (label_knn == j), axis=1)
                )

        proba = self.compute_proba(vote)
        self._query_dist = (dist, index_neighbors)
        
        return proba
    
    @property
    def dist(self, X=None):
        """
        return the distance of the query
        """
        check_is_fitted(self)
        if X is None:
            dist, index_neighbors = self._query_dist
        else:
            dist, index_neighbors = self.kneighbors(X, self.n_neighbors)

        return dist, index_neighbors

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)  # Input validation
        proba = self.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        return preds

    def filter_by_density(self, X):
        """
        https://papers.nips.cc/paper/2018/file/7180cffd6a8e829dacfc2a31b3f72ece-Paper.pdf
        Filter out points with low kNN density.
        Args:
        X: an array of sample points.
        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        original points of kNN density.
        """
        if self.alpha > 0.0:
            if self.tree_type == "kd_tree":
                tree = KDTree(X, metric=self.metric)
            elif self.tree_type == "ball_tree":
                tree = BallTree(X, metric=self.metric)
            else:
                raise NotImplementedError
            k = min(self.n_neighbors, X.shape[0] // 2)
            knn_radii = tree.query(X, k=k)[0][:, -1]
            eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
            X_filtered = X[np.where(knn_radii <= eps)[0], :]
            return X_filtered
        else:
            return X

    def filter_by_uncertainty(self, X, y):
        """
        https://papers.nips.cc/paper/2018/file/7180cffd6a8e829dacfc2a31b3f72ece-Paper.pdf
        Filter out points with high label disagreement amongst its kNN neighbors.
        Args:
        X: an array of sample points.
        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        samples with highest disagreement amongst its k nearest neighbors.
        """
        proba, _ = self.predict_proba(X)

        cutoff = np.percentile(proba, self.alpha * 100)
        unfiltered_idxs = np.where(proba >= cutoff)[0]
        # print(unfiltered_idxs.shape)
        return X[unfiltered_idxs, :], y[unfiltered_idxs]

    def _ensure_cls_cache(self):
        """
        ensure per-class cached arrays exist after loading trees from disk.
        query_sample_results() expects:

        - self.additional_labels_cls: list of per-class additional labels arrays
        - self.X_by_cls_used: list of per-class feature matrices (needed for cgMLST distance recompute)
        """

        if not hasattr(self, 'additional_labels_cls') or self.additional_labels_cls is None:
            index_by_label = [np.where(self._fit_y == label)[0] for label in self.train_labels]
            self.additional_labels_cls = [self._fit_additional_labels[idx] for idx in index_by_label]


        if not hasattr(self, 'X_by_cls_used') or self.X_by_cls_used is None:
            index_by_label = [np.where(self._fit_y == label)[0] for label in self.train_labels]
            self.X_by_cls_used = [self._fit_X[idx] for idx in index_by_label]


    def build_trees_by_cls(self, X, y, path=None):
        if self.tree_type == "kd_tree":
            Tree = KDTree
        elif self.tree_type == "ball_tree":
            Tree = BallTree
        else:
            raise NotImplementedError
        
        self.additional_labels_cls = [None] * self.n_labels
        index_by_label = [np.where(y == label)[0] for label in self.train_labels]
        for i, label in enumerate(self.train_labels):
            self.additional_labels_cls[i] = self._fit_additional_labels[
                index_by_label[i]
            ]
        self.trees = [None] * self.n_labels
        self.X_by_cls_used = [None] * self.n_labels  # keep per-class training matrix for penalty distance recompute
        with timer(f"Building tree per class"):
            if self.filtering_method == "uncertainty":
                X_filtered, _ = self.filter_by_uncertainty(X, y)
            for i, label in enumerate(self.train_labels):
                if self.filtering_method == "density":
                    X_to_use = self.filter_by_density(X[index_by_label[i]])
                    # print(f"X filtered: {X_to_use.shape}" )
                    self.trees[i] = Tree(X_to_use, metric=self.metric)
                elif self.filtering_method == "uncertainty":
                    X_to_use = X_filtered[index_by_label[i]]
                    self.trees[i] = Tree(X_to_use, metric=self.metric)
                else:
                    X_to_use = X[index_by_label[i]]
                    self.trees[i] = Tree(X_to_use, metric=self.metric)
                self.X_by_cls_used[i] = X_to_use  # keep per-class training matrix for cgMLST distance recompute

                if len(X_to_use) == 0:
                    raise ValueError(
                        "Filtered too much or missing examples from a label!\n Please lower alpha or check data."
                    )
        if self.verbose > 0:
            for i in range(self.n_labels):
                print(f"Tree size for class {self.train_labels[i]} size: {self.trees[i].__sizeof__()}")
            print(f"Saving tree per class to {path}")
        self.save_trees(self.trees, path)

    def filter(self, X, y, fold_id=None):
        # fold_id used to build/load fold-specific trees to avoid cross-fold leakage
        """
        Initialize trust score precomputations with training data
        X: an array of sample points.
        y: corresponding labels.
        """

        tree_by_cls_name = f"./cgMLST_tree_by_cls_Nsample_{len(X)}_Nfeat_{X.shape[1]}_Nclass_{len(self.train_labels)}" + (f"_fold{fold_id}" if fold_id is not None else "") + ".pkl"  # MOD(CV-fold)
        if self.tree_by_cls_path is not None:
            _tree_path_by_cls = os.path.join(self.tree_by_cls_path, tree_by_cls_name)
            try:
                with timer(f"Loading tree per class from {_tree_path_by_cls}"):
                    self.trees = self.load_trees(_tree_path_by_cls)
                    self._ensure_cls_cache()
            except:
                self.build_trees_by_cls(X, y, 
                                        path=_tree_path_by_cls)    
        else:
            self.build_trees_by_cls(X, y, path=tree_by_cls_name)
            

    def _filter(self):
        _fit_X = self._fit_X
        _fit_y = self._fit_y
        if self.n_filtered_samples is not None and int(self.n_filtered_samples) < len(
            _fit_X
        ):

            _fit_X, _, _fit_y, _ = train_test_split(
                _fit_X,
                _fit_y,
                train_size=self.n_filtered_samples/len(_fit_X),
                random_state=self.random_state,
                stratify=_fit_y,
            )
        self.filter(_fit_X, _fit_y, fold_id=getattr(self, 'fold_id', None))
        self.is_filtered = True

    def _get_filter_stats(self, max_query_neighbors=100, quantile=0.5):
        """
        get the shift for the distance ratio used in the trust score
        based on the training data
        """
        if not self.is_filtered:
            self._filter()

        X = self._fit_X
        y = self._fit_y
        max_query_neighbors = default(
            max_query_neighbors, self.score_max_query_neighbors
        )

        score_train_all = []
        eta = np.zeros(self.n_labels)
        sigma = np.zeros(self.n_labels)
        skewness = np.zeros(self.n_labels)

        for i, label in enumerate(self.train_labels):
            print(f"Get filter stats for label {label}")
            Xi = X[np.where(y == label)[0]]
            yi = y[np.where(y == label)[0]]

            d_to_label, d_to_closest_not_label = self.query_distance_by_label(
                Xi, yi, n_neighbors=max_query_neighbors
            )
            score = d_to_closest_not_label / (d_to_label + self.eps)
            score_train_all.append(score)
            eta[i] = np.quantile(
                d_to_closest_not_label / (d_to_label + self.eps), quantile
            )
            sigma[i] = np.nanstd(score)
            skewness[i] = skew(score)

        self.score_train_all = score_train_all
        self.score_mean_train = eta
        self.score_std_train = sigma
        self.score_skew_train = skewness

    def query_distance_by_label(
        self, X, y, n_neighbors=None, query_agg_func=np.nanmean
    ):
        d = np.tile(None, (X.shape[0], self.n_labels))
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        query_agg_func = (
            self.query_agg_func if query_agg_func is None else query_agg_func
        )

        for i, label in enumerate(self.train_labels):
            # candidate pool per class then recompute cgMLST distance and aggregate
            kcand_score = int(getattr(self, "kcand", 100))
            kcand_score = max(kcand_score, int(n_neighbors))
            dist0, ind = self.trees[i].query(X, k=kcand_score, return_distance=True)

            if getattr(self, "metric_mode", "hamming") == "hamming_cgmlst":
                Y_nb = self.X_by_cls_used[i][ind]
                dist_p = self.hamming_cgmlst_batch(X, Y_nb)
                order = np.argsort(dist_p, axis=1)[:, :n_neighbors]
                dist_top = np.take_along_axis(dist_p, order, axis=1)
                d[:, i] = query_agg_func(dist_top, axis=1)
            else:
                order0 = np.argsort(dist0, axis=1)[:, :n_neighbors]
                d[:, i] = query_agg_func(np.take_along_axis(dist0, order0, axis=1), axis=1)

        sorted_d = np.sort(d, axis=1)
        d_to_y = d[range(d.shape[0]), y].astype(float)
        d_to_closest_not_y = np.where(
            np.abs(sorted_d[:, 0] - d_to_y) > self.eps,
            sorted_d[:, 0],
            sorted_d[:, 1],
        ).astype(float)

        # compact trust-distance summary
        self._sanity_print(f'trust distances: d_to_y[min,max]=({float(d_to_y.min()):.6f},{float(d_to_y.max()):.6f}) d_to_not_y[min,max]=({float(d_to_closest_not_y.min()):.6f},{float(d_to_closest_not_y.max()):.6f})')
        # trust-score distances should be finite and within [0, 1] for normalized distance
        self._assert_(np.all(np.isfinite(d_to_y)), 'd_to_y has non-finite values')
        self._assert_(np.all(np.isfinite(d_to_closest_not_y)), 'd_to_closest_not_y has non-finite values')
        self._assert_(np.all(d_to_y >= 0) and np.all(d_to_y <= 1.0 + 1e-9), 'd_to_y out of range')
        self._assert_(np.all(d_to_closest_not_y >= 0) and np.all(d_to_closest_not_y <= 1.0 + 1e-9), 'd_to_closest_not_y out of range')
        return d_to_y, d_to_closest_not_y

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_score(
        self,
        X,
        y_preds,
        y_proba=None,
        n_neighbors_score=None,
        beta=None,
        score_mean=None,
        score_std=None,
        score_skew=None,
        query_agg_func=np.median,  # np.median, np.mean
    ):
        """
        Compute the trust scores.
        Given a set of points, determines the distance to each class.
        Args:
        - X: an array of sample points.
        - y_pred: The predicted labels for these points.
        - beta: exponential weighting factor for sigmoid or erf

        Optional if the score_func is 'normal':
        - score_mean: mean of the trust score for each class in train data
        - score_std: std of the trust score for each class in train data

        Returns:
            The trust score, which is ratio of distance to closest class that was not
            the predicted class to the distance to the predicted class.

        Notes:
        - If eta and sigma are not provided, they are computed from the training data which has a significant computational overhead.
        """

        check_is_fitted(self)
        if not self.is_filtered:
            self._filter()

        k = self.n_neighbors if n_neighbors_score is None else n_neighbors_score
        query_agg_func = (
            self.query_agg_func if query_agg_func is None else query_agg_func
        )
        d = np.tile(None, (X.shape[0], self.n_labels))

        d_to_pred, d_to_closest_not_pred = self.query_distance_by_label(
            X, y_preds, n_neighbors=k, query_agg_func=query_agg_func
        )

        score = d_to_closest_not_pred / (d_to_pred + self.eps)
        beta = default(beta, self.beta)
        if self.score_normalization == "normal":
            if score_mean is None:
                assert hasattr(self, "score_mean_train")
                score_mean = self.score_mean_train
            if score_std is None:
                assert hasattr(self, "score_std_train")
                score_std = self.score_std_train
            if score_skew is None:
                assert hasattr(self, "score_skew_train")
                score_skew = self.score_skew_train
            score_mean = np.asarray([score_mean[y_] for y_ in y_preds])
            score_std = np.asarray([score_std[y_] for y_ in y_preds])
            score_skew = np.asarray([score_skew[y_] for y_ in y_preds])

            score = (score - score_mean) / score_std + score_skew
            score = 0.5 * (1 + erf((beta * score) / np.sqrt(2)))

        elif self.score_normalization == "sigmoid":
            score = score - 1

            score = 1 / (1 + np.exp(-beta * score))
        return score

    def query_sample_results(
        self,
        sample={
            "features": None,
            "labels": None,
            "barcodes": None,
        },
        n_neighbors: int=None,  # how many neighbors from all classes to compute probability
        n_query_neighbors: int=None,  # how many neighbors to query in knn
        n_query_neighbors_score: int=None,  # how many neighbors to query each cluster in self.trees[i] for the scores
        sort_by=["Trust"],
        **score_kws,
    ):
        """
        compute the probability trust score and print the results
        """
        # ensure per-class caches exist (esp. when trees are loaded from disk)
        self._ensure_cls_cache()
        y_train = self._fit_y
        barcode_train = self._fit_additional_labels
        barcode_train_by_cls = self.additional_labels_cls
        X_sample = sample["features"]
        y_sample = sample["labels"]
        barcode = sample["barcodes"]
        n_query_samples = X_sample.shape[0]
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        if n_query_neighbors is None: n_query_neighbors = n_neighbors
        if n_query_neighbors_score is None: n_query_neighbors_score = n_neighbors

        proba = self.predict_proba(X_sample, n_neighbors)

        # kneighbors() internally uses candidate pool + re-ranking.
        if n_query_neighbors == n_neighbors:
            dist_norm, index_neighbors = self.dist
        else:
            dist_norm, index_neighbors = self.kneighbors(X_sample, n_query_neighbors)

        # distance is normalized by pairwise valid loci
        diff_cnt = getattr(self, "_last_diff_count", None)
        valid_cnt = getattr(self, "_last_valid_loci", None)

        if diff_cnt is not None and diff_cnt.shape == dist_norm.shape:
            dist_diff = diff_cnt.astype(int)
        else:
            # approximate diff from normalized dist using total loci (may be inaccurate for cgMLST)
            dist_diff = np.rint(dist_norm * self.n_features).astype(int)

        if valid_cnt is not None and valid_cnt.shape == dist_norm.shape:
            dist_valid = valid_cnt.astype(int)
        else:
            dist_valid = np.full_like(dist_diff, fill_value=self.n_features, dtype=int)

        dist_by_label_norm = np.tile(
            None, (self.n_labels, n_query_samples, n_query_neighbors_score)
        )
        dist_by_label_diff = np.tile(
            None, (self.n_labels, n_query_samples, n_query_neighbors_score)
        )
        dist_by_label_valid = np.tile(
            None, (self.n_labels, n_query_samples, n_query_neighbors_score)
        )
        index_nb_by_class = np.tile(
            None, (self.n_labels, n_query_samples, n_query_neighbors_score)
        )
        barcode_nb_by_class = np.tile(
            None, (self.n_labels, n_query_samples, n_query_neighbors_score)
        )
        for i, label in enumerate(self.train_labels):
            # use candidate pool + cgMLST distance for per-class neighbors
            kcand_score = int(getattr(self, 'kcand', 100))
            kcand_score = max(kcand_score, int(n_query_neighbors_score))
            _, cand_idx = self.trees[i].query(X_sample, k=kcand_score, return_distance=True)

            Y_cand = self.X_by_cls_used[i][cand_idx]
            dist_cand, diff_cand, valid_cand = self.hamming_cgmlst_batch(X_sample, Y_cand, return_counts=True)

            order = np.argsort(dist_cand, axis=1)[:, :n_query_neighbors_score]
            cand_idx = np.take_along_axis(cand_idx, order, axis=1)
            dist_cand = np.take_along_axis(dist_cand, order, axis=1)
            diff_cand = np.take_along_axis(diff_cand, order, axis=1)
            valid_cand = np.take_along_axis(valid_cand, order, axis=1)

            dist_by_label_norm[i] = dist_cand
            dist_by_label_diff[i] = diff_cand.astype(int)
            dist_by_label_valid[i] = valid_cand.astype(int)
            index_nb_by_class[i] = cand_idx
            barcode_nb_by_class[i] = barcode_train_by_cls[i][cand_idx]


        y_train_neighbors = y_train[index_neighbors]
        barcode_neighbors = barcode_train[index_neighbors]

        y_preds = np.argmax(proba, axis=1)
        score = self.get_score(X_sample, y_preds, **score_kws)

        result_df = pd.DataFrame(
            {
                "Barcode": barcode.squeeze(),
                "Label": y_sample,
                "Prediction": y_preds.squeeze(),
                "Trust": score.squeeze(),
            }
        )

        results = {
            "Prob (rounded)": proba.round(3),
            "Neighbor Index": index_neighbors,
            "Neighbor Barcodes": barcode_neighbors,
            "Neighbor Labels": y_train_neighbors,
            "Dist (norm)": dist_norm,
            "Diff count": dist_diff,
            "Valid loci": dist_valid,
        }

        results.update(
            {
                f"Dist(norm) to {label}": dist_by_label_norm[i]
                for i, label in enumerate(self.train_labels)
            }
        )
        results.update(
            {
                f"Diff count to {label}": dist_by_label_diff[i]
                for i, label in enumerate(self.train_labels)
            }
        )
        results.update(
            {
                f"Valid loci to {label}": dist_by_label_valid[i]
                for i, label in enumerate(self.train_labels)
            }
        )
        results.update(
            {
                f"Barcode {label}": barcode_nb_by_class[i]
                for i, label in enumerate(self.train_labels)
            }
        )

        for key, value in results.items():
            df_ = pd.DataFrame(columns=[key], index=range(n_query_samples))
            for i in range(n_query_samples):
                df_.loc[i, key] = value[i].tolist()
            result_df = pd.concat([result_df, df_], axis=1)

        if sort_by is not None:
            result_df = result_df.sort_values(by=sort_by)

        return result_df


if __name__ == "__main__":
    df = pd.read_parquet("Typhimurium_NH_USA_cgMLST_label.parquet")
    types = df["Sources"].unique()
    df = df[df["Sources"].isin(["Swine", "Poultry", "Bovine", "Avian"])]
    df.loc[:, "Sources"] = pd.Categorical(df.Sources.cat.remove_unused_categories())
    df.loc[:, "source_emb"] = df.Sources.cat.codes
    cat_codes = dict(zip(df["source_emb"], df["Sources"]))
    header = [cat_codes[x] for x in sorted(cat_codes.keys())]

    stmmw_cols = [col for col in df.columns if "STMMW" in col]
    feature_cols = [
        col for col in df.columns if col not in ["Sources", "source_emb", "ST"]
    ]
    df[feature_cols] = df[feature_cols].astype("int32")
    feat_df = df[feature_cols]
    feat_df = feat_df.fillna(0)

    y = df[["source_emb"]]
    y_label = df[["Sources"]]
    X = df.drop(["ST", "Sources", "source_emb"], axis=1)
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    k = 9
    result = []

    for fold, (train_id, test_id) in enumerate(folds.split(X, y)):
        X_train = X.iloc[train_id]
        y_train = y.iloc[train_id].values.ravel()
        X_valid = X.iloc[test_id]
        y_valid = y.iloc[test_id].values.ravel()
        if fold == 1:
            break

    result_fold = pd.DataFrame(
        {
            "fold": [fold, fold, fold, fold],
            "Sources": ["Avian", "Bovine", "Poultry", "Swine"],
            "Accuracy": [0, 0, 0, 0],
        }
    )

    weights = np.zeros(4)
    n_samples_class = np.asarray([np.sum(y_train == i) for i in range(4)])
    weights = (n_samples_class / n_samples_class.min()) ** (-1 / 2.5)
    knn_setting = dict(
        k=k,
        tol=1,
        weights="distance",
        class_weights=weights,
    )
    print(f"\nFold {fold}")
    print(knn_setting)
    for i in range(4):
        idx = y_valid == i
        proba = get_knn_probability(X_train, y_train, X_valid.loc[idx], **knn_setting)
        y_preds = np.argmax(proba, axis=1)
        accuracy = (y_preds == y_valid[idx]).mean()
        print(f"Accuracy for {cat_codes[i]} is {accuracy:.4f}")
        result_fold.loc[result_fold["Sources"] == cat_codes[i], "Accuracy"] = accuracy

    result.append(result_fold)
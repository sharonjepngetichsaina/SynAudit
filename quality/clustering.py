"""Stream clustering: CluStream and DenStream, implemented from their papers.

  CluStream  Aggarwal, Han, Wang, Yu. "A Framework for Clustering Evolving Data
             Streams", VLDB 2003.
  DenStream  Cao, Ester, Qian, Zhou. "Density-Based Clustering over an Evolving
             Data Stream with Noise", SIAM SDM 2006.

Both are fed one record at a time (online phase). Their micro-clusters are then
condensed into macro-clusters on demand (offline phase) and the cluster structure
learned from the real stream is compared with the one from the synthetic stream.

Simplifications, on purpose:
  * CluStream: no pyramidal time frame / snapshot differences (we only need the
    current clustering), and the paper's merge bookkeeping of cluster id lists is dropped.
  * Time is logical: one unit per record arriving in that stream.
  * Both algorithms get numeric features scaled to [0, 1] using the ranges in the
    SDV metadata, so one epsilon / one distance means the same for all columns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm
from sklearn.cluster import KMeans


# =============================================================================
# CluStream
# =============================================================================
class CluStream:
    """Online micro-clustering with cluster feature vectors (n, CF1x, CF2x, CF1t, CF2t)."""

    def __init__(self, dim, max_micro=50, boundary_factor=2.0, m=20, delta=1000):
        self.dim = dim
        self.q = max_micro          # maximum number of micro-clusters
        self.t = boundary_factor    # maximal boundary = t * RMS deviation
        self.m = m                  # "relevance stamp" looks at the m most recent arrivals
        self.delta = delta          # clusters whose relevance stamp is older than now-delta may be deleted
        self.n = np.zeros(max_micro)
        self.cf1 = np.zeros((max_micro, dim))
        self.cf2 = np.zeros((max_micro, dim))
        self.t1 = np.zeros(max_micro)   # sum of timestamps
        self.t2 = np.zeros(max_micro)   # sum of squared timestamps
        self.k = 0                      # number of used slots (slots 0..k-1)
        self.time = 0

    # ---- helpers on cluster feature vectors
    def _centroids(self):
        return self.cf1[:self.k] / self.n[:self.k, None]

    def _rms_radius(self, i):
        var = self.cf2[i] / self.n[i] - (self.cf1[i] / self.n[i]) ** 2
        return math.sqrt(max(float(var.sum()), 0.0))

    def _relevance_stamp(self, i):
        """Approximate time of the m-th most recent arrival, from CF1t/CF2t (normality assumption)."""
        n = self.n[i]
        mu = self.t1[i] / n
        if n < 2 * self.m:
            return mu
        sigma = math.sqrt(max(self.t2[i] / n - mu * mu, 0.0))
        return mu + sigma * norm.ppf(1.0 - self.m / (2.0 * n))

    def _put(self, slot, x):
        self.n[slot] = 1.0
        self.cf1[slot] = x
        self.cf2[slot] = x * x
        self.t1[slot] = self.time
        self.t2[slot] = float(self.time) ** 2

    def _absorb(self, slot, x):
        self.n[slot] += 1.0
        self.cf1[slot] += x
        self.cf2[slot] += x * x
        self.t1[slot] += self.time
        self.t2[slot] += float(self.time) ** 2

    def _merge(self, i, j):
        """Merge cluster j into i (CF vectors are additive) and free slot j by moving the last slot."""
        for arr in (self.n, self.cf1, self.cf2, self.t1, self.t2):
            arr[i] = arr[i] + arr[j]
        self._free(j)

    def _free(self, j):
        last = self.k - 1
        if j != last:
            for arr in (self.n, self.cf1, self.cf2, self.t1, self.t2):
                arr[j] = arr[last]
        self.n[last] = 0.0
        self.cf1[last] = 0.0
        self.cf2[last] = 0.0
        self.t1[last] = 0.0
        self.t2[last] = 0.0
        self.k -= 1

    # ---- online phase
    def insert(self, x):
        x = np.asarray(x, dtype=float)
        self.time += 1
        if self.k == 0:
            self._put(0, x)
            self.k = 1
            return

        cents = self._centroids()
        d = np.linalg.norm(cents - x, axis=1)
        j = int(np.argmin(d))

        # maximal boundary of the closest cluster
        if self.n[j] > 1:
            boundary = self.t * self._rms_radius(j)
        elif self.k > 1:
            dc = np.linalg.norm(cents - cents[j], axis=1)
            dc[j] = np.inf
            boundary = float(dc.min())   # singleton: distance to the closest other cluster
        else:
            boundary = 0.0

        if d[j] <= boundary:
            self._absorb(j, x)
            return

        # x starts a new micro-cluster; make room if we are at capacity
        if self.k >= self.q:
            stamps = np.array([self._relevance_stamp(i) for i in range(self.k)])
            oldest = int(np.argmin(stamps))
            if stamps[oldest] < self.time - self.delta:
                self._free(oldest)                       # delete an outdated cluster
            else:
                cents = self._centroids()                # otherwise merge the two closest clusters
                dist = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=2)
                np.fill_diagonal(dist, np.inf)
                a, b = np.unravel_index(int(np.argmin(dist)), dist.shape)
                a, b = int(min(a, b)), int(max(a, b))
                self._merge(a, b)
        self._put(self.k, x)
        self.k += 1

    # ---- offline phase
    def micro_clusters(self):
        """(centroids, weights) of the current micro-clusters."""
        if self.k == 0:
            return np.zeros((0, self.dim)), np.zeros(0)
        return self._centroids(), self.n[:self.k].copy()

    def macro_clusters(self, k=5, seed=0):
        """Weighted k-means over micro-cluster centroids -> (centers, weights)."""
        cents, w = self.micro_clusters()
        if len(cents) == 0:
            return np.zeros((0, self.dim)), np.zeros(0)
        k = min(k, len(cents))
        if k == 1:
            return np.average(cents, axis=0, weights=w)[None, :], np.array([w.sum()])
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(cents, sample_weight=w)
        weights = np.array([w[km.labels_ == c].sum() for c in range(k)])
        return km.cluster_centers_, weights


# =============================================================================
# DenStream
# =============================================================================
@dataclass
class _Micro:
    cf1: np.ndarray    # weighted linear sum
    cf2: np.ndarray    # weighted sum of squares
    w: float           # weight (decays with time)
    t0: int            # creation time
    t_last: int        # time of the last decay update

    def decay(self, now, lam):
        if now > self.t_last:
            f = 2.0 ** (-lam * (now - self.t_last))
            self.cf1 = self.cf1 * f
            self.cf2 = self.cf2 * f
            self.w *= f
            self.t_last = now

    def center(self):
        return self.cf1 / self.w

    def radius_with(self, x):
        cf1, cf2, w = self.cf1 + x, self.cf2 + x * x, self.w + 1.0
        var = cf2 / w - (cf1 / w) ** 2
        return math.sqrt(max(float(var.sum()), 0.0))


class DenStream:
    """Density-based stream clustering with potential and outlier micro-clusters.

    eps   maximum micro-cluster radius
    mu    weight threshold for a core micro-cluster (offline DBSCAN)
    beta  outlier threshold: a micro-cluster is "potential" when w >= beta*mu  (beta*mu > 1)
    lam   decay rate: a point's weight is 2^(-lam * age)
    """

    def __init__(self, eps=0.15, mu=10.0, beta=0.3, lam=0.002):
        if beta * mu <= 1:
            raise ValueError("DenStream needs beta * mu > 1")
        self.eps, self.mu, self.beta, self.lam = eps, mu, beta, lam
        self.p: list[_Micro] = []   # potential micro-clusters
        self.o: list[_Micro] = []   # outlier micro-clusters
        self.time = 0
        # pruning period Tp = ceil( (1/lam) * log2( beta*mu / (beta*mu - 1) ) )
        self.Tp = max(1, math.ceil((1.0 / lam) * math.log2(beta * mu / (beta * mu - 1.0))))

    @staticmethod
    def _nearest(clusters, x):
        cents = np.stack([c.center() for c in clusters])
        d = np.linalg.norm(cents - x, axis=1)
        return int(np.argmin(d))

    def insert(self, x):
        x = np.asarray(x, dtype=float)
        self.time += 1
        t = self.time

        merged = False
        if self.p:
            i = self._nearest(self.p, x)
            c = self.p[i]
            c.decay(t, self.lam)
            if c.radius_with(x) <= self.eps:
                c.cf1 = c.cf1 + x
                c.cf2 = c.cf2 + x * x
                c.w += 1.0
                merged = True
        if not merged and self.o:
            i = self._nearest(self.o, x)
            c = self.o[i]
            c.decay(t, self.lam)
            if c.radius_with(x) <= self.eps:
                c.cf1 = c.cf1 + x
                c.cf2 = c.cf2 + x * x
                c.w += 1.0
                merged = True
                if c.w >= self.beta * self.mu:      # outlier becomes a potential micro-cluster
                    self.o.pop(i)
                    self.p.append(c)
        if not merged:
            self.o.append(_Micro(x.copy(), x * x, 1.0, t, t))

        if t % self.Tp == 0:
            self._prune()

    def _prune(self):
        t = self.time
        for c in self.p + self.o:
            c.decay(t, self.lam)
        self.p = [c for c in self.p if c.w >= self.beta * self.mu]
        denom = 2.0 ** (-self.lam * self.Tp) - 1.0
        keep = []
        for c in self.o:
            xi = (2.0 ** (-self.lam * (t - c.t0 + self.Tp)) - 1.0) / denom
            if c.w >= xi:
                keep.append(c)
        self.o = keep

    # ---- offline phase
    def micro_clusters(self):
        """(centroids, weights) of the potential micro-clusters, decayed to now."""
        if not self.p:
            return np.zeros((0, 0)), np.zeros(0)
        for c in self.p:
            c.decay(self.time, self.lam)
        return np.stack([c.center() for c in self.p]), np.array([c.w for c in self.p])

    def macro_clusters(self):
        """DBSCAN over potential micro-clusters -> (centers, weights, labels).

        Core micro-clusters have w >= mu. Two cores are connected when their centers are
        within 2*eps. Non-core potential micro-clusters join the nearest core within
        2*eps, otherwise they are noise (label -1) and ignored.
        """
        cents, w = self.micro_clusters()
        n = len(cents)
        if n == 0:
            return np.zeros((0, 0)), np.zeros(0), np.zeros(0, dtype=int)
        core = w >= self.mu
        labels = -np.ones(n, dtype=int)
        if core.any():
            idx = np.flatnonzero(core)
            dist = np.linalg.norm(cents[idx][:, None, :] - cents[idx][None, :, :], axis=2)
            _, comp = connected_components(csr_matrix(dist <= 2 * self.eps), directed=False)
            labels[idx] = comp
            for i in np.flatnonzero(~core):
                dd = np.linalg.norm(cents[idx] - cents[i], axis=1)
                if dd.min() <= 2 * self.eps:
                    labels[i] = labels[idx[int(np.argmin(dd))]]
        ids = sorted(set(labels[labels >= 0]))
        if not ids:
            return np.zeros((0, cents.shape[1])), np.zeros(0), labels
        centers = np.stack([np.average(cents[labels == c], axis=0, weights=w[labels == c]) for c in ids])
        weights = np.array([w[labels == c].sum() for c in ids])
        return centers, weights, labels


# =============================================================================
# Real-vs-synthetic comparison
# =============================================================================
class RangeScaler:
    """Min-max scaling to [0, 1] with fixed ranges (from the SDV metadata)."""

    def __init__(self, columns, ranges):
        self.columns = [c for c in columns if c in ranges]
        self.lo = np.array([ranges[c][0] for c in self.columns], dtype=float)
        self.span = np.array([ranges[c][1] - ranges[c][0] for c in self.columns], dtype=float)

    @property
    def dim(self):
        return len(self.columns)

    def transform_record(self, record):
        v = np.array([_to_float(record.get(c)) for c in self.columns], dtype=float)
        return np.clip((v - self.lo) / self.span, 0.0, 1.0)

    def transform_frame(self, df):
        v = df[self.columns].apply(lambda s: s.map(_to_float)).to_numpy(dtype=float)
        return np.clip((v - self.lo) / self.span, 0.0, 1.0)


def _to_float(v):
    try:
        f = float(v)
        return f if not math.isnan(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def centroid_distance(a, b):
    """Mean distance between optimally matched centers (Hungarian matching). NaN if either side is empty."""
    if len(a) == 0 or len(b) == 0:
        return math.nan
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def membership_tvd(reference_centers, X_a, X_b):
    """Assign two samples to the nearest reference center; TVD between the cluster histograms."""
    if len(reference_centers) == 0 or len(X_a) == 0 or len(X_b) == 0:
        return math.nan
    k = len(reference_centers)

    def hist(X):
        d = np.linalg.norm(X[:, None, :] - reference_centers[None, :, :], axis=2)
        return np.bincount(np.argmin(d, axis=1), minlength=k) / len(X)

    return float(0.5 * np.abs(hist(X_a) - hist(X_b)).sum())


def coverage_ratio(reference_centers, X_real, X_synth):
    """Mean distance of synthetic points to the nearest real micro-cluster center, divided by the
    same quantity for real points. ~1 means synthetic points sit where real clusters are; large
    values mean synthetic data falls outside the regions the real stream occupies."""
    if len(reference_centers) == 0 or len(X_real) == 0 or len(X_synth) == 0:
        return math.nan

    def mean_nn(X):
        d = np.linalg.norm(X[:, None, :] - reference_centers[None, :, :], axis=2)
        return float(d.min(axis=1).mean())

    base = mean_nn(X_real)
    return mean_nn(X_synth) / base if base > 0 else math.nan


class StreamClusterer:
    """One CluStream and one DenStream per source (real / synthetic), same parameters."""

    SOURCES = ("real", "synthetic")

    def __init__(self, scaler: RangeScaler, k_macro=5, clustream=None, denstream=None):
        self.scaler = scaler
        self.k_macro = k_macro
        dim = scaler.dim
        self.clu = {s: CluStream(dim, **(clustream or {})) for s in self.SOURCES}
        self.den = {s: DenStream(**(denstream or {})) for s in self.SOURCES}

    def add(self, source, record):
        x = self.scaler.transform_record(record)
        self.clu[source].insert(x)
        self.den[source].insert(x)

    def compare(self, real_window, synth_window):
        """Scores for the current window (DataFrames of raw records)."""
        Xr = self.scaler.transform_frame(real_window)
        Xs = self.scaler.transform_frame(synth_window)

        cr, wr = self.clu["real"].macro_clusters(self.k_macro)
        cs, ws = self.clu["synthetic"].macro_clusters(self.k_macro)
        dr, dwr, _ = self.den["real"].macro_clusters()
        ds, dws, _ = self.den["synthetic"].macro_clusters()
        clu_micro_real, _ = self.clu["real"].micro_clusters()
        den_micro_real, _ = self.den["real"].micro_clusters()
        den_micro_synth, _ = self.den["synthetic"].micro_clusters()

        return {
            "clustream_micro_real": float(self.clu["real"].k),
            "clustream_micro_synth": float(self.clu["synthetic"].k),
            "clustream_centroid_dist": centroid_distance(cr, cs),
            "clustream_membership_tvd": membership_tvd(cr, Xr, Xs),
            "clustream_coverage_ratio": coverage_ratio(clu_micro_real, Xr, Xs),
            "denstream_micro_real": float(len(self.den["real"].p)),
            "denstream_micro_synth": float(len(self.den["synthetic"].p)),
            "denstream_outliers_real": float(len(self.den["real"].o)),
            "denstream_outliers_synth": float(len(self.den["synthetic"].o)),
            "denstream_macro_real": float(len(dr)),
            "denstream_macro_synth": float(len(ds)),
            "denstream_centroid_dist": centroid_distance(den_micro_real, den_micro_synth),
            "denstream_membership_tvd": membership_tvd(den_micro_real, Xr, Xs),
            "denstream_coverage_ratio": coverage_ratio(den_micro_real, Xr, Xs),
        }

    @staticmethod
    def empty_scores():
        keys = ["clustream_micro_real", "clustream_micro_synth", "clustream_centroid_dist",
                "clustream_membership_tvd", "clustream_coverage_ratio", "denstream_micro_real", "denstream_micro_synth",
                "denstream_outliers_real", "denstream_outliers_synth", "denstream_macro_real",
                "denstream_macro_synth", "denstream_centroid_dist", "denstream_membership_tvd",
                "denstream_coverage_ratio"]
        return {k: math.nan for k in keys}

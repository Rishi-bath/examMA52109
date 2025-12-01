###
## cluster_maker - algorithms.py 
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans


def initCentroids(
    X: np.ndarray,
    k: int,
    randomState: Optional[int] = None,
) -> np.ndarray:
    """
    Initialise centroids by randomly sampling points from X without replacement.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    nSamples = X.shape[0]
    if k > X.shape[0]:
        raise ValueError("k cannot be larger than the number of samples.")

    rng = np.random.RandomState(randomState)
    indices = rng.choice(nSamples, size=k, replace=False)
    return X[indices]


def assignClusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the nearest centroid (Euclidean distance).
    """
    # X: (n_samples, n_features)
    # centroids: (k, n_features)
    # Broadcast to compute distances
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # (n_samples, k)
    labels = np.argmin(distances, axis=1)
    return labels


def updateCentroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    randomState: Optional[int] = None,
) -> np.ndarray:
    """
    Update centroids by taking the mean of points in each cluster.
    If a cluster becomes empty, re-initialise its centroid randomly from X.
    """
    nFeatures = X.shape[1]
    newCentroids = np.zeros((k, nFeatures), dtype=float)
    rng = np.random.RandomState(randomState)

    for clusterId in range(k):
        mask = labels == clusterId
        if not np.any(mask):
            # Empty cluster: re-initialise randomly
            idx = rng.randint(0, X.shape[0])
            newCentroids[clusterId] = X[idx]
        else:
            newCentroids[clusterId] = X[mask].mean(axis=0)

    return newCentroids


def kmeans(
    X: np.ndarray,
    k: int,
    maxIter: int = 300,
    tol: float = 1e-4,
    randomState: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple manual K-means implementation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    maxIter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance on centroid movement.
    randomState : int or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    centroids = initCentroids(X, k, randomState=randomState)
    for _ in range(maxIter):
        labels = assignClusters(X, centroids)
        newCentroids = updateCentroids(X, labels, k, randomState=randomState)
        shift = np.linalg.norm(newCentroids - centroids)
        centroids = newCentroids
        if shift < tol:
            break

    labels = assignClusters(X, centroids)
    return labels, centroids


def sklearnKmeans(
    X: np.ndarray,
    k: int,
    randomState: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around scikit-learn's KMeans.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    model = KMeans(
        n_clusters=k,
        random_state=randomState,
        n_init=10,
    )
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return labels, centroids

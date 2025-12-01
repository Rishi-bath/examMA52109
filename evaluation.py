###
## cluster_maker - evaluation.py 
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score


def computeInertia(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """
    Compute inertia = Within-Cluster Sum of Squares (WCSS).
    
    Formula: Inertia = Σ ||x_i - c_{label_i}||²
    
    Inertia measures how tightly clustered the data is.
    Lower values = better clustering (tighter clusters).
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each point
    centroids : ndarray of shape (k, n_features)
        Cluster centers
    
    Returns
    -------
    inertia : float
        Sum of squared distances from points to their assigned centroids
    """
    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        centroid = centroids[labels[i]]
        distances[i] = np.linalg.norm(X[i] - centroid) ** 2
    return np.sum(distances)


def silhouetteScoreScikit(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score (cluster quality metric).
    
    Range: [-1, +1]
    - +1: Perfect clustering (tight, well-separated clusters)
    - 0: Overlapping clusters
    - -1: Points likely in wrong cluster
    
    For each point:
    - a = mean distance to other points in same cluster
    - b = mean distance to points in nearest other cluster
    - silhouette(i) = (b - a) / max(a, b)
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each point
    
    Returns
    -------
    score : float
        Average silhouette score (higher is better, range -1 to +1)
    
    Raises
    ------
    ValueError
        If less than 2 clusters
    """
    if len(np.unique(labels)) < 2:
        raise ValueError("Need at least 2 clusters for silhouette score.")
    return silhouette_score(X, labels)


def elbowCurve(
    X: np.ndarray,
    kValues: List[int],
    randomState: Optional[int] = None,
    useSklearn: bool = False,
) -> Dict[int, float]:
    """
    Compute inertia for multiple k values (used for elbow method).
    
    ELBOW METHOD:
    1. Compute clustering for k = 1, 2, 3, ..., max_k
    2. Plot inertia vs k
    3. Look for "elbow" (knee point)
    4. Choose k at elbow point
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points
    kValues : list of int
        Values of k to test (e.g., [1, 2, 3, 4, 5])
    randomState : int or None
        Random seed for reproducibility
    useSklearn : bool
        If True, use sklearn's KMeans
        If False, use manual kmeans implementation
    
    Returns
    -------
    inertias : dict
        Mapping of k -> inertia value
        Example: {1: 100.5, 2: 50.2, 3: 35.1, 4: 30.0, 5: 28.5}
    """
    from .algorithms import kmeans, sklearnKmeans
    
    inertias = {}
    for k in kValues:
        if useSklearn:
            labels, centroids = sklearnKmeans(X, k, randomState)
        else:
            labels, centroids = kmeans(X, k, randomState=randomState)
        inertias[k] = computeInertia(X, labels, centroids)
    return inertias

###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np

from cluster_maker.algorithms import (
    kmeans,
    sklearnKmeans,
    initCentroids,
    assignClusters,
)


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # Simple dataset with 3 obvious clusters
        self.X = np.vstack([
            np.random.RandomState(0).normal(loc=-5.0, scale=0.1, size=(10, 2)),
            np.random.RandomState(1).normal(loc=0.0, scale=0.1, size=(10, 2)),
            np.random.RandomState(2).normal(loc=5.0, scale=0.1, size=(10, 2)),
        ])

    # Test KMeans implementation.  
    def test_kmeans_manual(self):
        labels, centroids = kmeans(self.X, k=3, random_state=0)
        self.assertEqual(centroids.shape, (3, 2))
        self.assertEqual(labels.shape[0], self.X.shape[0])

    # Test sklearn KMeans wrapper.
    def test_kmeans_sklearn(self):
        labels, centroids = sklearn_kmeans(self.X, k=3, random_state=0)
        self.assertEqual(centroids.shape, (3, 2))
        self.assertEqual(labels.shape[0], self.X.shape[0])

    # Test init_centroids
    def test_init_centroids_samples_without_replacement(self):
        X_small = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        k = 3
        centroids = init_centroids(X_small, k=k, random_state=0)

        # Correct shape
        self.assertEqual(centroids.shape, (k, 2))

        # Each centroid must be exactly one of the rows in X_small
        # and rows must be unique (sampling without replacement)
        seen = []
        for c in centroids:
            # Check c is a row of X_small
            self.assertTrue(any(np.array_equal(c, row) for row in X_small))
            seen.append(tuple(c.tolist()))
        # Check we did not pick the same point twice
        self.assertEqual(len(seen), len(set(seen)))

        # Invalid k: k <= 0
        with self.assertRaises(ValueError):
            init_centroids(X_small, k=0, random_state=0)

        # Invalid k: k > n_samples
        with self.assertRaises(ValueError):
            init_centroids(X_small, k=5, random_state=0)

    # Test assign_clusters
    def test_assign_clusters_nearest_centroid(self):
        # Construct a tiny, easy-to-reason-about example in 2D
        X = np.array([
            [0.0, 0.0],   # closer to centroid A
            [0.1, -0.1],  # still closer to centroid A
            [10.0, 0.0],  # closer to centroid B
            [9.9, 0.2],   # still closer to centroid B
        ])

        centroids = np.array([
            [0.0, 0.0],   # centroid 0 (A)
            [10.0, 0.0],  # centroid 1 (B)
        ])

        labels = assign_clusters(X, centroids)

        # Expected assignments: first two to cluster 0, last two to cluster 1
        expected = np.array([0, 0, 1, 1])

        # Same length
        self.assertEqual(labels.shape, expected.shape)

        # Exact label match
        self.assertTrue(np.array_equal(labels, expected))


if __name__ == "__main__":
    unittest.main()
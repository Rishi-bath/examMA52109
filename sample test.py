###
## cluster_maker - tests/sample_test.py
## James Foadi - University of Bath
## November 2025
##
## COMPLETE TEST FILE - READY TO USE
###

import numpy as np
from cluster_maker.algorithms import initCentroids, assignClusters, kmeans

print("Testing cluster_maker module fixes...")
print("="*70)

# ============================================================================
# TEST 1: init_centroids() - Returns correct number of centroids
# ============================================================================
print("\n[TEST 1] init_centroids() - Shape validation")
print("-"*70)

X = np.random.randn(100, 2)
centroids = initCentroids(X, k=5)

try:
    assert centroids.shape[0] == 5, "Should return 5 centroids"
    assert centroids.shape[1] == 2, "Should return 2 features"
    print("PASS: init_centroids returns correct shape")
    print(f"      Centroids shape: {centroids.shape}")
except AssertionError as e:
    print(f"FAIL: {e}")


# ============================================================================
# TEST 2: init_centroids() - Validates k against n_samples
# ============================================================================
print("\n[TEST 2] init_centroids() - Validation (k > n_samples)")
print("-"*70)

X = np.random.randn(100, 2)

try:
    initCentroids(X, k=1000)
    print("FAIL: Should raise ValueError for k > n_samples")
except ValueError as e:
    print("PASS: Validation error raised correctly")
    print(f"      Error message: {str(e)}")


# ============================================================================
# TEST 3: assign_clusters() - Uses argmin (nearest centroid)
# ============================================================================
print("\n[TEST 3] assign_clusters() - Uses argmin for nearest centroid")
print("-"*70)

# Create simple test data
X_simple = np.array([[0, 0], [10, 10]])
centroids_simple = np.array([[0, 0], [10, 10]])

labels = assignClusters(X_simple, centroids_simple)

try:
    assert labels[0] == 0, "Point [0,0] should belong to cluster 0"
    assert labels[1] == 1, "Point [10,10] should belong to cluster 1"
    print("PASS: Points assigned to nearest centroids")
    print(f"      Labels: {labels}")
    print(f"      Point [0,0] -> Cluster {labels[0]}")
    print(f"      Point [10,10] -> Cluster {labels[1]}")
except AssertionError as e:
    print(f"FAIL: {e}")


# ============================================================================
# TEST 4: kmeans() - Complete algorithm works correctly
# ============================================================================
print("\n[TEST 4] kmeans() - Complete algorithm execution")
print("-"*70)

X_full = np.random.randn(100, 2)

try:
    labels, centroids = kmeans(X_full, k=3)
    assert labels.shape[0] == 100, "Should return 100 labels"
    assert centroids.shape == (3, 2), "Should return (3, 2) centroids"
    print("PASS: kmeans algorithm works correctly")
    print(f"      Input shape: {X_full.shape}")
    print(f"      Labels shape: {labels.shape}")
    print(f"      Centroids shape: {centroids.shape}")
    print(f"      Unique clusters: {np.unique(labels)}")
except AssertionError as e:
    print(f"FAIL: {e}")


# ============================================================================
# TEST 5: Evaluation functions exist and work
# ============================================================================
print("\n[TEST 5] Evaluation functions - Import and execution")
print("-"*70)

try:
    from cluster_maker.evaluation import (
        computeInertia,
        silhouetteScoreSklearn,
        elbowCurve
    )
    
    # Use results from previous test
    inertia = computeInertia(X_full, labels, centroids)
    silhouette = silhouetteScoreSklearn(X_full, labels)
    
    print("PASS: All evaluation functions imported successfully")
    print(f"      compute_inertia: OK")
    print(f"      silhouette_score_sklearn: OK")
    print(f"      elbow_curve: OK")
    print(f"\n      Evaluation results:")
    print(f"      - Inertia: {inertia:.2f}")
    print(f"      - Silhouette score: {silhouette:.3f}")
    
except ImportError as e:
    print(f"FAIL: Could not import evaluation functions")
    print(f"      Error: {str(e)}")
except Exception as e:
    print(f"FAIL: Evaluation functions failed")
    print(f"      Error: {str(e)}")


# ============================================================================
# BONUS TEST: Elbow curve
# ============================================================================
print("\n[BONUS] elbow_curve() - Test multiple k values")
print("-"*70)

try:
    inertias = elbowCurve(X_full, [1, 2, 3, 4, 5])
    print("PASS: elbow_curve computed successfully")
    print(f"      K values tested: {list(inertias.keys())}")
    print(f"      Inertia values:")
    for k, inertia_val in inertias.items():
        print(f"        k={k}: inertia={inertia_val:.2f}")
except Exception as e:
    print(f"FAIL: elbow_curve failed")
    print(f"      Error: {str(e)}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✓ All core functionality tests completed!")
print("✓ If all tests show PASS, your module is working correctly!")
print("\nNext steps:")
print("  1. Ensure all 5 fixes are applied:")
print("     - __init__.py: Remove stray comma")
print("     - tests/sample_test.py: Remove space before FAIL")
print("     - algorithms.py: Fix 4 lines (15, 18, 30, 31)")
print("     - evaluation.py: Replace with correct functions")
print("\n  2. Run: python tests/sample_test.py")
print("  3. All tests should show PASS!")
print("\n" + "="*70)

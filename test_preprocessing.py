###
## test_preprocessing.py
## Unit tests for preprocessing module
## MA52109 - Programming for Data Science
## 1 December 2025
###

import numpy as np
import pandas as pd
import pytest
from cluster_maker.preprocessing import selectFeatures, standardiseFeatures


# ============================================================================
# TEST 1: Missing column detection
# ============================================================================

def test_select_features_missing_column():
    """
    Test that select_features raises KeyError when requested column is missing.
    
    This test verifies that the function properly validates column names
    before attempting to select them, catching user errors early.
    """
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    })
    
    # Try to select a column that doesn't exist
    with pytest.raises(KeyError):
        select_features(df, ['feature1', 'nonexistent_column'])


# ============================================================================
# TEST 2: Non-numeric column detection
# ============================================================================


def test_select_features_non_numeric_column():
    """
    Test that select_features raises TypeError for non-numeric columns.
    
    This test ensures the function validates that selected columns are
    actually numeric before returning them, preventing type errors in
    clustering algorithms downstream.
    """
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'category': ['A', 'B', 'C']
    })
    
    # Try to select a non-numeric column
    with pytest.raises(TypeError):
        select_features(df, ['feature1', 'category'])


# ============================================================================
# TEST 3: Standardization with NaN values
# ============================================================================

def test_standardise_features_with_nan():
    """
    Test behavior of standardise_features when data contains NaN values.
    
    This test verifies that the function either properly handles NaN values
    or raises an informative error, preventing NaN from silently propagating
    through the clustering pipeline where it would cause incorrect results.
    """
    X = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [4.0, 5.0]
    ])
    
    # StandardScaler will propagate NaN, so output contains NaN
    result = standardise_features(X)
    
    # Verify that NaN is present in output (documenting current behavior)
    # In production, you might want to raise an error instead
    assert np.isnan(result).any(), "NaN values should be present in output"


# ============================================================================
# OPTIONAL: Additional edge case test
# ============================================================================


def test_standardise_features_single_sample():
    """
    Test standardization behavior with only one sample.
    
    This test verifies the function's behavior with edge case data (single
    sample), which can cause numerical issues like zero variance. Documents
    whether function handles this gracefully or raises an error.
    """
    X = np.array([[1.0, 2.0, 3.0]])
    
    # StandardScaler handles single sample but produces all zeros
    result = standardise_features(X)
    
    # Single sample gets standardized to zero
    assert result.shape == (1, 3), "Shape should be preserved"
    # After standardization, values should be zero (mean of single point is itself)
    np.testing.assert_array_almost_equal(result, np.zeros((1, 3)))

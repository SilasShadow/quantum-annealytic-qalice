"""Test numba optimizations."""

import numpy as np
import pytest

try:
    from _archive.numba_utils import (
        fast_decile_split,
        fast_silhouette_sample,
        fast_standard_scale,
    )

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not available")
def test_fast_standard_scale():
    """Test numba standard scaling."""
    data = np.random.randn(100, 5)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    result = fast_standard_scale(data, means, stds)

    # Check shape
    assert result.shape == data.shape

    # Check scaling (approximately zero mean, unit std)
    assert np.allclose(np.mean(result, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(result, axis=0), 1, atol=1e-10)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not available")
def test_fast_silhouette_sample():
    """Test numba silhouette calculation."""
    # Simple 2D data with clear clusters
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],  # Cluster 0
            [5, 5],
            [5, 6],
            [6, 5],
            [6, 6],  # Cluster 1
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    scores = fast_silhouette_sample(X, labels)

    # Check shape and range
    assert len(scores) == len(X)
    assert np.all(scores >= -1) and np.all(scores <= 1)

    # Should have positive silhouette (well-separated clusters)
    assert np.mean(scores) > 0


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not available")
def test_fast_decile_split():
    """Test numba decile splitting."""
    scores = np.random.rand(1000)
    deciles = fast_decile_split(scores)

    # Check shape and range
    assert len(deciles) == len(scores)
    assert np.all(deciles >= 1) and np.all(deciles <= 10)

    # Check roughly equal distribution
    unique, counts = np.unique(deciles, return_counts=True)
    assert len(unique) == 10
    assert np.all(counts >= 90) and np.all(counts <= 110)  # Allow some variance

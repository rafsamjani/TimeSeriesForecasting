"""
Test cases for preprocessing utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.transformers import Normalizer, Differencer, StationarityTester
from src.utils.helpers import generate_sample_data


class TestNormalizer:
    """Test cases for Normalizer."""
    
    def test_minmax_normalization(self):
        """Test MinMax normalization."""
        data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.fit_transform(data)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
    
    def test_standard_normalization(self):
        """Test Standard normalization."""
        data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        normalizer = Normalizer(method='standard')
        normalized = normalizer.fit_transform(data)
        
        # Check mean is approximately 0 and std is approximately 1
        assert abs(normalized.mean()) < 1e-10
        assert abs(normalized.std() - 1.0) < 0.1
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        normalizer = Normalizer(method='minmax')
        
        normalized = normalizer.fit_transform(data)
        original = normalizer.inverse_transform(normalized)
        
        np.testing.assert_array_almost_equal(data, original)


class TestDifferencer:
    """Test cases for Differencer."""
    
    def test_first_order_differencing(self):
        """Test first-order differencing."""
        data = pd.Series([1, 2, 4, 7, 11])
        differencer = Differencer(order=1)
        
        differenced = differencer.transform(data)
        
        expected = pd.Series([1, 2, 3, 4])
        pd.testing.assert_series_equal(differenced.reset_index(drop=True), 
                                      expected, check_dtype=False)


class TestStationarityTester:
    """Test cases for StationarityTester."""
    
    def test_adf_test(self):
        """Test ADF test."""
        data = generate_sample_data(n_samples=100, trend=False, seasonality=False)
        result = StationarityTester.adf_test(data)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result
        assert 'conclusion' in result
    
    def test_kpss_test(self):
        """Test KPSS test."""
        data = generate_sample_data(n_samples=100, trend=False, seasonality=False)
        result = StationarityTester.kpss_test(data)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result


if __name__ == "__main__":
    pytest.main([__file__])

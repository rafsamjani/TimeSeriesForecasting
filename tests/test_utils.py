"""
Test cases for utility functions.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.helpers import create_sequences, date_features, generate_sample_data


class TestCreateSequences:
    """Test cases for create_sequences function."""
    
    def test_create_sequences_shape(self):
        """Test that sequences have correct shape."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n_steps = 3
        
        X, y = create_sequences(data, n_steps=n_steps)
        
        expected_samples = len(data) - n_steps
        assert X.shape == (expected_samples, n_steps, 1)
        assert y.shape == (expected_samples,)
    
    def test_create_sequences_values(self):
        """Test that sequences contain correct values."""
        data = np.array([1, 2, 3, 4, 5])
        n_steps = 2
        
        X, y = create_sequences(data, n_steps=n_steps)
        
        # First sequence should be [1, 2] -> 3
        np.testing.assert_array_equal(X[0].flatten(), [1, 2])
        assert y[0] == 3


class TestDateFeatures:
    """Test cases for date_features function."""
    
    def test_date_features_extraction(self):
        """Test extraction of date features."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({'value': range(10)}, index=dates)
        
        df_with_features = date_features(df)
        
        # Check that new columns exist
        assert 'year' in df_with_features.columns
        assert 'month' in df_with_features.columns
        assert 'day' in df_with_features.columns
        assert 'dayofweek' in df_with_features.columns
        assert 'is_weekend' in df_with_features.columns


class TestGenerateSampleData:
    """Test cases for generate_sample_data function."""
    
    def test_generate_sample_data_length(self):
        """Test that generated data has correct length."""
        n_samples = 100
        data = generate_sample_data(n_samples=n_samples)
        
        assert len(data) == n_samples
    
    def test_generate_sample_data_index(self):
        """Test that generated data has datetime index."""
        data = generate_sample_data(n_samples=10)
        
        assert isinstance(data.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__])

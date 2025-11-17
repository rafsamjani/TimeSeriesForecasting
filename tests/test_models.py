"""
Test cases for time series models.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.arima_model import ARIMAForecaster
from src.utils.helpers import generate_sample_data


class TestARIMAForecaster:
    """Test cases for ARIMA model."""
    
    def test_arima_initialization(self):
        """Test ARIMA model initialization."""
        model = ARIMAForecaster(order=(1, 1, 1))
        assert model.order == (1, 1, 1)
        assert model.model is None
        assert model.fitted_model is None
    
    def test_arima_fit_predict(self):
        """Test ARIMA model fitting and prediction."""
        # Generate sample data
        data = generate_sample_data(n_samples=100, trend=True, seasonality=False)
        
        # Split data
        train = data[:80]
        test = data[80:]
        
        # Fit model
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(train)
        
        # Check model is fitted
        assert model.fitted_model is not None
        
        # Make predictions
        predictions = model.predict(steps=len(test))
        
        # Check predictions shape
        assert len(predictions) == len(test)
        assert isinstance(predictions, pd.Series)
    
    def test_arima_get_aic_bic(self):
        """Test getting AIC and BIC from fitted model."""
        data = generate_sample_data(n_samples=100, trend=True)
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(data)
        
        aic = model.get_aic()
        bic = model.get_bic()
        
        assert isinstance(aic, float)
        assert isinstance(bic, float)


if __name__ == "__main__":
    pytest.main([__file__])

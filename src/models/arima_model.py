"""
ARIMA Model Implementation

Autoregressive Integrated Moving Average (ARIMA) model for time series forecasting.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Tuple


class ARIMAForecaster:
    """
    ARIMA model wrapper for time series forecasting.
    
    Parameters
    ----------
    order : tuple
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters.
    seasonal_order : tuple, optional
        The (P,D,Q,s) seasonal order of the model.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: pd.Series) -> 'ARIMAForecaster':
        """
        Fit the ARIMA model to the data.
        
        Parameters
        ----------
        data : pd.Series
            Time series data to fit.
            
        Returns
        -------
        self : ARIMAForecaster
            Fitted model instance.
        """
        self.model = ARIMA(data, order=self.order, 
                          seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, steps: int = 1) -> pd.Series:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast.
            
        Returns
        -------
        predictions : pd.Series
            Forecasted values.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_model_summary(self) -> str:
        """
        Get summary statistics of the fitted model.
        
        Returns
        -------
        summary : str
            Model summary.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first.")
        
        return str(self.fitted_model.summary())
    
    def get_aic(self) -> float:
        """Get Akaike Information Criterion."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first.")
        return self.fitted_model.aic
    
    def get_bic(self) -> float:
        """Get Bayesian Information Criterion."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first.")
        return self.fitted_model.bic

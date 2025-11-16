"""
Prophet Model Implementation

Facebook Prophet model for time series forecasting with trend and seasonality.
"""

import pandas as pd
from prophet import Prophet
from typing import Optional


class ProphetForecaster:
    """
    Facebook Prophet model wrapper for time series forecasting.
    
    Parameters
    ----------
    growth : str
        'linear' or 'logistic' growth.
    changepoint_prior_scale : float
        Flexibility of the trend (default 0.05).
    seasonality_mode : str
        'additive' or 'multiplicative' seasonality.
    """
    
    def __init__(self, growth: str = 'linear',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_mode: str = 'additive'):
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.model = None
        
    def fit(self, data: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit the Prophet model to the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'ds' (date) and 'y' (value) columns.
            
        Returns
        -------
        self : ProphetForecaster
            Fitted model instance.
        """
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode
        )
        self.model.fit(data)
        return self
    
    def predict(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Generate forecasts.
        
        Parameters
        ----------
        periods : int
            Number of periods to forecast.
        freq : str
            Frequency of predictions ('D' for daily, 'M' for monthly, etc.).
            
        Returns
        -------
        forecast : pd.DataFrame
            DataFrame with predictions and confidence intervals.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast
    
    def plot_components(self) -> None:
        """Plot forecast components (trend, seasonality)."""
        if self.model is None:
            raise ValueError("Model must be fitted first.")
        
        forecast = self.predict(periods=0)
        return self.model.plot_components(forecast)
    
    def add_seasonality(self, name: str, period: float, 
                       fourier_order: int) -> 'ProphetForecaster':
        """
        Add custom seasonality.
        
        Parameters
        ----------
        name : str
            Name of the seasonality component.
        period : float
            Period of the seasonality in days.
        fourier_order : int
            Number of Fourier terms to use.
            
        Returns
        -------
        self : ProphetForecaster
        """
        if self.model is None:
            self.model = Prophet(
                growth=self.growth,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_mode=self.seasonality_mode
            )
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order
        )
        return self

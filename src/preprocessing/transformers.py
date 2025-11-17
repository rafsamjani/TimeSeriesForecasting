"""
Data Transformation Utilities

Classes and functions for transforming time series data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Tuple, Optional


class Normalizer:
    """
    Normalize time series data.
    
    Parameters
    ----------
    method : str
        Normalization method: 'minmax' or 'standard'.
    """
    
    def __init__(self, method: str = 'minmax'):
        self.method = method
        self.scaler = None
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to normalize.
            
        Returns
        -------
        normalized : np.ndarray
            Normalized data.
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        return self.scaler.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Parameters
        ----------
        data : np.ndarray
            Normalized data.
            
        Returns
        -------
        original : np.ndarray
            Original scale data.
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        return self.scaler.inverse_transform(data)


class Differencer:
    """
    Apply differencing to make time series stationary.
    
    Parameters
    ----------
    order : int
        Order of differencing (default 1).
    """
    
    def __init__(self, order: int = 1):
        self.order = order
        self.original_values = []
    
    def transform(self, data: pd.Series) -> pd.Series:
        """
        Apply differencing to the data.
        
        Parameters
        ----------
        data : pd.Series
            Time series data.
            
        Returns
        -------
        differenced : pd.Series
            Differenced time series.
        """
        self.original_values = []
        result = data.copy()
        
        for i in range(self.order):
            self.original_values.append(result.iloc[0])
            result = result.diff().dropna()
        
        return result
    
    def inverse_transform(self, data: pd.Series) -> pd.Series:
        """
        Reverse the differencing operation.
        
        Parameters
        ----------
        data : pd.Series
            Differenced time series.
            
        Returns
        -------
        original : pd.Series
            Original scale time series.
        """
        result = data.copy()
        
        for i in range(self.order - 1, -1, -1):
            result = result.cumsum() + self.original_values[i]
        
        return result


class StationarityTester:
    """
    Test time series stationarity using statistical tests.
    """
    
    @staticmethod
    def adf_test(data: pd.Series, significance_level: float = 0.05) -> dict:
        """
        Perform Augmented Dickey-Fuller test.
        
        Parameters
        ----------
        data : pd.Series
            Time series data.
        significance_level : float
            Significance level for the test (default 0.05).
            
        Returns
        -------
        results : dict
            Test results including statistic, p-value, and conclusion.
        """
        result = adfuller(data.dropna())
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < significance_level,
            'conclusion': 'Stationary' if result[1] < significance_level else 'Non-stationary'
        }
    
    @staticmethod
    def kpss_test(data: pd.Series, significance_level: float = 0.05) -> dict:
        """
        Perform KPSS test.
        
        Parameters
        ----------
        data : pd.Series
            Time series data.
        significance_level : float
            Significance level for the test (default 0.05).
            
        Returns
        -------
        results : dict
            Test results including statistic, p-value, and conclusion.
        """
        result = kpss(data.dropna(), regression='c')
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'is_stationary': result[1] > significance_level,
            'conclusion': 'Stationary' if result[1] > significance_level else 'Non-stationary'
        }
    
    @staticmethod
    def check_stationarity(data: pd.Series) -> dict:
        """
        Check stationarity using both ADF and KPSS tests.
        
        Parameters
        ----------
        data : pd.Series
            Time series data.
            
        Returns
        -------
        results : dict
            Combined results from both tests.
        """
        adf_results = StationarityTester.adf_test(data)
        kpss_results = StationarityTester.kpss_test(data)
        
        return {
            'adf_test': adf_results,
            'kpss_test': kpss_results,
            'both_agree': adf_results['is_stationary'] == kpss_results['is_stationary']
        }

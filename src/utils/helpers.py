"""
Helper Utilities

General utility functions for time series analysis.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List
from datetime import datetime


def create_sequences(data: np.ndarray, n_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for supervised learning from time series data.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data.
    n_steps : int
        Number of time steps in each sequence.
        
    Returns
    -------
    X : np.ndarray
        Input sequences with shape (samples, n_steps, features).
    y : np.ndarray
        Target values.
    """
    X, y = [], []
    
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to 3D for LSTM: [samples, timesteps, features]
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y


def date_features(data: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
    """
    Extract date-related features from datetime index or column.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index or column.
    date_column : str, optional
        Name of date column if not using index.
        
    Returns
    -------
    data : pd.DataFrame
        DataFrame with additional date features.
    """
    df = data.copy()
    
    if date_column:
        dates = pd.to_datetime(df[date_column])
    else:
        dates = pd.to_datetime(df.index)
    
    df['year'] = dates.year
    df['month'] = dates.month
    df['day'] = dates.day
    df['dayofweek'] = dates.dayofweek
    df['dayofyear'] = dates.dayofyear
    df['week'] = dates.isocalendar().week
    df['quarter'] = dates.quarter
    df['is_weekend'] = dates.dayofweek.isin([5, 6]).astype(int)
    
    return df


def lag_features(data: pd.Series, lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for time series.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    lags : list
        List of lag values to create.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with lagged features.
    """
    df = pd.DataFrame()
    df['value'] = data.values
    
    for lag in lags:
        df[f'lag_{lag}'] = data.shift(lag).values
    
    return df


def rolling_features(data: pd.Series, windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    windows : list
        List of window sizes.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with rolling features.
    """
    df = pd.DataFrame()
    df['value'] = data.values
    
    for window in windows:
        df[f'rolling_mean_{window}'] = data.rolling(window=window).mean().values
        df[f'rolling_std_{window}'] = data.rolling(window=window).std().values
        df[f'rolling_min_{window}'] = data.rolling(window=window).min().values
        df[f'rolling_max_{window}'] = data.rolling(window=window).max().values
    
    return df


def save_model(model, filepath: str) -> None:
    """
    Save a model to disk.
    
    Parameters
    ----------
    model : object
        Model object to save.
    filepath : str
        Path to save the model.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load a model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model.
        
    Returns
    -------
    model : object
        Loaded model object.
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def detect_outliers(data: pd.Series, method: str = 'iqr', 
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in time series data.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    method : str
        Method for outlier detection: 'iqr' or 'zscore'.
    threshold : float
        Threshold for outlier detection.
        
    Returns
    -------
    outliers : pd.Series
        Boolean series indicating outliers.
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers


def remove_outliers(data: pd.Series, method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Remove outliers from time series data.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    method : str
        Method for outlier detection.
    threshold : float
        Threshold for outlier detection.
        
    Returns
    -------
    cleaned : pd.Series
        Data with outliers removed.
    """
    outliers = detect_outliers(data, method, threshold)
    return data[~outliers]


def generate_sample_data(n_samples: int = 1000, trend: bool = True,
                        seasonality: bool = True, noise_level: float = 0.1,
                        seasonal_period: int = 12) -> pd.Series:
    """
    Generate synthetic time series data for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    trend : bool
        Whether to include a trend component.
    seasonality : bool
        Whether to include seasonality.
    noise_level : float
        Standard deviation of noise.
    seasonal_period : int
        Period of seasonality.
        
    Returns
    -------
    data : pd.Series
        Generated time series data.
    """
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Initialize series
    values = np.zeros(n_samples)
    
    # Add trend
    if trend:
        values += np.linspace(0, 10, n_samples)
    
    # Add seasonality
    if seasonality:
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / seasonal_period)
        values += seasonal
    
    # Add noise
    values += np.random.normal(0, noise_level, n_samples)
    
    # Create series
    data = pd.Series(values, index=dates, name='value')
    
    return data

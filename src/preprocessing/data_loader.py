"""
Data Loading Utilities

Functions for loading and splitting time series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


def load_time_series(filepath: str, date_column: str = 'date',
                     value_column: str = 'value',
                     date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Load time series data from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    date_column : str
        Name of the date column.
    value_column : str
        Name of the value column.
    date_format : str, optional
        Format of the date column.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with datetime index.
    """
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    if date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df.set_index(date_column, inplace=True)
    df.sort_index(inplace=True)
    
    return df


def train_test_split_ts(data: pd.Series, test_size: float = 0.2,
                        shuffle: bool = False) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series data into train and test sets.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    test_size : float
        Proportion of data to include in test set (default 0.2).
    shuffle : bool
        Whether to shuffle data before splitting (default False).
        
    Returns
    -------
    train : pd.Series
        Training data.
    test : pd.Series
        Test data.
    """
    if shuffle:
        data = data.sample(frac=1)
    
    split_idx = int(len(data) * (1 - test_size))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    return train, test


def create_dataset_for_prophet(data: pd.Series) -> pd.DataFrame:
    """
    Convert a time series to Prophet format.
    
    Parameters
    ----------
    data : pd.Series
        Time series with datetime index.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'ds' and 'y' columns for Prophet.
    """
    df = pd.DataFrame({
        'ds': data.index,
        'y': data.values
    })
    df.reset_index(drop=True, inplace=True)
    
    return df


def resample_time_series(data: pd.Series, freq: str = 'D',
                        agg_func: str = 'mean') -> pd.Series:
    """
    Resample time series to a different frequency.
    
    Parameters
    ----------
    data : pd.Series
        Time series with datetime index.
    freq : str
        Target frequency ('D' for daily, 'W' for weekly, 'M' for monthly).
    agg_func : str
        Aggregation function ('mean', 'sum', 'min', 'max').
        
    Returns
    -------
    resampled : pd.Series
        Resampled time series.
    """
    if agg_func == 'mean':
        return data.resample(freq).mean()
    elif agg_func == 'sum':
        return data.resample(freq).sum()
    elif agg_func == 'min':
        return data.resample(freq).min()
    elif agg_func == 'max':
        return data.resample(freq).max()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")


def handle_missing_values(data: pd.Series, method: str = 'interpolate') -> pd.Series:
    """
    Handle missing values in time series.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    method : str
        Method to handle missing values:
        - 'interpolate': Linear interpolation
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - 'drop': Drop missing values
        
    Returns
    -------
    data : pd.Series
        Time series with missing values handled.
    """
    if method == 'interpolate':
        return data.interpolate(method='linear')
    elif method == 'ffill':
        return data.fillna(method='ffill')
    elif method == 'bfill':
        return data.fillna(method='bfill')
    elif method == 'drop':
        return data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

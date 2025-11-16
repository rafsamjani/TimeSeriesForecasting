"""
Data preprocessing utilities for time series analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_time_series_data(filepath, date_column='date', value_column='value', 
                          parse_dates=True):
    """
    Load time series data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    date_column : str
        Name of the date column
    value_column : str
        Name of the value column
    parse_dates : bool
        Whether to parse dates
        
    Returns:
    --------
    pd.DataFrame : Loaded time series data
    """
    df = pd.read_csv(filepath, parse_dates=[date_column] if parse_dates else None)
    if date_column in df.columns:
        df.set_index(date_column, inplace=True)
    return df


def handle_missing_values(data, method='interpolate'):
    """
    Handle missing values in time series data
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    method : str
        Method to handle missing values ('interpolate', 'ffill', 'bfill', 'drop')
        
    Returns:
    --------
    pd.DataFrame or pd.Series : Data with missing values handled
    """
    if method == 'interpolate':
        return data.interpolate(method='time')
    elif method == 'ffill':
        return data.fillna(method='ffill')
    elif method == 'bfill':
        return data.fillna(method='bfill')
    elif method == 'drop':
        return data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_data(data, method='minmax'):
    """
    Normalize time series data
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    method : str
        Normalization method ('minmax' or 'standard')
        
    Returns:
    --------
    normalized_data : array-like
        Normalized data
    scaler : sklearn scaler object
        Fitted scaler for inverse transformation
    """
    values = data.values.reshape(-1, 1) if len(data.shape) == 1 else data.values
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    normalized_data = scaler.fit_transform(values)
    return normalized_data, scaler


def create_sequences(data, seq_length, target_column=None):
    """
    Create sequences for LSTM/RNN models
    
    Parameters:
    -----------
    data : array-like
        Time series data
    seq_length : int
        Length of input sequences
    target_column : int or None
        Column index for target variable
        
    Returns:
    --------
    X : np.array
        Input sequences
    y : np.array
        Target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        if target_column is not None:
            y.append(data[i+seq_length, target_column])
        else:
            y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def train_test_split_temporal(data, test_size=0.2):
    """
    Split time series data into train and test sets
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    train : pd.DataFrame or pd.Series
        Training data
    test : pd.DataFrame or pd.Series
        Testing data
    """
    split_idx = int(len(data) * (1 - test_size))
    train = data[:split_idx]
    test = data[split_idx:]
    return train, test


def check_stationarity(data, column=None):
    """
    Check if time series is stationary using Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    column : str or None
        Column name if data is DataFrame
        
    Returns:
    --------
    dict : Test results including test statistic, p-value, and conclusion
    """
    from statsmodels.tsa.stattools import adfuller
    
    if isinstance(data, pd.DataFrame):
        if column is None:
            column = data.columns[0]
        series = data[column]
    else:
        series = data
    
    result = adfuller(series.dropna())
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def difference_series(data, periods=1):
    """
    Apply differencing to make series stationary
    
    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data
    periods : int
        Number of periods to difference
        
    Returns:
    --------
    pd.Series or pd.DataFrame : Differenced series
    """
    return data.diff(periods=periods).dropna()

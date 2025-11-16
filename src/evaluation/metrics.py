"""
Evaluation Metrics

Functions for evaluating time series forecasting models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score as sklearn_r2
from typing import Dict, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    mae : float
        Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    rmse : float
        Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    mse : float
        Mean Squared Error.
    """
    return mean_squared_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    mape : float
        Mean Absolute Percentage Error (in percentage).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    r2 : float
        R-squared score.
    """
    return sklearn_r2(y_true, y_pred)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
        
    Returns
    -------
    smape : float
        Symmetric MAPE (in percentage).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def mase(y_true: np.ndarray, y_pred: np.ndarray, 
         y_train: Optional[np.ndarray] = None, seasonality: int = 1) -> float:
    """
    Calculate Mean Absolute Scaled Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    y_train : np.ndarray, optional
        Training data for scaling.
    seasonality : int
        Seasonal period (default 1 for non-seasonal).
        
    Returns
    -------
    mase : float
        Mean Absolute Scaled Error.
    """
    if y_train is None:
        y_train = y_true
    
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    
    if mae_naive == 0:
        return np.inf
    
    return mae_forecast / mae_naive


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    y_train : np.ndarray, optional
        Training data for MASE calculation.
        
    Returns
    -------
    metrics : dict
        Dictionary containing all metrics.
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred)
    }
    
    if y_train is not None:
        metrics['MASE'] = mase(y_true, y_pred, y_train)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics.
    """
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:10s}: {value:.4f}")
    
    print("="*50 + "\n")

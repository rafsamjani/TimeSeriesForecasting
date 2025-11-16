"""
Visualization Utilities

Functions for plotting time series data and forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Optional, Tuple, List

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_time_series(data: pd.Series, title: str = 'Time Series',
                     xlabel: str = 'Date', ylabel: str = 'Value',
                     figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot a time series.
    
    Parameters
    ----------
    data : pd.Series
        Time series data with datetime index.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size.
    """
    plt.figure(figsize=figsize)
    plt.plot(data.index, data.values, linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast(train: pd.Series, test: pd.Series,
                 predictions: np.ndarray,
                 title: str = 'Forecast vs Actual',
                 figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Plot training data, test data, and predictions.
    
    Parameters
    ----------
    train : pd.Series
        Training data.
    test : pd.Series
        Test data.
    predictions : np.ndarray
        Predicted values.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    plt.figure(figsize=figsize)
    
    # Plot training data
    plt.plot(train.index, train.values, label='Training Data', 
             linewidth=2, color='blue', alpha=0.7)
    
    # Plot test data
    plt.plot(test.index, test.values, label='Actual', 
             linewidth=2, color='green', marker='o', markersize=4)
    
    # Plot predictions
    plt.plot(test.index, predictions, label='Forecast', 
             linewidth=2, color='red', linestyle='--', marker='x', markersize=6)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(residuals: np.ndarray,
                  figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Plot residual diagnostics.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from the model.
    figsize : tuple
        Figure size.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals over time
    axes[0, 0].plot(residuals, linewidth=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Histogram of Residuals', fontweight='bold')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ACF of residuals
    plot_acf(residuals, lags=40, ax=axes[1, 1])
    axes[1, 1].set_title('ACF of Residuals', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(data: pd.Series, lags: int = 40,
                  figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Plot ACF and PACF.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    lags : int
        Number of lags to plot.
    figsize : tuple
        Figure size.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF
    plot_acf(data, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(data, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_decomposition(data: pd.Series, model: str = 'additive',
                      period: Optional[int] = None,
                      figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot time series decomposition.
    
    Parameters
    ----------
    data : pd.Series
        Time series data.
    model : str
        'additive' or 'multiplicative'.
    period : int, optional
        Seasonal period.
    figsize : tuple
        Figure size.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(data, model=model, period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Original
    decomposition.observed.plot(ax=axes[0], title='Original', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    decomposition.trend.plot(ax=axes[1], title='Trend', linewidth=2, color='orange')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', linewidth=2, color='green')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    decomposition.resid.plot(ax=axes[3], title='Residual', linewidth=2, color='red')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_multiple_forecasts(test: pd.Series, 
                           predictions_dict: dict,
                           title: str = 'Model Comparison',
                           figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Plot multiple model forecasts for comparison.
    
    Parameters
    ----------
    test : pd.Series
        Test data.
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    plt.figure(figsize=figsize)
    
    # Plot actual values
    plt.plot(test.index, test.values, label='Actual', 
             linewidth=3, color='black', marker='o', markersize=4)
    
    # Plot predictions from each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        plt.plot(test.index, predictions, label=model_name, 
                linewidth=2, linestyle='--', marker='x', 
                markersize=6, color=colors[i], alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

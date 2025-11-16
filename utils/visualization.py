"""
Visualization utilities for time series analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_time_series(data, title='Time Series Data', xlabel='Date', ylabel='Value', 
                     figsize=(12, 6)):
    """
    Plot time series data
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            plt.plot(data.index, data[col], label=col)
        plt.legend()
    else:
        plt.plot(data.index, data.values)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast(train, test, forecast, title='Time Series Forecast', 
                 figsize=(14, 7)):
    """
    Plot training data, test data, and forecast
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data
    forecast : array-like
        Forecasted values
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(train.index, train.values, label='Training Data', color='blue')
    plt.plot(test.index, test.values, label='Actual Test Data', color='green')
    plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(data, lags=40, figsize=(14, 6)):
    """
    Plot ACF and PACF
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    lags : int
        Number of lags
    figsize : tuple
        Figure size
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_acf(data.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    
    plot_pacf(data.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.show()


def plot_decomposition(data, model='additive', figsize=(14, 10)):
    """
    Plot time series decomposition
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    model : str
        Decomposition model ('additive' or 'multiplicative')
    figsize : tuple
        Figure size
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(data.dropna(), model=model, period=12)
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    decomposition.observed.plot(ax=axes[0], title='Observed')
    axes[0].grid(True, alpha=0.3)
    
    decomposition.trend.plot(ax=axes[1], title='Trend')
    axes[1].grid(True, alpha=0.3)
    
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    decomposition.resid.plot(ax=axes[3], title='Residual')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals(residuals, figsize=(14, 6)):
    """
    Plot residuals analysis
    
    Parameters:
    -----------
    residuals : array-like
        Model residuals
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals plot
    axes[0].plot(residuals)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title('Residuals Plot')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Residuals')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_title('Residuals Distribution')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_multiple_forecasts(test, forecasts, labels, title='Model Comparison', 
                            figsize=(14, 7)):
    """
    Plot multiple forecasts for comparison
    
    Parameters:
    -----------
    test : pd.Series
        Actual test data
    forecasts : list
        List of forecast arrays
    labels : list
        List of model names
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(test.index, test.values, label='Actual', color='black', linewidth=2)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (forecast, label) in enumerate(zip(forecasts, labels)):
        plt.plot(test.index, forecast, label=label, 
                color=colors[i % len(colors)], linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prediction_intervals(test, forecast, lower_bound, upper_bound, 
                              title='Forecast with Prediction Intervals', 
                              figsize=(14, 7)):
    """
    Plot forecast with prediction intervals
    
    Parameters:
    -----------
    test : pd.Series
        Actual test data
    forecast : array-like
        Forecasted values
    lower_bound : array-like
        Lower bound of prediction interval
    upper_bound : array-like
        Upper bound of prediction interval
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(test.index, test.values, label='Actual', color='black', linewidth=2)
    plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--')
    plt.fill_between(test.index, lower_bound, upper_bound, alpha=0.3, 
                     color='red', label='Prediction Interval')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

"""
Traditional Time Series Forecasting Methods

This module demonstrates classical forecasting methods:
1. Moving Average (MA)
2. Exponential Smoothing (Simple, Double, Triple/Holt-Winters)
3. Naive and Seasonal Naive methods

These methods are simple, interpretable, and often serve as good baselines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_sample_data():
    """
    Create sample time series data with seasonality
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    trend = np.linspace(100, 200, len(dates))
    seasonal = 15 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    noise = np.random.normal(0, 3, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({'value': values}, index=dates)
    return df


def moving_average_forecast(data, window=3, steps=12):
    """
    Simple Moving Average forecast
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    window : int
        Moving average window size
    steps : int
        Number of steps to forecast
        
    Returns:
    --------
    forecast : np.array
        Forecasted values
    """
    last_values = data.values[-window:]
    forecast = np.full(steps, last_values.mean())
    return forecast


def simple_exponential_smoothing(train, test, alpha=0.3):
    """
    Simple Exponential Smoothing
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data
    alpha : float
        Smoothing parameter
        
    Returns:
    --------
    forecast : np.array
        Forecasted values
    """
    model = ExponentialSmoothing(train, trend=None, seasonal=None)
    model_fit = model.fit(smoothing_level=alpha)
    forecast = model_fit.forecast(steps=len(test))
    return forecast.values


def double_exponential_smoothing(train, test):
    """
    Double Exponential Smoothing (Holt's method)
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data
        
    Returns:
    --------
    forecast : np.array
        Forecasted values
    """
    model = ExponentialSmoothing(train, trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast.values


def triple_exponential_smoothing(train, test, seasonal_periods=12):
    """
    Triple Exponential Smoothing (Holt-Winters method)
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data
    seasonal_periods : int
        Number of periods in a season
        
    Returns:
    --------
    forecast : np.array
        Forecasted values
    """
    model = ExponentialSmoothing(
        train, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=seasonal_periods
    )
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast.values


def naive_forecast(train, steps):
    """
    Naive forecast (last observation)
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    steps : int
        Number of steps to forecast
        
    Returns:
    --------
    forecast : np.array
        Forecasted values
    """
    return np.full(steps, train.values[-1])


def seasonal_naive_forecast(train, steps, seasonal_period=12):
    """
    Seasonal Naive forecast
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    steps : int
        Number of steps to forecast
    seasonal_period : int
        Number of periods in a season
        
    Returns:
    --------
    forecast : np.array
        Forecasted values
    """
    last_season = train.values[-seasonal_period:]
    forecast = np.tile(last_season, (steps // seasonal_period) + 1)[:steps]
    return forecast


def evaluate_model(actual, predicted):
    """
    Evaluate model performance
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def main():
    """
    Main function to demonstrate traditional forecasting methods
    """
    print("=" * 60)
    print("Traditional Time Series Forecasting Methods")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading sample data...")
    df = load_sample_data()
    print(f"Data shape: {df.shape}")
    
    # Split data
    print("\n2. Splitting data into train and test sets...")
    train_size = int(len(df) * 0.8)
    train = df['value'][:train_size]
    test = df['value'][train_size:]
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Dictionary to store forecasts and metrics
    forecasts = {}
    metrics_dict = {}
    
    # Moving Average
    print("\n3. Applying Moving Average...")
    ma_forecast = moving_average_forecast(train, window=3, steps=len(test))
    forecasts['Moving Average'] = ma_forecast
    metrics_dict['Moving Average'] = evaluate_model(test.values, ma_forecast)
    
    # Simple Exponential Smoothing
    print("4. Applying Simple Exponential Smoothing...")
    ses_forecast = simple_exponential_smoothing(train, test)
    forecasts['Simple Exponential Smoothing'] = ses_forecast
    metrics_dict['Simple Exponential Smoothing'] = evaluate_model(test.values, ses_forecast)
    
    # Double Exponential Smoothing
    print("5. Applying Double Exponential Smoothing...")
    des_forecast = double_exponential_smoothing(train, test)
    forecasts['Double Exponential Smoothing'] = des_forecast
    metrics_dict['Double Exponential Smoothing'] = evaluate_model(test.values, des_forecast)
    
    # Triple Exponential Smoothing
    print("6. Applying Triple Exponential Smoothing...")
    tes_forecast = triple_exponential_smoothing(train, test, seasonal_periods=12)
    forecasts['Triple Exponential Smoothing'] = tes_forecast
    metrics_dict['Triple Exponential Smoothing'] = evaluate_model(test.values, tes_forecast)
    
    # Naive Forecast
    print("7. Applying Naive Forecast...")
    naive_forecast_result = naive_forecast(train, len(test))
    forecasts['Naive'] = naive_forecast_result
    metrics_dict['Naive'] = evaluate_model(test.values, naive_forecast_result)
    
    # Seasonal Naive Forecast
    print("8. Applying Seasonal Naive Forecast...")
    snaive_forecast = seasonal_naive_forecast(train, len(test), seasonal_period=12)
    forecasts['Seasonal Naive'] = snaive_forecast
    metrics_dict['Seasonal Naive'] = evaluate_model(test.values, snaive_forecast)
    
    # Print all metrics
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)
    metrics_df = pd.DataFrame(metrics_dict).T
    print(metrics_df.to_string())
    
    # Find best model
    best_model = metrics_df['RMSE'].idxmin()
    print(f"\nBest Model (by RMSE): {best_model}")
    
    # Plot all forecasts
    print("\n9. Plotting results...")
    plt.figure(figsize=(16, 10))
    
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    plt.plot(train.index, train.values, label='Training Data', 
             color='blue', linewidth=2)
    plt.plot(test.index, test.values, label='Actual Test Data', 
             color='green', linewidth=2)
    
    for i, (method, forecast) in enumerate(forecasts.items()):
        plt.plot(test.index, forecast, label=method, 
                color=colors[i % len(colors)], linestyle='--', linewidth=1.5)
    
    plt.title('Comparison of Traditional Forecasting Methods')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('traditional_methods_comparison.png')
    print("Comparison plot saved to 'traditional_methods_comparison.png'")
    
    # Plot individual methods
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (method, forecast) in enumerate(forecasts.items()):
        axes[i].plot(train.index, train.values, label='Train', color='blue', alpha=0.5)
        axes[i].plot(test.index, test.values, label='Actual', color='green', linewidth=2)
        axes[i].plot(test.index, forecast, label='Forecast', 
                    color='red', linestyle='--', linewidth=2)
        axes[i].set_title(f'{method}\nRMSE: {metrics_dict[method]["RMSE"]:.2f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('traditional_methods_individual.png')
    print("Individual methods plot saved to 'traditional_methods_individual.png'")
    
    print("\n" + "=" * 60)
    print("Traditional methods forecasting completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

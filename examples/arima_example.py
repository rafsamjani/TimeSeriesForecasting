"""
ARIMA (AutoRegressive Integrated Moving Average) Example

ARIMA is one of the most popular statistical methods for time series forecasting.
It combines three components:
- AR (AutoRegressive): Uses past values to predict future
- I (Integrated): Differencing to make series stationary
- MA (Moving Average): Uses past forecast errors

Suitable for: Stationary or trend-stationary data, univariate forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_sample_data():
    """
    Create sample time series data
    """
    # Generate sample data with trend and seasonality
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({'value': values}, index=dates)
    return df


def fit_arima_model(train_data, order=(1, 1, 1)):
    """
    Fit ARIMA model to training data
    
    Parameters:
    -----------
    train_data : pd.Series
        Training time series data
    order : tuple
        ARIMA order (p, d, q)
        
    Returns:
    --------
    model_fit : ARIMAResults
        Fitted ARIMA model
    """
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit


def forecast_arima(model_fit, steps):
    """
    Generate forecasts using fitted ARIMA model
    
    Parameters:
    -----------
    model_fit : ARIMAResults
        Fitted ARIMA model
    steps : int
        Number of steps to forecast
        
    Returns:
    --------
    forecast : array
        Forecasted values
    conf_int : array
        Confidence intervals
    """
    forecast_result = model_fit.forecast(steps=steps)
    forecast = forecast_result
    
    # Get prediction intervals
    forecast_obj = model_fit.get_forecast(steps=steps)
    conf_int = forecast_obj.conf_int()
    
    return forecast, conf_int


def evaluate_model(actual, predicted):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns:
    --------
    dict : Evaluation metrics
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
    Main function to demonstrate ARIMA forecasting
    """
    print("=" * 60)
    print("ARIMA Time Series Forecasting Example")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading sample data...")
    df = load_sample_data()
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Split data
    print("\n2. Splitting data into train and test sets...")
    train_size = int(len(df) * 0.8)
    train = df['value'][:train_size]
    test = df['value'][train_size:]
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Plot ACF and PACF
    print("\n3. Analyzing ACF and PACF...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    max_lags = min(20, len(train) // 2 - 1)
    plot_acf(train, lags=max_lags, ax=axes[0])
    plot_pacf(train, lags=max_lags, ax=axes[1])
    plt.tight_layout()
    plt.savefig('arima_acf_pacf.png')
    print("ACF and PACF plots saved to 'arima_acf_pacf.png'")
    
    # Fit ARIMA model
    print("\n4. Fitting ARIMA model...")
    model_fit = fit_arima_model(train, order=(1, 1, 1))
    
    # Forecast
    print("\n5. Generating forecasts...")
    forecast, conf_int = forecast_arima(model_fit, steps=len(test))
    
    # Evaluate
    print("\n6. Evaluating model...")
    metrics = evaluate_model(test.values, forecast)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    print("\n7. Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train.values, label='Training Data', color='blue')
    plt.plot(test.index, test.values, label='Actual Test Data', color='green')
    plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
    plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                     alpha=0.3, color='red', label='95% Confidence Interval')
    plt.title('ARIMA Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('arima_forecast.png')
    print("Forecast plot saved to 'arima_forecast.png'")
    
    # Plot residuals
    residuals = model_fit.resid
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(residuals)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title('Residuals Plot')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Residuals')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_title('Residuals Distribution')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('arima_residuals.png')
    print("Residuals plot saved to 'arima_residuals.png'")
    
    print("\n" + "=" * 60)
    print("ARIMA forecasting completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

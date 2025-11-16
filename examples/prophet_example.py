"""
Prophet Example

Prophet is a forecasting tool developed by Facebook (Meta) designed for business forecasting.
It's particularly good for:
- Daily observations with seasonal patterns
- Multiple seasonality (weekly, yearly)
- Holiday effects
- Missing data and outliers

Suitable for: Business metrics, data with strong seasonal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_sample_data():
    """
    Create sample time series data
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Create trend
    trend = np.linspace(100, 250, len(dates))
    
    # Create yearly seasonality
    yearly_seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    
    # Create weekly seasonality
    weekly_seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Add noise
    noise = np.random.normal(0, 5, len(dates))
    
    values = trend + yearly_seasonal + weekly_seasonal + noise
    
    df = pd.DataFrame({
        'ds': dates,  # Prophet requires 'ds' column
        'y': values   # Prophet requires 'y' column
    })
    
    return df


def fit_prophet_model(train_data, yearly_seasonality=True, weekly_seasonality=True):
    """
    Fit Prophet model to training data
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data with 'ds' and 'y' columns
    yearly_seasonality : bool
        Whether to include yearly seasonality
    weekly_seasonality : bool
        Whether to include weekly seasonality
        
    Returns:
    --------
    model : Prophet
        Fitted Prophet model
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(train_data)
    return model


def forecast_prophet(model, periods):
    """
    Generate forecasts using Prophet model
    
    Parameters:
    -----------
    model : Prophet
        Fitted Prophet model
    periods : int
        Number of periods to forecast
        
    Returns:
    --------
    forecast : pd.DataFrame
        Forecast dataframe with predictions and confidence intervals
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
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
    Main function to demonstrate Prophet forecasting
    """
    print("=" * 60)
    print("Prophet Time Series Forecasting Example")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading sample data...")
    df = load_sample_data()
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['ds'].iloc[0]} to {df['ds'].iloc[-1]}")
    
    # Split data
    print("\n2. Splitting data into train and test sets...")
    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    test = df[train_size:]
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Fit Prophet model
    print("\n3. Fitting Prophet model...")
    model = fit_prophet_model(train)
    
    # Forecast
    print("\n4. Generating forecasts...")
    forecast = forecast_prophet(model, periods=len(test))
    
    # Extract test predictions
    test_forecast = forecast.iloc[-len(test):]
    
    # Evaluate
    print("\n5. Evaluating model...")
    metrics = evaluate_model(test['y'].values, test_forecast['yhat'].values)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    print("\n6. Plotting results...")
    
    # Plot 1: Forecast with components
    fig = model.plot(forecast)
    plt.title('Prophet Time Series Forecast')
    plt.tight_layout()
    plt.savefig('prophet_forecast_full.png')
    print("Full forecast plot saved to 'prophet_forecast_full.png'")
    
    # Plot 2: Components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('prophet_components.png')
    print("Components plot saved to 'prophet_components.png'")
    
    # Plot 3: Custom comparison plot
    plt.figure(figsize=(14, 7))
    plt.plot(train['ds'], train['y'], label='Training Data', color='blue')
    plt.plot(test['ds'], test['y'], label='Actual Test Data', color='green')
    plt.plot(test_forecast['ds'], test_forecast['yhat'], 
             label='Prophet Forecast', color='red', linestyle='--')
    plt.fill_between(test_forecast['ds'], 
                     test_forecast['yhat_lower'], 
                     test_forecast['yhat_upper'],
                     alpha=0.3, color='red', label='Prediction Interval')
    plt.title('Prophet Time Series Forecast - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prophet_forecast.png')
    print("Forecast comparison plot saved to 'prophet_forecast.png'")
    
    # Plot 4: Residuals
    residuals = test['y'].values - test_forecast['yhat'].values
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
    plt.savefig('prophet_residuals.png')
    print("Residuals plot saved to 'prophet_residuals.png'")
    
    print("\n" + "=" * 60)
    print("Prophet forecasting completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

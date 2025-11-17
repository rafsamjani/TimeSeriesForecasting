"""
Example Usage of Time Series Forecasting Library

This script demonstrates how to use the various components
of the time series forecasting library.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.utils.helpers import generate_sample_data
from src.preprocessing.data_loader import train_test_split_ts, create_dataset_for_prophet
from src.preprocessing.transformers import StationarityTester
from src.evaluation.metrics import calculate_metrics, print_metrics


def main():
    """Main function to demonstrate library usage."""
    
    print("="*60)
    print("Time Series Forecasting Library - Example Usage")
    print("="*60)
    
    # 1. Generate sample data
    print("\n1. Generating sample time series data...")
    data = generate_sample_data(
        n_samples=365,
        trend=True,
        seasonality=True,
        noise_level=0.5,
        seasonal_period=30
    )
    print(f"   Generated {len(data)} data points")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    
    # 2. Check stationarity
    print("\n2. Testing stationarity...")
    stationarity_results = StationarityTester.check_stationarity(data)
    print(f"   ADF Test: {stationarity_results['adf_test']['conclusion']}")
    print(f"   KPSS Test: {stationarity_results['kpss_test']['conclusion']}")
    
    # 3. Split data
    print("\n3. Splitting data into train and test sets...")
    train, test = train_test_split_ts(data, test_size=0.2)
    print(f"   Training set: {len(train)} samples")
    print(f"   Test set: {len(test)} samples")
    
    # 4. Train ARIMA model
    print("\n4. Training ARIMA model...")
    arima_model = ARIMAForecaster(order=(2, 1, 2))
    arima_model.fit(train)
    print("   Model fitted successfully")
    print(f"   AIC: {arima_model.get_aic():.2f}")
    print(f"   BIC: {arima_model.get_bic():.2f}")
    
    # 5. Make predictions
    print("\n5. Making predictions with ARIMA...")
    arima_predictions = arima_model.predict(steps=len(test))
    
    # 6. Evaluate ARIMA
    print("\n6. Evaluating ARIMA model...")
    arima_metrics = calculate_metrics(test.values, arima_predictions.values)
    print_metrics(arima_metrics)
    
    # 7. Train Prophet model
    print("\n7. Training Prophet model...")
    train_prophet = create_dataset_for_prophet(train)
    prophet_model = ProphetForecaster(
        growth='linear',
        changepoint_prior_scale=0.05,
        seasonality_mode='additive'
    )
    prophet_model.fit(train_prophet)
    print("   Model fitted successfully")
    
    # 8. Make predictions with Prophet
    print("\n8. Making predictions with Prophet...")
    prophet_forecast = prophet_model.predict(periods=len(test), freq='D')
    prophet_predictions = prophet_forecast['yhat'].tail(len(test)).values
    
    # 9. Evaluate Prophet
    print("\n9. Evaluating Prophet model...")
    prophet_metrics = calculate_metrics(test.values, prophet_predictions)
    print_metrics(prophet_metrics)
    
    # 10. Compare models
    print("\n10. Model Comparison:")
    comparison = pd.DataFrame({
        'ARIMA': arima_metrics,
        'Prophet': prophet_metrics
    }).T
    print(comparison)
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

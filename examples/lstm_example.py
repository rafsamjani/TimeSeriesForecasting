"""
LSTM (Long Short-Term Memory) Neural Network Example

LSTM is a type of recurrent neural network capable of learning long-term dependencies.
It's particularly effective for:
- Complex non-linear patterns
- Long sequences with dependencies
- Multivariate time series

Suitable for: Complex patterns, non-linear relationships, large datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


def load_sample_data():
    """
    Create sample time series data
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    trend = np.linspace(100, 300, len(dates))
    seasonal = 20 * np.sin(np.linspace(0, 20*np.pi, len(dates)))
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({'value': values}, index=dates)
    return df


def prepare_data(data, seq_length=30):
    """
    Prepare data for LSTM model
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    seq_length : int
        Length of input sequences
        
    Returns:
    --------
    X : np.array
        Input sequences
    y : np.array
        Target values
    scaler : MinMaxScaler
        Fitted scaler
    """
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


def build_lstm_model(seq_length, units=50):
    """
    Build LSTM model
    
    Parameters:
    -----------
    seq_length : int
        Length of input sequences
    units : int
        Number of LSTM units
        
    Returns:
    --------
    model : Sequential
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(units=units, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
    """
    Train LSTM model
    
    Parameters:
    -----------
    model : Sequential
        LSTM model
    X_train : np.array
        Training input sequences
    y_train : np.array
        Training target values
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    validation_split : float
        Validation split ratio
        
    Returns:
    --------
    history : History
        Training history
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history


def forecast_lstm(model, data, seq_length, scaler, steps):
    """
    Generate forecasts using LSTM model
    
    Parameters:
    -----------
    model : Sequential
        Trained LSTM model
    data : np.array
        Recent data for prediction
    seq_length : int
        Length of input sequences
    scaler : MinMaxScaler
        Fitted scaler
    steps : int
        Number of steps to forecast
        
    Returns:
    --------
    predictions : np.array
        Forecasted values
    """
    predictions = []
    current_sequence = data[-seq_length:].copy()
    
    for _ in range(steps):
        # Reshape for prediction
        current_input = current_sequence.reshape(1, seq_length, 1)
        
        # Predict next value
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()


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
    Main function to demonstrate LSTM forecasting
    """
    print("=" * 60)
    print("LSTM Time Series Forecasting Example")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading sample data...")
    df = load_sample_data()
    print(f"Data shape: {df.shape}")
    
    # Parameters
    seq_length = 30
    test_size = 90
    
    # Split data
    print("\n2. Splitting and preparing data...")
    train_data = df['value'][:-test_size]
    test_data = df['value'][-test_size:]
    
    # Prepare training data
    X, y, scaler = prepare_data(df['value'][:-test_size], seq_length)
    print(f"Training sequences shape: {X.shape}")
    
    # Build model
    print("\n3. Building LSTM model...")
    model = build_lstm_model(seq_length, units=50)
    print(model.summary())
    
    # Train model
    print("\n4. Training model...")
    history = train_model(model, X, y, epochs=50, batch_size=32)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    print("Training history plot saved to 'lstm_training_history.png'")
    
    # Prepare data for forecasting
    scaled_full_data = scaler.transform(df['value'].values.reshape(-1, 1))
    
    # Forecast
    print("\n5. Generating forecasts...")
    forecast = forecast_lstm(model, scaled_full_data[:-test_size], seq_length, scaler, test_size)
    
    # Evaluate
    print("\n6. Evaluating model...")
    metrics = evaluate_model(test_data.values, forecast)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    print("\n7. Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data.values, label='Training Data', color='blue')
    plt.plot(test_data.index, test_data.values, label='Actual Test Data', color='green')
    plt.plot(test_data.index, forecast, label='LSTM Forecast', color='red', linestyle='--')
    plt.title('LSTM Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_forecast.png')
    print("Forecast plot saved to 'lstm_forecast.png'")
    
    print("\n" + "=" * 60)
    print("LSTM forecasting completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

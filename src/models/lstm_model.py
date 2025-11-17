"""
LSTM Model Implementation

Long Short-Term Memory neural network for time series forecasting.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Tuple, Optional


class LSTMForecaster:
    """
    LSTM model for time series forecasting.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps to look back.
    n_features : int
        Number of features in the input.
    units : int
        Number of LSTM units (default 50).
    dropout : float
        Dropout rate (default 0.2).
    """
    
    def __init__(self, n_steps: int = 10, n_features: int = 1,
                 units: int = 50, dropout: float = 0.2):
        self.n_steps = n_steps
        self.n_features = n_features
        self.units = units
        self.dropout = dropout
        self.model = None
        self.history = None
        
    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        self.model = Sequential([
            LSTM(self.units, activation='relu', 
                 return_sequences=True,
                 input_shape=(self.n_steps, self.n_features)),
            Dropout(self.dropout),
            LSTM(self.units, activation='relu'),
            Dropout(self.dropout),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.1,
            verbose: int = 1) -> 'LSTMForecaster':
        """
        Fit the LSTM model to the training data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features with shape (samples, n_steps, n_features).
        y_train : np.ndarray
            Training targets.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        validation_split : float
            Fraction of data to use for validation.
        verbose : int
            Verbosity mode.
            
        Returns
        -------
        self : LSTMForecaster
            Fitted model instance.
        """
        if self.model is None:
            self.build_model()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input features with shape (samples, n_steps, n_features).
            
        Returns
        -------
        predictions : np.ndarray
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model must be built and fitted before prediction.")
        
        return self.model.predict(X)
    
    def predict_sequence(self, last_sequence: np.ndarray, 
                        n_steps_ahead: int = 1) -> np.ndarray:
        """
        Predict multiple steps ahead using recursive prediction.
        
        Parameters
        ----------
        last_sequence : np.ndarray
            Last known sequence with shape (n_steps, n_features).
        n_steps_ahead : int
            Number of steps to predict ahead.
            
        Returns
        -------
        predictions : np.ndarray
            Predicted values for future steps.
        """
        if self.model is None:
            raise ValueError("Model must be built and fitted before prediction.")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps_ahead):
            # Reshape for prediction
            X_input = current_sequence.reshape(1, self.n_steps, self.n_features)
            # Predict next value
            pred = self.model.predict(X_input, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred
        
        return np.array(predictions)
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("Model must be built before saving.")
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a model from a file."""
        self.model = keras.models.load_model(filepath)

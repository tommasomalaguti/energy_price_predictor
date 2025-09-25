"""
Advanced neural network models for electricity price forecasting.

This module implements state-of-the-art neural network architectures including
Transformers, N-BEATS, and other deep learning models for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neural network models will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Some neural network models will be disabled.")


class TransformerForecaster:
    """
    Transformer-based model for electricity price forecasting.
    """
    
    def __init__(self, sequence_length: int = 24, d_model: int = 64, 
                 n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1,
                 random_state: int = 42):
        """
        Initialize Transformer forecaster.
        
        Args:
            sequence_length: Length of input sequences
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            random_state: Random state for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Transformer models")
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _create_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Create Transformer model architecture."""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        x = self._add_positional_encoding(inputs)
        
        # Transformer layers
        for _ in range(self.n_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model,
                dropout=self.dropout
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed forward
            ffn = Dense(self.d_model * 4, activation='relu')(x)
            ffn = Dropout(self.dropout)(ffn)
            ffn = Dense(self.d_model)(ffn)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _add_positional_encoding(self, x):
        """Add positional encoding to input."""
        seq_len = tf.shape(x)[1]
        d_model = self.d_model
        
        # Create positional encoding matrix
        pos_encoding = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
        
        # Add positional encoding
        pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
        return x + pos_encoding
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting."""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_split: float = 0.2, epochs: int = 100, 
            batch_size: int = 32, verbose: int = 0) -> 'TransformerForecaster':
        """
        Train the Transformer model.
        
        Args:
            X: Feature matrix
            y: Target series
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            self
        """
        logger.info("Training Transformer model...")
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        y_array = y.values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_array)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Create model
        self.model = self._create_transformer_model((self.sequence_length, X.shape[1]))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("Transformer model trained successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the Transformer model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match original length
        full_predictions = np.full(len(X), np.nan)
        full_predictions[self.sequence_length:] = predictions.flatten()
        
        # Forward fill for initial predictions
        full_predictions[:self.sequence_length] = full_predictions[self.sequence_length]
        
        return full_predictions


class NBEATSForecaster:
    """
    N-BEATS (Neural Basis Expansion Analysis for Time Series) model.
    """
    
    def __init__(self, sequence_length: int = 24, forecast_length: int = 1,
                 n_blocks: int = 3, n_layers: int = 4, n_neurons: int = 512,
                 dropout: float = 0.1, random_state: int = 42):
        """
        Initialize N-BEATS forecaster.
        
        Args:
            sequence_length: Length of input sequences
            forecast_length: Length of forecast horizon
            n_blocks: Number of N-BEATS blocks
            n_layers: Number of layers per block
            n_neurons: Number of neurons per layer
            dropout: Dropout rate
            random_state: Random state for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for N-BEATS models")
        
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _create_nbeats_block(self, inputs, block_id: int):
        """Create a single N-BEATS block."""
        # Backcast and forecast networks
        x = inputs
        
        for _ in range(self.n_layers):
            x = Dense(self.n_neurons, activation='relu')(x)
            x = Dropout(self.dropout)(x)
        
        # Backcast (reconstruction)
        backcast = Dense(self.sequence_length, activation='linear', name=f'backcast_{block_id}')(x)
        
        # Forecast (prediction)
        forecast = Dense(self.forecast_length, activation='linear', name=f'forecast_{block_id}')(x)
        
        return backcast, forecast
    
    def _create_nbeats_model(self, input_shape: Tuple[int, int]) -> Model:
        """Create N-BEATS model architecture."""
        inputs = Input(shape=input_shape)
        
        # Flatten input
        x = tf.keras.layers.Flatten()(inputs)
        
        # N-BEATS blocks
        backcasts = []
        forecasts = []
        
        for i in range(self.n_blocks):
            backcast, forecast = self._create_nbeats_block(x, i)
            backcasts.append(backcast)
            forecasts.append(forecast)
            
            # Residual connection for next block
            if i < self.n_blocks - 1:
                x = tf.keras.layers.Subtract()([x, backcast])
        
        # Final forecast (sum of all block forecasts)
        final_forecast = tf.keras.layers.Add()(forecasts)
        
        model = Model(inputs=inputs, outputs=final_forecast)
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for N-BEATS."""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X) - self.forecast_length + 1):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i:i+self.forecast_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_split: float = 0.2, epochs: int = 100, 
            batch_size: int = 32, verbose: int = 0) -> 'NBEATSForecaster':
        """
        Train the N-BEATS model.
        
        Args:
            X: Feature matrix
            y: Target series
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            self
        """
        logger.info("Training N-BEATS model...")
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        y_array = y.values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_array)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Create model
        self.model = self._create_nbeats_model((self.sequence_length, X.shape[1]))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("N-BEATS model trained successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the N-BEATS model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match original length
        full_predictions = np.full(len(X), np.nan)
        full_predictions[self.sequence_length:] = predictions.flatten()
        
        # Forward fill for initial predictions
        full_predictions[:self.sequence_length] = full_predictions[self.sequence_length]
        
        return full_predictions


class NeuralModels:
    """
    Collection of advanced neural network models for electricity price forecasting.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize neural models collection.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
        # Initialize available models
        if TENSORFLOW_AVAILABLE:
            self.models['transformer'] = TransformerForecaster(random_state=random_state)
            self.models['nbeats'] = NBEATSForecaster(random_state=random_state)
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  epochs: int = 100, batch_size: int = 32, verbose: int = 0) -> Dict[str, Any]:
        """
        Train all neural network models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training neural network models...")
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                trained_model = model.fit(X_train, y_train, epochs=epochs, 
                                        batch_size=batch_size, verbose=verbose)
                self.trained_models[name] = trained_model
                logger.info(f"{name} trained successfully")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self.trained_models
    
    def predict_all(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with all trained models.
        
        Args:
            X_test: Test features
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(X_test)
                predictions[name] = pred
                logger.info(f"Predictions generated for {name}")
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        
        return predictions
    
    def evaluate_all(self, y_test: pd.Series, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Evaluate all models using multiple metrics.
        
        Args:
            y_test: True test values
            predictions: Dictionary of predictions
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for name, pred in predictions.items():
            if len(pred) != len(y_test):
                logger.warning(f"Length mismatch for {name}: pred={len(pred)}, test={len(y_test)}")
                continue
            
            # Calculate metrics
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, pred)
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
            
            # Directional accuracy
            if len(pred) > 1:
                y_diff = np.diff(y_test.values)
                pred_diff = np.diff(pred)
                directional_accuracy = np.mean((y_diff * pred_diff) > 0) * 100
            else:
                directional_accuracy = 0
            
            results.append({
                'model': name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            })
        
        self.results = pd.DataFrame(results)
        return self.results


def main():
    """Example usage of neural network models with real data."""
    print("Neural Network Models example requires real electricity price data.")
    print("Please use the ENTSO-E downloader to get real data first.")
    print("Example usage:")
    print("1. Download real data using ENTSOEDownloader")
    print("2. Preprocess data using DataPreprocessor") 
    print("3. Train models using NeuralModels.train_all()")
    print("4. Make predictions using NeuralModels.predict_all()")
    print("5. Evaluate results using NeuralModels.evaluate_all()")


if __name__ == "__main__":
    main()

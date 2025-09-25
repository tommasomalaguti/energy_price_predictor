"""
Time series models for electricity price forecasting.

This module implements classical time series models including
ARIMA, SARIMAX, and Prophet for electricity price forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logger.warning("TensorFlow not available. Deep learning models will be disabled.")


class TimeSeriesModels:
    """
    Collection of time series models for electricity price forecasting.
    
    This class provides a unified interface for training and evaluating
    various time series models including ARIMA, SARIMAX, Prophet, and LSTM.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize time series models collection.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.stationarity_results = {}
        
        # Set random seeds
        np.random.seed(random_state)
        if DEEP_LEARNING_AVAILABLE:
            tf.random.set_seed(random_state)
    
    def check_stationarity(self, series: pd.Series, test_type: str = 'adf') -> Dict[str, Any]:
        """
        Check stationarity of a time series.
        
        Args:
            series: Time series to test
            test_type: Type of test ('adf' for Augmented Dickey-Fuller, 'kpss' for KPSS)
            
        Returns:
            Dictionary with test results
        """
        if test_type == 'adf':
            result = adfuller(series.dropna())
            test_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            is_stationary = p_value < 0.05
            
        elif test_type == 'kpss':
            result = kpss(series.dropna(), regression='c')
            test_statistic = result[0]
            p_value = result[1]
            critical_values = result[3]
            
            is_stationary = p_value > 0.05
            
        else:
            raise ValueError("test_type must be 'adf' or 'kpss'")
        
        self.stationarity_results[test_type] = {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary
        }
        
        logger.info(f"{test_type.upper()} test: {'Stationary' if is_stationary else 'Non-stationary'} "
                   f"(p-value: {p_value:.4f})")
        
        return self.stationarity_results[test_type]
    
    def make_stationary(self, series: pd.Series, method: str = 'diff') -> pd.Series:
        """
        Make a time series stationary.
        
        Args:
            series: Time series to make stationary
            method: Method to use ('diff', 'log_diff', 'seasonal_diff')
            
        Returns:
            Stationary time series
        """
        if method == 'diff':
            return series.diff().dropna()
        elif method == 'log_diff':
            return np.log(series).diff().dropna()
        elif method == 'seasonal_diff':
            return series.diff(24).dropna()  # 24-hour seasonal difference
        else:
            raise ValueError("method must be 'diff', 'log_diff', or 'seasonal_diff'")
    
    def find_arima_order(self, series: pd.Series, max_p: int = 5, max_d: int = 2, 
                        max_q: int = 5) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using AIC.
        
        Args:
            series: Time series
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Tuple of (p, d, q) order
        """
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            
                    except Exception:
                        continue
        
        logger.info(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def find_sarimax_order(self, series: pd.Series, exog: Optional[pd.DataFrame] = None,
                          max_p: int = 3, max_d: int = 1, max_q: int = 3,
                          max_P: int = 2, max_D: int = 1, max_Q: int = 2,
                          seasonal_periods: int = 24) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Find optimal SARIMAX order using AIC.
        
        Args:
            series: Time series
            exog: Exogenous variables
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            max_P: Maximum seasonal AR order
            max_D: Maximum seasonal differencing order
            max_Q: Maximum seasonal MA order
            seasonal_periods: Seasonal period
            
        Returns:
            Tuple of ((p, d, q), (P, D, Q)) orders
        """
        best_aic = np.inf
        best_order = ((0, 0, 0), (0, 0, 0))
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    for P in range(max_P + 1):
                        for D in range(max_D + 1):
                            for Q in range(max_Q + 1):
                                try:
                                    model = SARIMAX(
                                        series,
                                        exog=exog,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, seasonal_periods)
                                    )
                                    fitted_model = model.fit(disp=False)
                                    aic = fitted_model.aic
                                    
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_order = ((p, d, q), (P, D, Q))
                                        
                                except Exception:
                                    continue
        
        logger.info(f"Best SARIMAX order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def train_arima(self, series: pd.Series, order: Optional[Tuple[int, int, int]] = None) -> Any:
        """
        Train ARIMA model.
        
        Args:
            series: Time series
            order: ARIMA order. If None, will be automatically determined
            
        Returns:
            Fitted ARIMA model
        """
        if order is None:
            order = self.find_arima_order(series)
        
        logger.info(f"Training ARIMA{order} model...")
        
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        self.trained_models['arima'] = fitted_model
        logger.info("ARIMA model trained successfully")
        
        return fitted_model
    
    def train_sarimax(self, series: pd.Series, exog: Optional[pd.DataFrame] = None,
                     order: Optional[Tuple[int, int, int]] = None,
                     seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Any:
        """
        Train SARIMAX model.
        
        Args:
            series: Time series
            exog: Exogenous variables
            order: ARIMA order. If None, will be automatically determined
            seasonal_order: Seasonal order. If None, will be automatically determined
            
        Returns:
            Fitted SARIMAX model
        """
        if order is None or seasonal_order is None:
            order, seasonal_order = self.find_sarimax_order(series, exog)
        
        logger.info(f"Training SARIMAX{order}x{seasonal_order} model...")
        
        model = SARIMAX(series, exog=exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        self.trained_models['sarimax'] = fitted_model
        logger.info("SARIMAX model trained successfully")
        
        return fitted_model
    
    def train_prophet(self, df: pd.DataFrame, exog_columns: Optional[List[str]] = None) -> Any:
        """
        Train Prophet model.
        
        Args:
            df: DataFrame with 'ds' (datetime) and 'y' (target) columns
            exog_columns: List of exogenous variable columns
            
        Returns:
            Fitted Prophet model
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        logger.info("Training Prophet model...")
        
        # Prepare data for Prophet
        prophet_df = df[['ds', 'y']].copy()
        
        # Add exogenous variables
        if exog_columns:
            for col in exog_columns:
                if col in df.columns:
                    prophet_df[col] = df[col]
        
        model = Prophet()
        
        # Add exogenous regressors
        if exog_columns:
            for col in exog_columns:
                if col in prophet_df.columns:
                    model.add_regressor(col)
        
        fitted_model = model.fit(prophet_df)
        
        self.trained_models['prophet'] = fitted_model
        logger.info("Prophet model trained successfully")
        
        return fitted_model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   sequence_length: int = 24, units: int = 50,
                   epochs: int = 100, batch_size: int = 32) -> Any:
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            sequence_length: Length of input sequences
            units: Number of LSTM units
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Fitted LSTM model
        """
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow is not available. Deep learning models are disabled.")
        
        logger.info("Training LSTM model...")
        
        # Prepare sequences
        def create_sequences(X, y, seq_len):
            X_seq, y_seq = [], []
            for i in range(seq_len, len(X)):
                X_seq.append(X[i-seq_len:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
        
        # Build model
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(units//2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.trained_models['lstm'] = model
        logger.info("LSTM model trained successfully")
        
        return model
    
    def predict_arima(self, steps: int) -> np.ndarray:
        """Make predictions with ARIMA model."""
        if 'arima' not in self.trained_models:
            raise ValueError("ARIMA model not trained")
        
        return self.trained_models['arima'].forecast(steps=steps)
    
    def predict_sarimax(self, steps: int, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions with SARIMAX model."""
        if 'sarimax' not in self.trained_models:
            raise ValueError("SARIMAX model not trained")
        
        return self.trained_models['sarimax'].forecast(steps=steps, exog=exog)
    
    def predict_prophet(self, periods: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Make predictions with Prophet model."""
        if 'prophet' not in self.trained_models:
            raise ValueError("Prophet model not trained")
        
        # Create future dataframe
        future = self.trained_models['prophet'].make_future_dataframe(periods=periods)
        
        # Add exogenous variables if provided
        if future_exog is not None:
            for col in future_exog.columns:
                future[col] = future_exog[col].values
        
        return self.trained_models['prophet'].predict(future)
    
    def predict_lstm(self, X: np.ndarray, sequence_length: int = 24) -> np.ndarray:
        """Make predictions with LSTM model."""
        if 'lstm' not in self.trained_models:
            raise ValueError("LSTM model not trained")
        
        # Create sequences
        X_seq = []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
        X_seq = np.array(X_seq)
        
        return self.trained_models['lstm'].predict(X_seq).flatten()
    
    def evaluate_all(self, y_test: pd.Series, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Evaluate all models using multiple metrics.
        
        Args:
            y_test: True test values
            predictions: Dictionary of predictions
            
        Returns:
            DataFrame with evaluation results
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
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
    
    def plot_diagnostics(self, model_name: str) -> None:
        """
        Plot model diagnostics.
        
        Args:
            model_name: Name of the model to diagnose
        """
        import matplotlib.pyplot as plt
        
        if model_name not in self.trained_models:
            logger.warning(f"Model {model_name} not found")
            return
        
        model = self.trained_models[model_name]
        
        if model_name in ['arima', 'sarimax']:
            # Plot residuals
            residuals = model.resid
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Residuals time series
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title('Residuals')
            axes[0, 0].grid(True)
            
            # Residuals histogram
            axes[0, 1].hist(residuals, bins=30, alpha=0.7)
            axes[0, 1].set_title('Residuals Distribution')
            axes[0, 1].grid(True)
            
            # ACF of residuals
            plot_acf(residuals, ax=axes[1, 0], lags=40)
            axes[1, 0].set_title('ACF of Residuals')
            
            # PACF of residuals
            plot_pacf(residuals, ax=axes[1, 1], lags=40)
            axes[1, 1].set_title('PACF of Residuals')
            
            plt.tight_layout()
            plt.show()
        
        elif model_name == 'prophet':
            # Plot Prophet components
            fig = model.plot_components()
            plt.show()
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all model results.
        
        Returns:
            DataFrame with model performance summary
        """
        if self.results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        return self.results.sort_values('rmse').round(4)


def main():
    """Example usage of time series models with real data."""
    print("Time Series Models example requires real electricity price data.")
    print("Please use the ENTSO-E downloader to get real data first.")
    print("Example usage:")
    print("1. Download real data using ENTSOEDownloader")
    print("2. Preprocess data using DataPreprocessor") 
    print("3. Train models using TimeSeriesModels.train_arima()")
    print("4. Make predictions using TimeSeriesModels.predict_arima()")
    print("5. Evaluate results using TimeSeriesModels.evaluate_all()")


if __name__ == "__main__":
    main()

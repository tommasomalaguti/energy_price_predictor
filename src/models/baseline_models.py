"""
Baseline models for electricity price forecasting.

This module implements simple baseline models that serve as benchmarks
for more sophisticated forecasting approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Naive forecaster that uses the last observed value as prediction.
    """
    
    def __init__(self, lag: int = 1):
        """
        Initialize naive forecaster.
        
        Args:
            lag: Number of periods to lag (default: 1 for next hour)
        """
        self.lag = lag
        self.last_value = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaiveForecaster':
        """
        Fit the naive forecaster.
        
        Args:
            X: Feature matrix (not used)
            y: Target series
            
        Returns:
            self
        """
        self.last_value = y.iloc[-self.lag] if len(y) >= self.lag else y.iloc[-1]
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the last observed value.
        
        Args:
            X: Feature matrix (not used)
            
        Returns:
            Array of predictions
        """
        if self.last_value is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return np.full(len(X), self.last_value)


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Seasonal naive forecaster that uses the value from the same period
    in the previous cycle (e.g., same hour yesterday).
    """
    
    def __init__(self, seasonal_period: int = 24):
        """
        Initialize seasonal naive forecaster.
        
        Args:
            seasonal_period: Length of seasonal cycle (default: 24 for daily)
        """
        self.seasonal_period = seasonal_period
        self.seasonal_values = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SeasonalNaiveForecaster':
        """
        Fit the seasonal naive forecaster.
        
        Args:
            X: Feature matrix (not used)
            y: Target series
            
        Returns:
            self
        """
        if len(y) < self.seasonal_period:
            # If not enough data, use simple naive
            self.seasonal_values = [y.iloc[-1]] * self.seasonal_period
        else:
            # Use the last complete seasonal cycle
            self.seasonal_values = y.iloc[-self.seasonal_period:].values
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using seasonal values.
        
        Args:
            X: Feature matrix (not used)
            
        Returns:
            Array of predictions
        """
        if self.seasonal_values is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for i in range(len(X)):
            idx = i % self.seasonal_period
            predictions.append(self.seasonal_values[idx])
        
        return np.array(predictions)


class MeanForecaster(BaseEstimator, RegressorMixin):
    """
    Mean forecaster that uses the historical mean as prediction.
    """
    
    def __init__(self):
        """Initialize mean forecaster."""
        self.mean_value = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MeanForecaster':
        """
        Fit the mean forecaster.
        
        Args:
            X: Feature matrix (not used)
            y: Target series
            
        Returns:
            self
        """
        self.mean_value = y.mean()
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the historical mean.
        
        Args:
            X: Feature matrix (not used)
            
        Returns:
            Array of predictions
        """
        if self.mean_value is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return np.full(len(X), self.mean_value)


class DriftForecaster(BaseEstimator, RegressorMixin):
    """
    Drift forecaster that extrapolates the trend from the training data.
    """
    
    def __init__(self):
        """Initialize drift forecaster."""
        self.slope = None
        self.last_value = None
        self.last_index = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DriftForecaster':
        """
        Fit the drift forecaster.
        
        Args:
            X: Feature matrix (not used)
            y: Target series
            
        Returns:
            self
        """
        if len(y) < 2:
            self.slope = 0
        else:
            # Calculate slope using first and last values
            self.slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        
        self.last_value = y.iloc[-1]
        self.last_index = len(y) - 1
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using drift extrapolation.
        
        Args:
            X: Feature matrix (not used)
            
        Returns:
            Array of predictions
        """
        if self.last_value is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for i in range(len(X)):
            steps_ahead = i + 1
            pred = self.last_value + self.slope * steps_ahead
            predictions.append(pred)
        
        return np.array(predictions)


class BaselineModels:
    """
    Collection of baseline forecasting models.
    
    This class provides a unified interface for training and evaluating
    various baseline models for electricity price forecasting.
    """
    
    def __init__(self):
        """Initialize baseline models collection."""
        self.models = {
            'naive': NaiveForecaster(lag=1),
            'naive_24h': NaiveForecaster(lag=24),
            'seasonal_naive': SeasonalNaiveForecaster(seasonal_period=24),
            'seasonal_naive_weekly': SeasonalNaiveForecaster(seasonal_period=168),
            'mean': MeanForecaster(),
            'drift': DriftForecaster()
        }
        
        self.trained_models = {}
        self.results = {}
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, BaseEstimator]:
        """
        Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training baseline models...")
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                trained_model = model.fit(X_train, y_train)
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
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, BaseEstimator]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison ('rmse', 'mae', 'mape')
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        if self.results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        best_idx = self.results[metric].idxmin()
        best_model_name = self.results.loc[best_idx, 'model']
        best_model = self.trained_models[best_model_name]
        
        return best_model_name, best_model
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all model results.
        
        Returns:
            DataFrame with model performance summary
        """
        if self.results.empty:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")
        
        return self.results.sort_values('rmse').round(4)
    
    def plot_predictions(self, y_test: pd.Series, predictions: Dict[str, np.ndarray], 
                        title: str = "Baseline Model Predictions", 
                        max_plot_points: int = 1000) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            y_test: True test values
            predictions: Dictionary of predictions
            title: Plot title
            max_plot_points: Maximum number of points to plot
        """
        import matplotlib.pyplot as plt
        
        # Limit points for plotting
        plot_indices = np.linspace(0, len(y_test)-1, min(max_plot_points, len(y_test)), dtype=int)
        
        plt.figure(figsize=(15, 10))
        
        # Plot actual values
        plt.subplot(2, 1, 1)
        plt.plot(plot_indices, y_test.iloc[plot_indices], label='Actual', alpha=0.7)
        
        # Plot predictions
        for name, pred in predictions.items():
            plt.plot(plot_indices, pred[plot_indices], label=name, alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Time Index')
        plt.ylabel('Price (€/MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 1, 2)
        for name, pred in predictions.items():
            residuals = y_test.iloc[plot_indices] - pred[plot_indices]
            plt.plot(plot_indices, residuals, label=f'{name} residuals', alpha=0.7)
        
        plt.title('Prediction Residuals')
        plt.xlabel('Time Index')
        plt.ylabel('Residual (€/MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of baseline models."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic electricity price data with trend and seasonality
    t = np.arange(n_samples)
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily seasonality
    noise = np.random.normal(0, 2, n_samples)
    price = 50 + trend + seasonal + noise
    
    # Create features and target
    X = pd.DataFrame({'feature': np.random.randn(n_samples)})
    y = pd.Series(price, name='price')
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train and evaluate baseline models
    baseline = BaselineModels()
    baseline.train_all(X_train, y_train)
    predictions = baseline.predict_all(X_test)
    results = baseline.evaluate_all(y_test, predictions)
    
    print("Baseline Model Results:")
    print(results)
    
    # Get best model
    best_name, best_model = baseline.get_best_model('rmse')
    print(f"\nBest model: {best_name}")
    
    # Plot results
    baseline.plot_predictions(y_test, predictions)


if __name__ == "__main__":
    main()

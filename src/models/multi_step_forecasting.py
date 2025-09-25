"""
Multi-step ahead forecasting for electricity price prediction.

This module provides methods for forecasting multiple time steps ahead
using various strategies including direct, recursive, and multi-output approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiStepForecaster:
    """
    Multi-step ahead forecasting using various strategies.
    """
    
    def __init__(self, base_model: BaseEstimator, strategy: str = 'recursive',
                 max_horizon: int = 24, random_state: int = 42):
        """
        Initialize multi-step forecaster.
        
        Args:
            base_model: Base forecasting model
            strategy: Forecasting strategy ('recursive', 'direct', 'multi_output')
            max_horizon: Maximum forecasting horizon
            random_state: Random state for reproducibility
        """
        self.base_model = base_model
        self.strategy = strategy
        self.max_horizon = max_horizon
        self.random_state = random_state
        self.trained_models = {}
        self.is_fitted = False
        
    def _create_lagged_features(self, X: pd.DataFrame, y: pd.Series, 
                               max_lag: int = 24) -> Tuple[pd.DataFrame, pd.Series]:
        """Create lagged features for multi-step forecasting."""
        # Create lagged features
        lagged_X = X.copy()
        
        for lag in range(1, max_lag + 1):
            lagged_X[f'y_lag_{lag}'] = y.shift(lag)
        
        # Remove rows with NaN values
        lagged_X = lagged_X.dropna()
        y_clean = y.iloc[len(y) - len(lagged_X):]
        
        return lagged_X, y_clean
    
    def _recursive_strategy(self, X: pd.DataFrame, y: pd.Series) -> Dict[int, BaseEstimator]:
        """Train models using recursive strategy."""
        models = {}
        
        for horizon in range(1, self.max_horizon + 1):
            logger.info(f"Training recursive model for horizon {horizon}")
            
            # Create target shifted by horizon
            y_shifted = y.shift(-horizon)
            
            # Remove NaN values
            valid_idx = ~y_shifted.isna()
            X_valid = X[valid_idx]
            y_valid = y_shifted[valid_idx]
            
            if len(X_valid) == 0:
                logger.warning(f"No valid data for horizon {horizon}")
                continue
            
            # Train model
            model = type(self.base_model)(**self.base_model.get_params())
            model.fit(X_valid, y_valid)
            models[horizon] = model
        
        return models
    
    def _direct_strategy(self, X: pd.DataFrame, y: pd.Series) -> Dict[int, BaseEstimator]:
        """Train models using direct strategy."""
        models = {}
        
        for horizon in range(1, self.max_horizon + 1):
            logger.info(f"Training direct model for horizon {horizon}")
            
            # Create target shifted by horizon
            y_shifted = y.shift(-horizon)
            
            # Remove NaN values
            valid_idx = ~y_shifted.isna()
            X_valid = X[valid_idx]
            y_valid = y_shifted[valid_idx]
            
            if len(X_valid) == 0:
                logger.warning(f"No valid data for horizon {horizon}")
                continue
            
            # Train model
            model = type(self.base_model)(**self.base_model.get_params())
            model.fit(X_valid, y_valid)
            models[horizon] = model
        
        return models
    
    def _multi_output_strategy(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """Train model using multi-output strategy."""
        logger.info("Training multi-output model")
        
        # Create multi-output targets
        y_multi = []
        for horizon in range(1, self.max_horizon + 1):
            y_shifted = y.shift(-horizon)
            y_multi.append(y_shifted)
        
        y_multi = pd.DataFrame(y_multi).T
        y_multi.columns = [f'horizon_{h}' for h in range(1, self.max_horizon + 1)]
        
        # Remove rows with any NaN values
        valid_idx = ~y_multi.isna().any(axis=1)
        X_valid = X[valid_idx]
        y_valid = y_multi[valid_idx]
        
        if len(X_valid) == 0:
            raise ValueError("No valid data for multi-output training")
        
        # Train multi-output model
        multi_model = MultiOutputRegressor(self.base_model)
        multi_model.fit(X_valid, y_valid)
        
        return multi_model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MultiStepForecaster':
        """
        Train multi-step forecasting models.
        
        Args:
            X: Feature matrix
            y: Target series
            
        Returns:
            self
        """
        logger.info(f"Training multi-step forecaster using {self.strategy} strategy")
        
        if self.strategy == 'recursive':
            self.trained_models = self._recursive_strategy(X, y)
        elif self.strategy == 'direct':
            self.trained_models = self._direct_strategy(X, y)
        elif self.strategy == 'multi_output':
            self.trained_models = {0: self._multi_output_strategy(X, y)}
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.is_fitted = True
        logger.info("Multi-step forecaster trained successfully")
        return self
    
    def predict(self, X: pd.DataFrame, horizon: Optional[int] = None) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Make multi-step ahead predictions.
        
        Args:
            X: Feature matrix
            horizon: Specific horizon to predict (if None, predicts all horizons)
            
        Returns:
            Predictions for specified horizon(s)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.strategy == 'multi_output':
            # Multi-output prediction
            model = self.trained_models[0]
            predictions = model.predict(X)
            
            if horizon is not None:
                return predictions[:, horizon - 1]
            else:
                return {h + 1: predictions[:, h] for h in range(self.max_horizon)}
        
        else:
            # Recursive or direct prediction
            if horizon is not None:
                if horizon not in self.trained_models:
                    raise ValueError(f"No model trained for horizon {horizon}")
                return self.trained_models[horizon].predict(X)
            else:
                predictions = {}
                for h, model in self.trained_models.items():
                    predictions[h] = model.predict(X)
                return predictions
    
    def predict_recursive(self, X: pd.DataFrame, initial_y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make recursive predictions using the trained model.
        
        Args:
            X: Feature matrix
            initial_y: Initial values for recursive prediction
            
        Returns:
            Recursive predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use the first horizon model for recursive prediction
        if 1 not in self.trained_models:
            raise ValueError("No model trained for horizon 1")
        
        model = self.trained_models[1]
        predictions = np.zeros((len(X), self.max_horizon))
        
        # Initialize with first prediction
        current_X = X.copy()
        if initial_y is not None:
            current_X['y_lag_1'] = initial_y[-1] if len(initial_y) > 0 else 0
        
        for h in range(self.max_horizon):
            # Make prediction
            pred = model.predict(current_X)
            predictions[:, h] = pred
            
            # Update features for next step
            if h < self.max_horizon - 1:
                current_X['y_lag_1'] = pred
        
        return predictions


class MultiStepEnsemble:
    """
    Ensemble of multi-step forecasters for improved accuracy.
    """
    
    def __init__(self, base_models: List[BaseEstimator], strategies: List[str] = None,
                 max_horizon: int = 24, random_state: int = 42):
        """
        Initialize multi-step ensemble.
        
        Args:
            base_models: List of base models
            strategies: List of forecasting strategies
            max_horizon: Maximum forecasting horizon
            random_state: Random state for reproducibility
        """
        self.base_models = base_models
        self.strategies = strategies or ['recursive', 'direct']
        self.max_horizon = max_horizon
        self.random_state = random_state
        self.forecasters = {}
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MultiStepEnsemble':
        """
        Train ensemble of multi-step forecasters.
        
        Args:
            X: Feature matrix
            y: Target series
            
        Returns:
            self
        """
        logger.info("Training multi-step ensemble")
        
        for i, base_model in enumerate(self.base_models):
            for strategy in self.strategies:
                forecaster = MultiStepForecaster(
                    base_model=base_model,
                    strategy=strategy,
                    max_horizon=self.max_horizon,
                    random_state=self.random_state
                )
                forecaster.fit(X, y)
                
                name = f"{type(base_model).__name__}_{strategy}_{i}"
                self.forecasters[name] = forecaster
        
        self.is_fitted = True
        logger.info("Multi-step ensemble trained successfully")
        return self
    
    def predict(self, X: pd.DataFrame, horizon: int, 
                ensemble_method: str = 'mean') -> np.ndarray:
        """
        Make ensemble predictions for a specific horizon.
        
        Args:
            X: Feature matrix
            horizon: Forecasting horizon
            ensemble_method: Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        
        for name, forecaster in self.forecasters.items():
            try:
                pred = forecaster.predict(X, horizon=horizon)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any forecaster")
        
        predictions = np.array(predictions)
        
        if ensemble_method == 'mean':
            return np.mean(predictions, axis=0)
        elif ensemble_method == 'median':
            return np.median(predictions, axis=0)
        elif ensemble_method == 'weighted':
            # Simple weighting based on prediction variance
            weights = 1.0 / (np.var(predictions, axis=0) + 1e-8)
            weights = weights / np.sum(weights)
            return np.average(predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def predict_all_horizons(self, X: pd.DataFrame, 
                           ensemble_method: str = 'mean') -> Dict[int, np.ndarray]:
        """
        Make predictions for all horizons.
        
        Args:
            X: Feature matrix
            ensemble_method: Ensemble method
            
        Returns:
            Dictionary of predictions for each horizon
        """
        predictions = {}
        
        for horizon in range(1, self.max_horizon + 1):
            predictions[horizon] = self.predict(X, horizon, ensemble_method)
        
        return predictions


class MultiStepEvaluator:
    """
    Evaluation metrics for multi-step ahead forecasting.
    """
    
    def __init__(self):
        """Initialize multi-step evaluator."""
        pass
    
    def evaluate_horizon(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        horizon: int) -> Dict[str, float]:
        """
        Evaluate predictions for a specific horizon.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            horizon: Forecasting horizon
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'horizon': horizon,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def evaluate_all_horizons(self, y_true: Dict[int, np.ndarray], 
                            y_pred: Dict[int, np.ndarray]) -> pd.DataFrame:
        """
        Evaluate predictions for all horizons.
        
        Args:
            y_true: Dictionary of true values for each horizon
            y_pred: Dictionary of predicted values for each horizon
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for horizon in y_true.keys():
            if horizon in y_pred:
                metrics = self.evaluate_horizon(y_true[horizon], y_pred[horizon], horizon)
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_horizon_performance(self, results_df: pd.DataFrame, 
                                metric: str = 'rmse') -> None:
        """
        Plot performance across horizons.
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['horizon'], results_df[metric], 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Forecasting Horizon (hours)')
        plt.ylabel(metric.upper())
        plt.title(f'Forecasting Performance Across Horizons')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of multi-step forecasting with real data."""
    print("Multi-step Forecasting example requires real electricity price data.")
    print("Please use the ENTSO-E downloader to get real data first.")
    print("Example usage:")
    print("1. Download real data using ENTSOEDownloader")
    print("2. Preprocess data using DataPreprocessor") 
    print("3. Create MultiStepForecaster with base model")
    print("4. Train using fit() method")
    print("5. Make predictions using predict() method")
    print("6. Evaluate using MultiStepEvaluator")


if __name__ == "__main__":
    main()

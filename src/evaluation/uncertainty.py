"""
Uncertainty quantification for electricity price forecasting models.

This module provides methods for calculating prediction intervals,
confidence intervals, and uncertainty measures for forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Class for quantifying uncertainty in electricity price forecasts.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize uncertainty quantifier.
        
        Args:
            confidence_level: Confidence level for prediction intervals (0-1)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def bootstrap_prediction_intervals(
        self, 
        model, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using bootstrap resampling.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        n_samples = len(X_test)
        predictions = np.zeros((n_bootstrap, n_samples))
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train.iloc[indices]
            y_boot = y_train.iloc[indices]
            
            # Train model on bootstrap sample
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_boot, y_boot)
            
            # Make predictions
            predictions[i] = model_copy.predict(X_test)
        
        # Calculate statistics
        mean_predictions = np.mean(predictions, axis=0)
        lower_bounds = np.percentile(predictions, (self.alpha / 2) * 100, axis=0)
        upper_bounds = np.percentile(predictions, (1 - self.alpha / 2) * 100, axis=0)
        
        return mean_predictions, lower_bounds, upper_bounds
    
    def quantile_regression_intervals(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame,
        quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals using quantile regression.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            quantiles: List of quantiles to calculate
            
        Returns:
            Dictionary with quantile predictions
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        quantile_predictions = {}
        
        for quantile in quantiles:
            # Train quantile regressor
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=100,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            quantile_predictions[f'q{int(quantile*100)}'] = model.predict(X_test)
        
        return quantile_predictions
    
    def conformal_prediction_intervals(
        self, 
        model, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame,
        X_cal: pd.DataFrame,
        y_cal: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using conformal prediction.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            X_cal: Calibration features
            y_cal: Calibration targets
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions on calibration set
        cal_predictions = model.predict(X_cal)
        cal_errors = np.abs(y_cal - cal_predictions)
        
        # Calculate conformal quantile
        conformal_quantile = np.quantile(cal_errors, self.confidence_level)
        
        # Make predictions on test set
        test_predictions = model.predict(X_test)
        
        # Calculate intervals
        lower_bounds = test_predictions - conformal_quantile
        upper_bounds = test_predictions + conformal_quantile
        
        return test_predictions, lower_bounds, upper_bounds
    
    def ensemble_uncertainty(
        self, 
        models: Dict[str, any], 
        X_test: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Calculate uncertainty using ensemble of models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            
        Returns:
            Dictionary with ensemble statistics
        """
        predictions = {}
        
        # Get predictions from all models
        for name, model in models.items():
            try:
                predictions[name] = model.predict(X_test)
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Calculate ensemble statistics
        pred_array = np.array(list(predictions.values()))
        
        ensemble_stats = {
            'mean': np.mean(pred_array, axis=0),
            'std': np.std(pred_array, axis=0),
            'min': np.min(pred_array, axis=0),
            'max': np.max(pred_array, axis=0),
            'median': np.median(pred_array, axis=0),
            'q25': np.percentile(pred_array, 25, axis=0),
            'q75': np.percentile(pred_array, 75, axis=0)
        }
        
        # Calculate confidence intervals
        ensemble_stats['lower_bound'] = ensemble_stats['mean'] - 1.96 * ensemble_stats['std']
        ensemble_stats['upper_bound'] = ensemble_stats['mean'] + 1.96 * ensemble_stats['std']
        
        return ensemble_stats
    
    def monte_carlo_dropout_intervals(
        self, 
        model, 
        X_test: pd.DataFrame, 
        n_samples: int = 100,
        dropout_rate: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using Monte Carlo dropout.
        
        Note: This method is designed for neural networks with dropout layers.
        
        Args:
            model: Trained neural network model
            X_test: Test features
            n_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate for uncertainty estimation
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        # This is a placeholder for Monte Carlo dropout
        # In practice, you would need a neural network with dropout layers
        logger.warning("Monte Carlo dropout requires a neural network with dropout layers")
        
        # Fallback to simple prediction
        predictions = model.predict(X_test)
        std = np.std(predictions) * 0.1  # Simple uncertainty estimate
        
        lower_bounds = predictions - 1.96 * std
        upper_bounds = predictions + 1.96 * std
        
        return predictions, lower_bounds, upper_bounds
    
    def calculate_prediction_uncertainty(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_lower: np.ndarray, 
        y_pred_upper: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate uncertainty metrics for prediction intervals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_lower: Lower bounds of prediction intervals
            y_pred_upper: Upper bounds of prediction intervals
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Coverage probability
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        
        # Average interval width
        interval_width = np.mean(y_pred_upper - y_pred_lower)
        
        # Mean absolute deviation from interval center
        interval_center = (y_pred_upper + y_pred_lower) / 2
        mad_center = np.mean(np.abs(y_pred - interval_center))
        
        # Prediction interval score (PIS)
        pis = np.mean(
            (y_pred_upper - y_pred_lower) + 
            (2 / self.alpha) * (
                (y_pred_lower - y_true) * (y_true < y_pred_lower) +
                (y_true - y_pred_upper) * (y_true > y_pred_upper)
            )
        )
        
        return {
            'coverage_probability': coverage,
            'average_interval_width': interval_width,
            'mad_from_center': mad_center,
            'prediction_interval_score': pis,
            'target_coverage': self.confidence_level
        }
    
    def plot_uncertainty_intervals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_lower: np.ndarray, 
        y_upper: np.ndarray,
        title: str = "Prediction Intervals"
    ) -> None:
        """
        Plot prediction intervals with uncertainty visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_lower: Lower bounds
            y_upper: Upper bounds
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data
        x = np.arange(len(y_true))
        
        # Plot prediction intervals
        ax.fill_between(x, y_lower, y_upper, alpha=0.3, color='blue', 
                       label=f'{int(self.confidence_level*100)}% Prediction Interval')
        
        # Plot predictions and true values
        ax.plot(x, y_pred, 'b-', label='Predictions', linewidth=2)
        ax.plot(x, y_true, 'r-', label='True Values', linewidth=2)
        
        # Calculate and display coverage
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Price (€/MWh)')
        ax.set_title(f'{title}\nCoverage: {coverage:.1%}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of uncertainty quantification."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    # Create target with some noise
    y = 2 * X['feature1'] + 3 * X['feature2'] + np.random.randn(n_samples) * 0.5
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train a simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize uncertainty quantifier
    uq = UncertaintyQuantifier(confidence_level=0.95)
    
    # Calculate bootstrap prediction intervals
    pred_mean, pred_lower, pred_upper = uq.bootstrap_prediction_intervals(
        model, X_train, y_train, X_test, n_bootstrap=50
    )
    
    # Calculate uncertainty metrics
    uncertainty_metrics = uq.calculate_prediction_uncertainty(
        y_test.values, pred_mean, pred_lower, pred_upper
    )
    
    print("Uncertainty Metrics:")
    for metric, value in uncertainty_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    uq.plot_uncertainty_intervals(
        y_test.values, pred_mean, pred_lower, pred_upper,
        title="Bootstrap Prediction Intervals"
    )


if __name__ == "__main__":
    main()
